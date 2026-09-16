"""Real image-conditioned 9B PSD updates and same-horizon optimizer resume.

Batch two complete admitted datums. This is not the DP4/global-batch32 gate.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('datums','datum-manifest','serving-profile','checkpoint-manifest','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deterministic-attention',action='store_true')
    args=p.parse_args()
    if args.deterministic_attention:
        os.environ['FLASH_ATTENTION_DETERMINISTIC']='1'
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    from peft import get_peft_model,LoraConfig,PeftModel
    from ifv_training.io import load_json,load_jsonl,write_json,sha256_file
    from ifv_training.psd_preflight import verify_psd_training_input
    from ifv_training.psd_initialization import verify_initialization
    from ifv_training.psd_ms_swift import install_ms_swift_psd_plugin,prepare_psd_logits_to_keep,sparse_topk_cross_entropy
    from swift.template.register import TEMPLATE_MAPPING
    started=time.monotonic();device='cuda:0'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    profile=load_json(args.serving_profile);base=profile.get('engine_model_path') or profile['model_path']
    assert not profile.get('adapter_path'), 'This diagnostic is bound to full epoch3 export'
    initialization=verify_initialization(datum_manifest_path=args.datum_manifest,
        serving_profile_path=args.serving_profile,checkpoint_manifest_path=args.checkpoint_manifest,
        model_path=base,adapter_path=None)
    assert verify_psd_training_input(datums_path=args.datums,manifest_path=args.datum_manifest,
        expected_topk=20,max_context=131072)['passed']
    rows=load_jsonl(args.datums)
    selected=[min((r for r in rows if r['kind']==kind),key=lambda r:(len(r['input_ids']),r['target_id']))
              for kind in ('repair','preserve')]
    args.output_dir.mkdir(exist_ok=False)
    write_json(args.output_dir/'selection.json',{'source_sha256':sha256_file(args.datums),
        'rule':'shortest complete admitted datum per kind, batch two; no truncation',
        'targets':[{'target_id':r['target_id'],'kind':r['kind'],'tokens':len(r['input_ids'])} for r in selected],
        'production_global_batch_tested':False})
    install_ms_swift_psd_plugin()
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    def load_base():
        return AutoModelForImageTextToText.from_pretrained(base,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device)
    def configure(model):
        model.train();model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),
            lr=4e-5,betas=(.9,.95),eps=1e-12,weight_decay=0)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        return optimizer,scheduler
    def snapshot(model):
        return {name:v.detach().cpu().clone() for name,v in model.named_parameters() if v.requires_grad}
    def fingerprint(state):
        h=hashlib.sha256()
        for name,v in state.items():h.update(name.encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    def step(model,optimizer,scheduler,phase):
        write_json(args.output_dir/'progress.json',{'phase':phase,'seconds':time.monotonic()-started})
        cls=TEMPLATE_MAPPING['ifv_psd_topk'].template_cls
        template=cls.__new__(cls)
        template.processor,template.padding_free,template.sequence_parallel_size=processor,False,1
        template._get_get_rope_index=lambda:model.get_base_model().model.get_rope_index
        batch=template.data_collator([template.encode(r) for r in selected])
        prepare_psd_logits_to_keep(batch)
        batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        tokens,weights=batch.pop('psd_target_tokens'),batch.pop('psd_weights');batch.pop('labels')
        optimizer.zero_grad(set_to_none=True)
        output=model(**batch,use_cache=False)
        loss=sparse_topk_cross_entropy(output,psd_target_tokens=tokens,psd_weights=weights)
        assert torch.isfinite(loss),'Nonfinite actual GPU PSD loss'
        loss.backward()
        visual=sum(float(v.grad.abs().sum()) for name,v in model.named_parameters() if 'visual' in name and v.grad is not None)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
        return {'loss':float(loss.detach()),'gradient_norm':float(norm),'visual_gradient_l1':visual,
                'max_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
    model=get_peft_model(load_base(),LoraConfig(r=32,lora_alpha=32,lora_dropout=0,
        target_modules='all-linear',task_type='CAUSAL_LM'))
    optimizer,scheduler=configure(model)
    before=fingerprint(snapshot(model));first=step(model,optimizer,scheduler,'first_gpu_update')
    updated=fingerprint(snapshot(model));assert before!=updated and first['visual_gradient_l1']>0
    checkpoint=args.output_dir/'checkpoint-1';model.save_pretrained(checkpoint)
    torch.save(optimizer.state_dict(),checkpoint/'optimizer.pt');torch.save(scheduler.state_dict(),checkpoint/'scheduler.pt')
    torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},checkpoint/'rng_state.pth')
    write_json(checkpoint/'trainer_state.json',{'global_step':1,'scope':'diagnostic_batch_two','max_steps':2})
    write_json(args.output_dir/'first-step.json',first)
    second=step(model,optimizer,scheduler,'uninterrupted_second_gpu_update');expected=snapshot(model)
    del model,optimizer,scheduler;gc.collect();torch.cuda.empty_cache()
    model=PeftModel.from_pretrained(load_base(),checkpoint,is_trainable=True)
    optimizer,scheduler=configure(model)
    assert fingerprint(snapshot(model))==updated and not optimizer.state, 'Restored adapter mismatch'
    optimizer.load_state_dict(torch.load(checkpoint/'optimizer.pt',weights_only=True,map_location=device))
    scheduler.load_state_dict(torch.load(checkpoint/'scheduler.pt',weights_only=True))
    rng=torch.load(checkpoint/'rng_state.pth',weights_only=True)
    torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state_all(rng['cuda'])
    resumed=step(model,optimizer,scheduler,'resumed_second_gpu_update');actual=snapshot(model)
    diffs={name:float((actual[name]-v).abs().max()) for name,v in expected.items()}
    exact=fingerprint(actual)==fingerprint(expected)
    # GPU kernels may differ at round-off scale; report, never disguise as exact.
    close=all(torch.allclose(actual[name],v,rtol=1e-5,atol=1e-7) for name,v in expected.items())
    passed=close and abs(resumed['loss']-second['loss'])<=max(1e-4,abs(second['loss'])*1e-5)
    result={'passed':passed,'scope':'real_9b_gpu_batch2_sparse_psd_optimizer_resume',
        'flash_attention_deterministic':os.environ.get('FLASH_ATTENTION_DETERMINISTIC')=='1',
        'production_round_completed':False,'dp4_global_batch32_verified':False,
        'initialization':initialization,'first':first,'second':second,'resumed':resumed,
        'resume_bitwise_exact':exact,'resume_weights_close_rtol1e5_atol1e7':close,
        'max_adapter_absolute_difference':max(diffs.values()),'seconds':time.monotonic()-started}
    write_json(args.output_dir/'result.json',result);print(json.dumps(result),flush=True)
    assert passed,'Real GPU resume differs beyond documented numerical tolerance'


if __name__=='__main__':main()
