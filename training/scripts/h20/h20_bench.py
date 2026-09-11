import subprocess,os,time,json,sys,threading
from pathlib import Path
root=Path(os.environ['IFV_H20_ROOT']);out=Path(os.environ['IFV_H20_RUN_ROOT'])
batch=int(sys.argv[1]) if len(sys.argv)>1 else 1
name=sys.argv[2] if len(sys.argv)>2 else f'sdpa-zero2-b{batch}'
run=out/name;run.mkdir(exist_ok=False)
args=['swift','sft','--model',str(root/'models/Qwen3.5-9B-local'),'--model_type','qwen3_5','--tuner_type','full','--dataset',str(out/'train.jsonl'),'--split_dataset_ratio','0','--torch_dtype','bfloat16','--bf16','true','--freeze_llm','false','--freeze_vit','false','--freeze_aligner','false','--deepspeed','zero2','--attn_impl','sdpa','--gradient_checkpointing','true','--vit_gradient_checkpointing','true','--per_device_train_batch_size',str(batch),'--gradient_accumulation_steps','1','--max_steps','6','--learning_rate','1e-5','--warmup_ratio','0','--max_length','16384','--truncation_strategy','raise','--group_by_length','true','--padding_free','false','--packing','false','--loss_scale','ignore_empty_think','--enable_thinking','false','--add_non_thinking_prefix','false','--use_logits_to_keep','true','--dataset_num_proc','4','--dataloader_num_workers','4','--dataloader_persistent_workers','true','--load_from_cache_file','true','--logging_steps','1','--eval_strategy','no','--save_strategy','no','--report_to','none','--output_dir',str(run/'output'),'--add_version','false','--seed','42']
def setarg(k,v):
    if k in args: args[args.index(k)+1]=str(v)
    else: args.extend([k,str(v)])
# H20-specific 128K baseline; no A100 profile is sourced.
i=args.index('--deepspeed');del args[i:i+2]
setarg('--fsdp','fsdp2')
if os.environ.get('H20_RESHARD')=='false':
    config={'fsdp':'shard_grad_op auto_wrap','fsdp_config':{'fsdp_version':2,'reshard_after_forward':False,'auto_wrap_policy':'TRANSFORMER_BASED_WRAP','cpu_ram_efficient_loading':True,'state_dict_type':'SHARDED_STATE_DICT','activation_checkpointing':True}}
    (run/'fsdp.json').write_text(json.dumps(config,indent=2))
    setarg('--fsdp',run/'fsdp.json')
setarg('--gradient_checkpointing','false')
setarg('--vit_gradient_checkpointing','false')
setarg('--attn_impl','flash_attn')
setarg('--max_length',131072)
setarg('--truncation_strategy','delete')
setarg('--padding_free','true')
setarg('--sequence_parallel_size',os.environ.get('H20_SP','4'))
setarg('--dataset',str(out/os.environ.get('H20_DATA','train.jsonl')))
setarg('--max_steps',os.environ.get('H20_STEPS','6'))
setarg('--lazy_tokenize','false')
setarg('--use_liger_kernel','true')
setarg('--use_logits_to_keep','false')
setarg('--optim','adamw_torch_fused')
setarg('--include_num_input_tokens_seen','true')
setarg('--max_pixels','262144')
if os.environ.get('H20_PACKING')=='true':
    setarg('--packing','true')
    setarg('--packing_length',os.environ.get('H20_PACKING_LENGTH','120000'))
if os.environ.get('H20_SAVE')=='1':
    setarg('--save_strategy','steps')
    setarg('--save_steps',os.environ.get('H20_STEPS','6'))
    setarg('--save_total_limit','1')
if os.environ.get('H20_RESUME'):
    setarg('--resume_from_checkpoint',os.environ['H20_RESUME'])
env=dict(os.environ,NPROC_PER_NODE='4',CUDA_VISIBLE_DEVICES='0,1,2,3',MASTER_PORT='29641')
(run/'command.json').write_text(json.dumps(args,indent=2))
(run/'provenance.json').write_text(json.dumps({'sequence_parallel_size':int(env.get('H20_SP','4')),'data':args[args.index('--dataset')+1],'max_length':131072,'commit':subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[3]),'rev-parse','HEAD'],text=True).strip()},indent=2))
start=time.time()
with (run/'train.log').open('w') as log,(run/'gpu.csv').open('w') as gpu:
    p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
    (run/'pid').write_text(str(p.pid))
    while p.poll() is None:
        r=subprocess.run(['nvidia-smi','--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
        gpu.write(r.stdout);gpu.flush();time.sleep(3)
    (run/'result.json').write_text(json.dumps({'exit_code':p.returncode,'wall_seconds':time.time()-start,'batch_per_device':batch,'gpus':4},indent=2))
print('BENCHMARK_EXIT',p.returncode,flush=True)
sys.exit(p.returncode)
