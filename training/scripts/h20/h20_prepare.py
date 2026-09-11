import json, os, statistics, hashlib, importlib.util
from pathlib import Path
root=Path(os.environ['IFV_H20_ROOT'])
pkg=Path(os.environ['IFV_SFT_PACKAGE'])
out=Path(os.environ['IFV_H20_RUN_ROOT'])
out.mkdir(exist_ok=True,parents=True)
src=pkg/'ms-swift-policy'
rows=[json.loads(l) for l in (src/'train.jsonl').open() if l.strip()]
print('rows',len(rows),'first_images',rows[0]['images'][:2],flush=True)
image_map={p.name:p for p in (pkg/'images').rglob('*') if p.is_file()}
missing=[]
for row in rows:
    paths=[]
    for s in row['images']:
        p=Path(s)
        if not p.is_file():
            p=(pkg/p).resolve() if not p.is_absolute() else image_map.get(p.name,p)
        if not p.is_file(): p=image_map.get(Path(s).name,p)
        if not p.is_file(): missing.append(s)
        paths.append(str(p))
    row['images']=paths
print('missing_images',len(missing), 'examples',missing[:3],flush=True)
assert not missing
from swift import get_processor, get_template
processor=get_processor(str(root/'models/Qwen3.5-9B-local'),model_type='qwen3_5')
template=get_template(processor,max_length=131072,truncation_strategy='raise',max_pixels=262144,padding_free=False,loss_scale='ignore_empty_think',enable_thinking=False,add_non_thinking_prefix=False)
template.set_mode('train')
spec=importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'probe/verify_ms_swift_agent_dataset.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
# Stratified by raw episode size, preserve complete episodes; only train split is sampled.
order=sorted(range(len(rows)),key=lambda i:sum(len(m['content']) for m in rows[i]['messages']))
selected=[]; stats=[];long_rows=[]
positions=sorted(set(list(range(0,min(100,len(order)),3))+[int((len(order)-1)*q) for q in [.25,.5,.75,.9,.95,.99,1.0]]))
for pos in positions:
    i=order[pos]; row=rows[i]
    try:
        probe._validate_roles(row['messages'],kind='policy')
        encoded=template.encode(row,return_template_inputs=True)
        ids=probe._as_list(encoded['input_ids']);labels=probe._as_list(encoded['labels'])
        decoded=processor.tokenizer.decode(ids,skip_special_tokens=False)
        probe._verify_rendered_tool_calls(row['messages'],decoded)
        assert any(x!=-100 for x in labels)
        assert probe._contains_supervised_subsequence(ids,labels,probe._encode(processor.tokenizer,'<think>'))
        for m in row['messages']:
            if m['role']=='tool_response':
                v=probe._encode(processor.tokenizer,probe._text_for_presence_check(m['content']))
                assert not v or probe._contains_subsequence(ids,v)
        stat={'source_row':i,'size_rank':pos,'tokens':len(ids),'labels':sum(x!=-100 for x in labels),'images':len(row['images'])}
        stats.append(stat)
        if pos>=644: long_rows.append(row)
        if len(ids)<=16384 and len(selected)<32: selected.append((row,stat))
        print(stat,flush=True)
    except Exception as e:
        stats.append({'source_row':i,'error':str(e)});print('ERROR',i,str(e),flush=True)
assert len(selected)>=8, len(selected)
with (out/'train.jsonl').open('w') as f:
    for row,s in selected: f.write(json.dumps(row,ensure_ascii=False)+'\n')
(out/'data-audit.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256((src/'train.jsonl').read_bytes()).hexdigest(),'missing_images':0,'selection':'short complete train episodes, no truncation, not a full-dataset benchmark','selected': [s for _,s in selected],'stratified_probe':stats},indent=2))
(out/'long-real.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in long_rows))
print('PREPARED',len(selected),flush=True)
