import json, os
from pathlib import Path
from swift import get_processor,get_template
root=Path(os.environ['IFV_H20_ROOT']);out=Path(os.environ['IFV_H20_RUN_ROOT'])
p=get_processor(str(root/'models/Qwen3.5-9B-local'),model_type='qwen3_5')
t=get_template(p,max_length=131072,truncation_strategy='raise',max_pixels=262144,padding_free=True,loss_scale='ignore_empty_think',enable_thinking=False,add_non_thinking_prefix=False)
t.set_mode('train')
image=json.loads((out/'train.jsonl').read_text().splitlines()[0])['images'][0]
# Explicitly synthetic stress input, never included in production SFT.
unit='This is synthetic context for GPU capacity measurement only. '
response='<think>\n'+('This is a synthetic supervised training token sequence. '*2200)+'\n</think>\n<answer>capacity test only</answer>'
def make(n):
 return {'messages':[{'role':'user','content':'<image>\n'+unit*n},{'role':'assistant','content':response}],'images':[image],'tools':[]}
lo,hi=1,14000
while lo<hi:
 mid=(lo+hi+1)//2
 try: n=len(t.encode(make(mid))['input_ids'])
 except Exception: n=10**9
 if n<=130000:lo=mid
 else:hi=mid-1
row=make(lo);enc=t.encode(row)
stats={'synthetic':True,'purpose':'single-sequence near-128K capacity, not model quality','input_tokens':len(enc['input_ids']),'supervised_tokens':sum(v!=-100 for v in enc['labels'])}
assert 128000<=stats['input_tokens']<=131072
(out/'capacity-synthetic.jsonl').write_text((json.dumps(row)+'\n')*8)
(out/'capacity-data.json').write_text(json.dumps(stats,indent=2))
print(stats,flush=True)
