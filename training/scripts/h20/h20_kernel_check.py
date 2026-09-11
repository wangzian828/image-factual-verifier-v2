import json,os,torch
from pathlib import Path
from flash_attn import flash_attn_func
import causal_conv1d, fla
rank=int(os.environ.get('LOCAL_RANK',0))
torch.cuda.set_device(rank)
torch.distributed.init_process_group('nccl')
torch.manual_seed(42)
q=torch.randn(1,1024,4,256,device='cuda',dtype=torch.bfloat16,requires_grad=True)
k=torch.randn_like(q,requires_grad=True);v=torch.randn_like(q,requires_grad=True)
y=flash_attn_func(q,k,v,causal=True)
ref=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),is_causal=True).transpose(1,2)
diff=(y-ref).abs().max().item();assert diff<0.08,diff
y.float().square().mean().backward()
assert all(torch.isfinite(x.grad).all().item() for x in [q,k,v])
value=torch.ones(1,device='cuda');torch.distributed.all_reduce(value);assert value.item()==4
result={'rank':rank,'gpu':torch.cuda.get_device_name(rank),'flash_max_abs_error_vs_sdpa':diff,'finite_backward':True,'nccl_allreduce':value.item()}
print(json.dumps(result),flush=True)
(Path(os.environ['IFV_H20_RUN_ROOT'])/f'kernel-rank{rank}.json').write_text(json.dumps(result,indent=2))
torch.distributed.destroy_process_group()
