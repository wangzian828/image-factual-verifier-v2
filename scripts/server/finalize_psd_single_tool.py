"""Persist the completed eight-episode runtime gate; never claim full PSD success."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT=Path('/volume/ybo/wza')
CODE=ROOT/'training-artifacts/psd-adjustments-20260915/code'
RUN=ROOT/'runs/psd-single-tool4x2-20260916'
WIRE=ROOT/'inference/psd-sft2056-safety-20260916/wire-single-tool-v1/wire'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path.insert(0,str(CODE))
    from scripts.audit_real_trace import audit_trace,SourceAccessPolicy,_json_summary
    from tokenizers import Tokenizer
    output=RUN/'runtime-acceptance.json'
    assert not output.exists(),'Preserve completed acceptance'
    binding=json.loads((RUN/'binding.json').read_text())
    state=json.loads((RUN/'state.json').read_text())
    assert state['phase']=='completed_requires_raw_review' and state['summary']['num_errors']==0
    paths=sorted((RUN/'episodes/traces').glob('*.json'))
    assert len(paths)==8
    policy=SourceAccessPolicy.load(Path(binding['source_access_policy']))
    report=_json_summary([audit_trace(p,source_access_policy=policy) for p in paths],strict_scheduler=True)
    assert report['passed']
    cases=Counter();cache=Counter()
    for path in paths:
        trace=json.loads(path.read_text());assert not trace.get('error')
        cases[trace['case_id']]+=1
        for step in trace.get('state',{}).get('all_steps',[]):
            if step.get('tool_name') in ['perceive_scene','ocr_with_position']:
                cache[str(step.get('metadata',{}).get('cache_hit'))]+=1
    assert set(cases)==set(binding['case_ids']) and set(cases.values())=={2}
    assert cache and set(cache)=={'False'}
    for relative,expected in binding['source_sha256'].items():assert sha(CODE/relative)==expected
    tokenizer=Tokenizer.from_file(str(ROOT/'exports/h20-sft-merged4872-epoch2-step2056-20260915/model/tokenizer.json'))
    end=tokenizer.token_to_id('</think>');required=0;reasons=Counter();closures=[];lengths=[];wire_hashes={}
    for directory in sorted(WIRE.iterdir()):
        assert (directory/'response-meta.json').exists() and not (directory/'error.json').exists()
        bodies={}
        for side in ['request','response']:
            meta=json.loads((directory/(side+'-meta.json')).read_text())
            raw=gzip.decompress((directory/(side+'.json.gz')).read_bytes())
            assert hashlib.sha256(raw).hexdigest()==meta['sha256'] and len(raw)==meta['bytes']
            bodies[side]=json.loads(raw);wire_hashes[f'{directory.name}/{side}']=meta['sha256']
        for choice in bodies['response']['choices']:
            reasons[choice['finish_reason']]+=1
            assert choice['finish_reason'] in ['stop','tool_calls']
            ids=choice.get('token_ids') or [];assert ids
            lengths.append(len(ids))
            if end in ids:closures.append(ids.index(end))
            if bodies['request'].get('tool_choice')=='required':
                required+=1;assert end in ids and choice['finish_reason']=='tool_calls'
                array=json.loads(tokenizer.decode(ids[ids.index(end)+1:],skip_special_tokens=True).strip())
                assert isinstance(array,list) and len(array)==1
    with urllib.request.urlopen('http://127.0.0.1:19016/health',timeout=5) as response:health=json.load(response)
    assert all(r['inflight']==0 for r in health['replicas']) and health['wire_capture']['failed']==0
    assert health['wire_capture']['completed']==health['wire_capture']['started']==sum(reasons.values())
    result={'runtime_gate_passed':True,'full_psd_gate_passed':False,'training_started':False,
        'formal_source_bank':False,'cases':4,'episodes':8,'strict_audit':report,
        'perception_cache_flags':dict(cache),'finish_reasons':dict(reasons),'required_raw_single_calls':required,
        'max_output_tokens':max(lengths),'max_endthink_index':max(closures),
        'wire_body_sha256':wire_hashes,'trace_sha256':{p.name:sha(p) for p in paths},
        'binding_sha256':sha(RUN/'binding.json'),'script_sha256':sha(Path(__file__)),
        'time':time.time(),'next':'4x8 source collection/checker, multi-position repair, exact targets, top20 and GPU save/resume'}
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['wire_body_sha256','trace_sha256','strict_audit']}))


if __name__=='__main__':main()
