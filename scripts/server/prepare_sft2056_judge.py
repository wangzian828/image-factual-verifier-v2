"""Prepare the 1,524 frozen epoch2 successes for v3 judge, without API calls."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/volume/ybo/wza')
REPO = ROOT/'image-factual-verifier-v2'
SOURCE = ROOT/'evaluation/qwen35-sft2056-epoch2-agent-budgeted1524-20260916'
RUN = ROOT/'runs/eval/qwen35-sft2056-epoch2-agent-formal1527-20260915'
OUT = ROOT/'evaluation/sft2056-v3-judge-20260916'
NAME = 'qwen35-9b-sft2056-agent'
CASES = ROOT/'evaluation/factcheck-formal1527-available1526-20260912/runtime-release/runtime_input/cases.jsonl'
GOLD = ROOT/'data/factcheck-test-1527-filtered-frozen-20260909/evaluator_private/private-gold-v1/private-gold.jsonl'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.path.insert(0, str(REPO))
    from src.eval.agent_private_gold import build_agent_private_gold_candidate, agent_candidate_answer
    from src.eval.private_gold_judge_contract import PRIVATE_GOLD_JUDGE_PROMPT, PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA
    from scripts.audit_direct_qa_baseline import _private_gold
    assert digest(CASES) == 'c6568c302147893f7a648ea4e5cccd29e4c11dae831e1344399330ef75d7cf32'
    assert digest(GOLD) == '49a5785522b982dc4f97b28270a4b5bf5f4da0d7247f8dc7fd30cdf0de5ea671'
    assert digest(REPO/'src/eval/agent_private_gold.py') == 'a7072d30cfaef2af5add48648cf376cbe316e93deeab33878873fd347900117e'
    assert digest(REPO/'src/eval/private_gold_judge_contract.py') == '09c97955e292786b0a2259a653c0d198b86eeff33d3984e6e451dd4fcdc3577a'
    assert digest(SOURCE/'selected-traces.json') == '104bd65fec08f13966156beea4e012363d2ec02ddc67ec7a8f50066efcc210b0'
    selected = json.loads((SOURCE/'selected-traces.json').read_text())
    cases = [json.loads(line) for line in CASES.read_text().splitlines()]
    gold = {r['case_id']: r for r in map(json.loads, GOLD.read_text().splitlines())}
    eligible = [r for r in cases if r['case_id'] in selected]
    assert len(selected) == len(eligible) == 1524
    excluded = [case for case in gold if case not in selected]
    assert len(excluded) == 3
    OUT.mkdir()  # Existing/partial preparation must be inspected, not overwritten.
    raw_actions = 0
    candidate_file = OUT/(NAME+'.jsonl.gz')
    with gzip.open(candidate_file, 'wt', encoding='utf-8') as handle:
        for row in eligible:
            case = row['case_id']
            info = selected[case]
            path = Path(info['path']).resolve()
            path.relative_to(RUN)
            assert digest(path) == info['sha256']
            trace = json.loads(path.read_text())
            assert (trace.get('case_id') or trace.get('image_id')) == case
            assert not trace.get('error') and trace.get('fact_check_report')
            candidate = build_agent_private_gold_candidate(trace)
            answer = agent_candidate_answer(candidate)
            assert candidate['projection_schema_version'] == 'ifv-agent-private-gold-candidate-v3'
            assert answer['verdict'] == trace['verdict']
            raw_actions += len(candidate['raw_observations'])
            private = _private_gold(gold[case])
            assert private['auditable']
            handle.write(json.dumps({'case_id': case, 'source_status': 'success',
                'source_model': 'ifv-qwen3.5-9b-sft-2056', 'source_trace_path': str(path),
                'source_trace_sha256': info['sha256'], 'gold': private,
                'candidate_output': candidate, 'candidate_answer': answer}, ensure_ascii=False)+'\n')
    record = {'source_name': NAME, 'count': 1524, 'formal_denominator': 1527,
              'model': 'gemini-3.7-flash', 'thinking_level': 'low', 'max_output_tokens': 32768,
              'candidate_schema': 'ifv-agent-private-gold-candidate-v3',
              'prompt_sha256': hashlib.sha256(PRIVATE_GOLD_JUDGE_PROMPT.encode()).hexdigest(),
              'response_schema_sha256': hashlib.sha256(json.dumps(PRIVATE_GOLD_JUDGE_RESPONSE_SCHEMA, sort_keys=True).encode()).hexdigest(),
              'source_sha256': digest(candidate_file), 'selected_sha256': digest(SOURCE/'selected-traces.json'),
              'cases_sha256': digest(CASES), 'gold_sha256': digest(GOLD),
              'raw_actions': raw_actions, 'excluded_case_ids': excluded,
              'submitted': False, 'api_calls': 0, 'bytes_gzip': candidate_file.stat().st_size}
    (OUT/'prepared.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
