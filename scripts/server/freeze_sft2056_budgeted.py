"""Freeze completed epoch2 attempts without disguising its two exhausted cases."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/volume/ybo/wza')
RUN = ROOT/'runs/eval/qwen35-sft2056-epoch2-agent-formal1527-20260915'
OUT = ROOT/'evaluation/qwen35-sft2056-epoch2-agent-budgeted1524-20260916'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    sys.path.insert(0, str(ROOT/'image-factual-verifier-v2'))
    from scripts.run_direct_qa_baseline import _gold_verdict
    progress = json.loads((RUN/'progress.json').read_text())
    assert progress['phase'] == 'retry_budget_exhausted' and progress['success'] == 1524
    selected = json.loads((RUN/'selected-traces.json').read_text())
    assert len(selected) == 1524 and len(progress['remaining']) == 2
    gold_path = ROOT/'data/factcheck-test-1527-filtered-frozen-20260909/evaluator_private/private-gold-v1/private-gold.jsonl'
    assert digest(gold_path) == '49a5785522b982dc4f97b28270a4b5bf5f4da0d7247f8dc7fd30cdf0de5ea671'
    gold = {r['case_id']: _gold_verdict(r) for r in map(json.loads, gold_path.read_text().splitlines())}
    assert len(gold) == 1527 and sum(v == 'real' for v in gold.values()) == 377
    confusion = {'real': {'real': 0, 'fake': 0, 'missing_or_failed': 0},
                 'fake': {'real': 0, 'fake': 0, 'missing_or_failed': 0}}
    for case, label in gold.items():
        if case not in selected:
            confusion[label]['missing_or_failed'] += 1
            continue
        item = selected[case]
        path = Path(item['path']).resolve()
        path.relative_to(RUN)
        assert digest(path) == item['sha256']
        trace = json.loads(path.read_text())
        assert (trace.get('case_id') or trace.get('image_id')) == case
        assert not trace.get('error') and trace.get('fact_check_report')
        confusion[label][trace['verdict']] += 1
    real = confusion['real']['real']/377
    fake = confusion['fake']['fake']/1150
    record = {'schema': 'ifv-budgeted-eval-freeze-v1', 'successful': 1524,
              'executable': 1526, 'reported_denominator': 1527,
              'inference_complete': False, 'retry_budget_finished': True,
              'remaining_failed': progress['remaining'], 'confusion': confusion,
              'bacc_percent': (real+fake)*50, 'real_recall_percent': real*100,
              'fake_recall_percent': fake*100, 'sesr_percent': None,
              'source_run': str(RUN), 'selected_sha256': digest(RUN/'selected-traces.json'),
              'inputs_sha256': digest(RUN/'inputs.json'),
              'thinking_budget_requested': 8192, 'thinking_budget_enforced': False,
              'note': 'No extra retries, changed-protocol predictions, or fabricated successes.'}
    if OUT.exists():
        assert json.loads((OUT/'summary.json').read_text()) == record
    else:
        OUT.mkdir()
        (OUT/'selected-traces.json').write_bytes((RUN/'selected-traces.json').read_bytes())
        (OUT/'summary.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    freeze()
