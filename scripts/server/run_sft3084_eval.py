"""Frozen step-3084 evaluation orchestration; never changes the Agent runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/volume/ybo/wza')
REPO = ROOT / 'image-factual-verifier-v2'
OUTPUT = ROOT / 'runs/eval/qwen35-sft3084-3epoch-agent-formal1527-20260915'
EXPORT = ROOT / 'exports/h20-sft-merged4872-3epoch-step3084-20260915/export.json'
BENCHMARK = ROOT / 'evaluation/factcheck-formal1527-available1526-20260912/runtime-release/runtime_input/cases.jsonl'
CANARY = ROOT / 'runs/eval/qwen35-sft1028-agent-canary4-20260914/target-case-list.txt'
ALIAS = 'ifv-qwen3.5-9b-sft-3084'
BENCHMARK_SHA = 'c6568c302147893f7a648ea4e5cccd29e4c11dae831e1344399330ef75d7cf32'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def successful_traces(directories: list[Path], expected: set[str]) -> dict[str, Path]:
    selected = {}
    for directory in directories:
        for path in sorted((directory / 'traces').glob('*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            case = str(row.get('case_id') or row.get('image_id') or '')
            if case not in expected:
                raise ValueError('Unknown case in evaluation output')
            if row.get('error') or row.get('verdict') not in ('real', 'fake') or not row.get('fact_check_report'):
                continue
            if case in selected:
                raise ValueError('A successful case was resampled')
            selected[case] = path
    return selected


def verify_acceptance(path: Path, smoke: Path, expected: set[str], export_sha: str) -> None:
    record = json.loads(path.read_text(encoding='utf-8'))
    selected = successful_traces([smoke], expected)
    actual = {case: digest(trace) for case, trace in selected.items()}
    if not (record.get('passed') is True and record.get('export_sha256') == export_sha
            and set(selected) == expected and record.get('trace_sha256') == actual
            and record.get('strict_audit_passed') is True
            and record.get('external_tools_verified') is True
            and record.get('multimodal_verified') is True):
        raise ValueError('Smoke acceptance is absent, incomplete or no longer bound')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['smoke', 'full'], required=True)
    args = parser.parse_args()
    import fcntl
    from dotenv import dotenv_values

    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUTPUT / 'controller.pid').write_text(str(os.getpid()), encoding='utf-8')
    export = json.loads(EXPORT.read_text())
    if not (export.get('passed') and export.get('global_step') == 3084
            and export.get('epoch') == 3 and export.get('source_full_state_preserved')):
        raise ValueError('Wrong or incomplete three-epoch export')
    if digest(BENCHMARK) != BENCHMARK_SHA:
        raise ValueError('Frozen benchmark changed')
    benchmark = [json.loads(line) for line in BENCHMARK.read_text().splitlines() if line.strip()]
    expected = [str(row['case_id']) for row in benchmark]
    canary = [line.strip() for line in CANARY.read_text().splitlines() if line.strip()]
    if len(expected) != 1526 or len(set(expected)) != 1526 or len(set(canary)) != 4 or not set(canary) <= set(expected):
        raise ValueError('Frozen case selection is invalid')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    if revision != '1d61abb41f147e871b5361ee3e917aaa5c06ccc5':
        raise ValueError('Frozen server runtime revision changed')
    identity = {'export_sha256': digest(EXPORT), 'benchmark_sha256': BENCHMARK_SHA,
                'runtime_commit': revision, 'model': ALIAS, 'canary_ids': canary,
                'model_path': export['model_path'], 'formal_denominator': 1527,
                'output_tokens': 32768, 'thinking_budget': 8192,
                'timeout_seconds': 3000, 'concurrency': [40, 32, 24, 16],
                'seeds': [1903, 2903, 3903, 4903]}
    binding = OUTPUT / 'inputs.json'
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError('Controller inputs changed')
    save(binding, identity)
    env = {**os.environ, **{k: v for k, v in dotenv_values(ROOT / 'private/runtime.env').items() if v is not None}}
    present = lambda *keys: any(str(env.get(key, '')).strip() for key in keys)
    checks = {'serper': present('SERPER_API_KEY', 'SERPER_KEY_ID'),
              'jina': present('JINA_API_KEY', 'JINA_API_KEYS'),
              'gemini_tools': present('GEMINI_API_KEY', 'GOOGLE_API_KEY'),
              'baidu_ocr': present('BAIDU_OCR_API_KEY') and present('BAIDU_OCR_SECRET_KEY')}
    save(OUTPUT / 'credential-presence.json', checks)
    if not all(checks.values()):
        raise ValueError('Required external tool credentials are missing; values are not logged')
    env.update(QWEN35_LOCAL_BASE_URL='http://127.0.0.1:8901/v1', QWEN35_LOCAL_MODEL=ALIAS,
               QWEN_UNIFIED_REACT_MAX_OUTPUT_TOKENS='32768', QWEN_UNIFIED_REACT_THINKING_TOKEN_BUDGET='8192',
               QWEN_UNIFIED_JUDGMENT_MAX_OUTPUT_TOKENS='32768', QWEN_UNIFIED_JUDGMENT_THINKING_TOKEN_BUDGET='8192',
               AGENT_STAGE_REQUEST_TIMEOUT_SECONDS='1500', TOOL_CACHE_ENABLED='0', OMP_NUM_THREADS='1',
               PYTHONPATH=str(REPO), TMPDIR=str(ROOT / 'tmp'))
    deadline = time.monotonic() + 1200
    save(OUTPUT / 'progress.json', {'phase': 'waiting_for_services', 'stage': args.stage})
    for port in (8902, 8903, 8904, 8905):
        while True:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=5) as response:
                    cards = json.load(response)['data']
                if not any(c['id'] == ALIAS and c.get('root') == export['model_path']
                           and c.get('max_model_len', 0) >= 131072 for c in cards):
                    raise ValueError('Service model identity or context length mismatch')
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Service readiness timeout')
                time.sleep(10)

    def run(directory: Path, cases: list[str], concurrency: int, seed: int, previous: Path | None = None) -> None:
        if directory.exists():
            raise ValueError('Refusing to overwrite an existing attempt')
        case_list = OUTPUT / (directory.name + '-cases.txt')
        case_list.write_text(''.join(case + '\n' for case in cases), encoding='utf-8')
        command = [sys.executable, '-m', 'src.eval.run_cases', '--benchmark', str(BENCHMARK),
                   '--profile', 'student-qwen3.5-local', '--output-dir', str(directory),
                   '--case-list', str(case_list), '--concurrency', str(min(len(cases), concurrency)),
                   '--base-sampling-seed', str(seed), '--timeout', '3000']
        if previous is not None:
            command += ['--resume-from', str(previous)]
        save(OUTPUT / 'progress.json', {'phase': 'running', 'stage': args.stage, 'attempt': directory.name,
                                      'pending_at_start': len(cases), 'concurrency': min(len(cases), concurrency)})
        with (OUTPUT / (directory.name + '.log')).open('xb') as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        if not (directory / 'summary.json').is_file():
            raise RuntimeError(f'Attempt did not finish normally (exit {result.returncode}); inspect preserved output')

    smoke = OUTPUT / 'smoke'
    if args.stage == 'smoke':
        run(smoke, canary, 4, 1903)
        selected = successful_traces([smoke], set(canary))
        save(OUTPUT / 'progress.json', {'phase': 'smoke_requires_audit', 'success': len(selected), 'expected': 4})
        return
    verify_acceptance(OUTPUT / 'smoke-acceptance.json', smoke, set(canary), digest(EXPORT))
    directories = [smoke]
    for attempt, concurrency in enumerate((40, 32, 24, 16)):
        directory = OUTPUT / f'attempt-{attempt}'
        if directory.exists():
            if not (directory / 'summary.json').is_file():
                raise RuntimeError('Interrupted attempt exists; inspect before explicit recovery')
            directories.append(directory)
            continue
        selected = successful_traces(directories, set(expected))
        pending = [case for case in expected if case not in selected]
        if not pending:
            break
        run(directory, pending, concurrency, 1903 + attempt * 1000, directories[-1] if attempt else None)
        directories.append(directory)
    selected = successful_traces(directories, set(expected))
    save(OUTPUT / 'selected-traces.json', {case: {'path': str(path), 'sha256': digest(path)} for case, path in selected.items()})
    save(OUTPUT / 'progress.json', {'phase': 'inference_complete' if len(selected) == len(expected) else 'retry_budget_exhausted',
                                  'success': len(selected), 'expected': len(expected),
                                  'remaining': [case for case in expected if case not in selected], 'judge_submitted': False})


if __name__ == '__main__':
    main()
