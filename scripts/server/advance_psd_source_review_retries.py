"""Drain only the old PSD reviewer, then follow v6 with bounded auto-retries.

Never restarts collection, vLLM, or GPU protection. Never interrupts a paid
in-flight review. All paths are within the user's owned server directory.
"""
import importlib.util
import os
from pathlib import Path
import sys
import time

ROOT = Path('/volume/ybo/wza')
OUT = ROOT/'training-artifacts/psd-review-autoretry-20260917-v42'
CODE = OUT/'code'
OLD = ROOT/'runs/psd-production400x8-20260917-v5'
NEW = ROOT/'runs/psd-production400x8-20260917-v6'
OUTPUT = NEW/'source-review-prefetch-v2-auto-retry'


def main():
    spec = importlib.util.spec_from_file_location('owner',
        ROOT/'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)
    def status(phase, **kwargs):
        o.save(OUT/'review-handoff.json', {'phase':phase, 'time':time.time(), **kwargs})
    try:
        if o.load(OUT/'stage-state.json')['tests_returncode'] != 0:
            raise ValueError('Stage tests did not pass')
        if OUTPUT.exists() or (NEW/'source-review-prefetch-process.json').exists():
            raise ValueError('Destination reviewer already exists; do not launch twice')
        old = o.load(OLD/'source-review-prefetch-process.json')
        status('waiting_for_old_paid_reviews_to_drain')
        deadline = time.monotonic()+7200
        while True:
            o.checked(old)
            p = o.load(OLD/'source-review-prefetch-v1/progress.json')
            if (p['active'] == p['queued'] == 0 and p['phase'] == 'waiting_for_completed_sources'
                    and (NEW/'slot-recovery.json').exists()):
                old_collection = o.load(OLD/'process.json')
                command = Path('/proc')/str(old_collection['pid'])/'cmdline'
                if command.exists() and command.read_bytes():
                    raise ValueError('Old source collector is still active')
                # Old sources are immutable now; no new old-run slots can appear.
                o.stop(old)
                break
            if time.monotonic() > deadline: raise TimeoutError('Old reviewer did not safely drain')
            time.sleep(10)
        status('starting_v6_reviewer_with_transport_retries')
        helper = o.helper(); helper.CODE = CODE; helper.GATEWAY = 'http://127.0.0.1:19025'
        env, checks = helper.environment()
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=str(CODE)+':'+str(CODE/'training'), TMPDIR=str(ROOT/'tmp'))
        binding = o.load(NEW/'binding.json')
        command = [sys.executable, '-u', str(CODE/'scripts/prefetch_psd_source_reviews.py'),
            '--run-dir', str(NEW/'episodes'), '--benchmark', binding['benchmark'],
            '--train-cases', binding['train_cases'], '--private-gold', binding['private_gold'],
            '--output', str(OUTPUT), '--cache-source', str(OLD/'source-review-prefetch-v1'),
            '--model', 'gemini-3.1-pro-preview', '--concurrency', '16', '--follow',
            '--retry-errors', '--max-review-attempts', '4', '--retry-delay-seconds', '60']
        receipt = o.spawn(command, env, NEW/'source-review-prefetch-v2-auto-retry.log')
        o.save(NEW/'source-review-prefetch-process.json', receipt)
        deadline = time.monotonic()+600
        while True:
            o.checked(receipt)
            p = OUTPUT/'progress.json'
            if p.exists():
                state = o.load(p)
                if state.get('retry_errors') is not True or state.get('concurrency') != 16:
                    raise ValueError('Retry policy or concurrency is not active')
                status('v6_reviewer_autoretry_running', output=str(OUTPUT), pid=receipt['pid'],
                       collector_restarted=False, model_services_restarted=False)
                return
            if time.monotonic() > deadline: raise TimeoutError('Reviewer did not publish progress')
            time.sleep(5)
    except BaseException as error:
        status('requires_diagnosis', error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    os.umask(0o077)
    main()
