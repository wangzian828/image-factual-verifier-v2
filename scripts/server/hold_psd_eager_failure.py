"""Stop the exact owned recovery after eager FSM recurrence, preserving artifacts."""
import importlib.util
from pathlib import Path
import time

ROOT = Path('/volume/ybo/wza')
RUN = ROOT / 'runs/psd-sft3084-captured-canary4x8-20260916/psd-observed-positions-v14'
SERVICE = ROOT / 'inference/psd-sft3084-20260916'


def main():
    spec = importlib.util.spec_from_file_location('owner',
        ROOT / 'training-artifacts/psd-epoch3-20260916-v1/psd_epoch3_canary.py')
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    output = RUN / 'eager-failure-hold-v1'
    output.mkdir(exist_ok=False)
    receipt = owner.load(RUN / 'execution-recovery-v1/process.json')
    assert receipt['command'][2] == str(ROOT / 'training-artifacts/psd-observed-positions-20260916-v14/resume_psd_pending_cases.py')
    owner.checked(receipt)
    evidence = SERVICE / 'validated-execution-candidate-v1/backend-1.log'
    assert 'Failed to advance FSM' in evidence.read_text(errors='replace')
    snapshots = {}
    for directory in (RUN / 'search/repairs').iterdir():
        if directory.is_dir() and (directory / 'slate-state.json').exists():
            state = owner.load(directory / 'slate-state.json')
            snapshots[directory.name] = {'state': state,
                'completed_round_files': {path: digest for row in state['payload']['rounds']
                                          for path, digest in row['files'].items()}}
    owner.save(output / 'before.json', {'controller': receipt, 'cases': snapshots,
        'reason': 'eager actual long request FSM token0 recurrence; not a validated fix',
        'source_wire_ticket': '1789551642580005259-b2b919425c11426eb34c440879f2f248',
        'training_started': False, 'time': time.time()})
    owner.stop(receipt)
    for case in snapshots.values():
        for path, digest in case['completed_round_files'].items():
            assert owner.sha(path) == digest, 'A completed round changed'
    owner.save(output / 'state.json', {'phase': 'owned_recovery_stopped_for_numerical_diagnosis',
        'completed_results_preserved': True, 'budget_reset': False, 'training_started': False,
        'time': time.time()})
    print('OWNED_RECOVERY_STOPPED_RESULTS_PRESERVED')


if __name__ == '__main__':
    main()
