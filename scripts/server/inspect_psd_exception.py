"""Summarize private diagnostic frames without displaying trajectory material."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def inspect(receipt):
    metadata = json.loads(receipt.read_text())
    with gzip.open(metadata['path'], 'rb') as stream:
        raw = stream.read()
    assert hashlib.sha256(raw).hexdigest() == metadata['sha256']
    value = json.loads(raw)
    rows = []
    for frame in value['selected_model_context']:
        local = frame['locals']
        row = {'function': frame['function'], 'keys': list(local)}
        if frame['function'] == 'capture_target':
            messages = local.get('messages', [])
            prefix = local.get('prefix', [])
            row['prefix_equal'] = messages[:len(prefix)] == prefix
            row['non_system_prefix_equal'] = messages[1:len(prefix)] == prefix[1:]
            row['system_chars'] = [len(item[0].get('content', '')) if item else None
                                   for item in (messages, prefix)]
            row['system_availability_blocks'] = [item[0].get('content', '').count('Runtime tool availability:')
                                                  if item else None for item in (messages, prefix)]
        if frame['function'] == 'run_hinted_episode':
            row['teacher_step_count'] = len(local.get('teacher_steps', []))
            row['judgment_step_actions'] = [step.get('action_type') for step in local.get('judgment_steps', [])]
            row['judgment_error_classes'] = [step.get('metadata', {}).get('error_class')
                                             for step in local.get('judgment_steps', [])]
            row['unknown_observation_rejections'] = sum('unknown verdict observation IDs' in
                step.get('metadata', {}).get('rejection_reason', '') for step in local.get('judgment_steps', []))
            observations = [step for step in local.get('teacher_steps', [])
                            if step.get('action_type') == 'tool_call' and step.get('tool_name') != 'finish_investigation'
                            and step.get('metadata', {}).get('function_call_id')]
            row['repair_observations_ignored_by_frozen_stage_filter'] = sum(
                step.get('stage_name') == 'psd_teacher_repair' for step in observations)
        rows.append(row)
    return {'receipt': str(receipt), 'error_type': value['error_type'],
            'frames': value['frames'], 'context_summary': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('deployment', type=Path)
    args = parser.parse_args()
    for receipt in sorted(args.deployment.glob('*-failure.json')):
        print(json.dumps(inspect(receipt)), flush=True)
