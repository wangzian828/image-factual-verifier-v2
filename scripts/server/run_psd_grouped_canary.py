"""Collect four preselected training cases x eight slots; no repair or training.

The frozen Agent is imported, never edited. All server outputs stay under ROOT.
Run --prepare first, then --launch. Neither command resumes or replaces a run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/volume/ybo/wza')
CODE = ROOT/'training-artifacts/psd-adjustments-20260915/code'
HELPERS = ROOT/'training-artifacts/psd-serving-safety-20260916'
PREP = ROOT/'runs/psd-pilot400-preparation-20260915-v1'
SERVICE = ROOT/'inference/psd-sft2056-safety-20260916'
RUN = ROOT/'runs/psd-slate-canary4x8-20260916'
EXPORT = ROOT/'exports/h20-sft-merged4872-epoch2-step2056-20260915/export.json'
EXPORT_SHA = '55dfb77f56cb175573c5e966816c8a0a0c384385190ae5b62795963f681fbd28'
ALIAS = 'ifv-qwen3.5-9b-sft-2056'
CACHE_FREE = False


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def select_fixed_cases(canary, public, splits):
    if len(canary) != 32 or any(row.get('split') != 'train' for row in canary):
        raise ValueError('Expected the frozen 32-case training canary')
    ids = [row['case_id'] for row in canary[:4]]
    if len(set(ids)) != 4:
        raise ValueError('Duplicate selected case')
    for rows in (public, splits):
        if len({row['case_id'] for row in rows}) != len(rows):
            raise ValueError('Duplicate source membership')
    split = {row['case_id']: row for row in splits}
    if not set(ids) <= {row['case_id'] for row in public}:
        raise ValueError('Canary absent from public training input')
    if any(split.get(case, {}).get('split') != 'train' for case in ids):
        raise ValueError('Selected input is not explicitly training-only')
    return ids


def source_revision(hashes):
    encoded = json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()
    return 'source-sha256:' + hashlib.sha256(encoded).hexdigest()


def helpers():
    version = 'run_psd_runtime_gate_v4.py' if CACHE_FREE else 'run_psd_runtime_gate_v3.py'
    spec = importlib.util.spec_from_file_location('runtime_gate', HELPERS/version)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.GATEWAY = 'http://127.0.0.1:19012'
    if CACHE_FREE:
        module.DISABLE_PERCEPTION_CACHE = True
    return module


def preflight():
    assert not (RUN/'cache-contamination-hold').exists(), (
        'This canary is contaminated; preserve it and validate a new cache-free protocol before recollection')
    h = helpers()
    binding = h.preflight()
    if CACHE_FREE:
        gate = ROOT/'runs/psd-real-runtime-gate-no-perception-cache-20260916'
        assert load(gate/'strict-runtime-audit.json')['passed']
        config = load(gate/'effective-cache-config.json')
        assert config['tool_cache_enabled'] is False and config['cacheable_tools'] == []
        assert load(gate/'cache-identity-audit.json')['passed']
    for name in ('strict-runtime-audit.json', 'summary.json'):
        assert load(SERVICE/'tokenizer-gateway-validation-v1'/name)['passed']
    backend = load(SERVICE/'replica-2.json')
    actual = [part.decode() for part in Path(f'/proc/{backend["pid"]}/cmdline').read_bytes().split(b'\0') if part]
    assert actual == backend['command'] and '--no-enable-prefix-caching' in actual
    assert actual[actual.index('--mamba-cache-mode') + 1] == 'none'
    assert digest(EXPORT) == EXPORT_SHA
    return h, binding


def prepare():
    h, runtime = preflight()
    assert not RUN.exists(), 'Inspect existing canary; never overwrite/resample it'
    sys.path[:0] = [str(CODE), str(CODE/'training')]
    from ifv_training.io import load_jsonl, write_jsonl
    from ifv_training.checkpoints import build_checkpoint_manifest, build_serving_profile
    from src.eval.public_release import load_public_release
    source = load(PREP/'prepared.json')['payload']
    benchmark = Path(source['benchmark'])
    public = load_jsonl(benchmark)
    splits = load_jsonl(Path(source['train_cases']))
    ids = select_fixed_cases(load_jsonl(PREP/'training-canary-case-list.jsonl'), public, splits)
    by_id = {row['case_id']: row for row in public}
    selected = []
    for case in ids:
        row = by_id[case]
        image = (benchmark.parent/row['image_path']).resolve()
        image.relative_to(ROOT)
        assert digest(image) == row['image_sha256']
        destination = RUN/'runtime-release/runtime_input/assets'/(row['image_sha256']+image.suffix.lower())
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(image, destination)
        selected.append({'case_id': case, 'image_path': 'assets/'+destination.name,
                         'image_sha256': row['image_sha256']})
    public_path = RUN/'runtime-release/runtime_input/cases.jsonl'
    write_jsonl(public_path, selected)
    # Private files never appear in the collector's public input or prompts.
    for key, name in [('train_cases', 'case_split.jsonl'), ('private_gold', 'private_gold.jsonl')]:
        rows = load_jsonl(Path(source[key]))
        mapping = {row['case_id']: row for row in rows}
        assert len(mapping) == len(rows) and set(ids) <= set(mapping)
        write_jsonl(RUN/'evaluator_private'/name, [mapping[case] for case in ids])
    policy = RUN/'runtime-release/evaluator_private/source_access_policy.json'
    save(policy, load(Path(source['source_access_policy'])))
    release = load_public_release(benchmark)
    save(RUN/'runtime-release/manifest.json', {**release.manifest,
        'release_id': 'psd-slate-canary4x8-20260916', 'release_stage': 'psd_training_canary',
        'source_access_policy': {'active': True, 'path': 'evaluator_private/source_access_policy.json',
                                 'sha256': digest(policy)}})
    load_public_release(public_path)
    exported = load(EXPORT)
    snapshot = RUN/'snapshot'
    provenance = snapshot/'sft-provenance.json'
    save(provenance, {'dataset_version': None, 'local_training_performed': True,
        'source': 'existing full-parameter SFT epoch2 export, not a pretrained base',
        'dataset_identity': 'not reconstructed here; original source checkpoint retained',
        'source_checkpoint': exported['source_checkpoint'], 'export': str(EXPORT),
        'export_sha256': EXPORT_SHA, 'epoch': exported['epoch'], 'global_step': exported['global_step'],
        'source_full_state_preserved': exported['source_full_state_preserved']})
    manifest_path = snapshot/'checkpoint-manifest.json'
    manifest = build_checkpoint_manifest(checkpoint_dir=EXPORT.parent/'model',
        dataset_manifest_path=provenance, output_path=manifest_path,
        base_model_id=exported['base_model'], model_revision='exact exported file hashes authoritative',
        processor_revision='exact exported file hashes authoritative', method='full',
        framework_version='not inferred from the current environment')
    manifest['framework']['version'] = None
    manifest['checkpoint'].update(global_step=2056, epoch=2.0)
    manifest['source_export'] = {'path': str(EXPORT), 'sha256': EXPORT_SHA,
        'source_checkpoint': exported['source_checkpoint'], 'full_state_retained_separately': True}
    for item in manifest['artifacts']:
        if item['scope'] == 'model':
            assert exported['model_artifacts'][item['path']]['sha256'] == item['sha256']
    index = EXPORT.parent/'model/model.safetensors.index.json'
    assert digest(index) == exported['model_artifacts'][index.name]['sha256']
    manifest['artifacts'].append({'scope': 'model', 'path': index.name,
        'bytes': index.stat().st_size, 'sha256': digest(index)})
    save(manifest_path, manifest)
    build_serving_profile(output_path=snapshot/'serving-profile.json', profile_id=ALIAS,
        model_path=str(EXPORT.parent/'model'), engine='vllm', port=19012, tensor_parallel_size=1,
        dtype='bfloat16', context_length=131072, tool_call_parser='qwen3_coder', reasoning_parser='qwen3',
        thinking_enabled=True, checkpoint_manifest_path=manifest_path)
    binding = {**runtime, 'case_ids': ids, 'benchmark': str(public_path),
        'source_access_policy': str(policy), 'train_cases': str(RUN/'evaluator_private/case_split.jsonl'),
        'private_gold': str(RUN/'evaluator_private/private_gold.jsonl'), 'snapshot': str(snapshot),
        'purpose': 'bounded multi-position PSD canary; not the 400x8 collection',
        'rollouts_per_case': 8, 'slots': 32, 'concurrency': 8,
        'policy_revision': source_revision(runtime['source_sha256']),
        'policy_revision_kind': 'source_content_hash_not_git_commit',
        'selection': 'first four of frozen 32-case canary, before current grouped outcomes',
        'private_gold_loaded': 'preparation only; never supplied to collector model',
        'files': {str(path): digest(path) for path in RUN.rglob('*') if path.is_file()}}
    save(RUN/'binding.json', binding)
    save(RUN/'state.json', {'phase': 'prepared_not_started', 'time': time.time(),
        'slots': 32, 'psd_training_started': False, 'full_collection_started': False})
    print(json.dumps({'prepared': True, 'cases': len(ids), 'slots': 32, 'snapshot': str(snapshot)}), flush=True)


def verify_binding(binding):
    for name, expected in binding['files'].items():
        path = Path(name).resolve()
        path.relative_to(RUN)
        assert digest(path) == expected, name
    assert digest(EXPORT) == EXPORT_SHA
    for relative, expected in binding['source_sha256'].items():
        assert digest(CODE/relative) == expected, relative


def execute():
    h, _ = preflight()
    binding = load(RUN/'binding.json')
    verify_binding(binding)
    if CACHE_FREE:
        assert os.environ.get('PERCEPTION_CACHE_ENABLED') == os.environ.get('TOOL_CACHE_ENABLED') == '0'
    sys.path[:0] = [str(CODE), str(CODE/'training')]
    from src.eval import run_cases
    from scripts.collect_psd_rollouts import main as collect
    # Legacy manifest field is named git_commit, but carries an explicitly
    # prefixed content identity here. Never fabricate a commit or invoke Git.
    assert binding['policy_revision'] == source_revision(binding['source_sha256'])
    run_cases._git_commit = lambda: binding['policy_revision']
    sys.argv = ['psd-grouped-canary', '--train-cases', binding['train_cases'],
        '--benchmark', binding['benchmark'], '--source-access-policy', binding['source_access_policy'],
        '--profile', 'student-qwen3.5-local', '--output-dir', str(RUN/'episodes'),
        '--concurrency', '8', '--rollouts-per-case', '8', '--base-sampling-seed', '0', '--timeout', '3000']
    save(RUN/'state.json', {'phase': 'collecting_canary32', 'time': time.time(),
        'slots': 32, 'psd_training_started': False, 'full_collection_started': False})
    collect()
    manifest_path = RUN/'episodes/run_manifest.json'
    manifest = load(manifest_path)
    manifest['policy_revision_kind'] = binding['policy_revision_kind']
    manifest['source_binding_sha256'] = digest(RUN/'binding.json')
    save(manifest_path, manifest)
    from ifv_training.psd_collection import verify_collection
    from ifv_training.io import load_jsonl
    verification = verify_collection(load_jsonl(RUN/'episodes/run_results.jsonl'), case_ids=binding['case_ids'],
        manifest=load(RUN/'episodes/run_manifest.json'), expected_rollouts=8)
    save(RUN/'collection-verification.json', verification)
    save(RUN/'state.json', {'phase': 'collection_complete_requires_source_checker', 'time': time.time(),
        'slots': 32, 'psd_training_started': False, 'full_collection_started': False,
        'next': 'source checker, multi-position repair, exact targets, GPU update and native save/resume'})


def launch():
    h, _ = preflight()
    binding = load(RUN/'binding.json')
    verify_binding(binding)
    assert load(RUN/'state.json')['phase'] == 'prepared_not_started'
    assert not (RUN/'episodes').exists() and not (RUN/'process.json').exists()
    env, checks = h.environment()
    save(RUN/'credential-presence.json', checks)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execute']
    if CACHE_FREE:
        command += ['--cache-free']
    with (RUN/'run.log').open('xb') as log:
        child = subprocess.Popen(command, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(RUN/'process.json', {'pid': child.pid, 'command': command, 'time': time.time(),
        'script_sha256': digest(Path(__file__))})
    state = load(SERVICE/'state.json')
    state.update(phase='grouped_psd_canary32_running', grouped_canary=str(RUN),
        gpu_verified=False, source_collection_started=False)
    save(SERVICE/'state.json', state)
    print(json.dumps({'started': True, 'pid': child.pid, 'slots': 32, 'full_collection_started': False}), flush=True)


def main():
    global RUN, CACHE_FREE
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ('prepare', 'launch', 'execute'):
        mode.add_argument('--'+flag, action='store_true')
    parser.add_argument('--cache-free', action='store_true')
    args = parser.parse_args()
    if args.cache_free:
        CACHE_FREE = True
        RUN = ROOT/'runs/psd-slate-canary4x8-no-perception-cache-20260916'
    if args.execute:
        try:
            execute()
        except BaseException as error:
            save(RUN/'state.json', {'phase': 'held_requires_inspection', 'error_type': type(error).__name__,
                'time': time.time(), 'psd_training_started': False, 'full_collection_started': False})
            raise
    elif args.prepare:
        prepare()
    else:
        launch()


if __name__ == '__main__':
    main()
