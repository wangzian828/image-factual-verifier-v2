"""Prepare, but never execute, the first full-data H20 SFT command.

Use the measured H20 command as a bound template. No A100 profile or test
result is used to choose training hyperparameters. Check judge completion and
actual storage quota, stop GPU keepers and verify idle devices before execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


TEMPLATE_CONTRACT = {
    'max_length': 131072,
    'truncation_strategy': 'raise',
    'max_pixels': 262144,
    'padding_free': True,
    'sequence_parallel_size': 4,
    'loss_scale': 'ignore_empty_think',
    'enable_thinking': False,
    'add_non_thinking_prefix': False,
    'image_max_token_num': 1024,
}


def command_from_benchmark(
        command, train, output, *, save_only_model=False, save_steps=None,
        save_total_limit=1):
    if command[:2] != ['swift', 'sft'] or len(command[2:]) % 2:
        raise ValueError('expected paired swift sft options')
    options = dict(zip(command[2::2], command[3::2]))
    if len(options) * 2 != len(command) - 2:
        raise ValueError('duplicate benchmark options')
    required = {'--model_type':'qwen3_5', '--tuner_type':'full', '--fsdp':'fsdp2', '--sequence_parallel_size':'4',
        '--max_length':'131072', '--packing':'true', '--packing_length':'65536',
        '--padding_free':'true', '--attn_impl':'flash_attn', '--strict':'true',
        '--freeze_llm':'false', '--freeze_vit':'false', '--freeze_aligner':'false',
        '--per_device_train_batch_size':'1', '--gradient_accumulation_steps':'1',
        '--truncation_strategy':'delete', '--loss_scale':'ignore_empty_think',
        '--enable_thinking':'false', '--add_non_thinking_prefix':'false',
        '--max_pixels':'262144', '--split_dataset_ratio':'0',
        '--torch_dtype':'bfloat16', '--bf16':'true', '--use_logits_to_keep':'false',
        '--lazy_tokenize':'false', '--learning_rate':'1e-5'}
    if any(options.get(k) != v for k, v in required.items()):
        raise ValueError('benchmark is not the accepted H20 SP4/64K full-parameter configuration')
    if any(k in options for k in ('--resume_from_checkpoint', '--adapters', '--val_dataset')):
        raise ValueError('unexpected adapter/resume/validation option')
    if save_only_model:
        if not isinstance(save_steps, int) or save_steps <= 0:
            raise ValueError('model-only intermediate checkpoints require positive save_steps')
        if not isinstance(save_total_limit, int) or save_total_limit < 2:
            raise ValueError('model-only intermediate checkpoints require save_total_limit >= 2')
        save_options = {
            '--save_strategy': 'steps',
            '--save_steps': str(save_steps),
            '--save_total_limit': str(save_total_limit),
            '--save_only_model': 'true',
        }
    else:
        if save_steps is not None or save_total_limit != 1:
            raise ValueError('full-state epoch saving does not accept step checkpoint options')
        save_options = {
            '--save_strategy': 'epoch',
            '--save_total_limit': '1',
            '--save_only_model': 'false',
        }
    options.update({'--dataset':str(train), '--output_dir':str(output),
        '--max_steps':'-1', '--num_train_epochs':'1',
        '--load_best_model_at_end':'false', **save_options})
    # No best-checkpoint selection from the one-case validation or the test set.
    return ['swift', 'sft', *[x for pair in options.items() for x in pair]]


def prepare(
        benchmark, gate_path, processor_path, output, *,
        expected_train_rows=2578, save_only_model=False, save_steps=None,
        save_total_limit=1):
    gate = json.loads(gate_path.read_text())
    processor = json.loads(processor_path.read_text())
    if gate.get('passed') is not True or processor.get('passed') is not True:
        raise ValueError('data and processor gates must pass')
    if gate.get('schema_version') != 'ifv-sft-raw-data-gate-v2':
        raise ValueError('formal SFT requires the causal-contract v2 data gate')
    if processor.get('schema_version') != 'ifv-ms-swift-agent-processor-verification-v3':
        raise ValueError('formal SFT requires the v3 processor verification')
    if gate['processor_report']['sha256'] != sha(processor_path):
        raise ValueError('processor report changed')
    for entry in [*gate['datasets'].values(), gate['dataset_manifest']]:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('frozen dataset changed')
    if not isinstance(expected_train_rows, int) or expected_train_rows <= 0:
        raise ValueError('expected_train_rows must be a positive integer')
    if processor['counts'].get('policy:train') != expected_train_rows:
        raise ValueError(
            f'expected exactly {expected_train_rows} complete training episodes'
        )
    train_path = str(Path(gate['datasets']['train']['path']).resolve())
    train_stats = processor.get('input_tokens_by_dataset', {}).get(train_path)
    if not train_stats or train_stats.get('count') != expected_train_rows:
        raise ValueError('processor report does not bind all formal training rows')
    if train_stats['max'] > 65536:
        raise ValueError('long episode needs a separately validated packing route')
    if gate['template_contract'] != processor['template_contract']:
        raise ValueError('template binding mismatch')
    if gate['template_contract'] != TEMPLATE_CONTRACT:
        raise ValueError('formal SFT template contract changed')
    causal = gate.get('dataset_audit', {}).get('causal_contract', {})
    blockers = causal.get('production_blockers', {})
    if causal.get('passed') is not True or not blockers or any(blockers.values()):
        raise ValueError('causal policy audit has production blockers')
    release_manifest = gate_path.parent / 'manifest.json'
    independent_audit = gate_path.parent / 'independent-audit.json'
    policy_audit = gate_path.parent / 'policy-audit.json'
    for artifact in (release_manifest, independent_audit, policy_audit):
        if not artifact.is_file():
            raise ValueError(f'missing canonical release artifact: {artifact.name}')
    independent = json.loads(independent_audit.read_text())
    if independent.get('passed') is not True or independent.get('error_count') != 0:
        raise ValueError('independent canonical rebuild audit did not pass')
    release = json.loads(release_manifest.read_text())
    release_policy_audit = release.get('policy_audit', {})
    if (release_policy_audit.get('passed') is not True or
            release_policy_audit.get('sha256') != sha(policy_audit)):
        raise ValueError('canonical release does not bind the passing policy audit')
    command = command_from_benchmark(
        json.loads(benchmark.read_text()),
        gate['datasets']['train']['path'],
        output / 'output',
        save_only_model=save_only_model,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
    )
    if command[command.index('--model')+1] != gate['model']:
        raise ValueError('model does not match data preflight')
    result = {'stage':'prepared_not_started', 'training_started':False, 'command':command,
        'bindings':{'benchmark_command':sha(benchmark), 'data_preflight':sha(gate_path),
                    'processor_report':sha(processor_path), 'train':gate['datasets']['train']['sha256']},
        'canonical_release_bindings':{'manifest':sha(release_manifest),
                                      'independent_audit':sha(independent_audit),
                                      'policy_audit':sha(policy_audit)},
        'runtime_contract':{**TEMPLATE_CONTRACT,
                            'cli_truncation_strategy':'delete_maps_to_template_raise_in_ms_swift_4.4.2'},
        'distributed_environment':{'NPROC_PER_NODE':'4', 'CUDA_VISIBLE_DEVICES':'0,1,2,3',
                                   'MASTER_PORT':'29641'},
        'save_policy':(
            {
                'strategy': 'steps',
                'save_steps': save_steps,
                'save_total_limit': save_total_limit,
                'save_only_model': True,
                'optimizer_scheduler_rng_saved': False,
            }
            if save_only_model
            else {
                'strategy': 'epoch',
                'save_total_limit': 1,
                'save_only_model': False,
                'optimizer_scheduler_rng_saved': True,
            }
        ),
        'remaining_launch_gates':['complete frozen pre-training judge coverage',
            ('actual user storage allowance for the model-only checkpoint ring and later inference export'
             if save_only_model else
             'actual user storage allowance for full-state checkpoint and later inference export'),
            'stop keepers; verify all four GPUs idle; source H20 environment',
            'capture command/environment, resource samples and training exit status'],
        'limitations':([
            'model-only checkpoints cannot resume optimizer/scheduler/RNG state',
            'save_steps is an optimizer-step interval, not a percentage of the epoch',
            'preparing the command is not training or checkpoint-load acceptance',
        ] if save_only_model else [
            'epoch-only saving does not provide mid-epoch recovery',
            'preparing the command is not training or checkpoint-load acceptance',
        ])}
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'training-plan.json'
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError('existing training plan changed')
    path.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark-command','data-preflight','processor-report','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--expected-train-rows', type=int, default=2578)
    parser.add_argument('--save-only-model', action='store_true')
    parser.add_argument('--save-steps', type=int)
    parser.add_argument('--save-total-limit', type=int, default=1)
    args = parser.parse_args()
    result = prepare(
        args.benchmark_command,
        args.data_preflight,
        args.processor_report,
        args.output,
        expected_train_rows=args.expected_train_rows,
        save_only_model=args.save_only_model,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )
    print(json.dumps({'stage':result['stage'], 'bindings':result['bindings']}))
