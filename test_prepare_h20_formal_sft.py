import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('formal_sft', Path(__file__).parent / 'training/scripts/h20/prepare_formal_sft.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def template():
    d = {'--model_type':'qwen3_5','--tuner_type':'full','--fsdp':'fsdp2','--sequence_parallel_size':'4',
         '--max_length':'131072','--packing':'true','--packing_length':'65536',
         '--padding_free':'true','--attn_impl':'flash_attn','--strict':'true',
         '--freeze_llm':'false','--freeze_vit':'false','--freeze_aligner':'false',
         '--per_device_train_batch_size':'1','--gradient_accumulation_steps':'1',
         '--truncation_strategy':'delete','--loss_scale':'ignore_empty_think',
         '--enable_thinking':'false','--add_non_thinking_prefix':'false',
         '--max_pixels':'262144','--split_dataset_ratio':'0',
         '--torch_dtype':'bfloat16','--bf16':'true','--use_logits_to_keep':'false',
         '--lazy_tokenize':'false','--learning_rate':'1e-5',
         '--dataset':'old','--max_steps':'12','--save_strategy':'no'}
    return ['swift','sft',*[x for p in d.items() for x in p]]


def test_full_epoch_and_full_state():
    original = template()
    c = module.command_from_benchmark(original, 'train', 'output')
    opts = dict(zip(c[2::2], c[3::2]))
    assert opts['--dataset'] == 'train'
    assert opts['--max_steps'] == '-1' and opts['--num_train_epochs'] == '1'
    assert opts['--save_only_model'] == 'false' and opts['--save_strategy'] == 'epoch'
    assert original == template()


def test_full_epoch_with_bounded_model_only_step_checkpoints():
    c = module.command_from_benchmark(
        template(), 'train', 'output', save_only_model=True,
        save_steps=400, save_total_limit=3,
    )
    opts = dict(zip(c[2::2], c[3::2]))
    assert opts['--save_only_model'] == 'true'
    assert opts['--save_strategy'] == 'steps'
    assert opts['--save_steps'] == '400'
    assert opts['--save_total_limit'] == '3'
    assert opts['--max_steps'] == '-1' and opts['--num_train_epochs'] == '1'


@pytest.mark.parametrize(
    ('save_steps', 'save_total_limit'),
    [(None, 3), (0, 3), (400, 1)],
)
def test_model_only_checkpoints_require_bounded_intermediate_policy(
        save_steps, save_total_limit):
    with pytest.raises(ValueError):
        module.command_from_benchmark(
            template(), 'train', 'output', save_only_model=True,
            save_steps=save_steps, save_total_limit=save_total_limit,
        )


@pytest.mark.parametrize('suffix', [['--fsdp','fsdp2'],['--adapters','diagnostic'], ['--resume_from_checkpoint','old']])
def test_reject_unbound_options(suffix):
    with pytest.raises(ValueError):
        module.command_from_benchmark(template()+suffix,'train','out')


def test_no_silent_configuration_change():
    c = template()
    c[c.index('--sequence_parallel_size')+1] = '2'
    with pytest.raises(ValueError):
        module.command_from_benchmark(c,'train','out')
