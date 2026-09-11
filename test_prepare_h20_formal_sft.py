import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('formal_sft', Path(__file__).parent / 'training/scripts/h20/prepare_formal_sft.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def template():
    d = {'--tuner_type':'full','--fsdp':'fsdp2','--sequence_parallel_size':'4',
         '--max_length':'131072','--packing':'true','--packing_length':'65536',
         '--padding_free':'true','--attn_impl':'flash_attn','--strict':'true',
         '--freeze_llm':'false','--freeze_vit':'false','--freeze_aligner':'false',
         '--per_device_train_batch_size':'1','--gradient_accumulation_steps':'1',
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


@pytest.mark.parametrize('suffix', [['--fsdp','fsdp2'],['--adapters','diagnostic'], ['--resume_from_checkpoint','old']])
def test_reject_unbound_options(suffix):
    with pytest.raises(ValueError):
        module.command_from_benchmark(template()+suffix,'train','out')


def test_no_silent_configuration_change():
    c = template()
    c[c.index('--sequence_parallel_size')+1] = '2'
    with pytest.raises(ValueError):
        module.command_from_benchmark(c,'train','out')
