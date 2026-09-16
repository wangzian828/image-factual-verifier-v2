from types import SimpleNamespace
import pytest
from ifv_training.psd_fsdp_resume import prepare_fsdp_resume


def resume_case(tmp_path):
    checkpoint = tmp_path / 'checkpoint-1'
    (checkpoint / 'pytorch_model_fsdp_0').mkdir(parents=True)
    (checkpoint / 'pytorch_model_fsdp_0/.metadata').write_bytes(b'metadata')
    (tmp_path / 'psd-resume-binding.json').write_text('{}')
    args = SimpleNamespace(resume_from_checkpoint=str(checkpoint), adapters=['prior-round'],
        template='ifv_psd_topk', tuner_type='lora', tuner_backend='peft')
    env = {'FSDP_VERSION': '2', 'ACCELERATE_USE_FSDP': 'true'}
    return args, env


def test_reconstructs_lora_but_keeps_native_resume_for_trainer(tmp_path):
    args, env = resume_case(tmp_path)
    saved = dict(vars(args))
    model = SimpleNamespace(peft_config={'default': True}, named_parameters=lambda: [
        ('lora_A.weight', SimpleNamespace(requires_grad=True)),
        ('base.weight', SimpleNamespace(requires_grad=False))])
    def original(cls, current, base, **kwargs):
        assert current.resume_from_checkpoint is None and current.adapters == []
        return model
    result = prepare_fsdp_resume(original, object, args, None, environ=env)
    assert result is model and vars(args) == saved


def test_restores_arguments_even_when_adapter_preparation_fails(tmp_path):
    args, env = resume_case(tmp_path); saved = dict(vars(args))
    def fail(*args, **kwargs): raise RuntimeError('setup failed')
    with pytest.raises(RuntimeError, match='setup failed'):
        prepare_fsdp_resume(fail, object, args, None, environ=env)
    assert vars(args) == saved


def test_never_accepts_frozen_zero_parameter_wrapper(tmp_path):
    args, env = resume_case(tmp_path)
    with pytest.raises(ValueError, match='trainable LoRA'):
        prepare_fsdp_resume(lambda *a, **k: SimpleNamespace(peft_config={}, named_parameters=lambda: []),
            object, args, None, environ=env)


def test_unrelated_sft_and_plain_peft_remain_unchanged(tmp_path):
    args, env = resume_case(tmp_path); args.template = 'qwen3_5'
    def original(cls, current, model, **kwargs):
        assert current.resume_from_checkpoint == args.resume_from_checkpoint
        return 'untouched'
    assert prepare_fsdp_resume(original, object, args, None, environ=env) == 'untouched'
