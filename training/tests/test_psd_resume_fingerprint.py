import sys
from types import ModuleType, SimpleNamespace

from ifv_training import psd_resume_fingerprint


def test_resume_fingerprint_is_bounded_and_does_not_change_loss(monkeypatch, caplog):
    class Trainer:
        state = SimpleNamespace(global_step=1)

        def compute_loss(self, model, inputs):
            return SimpleNamespace(detach=lambda: SimpleNamespace(item=lambda: 6.0))

    class Model:
        def named_parameters(self):
            yield "layer.lora_A", 7

    for name in ("swift", "swift.trainers"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    module = ModuleType("swift.trainers.seq2seq_trainer")
    module.Seq2SeqTrainer = Trainer
    monkeypatch.setitem(sys.modules, "swift.trainers.seq2seq_trainer", module)
    monkeypatch.setenv("IFV_PSD_RESUME_FINGERPRINT", "1")
    monkeypatch.setattr(psd_resume_fingerprint, "_summary", lambda value: {"summary": value})
    psd_resume_fingerprint.install_resume_fingerprint()
    psd_resume_fingerprint.install_resume_fingerprint()
    trainer = Trainer()
    for _ in range(5):
        loss = trainer.compute_loss(Model(), {"input_ids": 6})
        assert loss.detach().item() == 6
    assert caplog.text.count("IFV PSD resume fingerprint") == 4
    assert caplog.text.count("IFV PSD resume loss") == 4
    assert "lora_A" in caplog.text
