import sys
from types import ModuleType, SimpleNamespace

from ifv_training.psd_kernel_trace import install_fla_tuning_trace


def test_trace_records_only_first_kernel_choice(monkeypatch, caplog):
    class Key:
        @staticmethod
        def build(arg_names, keys, args, kwargs):
            return SimpleNamespace(autotune_key=(args[0],))

        @staticmethod
        def key_hash(key):
            return str(key[0])

    class Tuner:
        arg_names = ["shape"]
        keys = ["shape"]
        kernel_name = "kernel"

        def __init__(self):
            self.cache = {}

        def run(self, shape):
            self.cache[(shape,)] = SimpleNamespace(
                kwargs={"block": 32}, num_warps=4, num_stages=2, num_ctas=1)
            return shape

    for name in ("fla", "fla.ops", "fla.ops.utils"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    cache = ModuleType("fla.ops.utils.cache")
    cache.AutotuneKey = Key
    cache.CachedAutotuner = Tuner
    monkeypatch.setitem(sys.modules, "fla.ops.utils.cache", cache)
    monkeypatch.setenv("IFV_PSD_FLA_TRACE", "1")
    install_fla_tuning_trace()
    install_fla_tuning_trace()
    tuner = Tuner()
    assert tuner.run(7) == 7
    assert tuner.run(7) == 7
    assert caplog.text.count("IFV PSD FLA") == 1
    assert "key=7" in caplog.text
    assert "('block', 32)" in caplog.text
