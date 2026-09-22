from pathlib import Path

import pytest

from scripts.server import recover_old1000_cache_off as recovery


def test_only_attested_sft3_command_can_be_switched(monkeypatch, tmp_path):
    model = tmp_path / "frozen-sft3"
    monkeypatch.setattr(recovery, "MODEL", model)
    original = ["python", "-m", "serve", str(model), "--no-enable-prefix-caching",
                "--mamba-cache-mode", "none", "--port", "19002"]
    assert recovery.cache_on_from_original(original) == [
        "python", "-m", "serve", str(model),
        "--mamba-cache-mode", "align", "--port", "19002", "--enable-prefix-caching"]
    with pytest.raises(ValueError, match="frozen cache-off"):
        recovery.cache_on_from_original(original[:-1] + ["19003", "--enable-prefix-caching"])
    with pytest.raises(ValueError, match="frozen cache-off"):
        recovery.cache_on_from_original(original[:3] + [str(tmp_path / "wrong-model")] + original[4:])
