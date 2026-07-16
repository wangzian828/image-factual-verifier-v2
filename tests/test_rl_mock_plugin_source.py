from __future__ import annotations

from pathlib import Path


def test_rl_mock_is_an_ms_swift_gym_plugin_not_a_trainer() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "ifv_training"
        / "rl_mock_plugin.py"
    ).read_text(encoding="utf-8")

    assert 'envs["ifv_mock_search"]' in source
    assert "class IFVMockSearchEnv(Env)" in source
    assert "class IFVMockRolloutReward" not in source
    assert "Trainer" not in source
    assert "optimizer" not in source.casefold()
