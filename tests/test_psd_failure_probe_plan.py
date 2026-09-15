from scripts.server.probe_psd_cache_free_failure import probe_plan


def test_serial_control_is_exactly_one_budget_on_and_one_off():
    assert probe_plan(False) == [('budget-on', True), ('budget-off-control', False)]


def test_concurrent_diagnostic_is_bounded_balanced_and_uniquely_named():
    plan = probe_plan(True)
    assert len(plan) == len({label for label, _ in plan}) == 8
    assert sum(enabled for _, enabled in plan) == 4
