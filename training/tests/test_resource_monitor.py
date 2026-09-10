from __future__ import annotations

from ifv_training.resource_monitor import summarize_resource_samples


def test_resource_summary_aggregates_process_tree_and_gpu_peaks() -> None:
    samples = [
        {
            "timestamp_epoch": 10.0,
            "process_count": 3,
            "process_tree_rss_mib": 100.0,
            "process_tree_cpu_seconds": 4.0,
            "gpu": {
                "process_memory_mib_by_physical_gpu": {"4": 1000, "5": 900},
                "whole_gpu_memory_mib_by_physical_gpu": {"4": 1100, "5": 950},
                "utilization_percent_by_physical_gpu": {"4": 20, "5": 30},
            },
        },
        {
            "timestamp_epoch": 12.0,
            "process_count": 5,
            "process_tree_rss_mib": 250.0,
            "process_tree_cpu_seconds": 10.0,
            "gpu": {
                "process_memory_mib_by_physical_gpu": {"4": 1500, "5": 1200},
                "whole_gpu_memory_mib_by_physical_gpu": {"4": 1600, "5": 1300},
                "utilization_percent_by_physical_gpu": {"4": 60, "5": 50},
            },
        },
    ]

    result = summarize_resource_samples(
        samples,
        command=["swift", "sft"],
        exit_code=0,
        started_at="start",
        finished_at="finish",
        wall_seconds=3.0,
        selected_gpu_ids=[4, 5],
    )

    assert result["process_tree_peak_rss_mib"] == 250.0
    assert result["process_tree_peak_process_count"] == 5
    assert result["gpu_peak_process_memory_mib_by_physical_gpu"] == {
        "4": 1500,
        "5": 1200,
    }
    assert result["gpu_mean_utilization_percent_by_physical_gpu"] == {
        "4": 40,
        "5": 40,
    }
    assert result["process_tree_cpu_core_equivalents"] == {
        "count": 1,
        "mean": 3.0,
        "median": 3.0,
        "p90": 3.0,
        "max": 3.0,
    }
    assert result["gpu_utilization_percent_by_physical_gpu"]["4"] == {
        "count": 2,
        "mean": 40,
        "median": 40,
        "p90": 56.0,
        "max": 60,
        "busy_fraction_ge_80": 0.0,
    }


def test_resource_summary_enforces_memory_tuning_envelope() -> None:
    samples = [
        {
            "timestamp_epoch": 10.0,
            "gpu": {
                "total_memory_mib_by_physical_gpu": {
                    "0": 40960,
                    "1": 40960,
                },
                "whole_gpu_memory_mib_by_physical_gpu": {
                    "0": 37000,
                    "1": 37200,
                },
                "process_memory_mib_by_physical_gpu": {
                    "0": 36900,
                    "1": 37100,
                },
                "utilization_percent_by_physical_gpu": {"0": 95, "1": 96},
                "temperature_c_by_physical_gpu": {"0": 66, "1": 67},
                "power_draw_w_by_physical_gpu": {"0": 350.5, "1": 352.0},
            },
        }
    ]

    result = summarize_resource_samples(
        samples,
        command=["swift", "sft"],
        exit_code=0,
        started_at="start",
        finished_at="finish",
        wall_seconds=3.0,
        selected_gpu_ids=[0, 1],
        memory_target_min_mib=36000,
        memory_target_max_mib=38912,
        memory_max_imbalance_mib=1024,
    )

    assert result["acceptance"]["passed"] is True
    assert result["gpu_peak_memory_imbalance_mib"] == 200
    assert result["gpu_peak_memory_fraction_by_physical_gpu"]["0"] == round(
        37000 / 40960,
        6,
    )
    assert result["gpu_temperature_c_by_physical_gpu"]["1"]["max"] == 67


def test_resource_summary_rejects_low_or_imbalanced_memory() -> None:
    samples = [
        {
            "timestamp_epoch": 10.0,
            "gpu": {
                "total_memory_mib_by_physical_gpu": {
                    "0": 40960,
                    "1": 40960,
                },
                "whole_gpu_memory_mib_by_physical_gpu": {
                    "0": 34000,
                    "1": 37200,
                },
            },
        }
    ]

    result = summarize_resource_samples(
        samples,
        command=["swift", "sft"],
        exit_code=0,
        started_at="start",
        finished_at="finish",
        wall_seconds=3.0,
        selected_gpu_ids=[0, 1],
        memory_target_min_mib=36000,
        memory_target_max_mib=38912,
        memory_max_imbalance_mib=1024,
    )

    assert result["acceptance"]["passed"] is False
    assert result["acceptance"]["checks"]["minimum_peak_memory_reached"] is False
    assert (
        result["acceptance"]["checks"]["peak_memory_imbalance_respected"]
        is False
    )
