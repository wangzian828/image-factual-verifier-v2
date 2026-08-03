from __future__ import annotations

from ifv_training.resource_monitor import summarize_resource_samples


def test_resource_summary_aggregates_process_tree_and_gpu_peaks() -> None:
    samples = [
        {
            "process_count": 3,
            "process_tree_rss_mib": 100.0,
            "gpu": {
                "process_memory_mib_by_physical_gpu": {"4": 1000, "5": 900},
                "whole_gpu_memory_mib_by_physical_gpu": {"4": 1100, "5": 950},
                "utilization_percent_by_physical_gpu": {"4": 20, "5": 30},
            },
        },
        {
            "process_count": 5,
            "process_tree_rss_mib": 250.0,
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
