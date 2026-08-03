#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import inspect
import json
import math
from pathlib import Path
import re
import textwrap
from typing import Any, Mapping


WARNING_TEXT = (
    "Detected call of `lr_scheduler.step()` before `optimizer.step()`"
)


def _load_torch(path: Path) -> Mapping[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint state must be an object: {path}")
    return value


def _source_contract() -> dict[str, Any]:
    from accelerate import Accelerator
    from accelerate.utils.deepspeed import (
        DeepSpeedEngineWrapper,
        DeepSpeedOptimizerWrapper,
    )
    from transformers import Trainer

    engine_backward = inspect.getsource(DeepSpeedEngineWrapper.backward)
    optimizer_step = inspect.getsource(DeepSpeedOptimizerWrapper.step)
    accelerator_backward = inspect.getsource(Accelerator.backward)
    trainer_epoch = inspect.getsource(Trainer._run_epoch)
    optimizer_tree = ast.parse(textwrap.dedent(optimizer_step))
    optimizer_functions = [
        item
        for item in optimizer_tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    optimizer_step_is_noop = bool(
        optimizer_functions
        and len(optimizer_functions[0].body) == 1
        and isinstance(optimizer_functions[0].body[0], ast.Pass)
    )
    checks = {
        "accelerator_routes_backward_to_deepspeed": (
            "self.deepspeed_engine_wrapped.backward" in accelerator_backward
        ),
        "deepspeed_backward_performs_engine_step": (
            "self.engine.step()" in engine_backward
        ),
        "deepspeed_optimizer_wrapper_step_is_noop": optimizer_step_is_noop,
        "trainer_calls_wrapper_step_before_scheduler": (
            0
            <= trainer_epoch.find("self.optimizer.step()")
            < trainer_epoch.find("self.lr_scheduler.step()")
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "source_files": {
            "accelerator": str(inspect.getsourcefile(Accelerator.backward)),
            "deepspeed_wrapper": str(
                inspect.getsourcefile(DeepSpeedEngineWrapper.backward)
            ),
            "trainer": str(inspect.getsourcefile(Trainer._run_epoch)),
        },
    }


def audit_scheduler(
    *,
    train_log: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    train_log = train_log.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    log_text = train_log.read_text(encoding="utf-8", errors="replace")
    warning_count = log_text.count(WARNING_TEXT)

    trainer_state_path = checkpoint / "trainer_state.json"
    scheduler_path = checkpoint / "scheduler.pt"
    model_state_paths = sorted(checkpoint.rglob("*model_states.pt"))
    if not trainer_state_path.is_file():
        raise FileNotFoundError(trainer_state_path)
    if not scheduler_path.is_file():
        raise FileNotFoundError(scheduler_path)
    if not model_state_paths:
        raise FileNotFoundError(f"no DeepSpeed model state under {checkpoint}")

    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    scheduler_state = _load_torch(scheduler_path)
    model_state = _load_torch(model_state_paths[0])
    global_step = int(trainer_state.get("global_step", -1))
    scheduler_last_epoch = int(scheduler_state.get("last_epoch", -1))
    scheduler_step_count = int(scheduler_state.get("_step_count", -1))
    deepspeed_global_steps = int(model_state.get("global_steps", -1))
    deepspeed_skipped_steps = int(model_state.get("skipped_steps", -1))
    deepspeed_global_samples = int(model_state.get("global_samples", 0))
    learning_rates = [
        float(item["learning_rate"])
        for item in trainer_state.get("log_history", []) or []
        if isinstance(item, Mapping)
        and isinstance(item.get("learning_rate"), (int, float))
    ]
    checkpoint_checks = {
        "positive_global_step": global_step > 0,
        "scheduler_last_epoch_matches_global_step": (
            scheduler_last_epoch == global_step
        ),
        "scheduler_step_count_matches_initial_plus_global_step": (
            scheduler_step_count == global_step + 1
        ),
        "deepspeed_global_steps_match_trainer": (
            deepspeed_global_steps == global_step
        ),
        "deepspeed_skipped_steps_zero": deepspeed_skipped_steps == 0,
        "deepspeed_global_samples_positive": deepspeed_global_samples > 0,
        "logged_learning_rates_finite": bool(learning_rates)
        and all(math.isfinite(value) and value >= 0 for value in learning_rates),
    }
    source_contract = _source_contract()
    passed = source_contract["passed"] and all(checkpoint_checks.values())
    classification = (
        "wrapper_false_positive"
        if passed and warning_count
        else "order_correct_no_warning"
        if passed
        else "unresolved"
    )
    return {
        "schema_version": "ifv-deepspeed-scheduler-audit-v1",
        "train_log": str(train_log),
        "checkpoint": str(checkpoint),
        "warning_count": warning_count,
        "classification": classification,
        "passed": passed,
        "source_contract": source_contract,
        "checkpoint_checks": checkpoint_checks,
        "state": {
            "trainer_global_step": global_step,
            "scheduler_last_epoch": scheduler_last_epoch,
            "scheduler_step_count": scheduler_step_count,
            "scheduler_last_lr": scheduler_state.get("_last_lr", []),
            "deepspeed_global_steps": deepspeed_global_steps,
            "deepspeed_skipped_steps": deepspeed_skipped_steps,
            "deepspeed_global_samples": deepspeed_global_samples,
            "logged_learning_rates": learning_rates,
            "model_state_path": str(model_state_paths[0]),
        },
        "explanation": (
            "Accelerate performs DeepSpeed engine.step() inside backward at the "
            "gradient-accumulation boundary. Its public optimizer wrapper step is "
            "intentionally a no-op, so PyTorch's _opt_called bookkeeping remains "
            "false even though the persisted DeepSpeed optimizer step completed "
            "before Transformers advanced the external scheduler."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove or reject the DeepSpeed scheduler-order warning."
    )
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_scheduler(
        train_log=args.train_log,
        checkpoint=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
