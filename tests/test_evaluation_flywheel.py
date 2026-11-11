"""Flywheel guards, ledger traceability and the no-training invariant."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from sim2real.evaluation import FlywheelController, load_evaluation_config

ROOT = Path(__file__).resolve().parents[1]


def _controller(tmp_path, **overrides) -> FlywheelController:
    config = load_evaluation_config()
    config = replace(config, flywheel=replace(config.flywheel, **overrides))
    return FlywheelController(
        config,
        ledger_path=tmp_path / "ledger.jsonl",
        stop_file=tmp_path / "STOP",
        now=lambda: 1_000_000.0,
    )


def _decide(controller, run_id: str, *, success: float = 0.5, failures: int = 5):
    return controller.decide(
        eval_run_id=run_id,
        success_rate=success,
        n_failures=failures,
        policy_only_failures=failures,
        requested_rollouts=100,
    )


def test_max_cycles_and_per_request_cap(tmp_path):
    controller = _controller(tmp_path, max_cycles=2, max_rollouts_per_request=7)
    first = _decide(controller, "run-a")
    second = _decide(controller, "run-b")
    third = _decide(controller, "run-c", success=0.9)
    assert first.decision == "REQUEST_TARGETED_GENERATION"
    assert first.rollouts_requested == 7
    assert second.decision == "REQUEST_TARGETED_GENERATION"
    assert third.decision == "STOP"
    assert "max_cycles" in third.reason


def test_total_rollout_budget_is_enforced(tmp_path):
    controller = _controller(
        tmp_path, max_cycles=10, max_rollouts_per_request=7, max_total_rollouts=10
    )
    first = _decide(controller, "run-a")
    second = _decide(controller, "run-b")
    third = _decide(controller, "run-c")
    assert first.rollouts_requested == 7
    assert second.rollouts_requested == 3
    assert third.decision == "STOP"
    assert "max_total_rollouts" in third.reason


def test_same_eval_run_id_is_refused_twice(tmp_path):
    controller = _controller(tmp_path)
    first = _decide(controller, "run-a")
    second = _decide(controller, "run-a")
    assert first.decision == "REQUEST_TARGETED_GENERATION"
    assert second.decision == "STOP"
    assert "idempotency" in second.reason


def test_stop_file_wins(tmp_path):
    controller = _controller(tmp_path)
    (tmp_path / "STOP").write_text("stop\n", encoding="utf-8")
    record = _decide(controller, "run-a")
    assert record.decision == "STOP"
    assert "STOP file" in record.reason


def test_consecutive_no_improvement_stops_the_loop(tmp_path):
    controller = _controller(
        tmp_path,
        max_cycles=10,
        min_success_improvement=0.1,
        max_consecutive_no_improvement=2,
    )
    first = _decide(controller, "run-a", success=0.2)
    second = _decide(controller, "run-b", success=0.2)
    third = _decide(controller, "run-c", success=0.2)
    fourth = _decide(controller, "run-d", success=0.2)
    assert first.decision == "REQUEST_TARGETED_GENERATION"
    assert second.decision == "REQUEST_TARGETED_GENERATION"
    assert third.decision == "REQUEST_TARGETED_GENERATION"
    assert fourth.decision == "STOP"
    assert "no success improvement" in fourth.reason


def test_low_failures_do_not_trigger_generation(tmp_path):
    controller = _controller(tmp_path, min_failures_to_trigger=10)
    record = _decide(controller, "run-a", failures=4)
    assert record.decision == "STOP"
    assert "below min_failures_to_trigger" in record.reason


def test_ledger_is_append_only_and_links_parent_to_child(tmp_path):
    controller = _controller(tmp_path)
    _decide(controller, "run-a")
    _decide(controller, "run-b")
    records, corrupt = controller.load_ledger()
    assert corrupt == []
    assert [record.cycle_id for record in records] == ["cycle_0001", "cycle_0002"]
    assert records[1].parent_cycle_id == records[0].cycle_id
    for record in records:
        for field in (
            "config_hash",
            "source_dataset_versions",
            "episode_ids",
            "checkpoint_hash",
            "eval_run_id",
            "failure_distribution_id",
            "rollouts_requested",
            "rollouts_generated",
            "decision",
        ):
            assert hasattr(record, field)


def test_corrupt_ledger_line_is_reported_and_skipped(tmp_path):
    controller = _controller(tmp_path)
    (tmp_path / "ledger.jsonl").write_text('{"broken": true}\n', encoding="utf-8")
    records, corrupt = controller.load_ledger()
    assert records == []
    assert len(corrupt) == 1
    record = _decide(controller, "run-a")
    assert record.decision == "REQUEST_TARGETED_GENERATION"
    status = controller.status()
    assert status["num_cycles"] == 1
    assert len(status["corrupt_lines"]) == 1


def test_no_code_path_starts_training(tmp_path):
    source = (ROOT / "src" / "sim2real" / "evaluation" / "flywheel.py").read_text(encoding="utf-8")
    for token in ("start_training", "train_vla", "run_training", "trainer.run"):
        assert token not in source
    controller = _controller(tmp_path)
    assert controller.status()["will_train"] is False
    _decide(controller, "run-a")
    status = json.loads(json.dumps(controller.status()))
    assert status["will_train"] is False
