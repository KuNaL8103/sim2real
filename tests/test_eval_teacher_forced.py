"""CPU-only tests for the teacher-forced evaluation and pre-registered criteria."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "configs/evaluation/overfit_rule.yaml"


def _load_script():
    path = ROOT / "scripts" / "eval_teacher_forced.py"
    spec = importlib.util.spec_from_file_location("eval_teacher_forced_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rule(module):
    return module.load_rule(RULE)


def _rows(*, arm: float, beats: int, total: int, gripper: float, clip: float):
    rows = []
    for index in range(total):
        rows.append(
            {
                "first_step_l1_arm": arm,
                "first10_l1": 0.5 if index < beats else 1.0,
                "hold_last_first10_l1": 1.0,
                "gripper_first_step_l1": gripper,
                "gripper_clip_delta": clip,
            }
        )
    return rows


def test_known_error_gives_exact_l1_and_hold_last_baseline():
    module = _load_script()
    rule = _rule(module)
    recorded = np.zeros((30, 8), dtype=np.float64)
    recorded[0] = np.array([0.02, -0.03, 0.04, 0.0, 0.0, 0.0, 0.0, 0.01])
    recorded[1:] = np.array([0.04, -0.06, 0.08, 0.0, 0.0, 0.0, 0.0, 0.02])
    state = np.zeros(8, dtype=np.float64)
    predictions = np.zeros((50, 8), dtype=np.float64)
    predictions[:, :] = recorded[1]
    predictions[0] = np.array([0.05, -0.09, 0.12, 0.0, 0.0, 0.0, 0.0, 0.03])
    metrics = module.compute_frame_metrics(predictions, recorded, state, 0, rule)
    expected_first = np.abs(predictions[0] - recorded[0])
    assert metrics["first_step_abs_error_per_dim"] == pytest.approx(expected_first.tolist())
    assert metrics["first_step_l1_arm"] == pytest.approx(float(expected_first[:7].mean()))
    expected_first10 = float(np.abs(predictions[:10] - recorded[:10]).mean())
    assert metrics["first10_l1"] == pytest.approx(expected_first10)
    hold = np.repeat(state[None, :], 10, axis=0)
    assert metrics["hold_last_first10_l1"] == pytest.approx(
        float(np.abs(hold - recorded[:10]).mean())
    )
    assert metrics["hold_last_first_step_l1"] == pytest.approx(
        float(np.abs(state - recorded[0]).mean())
    )


def test_first_criterion_flips_at_threshold():
    module = _load_script()
    rule = _rule(module)
    at_limit = module.evaluate_criteria(
        _rows(arm=0.010, beats=10, total=10, gripper=0.001, clip=0.001), rule
    )
    assert at_limit["criteria"]["first_step_arm_l1"]["passed"] is True
    above = module.evaluate_criteria(
        _rows(arm=0.0101, beats=10, total=10, gripper=0.001, clip=0.001), rule
    )
    assert above["criteria"]["first_step_arm_l1"]["passed"] is False


def test_second_criterion_flips_at_threshold():
    module = _load_script()
    rule = _rule(module)
    at_limit = module.evaluate_criteria(
        _rows(arm=0.001, beats=9, total=10, gripper=0.001, clip=0.001), rule
    )
    assert at_limit["criteria"]["first10_beats_hold_fraction"]["passed"] is True
    below = module.evaluate_criteria(
        _rows(arm=0.001, beats=8, total=10, gripper=0.001, clip=0.001), rule
    )
    assert below["criteria"]["first10_beats_hold_fraction"]["passed"] is False


def test_gripper_criteria_flip_at_threshold():
    module = _load_script()
    rule = _rule(module)
    at_limit = module.evaluate_criteria(
        _rows(arm=0.001, beats=10, total=10, gripper=0.005, clip=0.0019), rule
    )
    assert at_limit["criteria"]["gripper_first_step_l1"]["passed"] is True
    assert at_limit["criteria"]["gripper_clip_delta"]["passed"] is True
    above = module.evaluate_criteria(
        _rows(arm=0.001, beats=10, total=10, gripper=0.0051, clip=0.002), rule
    )
    assert above["criteria"]["gripper_first_step_l1"]["passed"] is False
    assert above["criteria"]["gripper_clip_delta"]["passed"] is False


def test_default_frame_list_matches_pre_registered_rule():
    module = _load_script()
    assert module.default_frame_list(60) == [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    assert module.default_frame_list(35) == [0, 1, 2, 3, 4, 5, 10, 20, 30]
    assert module.parse_frames("0, 5, 10", 12) == [0, 5, 10]


def test_malformed_inputs_fail_loudly(tmp_path):
    module = _load_script()
    rule = _rule(module)
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\noverfit_rule: {}\n", encoding="utf-8")
    with pytest.raises(module.TeacherForcedError, match="missing"):
        module.load_rule(bad)
    with pytest.raises(module.TeacherForcedError, match="frames must be <"):
        module.parse_frames("99", 10)
    with pytest.raises(module.TeacherForcedError, match="episode length"):
        module.default_frame_list(0)
    with pytest.raises(module.TeacherForcedError, match="predictions"):
        module.compute_frame_metrics(np.zeros((10, 7)), np.zeros((20, 8)), np.zeros(8), 0, rule)
    with pytest.raises(module.TeacherForcedError, match="frame 30 outside"):
        module.compute_frame_metrics(np.zeros((10, 8)), np.zeros((20, 8)), np.zeros(8), 30, rule)
    with pytest.raises(module.TeacherForcedError, match="NaN"):
        module.compute_frame_metrics(
            np.full((10, 8), np.nan), np.zeros((20, 8)), np.zeros(8), 0, rule
        )
    with pytest.raises(module.TeacherForcedError, match="no frame metrics"):
        module.evaluate_criteria([], rule)
