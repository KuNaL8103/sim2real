"""Feature-space round-trip, clipping and determinism tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from sim2real.evaluation import load_profile_config
from sim2real.randomization import (
    FEATURE_DIM,
    FEATURE_NAMES,
    feature_bounds,
    from_feature_vector,
    sample,
    to_feature_vector,
)

PROFILE = load_profile_config("default")


def test_feature_names_and_bounds_match_the_profile():
    assert len(FEATURE_NAMES) == FEATURE_DIM
    bounds = feature_bounds(PROFILE)
    assert set(bounds) == set(FEATURE_NAMES)
    for low, high in bounds.values():
        assert low <= high
    assert bounds["object_x"] == tuple(PROFILE.object_pose.translation_bounds_m[0])


def test_round_trip_reproduces_the_feature_vector():
    for index in range(10):
        sample_obj = sample(PROFILE, index)
        vector = to_feature_vector(sample_obj, PROFILE)
        rebuilt, report = from_feature_vector(vector, PROFILE, PROFILE.seed, index)
        restored = to_feature_vector(rebuilt, PROFILE)
        assert np.allclose(restored, vector, atol=1.0e-9)
        assert report["clipped_features"] == []
        assert report["separation_repaired"] == []


def test_out_of_bounds_features_are_clipped_and_reported():
    vector = np.full(FEATURE_DIM, 0.5)
    vector[0] = -0.25
    vector[1] = 1.75
    rebuilt, report = from_feature_vector(vector, PROFILE, 1234, 0)
    assert set(report["clipped_features"]) == {"object_x", "object_y"}
    restored = to_feature_vector(rebuilt, PROFILE)
    assert np.all((restored >= 0.0) & (restored <= 1.0))
    assert restored[0] == 0.0
    assert restored[1] == 1.0


def test_groups_outside_the_feature_space_stay_deterministic():
    vector = np.full(FEATURE_DIM, 0.4)
    first, first_report = from_feature_vector(vector, PROFILE, 77, 3)
    second, second_report = from_feature_vector(vector, PROFILE, 77, 3)
    assert first.to_dict() == second.to_dict()
    assert first_report == second_report
    # Texture and distractors are not in the feature space and come from the
    # normal seeded sampler for (base_seed, episode_index).
    base = sample(replace(PROFILE, seed=77), 3)
    assert first.texture == base.texture
    assert first.distractors == base.distractors


def test_a_targeted_position_too_close_to_another_cube_is_repaired():
    # Force the target cube onto the nominal green cube position.
    vector = to_feature_vector(sample(PROFILE, 0), PROFILE)
    bounds = feature_bounds(PROFILE)
    nominal_green = PROFILE.scene.nominal_positions["cube_green"]
    vector[0] = (nominal_green[0] - bounds["object_x"][0]) / (
        bounds["object_x"][1] - bounds["object_x"][0]
    )
    vector[1] = (nominal_green[1] - bounds["object_y"][0]) / (
        bounds["object_y"][1] - bounds["object_y"][0]
    )
    rebuilt, report = from_feature_vector(vector, PROFILE, 99, 0)
    assert report["separation_repaired"]
    target = rebuilt.object_pose.poses["cube_red"]
    for name, pose in rebuilt.object_pose.poses.items():
        if name == "cube_red":
            continue
        distance = np.linalg.norm(target[:2, 3] - pose[:2, 3])
        assert distance >= PROFILE.object_pose.min_separation_m
