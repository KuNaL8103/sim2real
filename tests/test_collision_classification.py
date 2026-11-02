"""Pure contact classification, config validation, and fake-backend analogue."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from sim2real.sim.collision import (
    CollisionConfigError,
    classify_contacts,
    load_collision_config,
)
from sim2real.sim.fake_backend import FakeBackend

CONFIG = load_collision_config("configs/sim/collision.yaml")


def _classify(contacts, target="cube_red"):
    return classify_contacts(contacts, CONFIG, target)


@pytest.mark.parametrize(
    ("contacts", "target", "expected_count", "detail_word"),
    [
        ([("left_finger/geom71", "cube_red_geom", -0.001)], "cube_red", 0, "allowed"),
        ([("link5/geom40", "table", -0.002)], "cube_red", 1, "unintended"),
        ([("left_finger/geom71", "table", -0.001)], "cube_red", 1, "unintended"),
        ([("left_finger/geom71", "cube_blue_geom", -0.001)], "cube_red", 1, "unintended"),
        ([("cube_red_geom", "table", -0.0002)], "cube_red", 0, "allowed"),
        ([("cube_red_geom", "cube_green_geom", -0.001)], "cube_red", 1, "unintended"),
        ([("mystery_geom", "table", -0.001)], "cube_red", 1, "unknown"),
        ([("link1/geom10", "link2/geom12", -0.001)], "cube_red", 0, "self"),
        ([("left_finger/geom71", "table", 0.0)], "cube_red", 0, "touch"),
    ],
)
def test_classify_contacts_table(contacts, target, expected_count, detail_word):
    info = _classify(contacts, target)
    assert info.num_unintended_contacts == expected_count
    assert any(detail_word in detail for detail in info.details)


def test_target_object_accepts_a_geom_name():
    info = classify_contacts(
        [("right_finger/geom83", "cube_red_geom", -0.001)],
        CONFIG,
        "cube_red_geom",
    )
    assert info.num_unintended_contacts == 0
    assert "allowed" in info.details[0]


def test_robot_self_exclusion_can_be_disabled():
    from dataclasses import replace

    config = replace(CONFIG, ignore_robot_self=False)
    info = classify_contacts([("link1/geom10", "link2/geom12", -0.001)], config, "cube_red")
    assert info.num_unintended_contacts == 1


def test_multiple_contacts_count_each_penetration():
    info = _classify(
        [
            ("link5/geom40", "table", -0.002),
            ("left_finger/geom71", "table", -0.001),
            ("cube_red_geom", "table", -0.0002),
            ("left_finger/geom71", "cube_red_geom", -0.001),
        ]
    )
    assert info.num_unintended_contacts == 2
    assert sum(detail.startswith("unintended") for detail in info.details) == 2


def test_empty_contacts_are_clean():
    info = _classify([])
    assert info.num_unintended_contacts == 0
    assert info.details == ()


@pytest.mark.parametrize(
    "body",
    [
        {"schema_version": 2, "collision": {}},
        {
            "schema_version": 1,
            "collision": {
                "groups": {"cubes": ["cube_red_geom"]},
                "target_group": "missing",
                "allowed_contacts": ["cube_table"],
            },
        },
        {
            "schema_version": 1,
            "collision": {
                "groups": {"cubes": ["cube_red_geom"]},
                "target_group": "cubes",
                "allowed_contacts": ["not_a_rule"],
            },
        },
        {
            "schema_version": 1,
            "collision": {
                "groups": {"cubes": []},
                "target_group": "cubes",
                "allowed_contacts": ["cube_table"],
            },
        },
    ],
)
def test_invalid_collision_configs_raise(tmp_path, body):
    path = tmp_path / "collision.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(CollisionConfigError):
        load_collision_config(path)


def test_fake_backend_uses_the_same_classifier(monkeypatch):
    backend = FakeBackend("configs/sim")
    backend.set_collision_context("cube_red")
    # Force the TCP below the table top so the fake contact is generated.
    monkeypatch.setattr(
        backend,
        "_tcp",
        lambda: np.array(
            [
                [1.0, 0.0, 0.0, 0.45],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.30],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    info = backend._table_collision()
    assert info.num_unintended_contacts == 1
    assert "unintended" in info.details[0]
    assert "hand/geom_tcp" in info.details[0]
    backend.close()
