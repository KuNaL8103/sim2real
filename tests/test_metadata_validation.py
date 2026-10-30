"""Metadata JSON validation: bounded, safe, object-rooted JSON only."""

from __future__ import annotations

import pytest

from sim2real.metadata import MetadataError, validate_metadata_json


def test_accepts_valid_object():
    parsed = validate_metadata_json('{"source": "sim", "count": 2, "ok": true}')
    assert parsed == {"source": "sim", "count": 2, "ok": True}


def test_empty_string_means_no_metadata():
    assert validate_metadata_json("") == {}


@pytest.mark.parametrize("payload", ['{"a": 1', "not json", '{"a": }'])
def test_rejects_malformed_json(payload: str):
    with pytest.raises(MetadataError, match="invalid JSON"):
        validate_metadata_json(payload)


@pytest.mark.parametrize("payload", ["[]", "[1, 2]", "3", '"text"', "null", "true"])
def test_rejects_non_object_root(payload: str):
    with pytest.raises(MetadataError, match="root must be a JSON object"):
        validate_metadata_json(payload)


@pytest.mark.parametrize("payload", ['{"x": NaN}', '{"x": Infinity}', '{"x": -Infinity}'])
def test_rejects_non_finite_numbers(payload: str):
    with pytest.raises(MetadataError, match="non-finite"):
        validate_metadata_json(payload)


def test_rejects_oversize_payload():
    payload = '{"x": "' + ("a" * 70000) + '"}'
    with pytest.raises(MetadataError, match="exceeding"):
        validate_metadata_json(payload)


def test_rejects_non_string_keys():
    with pytest.raises(MetadataError, match="JSON string"):
        validate_metadata_json({1: "value"})  # type: ignore[arg-type]


def test_numeric_looking_json_keys_remain_strings():
    parsed = validate_metadata_json('{"1": "value"}')
    assert parsed == {"1": "value"}
    assert all(isinstance(key, str) for key in parsed)


def test_respects_custom_byte_limit():
    with pytest.raises(MetadataError, match="exceeding the 4-byte limit"):
        validate_metadata_json('{"a": 1}', max_bytes=4)
