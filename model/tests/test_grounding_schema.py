"""Schema-level tests: BBox normalization, GroundingResult (de)serialization."""

from __future__ import annotations

import json

import pytest

from stvg.schemas import (
    GROUNDING_METADATA_KEYS,
    BBox,
    GroundingResult,
)


def test_bbox_casts_to_int_and_computes_geometry():
    bbox = BBox(10.0, 20.0, 110.0, 220.0)
    assert (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax) == (10, 20, 110, 220)
    assert bbox.width == 100
    assert bbox.height == 200
    assert bbox.area == 20000


def test_bbox_rejects_inverted_coordinates():
    with pytest.raises(ValueError):
        BBox(100, 10, 10, 100)


def test_bbox_from_xyxy_and_dict_roundtrip():
    bbox = BBox.from_xyxy((1, 2, 3, 4))
    assert bbox.to_dict() == {"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}
    assert BBox.from_dict(bbox.to_dict()).to_dict() == bbox.to_dict()
    assert BBox.from_xyxy(None) is None
    assert BBox.from_dict(None) is None


def test_bbox_from_dict_missing_key_raises():
    with pytest.raises(KeyError):
        BBox.from_dict({"xmin": 0, "ymin": 0, "xmax": 5})


def test_grounding_metadata_has_canonical_keys():
    result = GroundingResult(
        vid="v1",
        text_query="a cat",
        success=True,
        method="qwen",
        bbox=BBox(0, 0, 10, 10),
    )
    meta = result.to_metadata()
    assert tuple(meta.keys()) == GROUNDING_METADATA_KEYS


def test_grounding_result_json_roundtrip():
    result = GroundingResult(
        vid="v1",
        text_query="a cat",
        target_object="cat",
        success=True,
        method="zeroov",
        bbox=BBox(1, 2, 30, 40),
        confidence=0.87,
        answer=None,
        mask_path="/tmp/mask.npy",
        extras={"clip_box_similarity": 0.9},
    )
    meta = result.to_metadata()
    # Must be JSON serializable.
    restored = GroundingResult.from_metadata(json.loads(json.dumps(meta)))
    assert restored.to_metadata() == meta
    assert restored.bbox.to_dict() == {"xmin": 1, "ymin": 2, "xmax": 30, "ymax": 40}


def test_failed_grounding_result_has_no_bbox():
    result = GroundingResult(
        vid="v1",
        text_query="a cat",
        success=False,
        method="qwen",
        failure_reason="subject_not_found",
    )
    meta = result.to_metadata()
    assert meta["bbox"] is None
    assert meta["success"] is False
    assert meta["failure_reason"] == "subject_not_found"
