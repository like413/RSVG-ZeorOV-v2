"""Contract tests: the ZeroOV strategy must emit the canonical Stage 2 schema.

Inference is stubbed (no model weights loaded) so we only exercise the
normalization path that defines the data contract Stage 3 consumes.
"""

from __future__ import annotations

from stvg.grounding.zeroov_strategy import ZeroOVGroundingStrategy
from stvg.schemas import GROUNDING_METADATA_KEYS, GroundingResult


def _zeroov_result(record, monkeypatch, metadata):
    strategy = ZeroOVGroundingStrategy({}, work_dir="/tmp/zeroov")
    monkeypatch.setattr(strategy, "_infer", lambda r: metadata)
    monkeypatch.setattr(strategy, "_mask_path_for", lambda vid: "/tmp/zeroov/mask.npy")
    return strategy.ground(record)


def test_zeroov_success_emits_canonical_schema(stage1_record, monkeypatch):
    result = _zeroov_result(
        stage1_record,
        monkeypatch,
        {
            "success": True,
            "bbox": {"xmin": 10, "ymin": 20, "xmax": 110, "ymax": 220},
            "clip_box_similarity": 0.81,
            "clip_mask_similarity": 0.75,
            "mask_coverage": 12.3,
        },
    )

    assert isinstance(result, GroundingResult)
    meta = result.to_metadata()
    assert tuple(meta.keys()) == GROUNDING_METADATA_KEYS
    assert result.success is True
    assert meta["bbox"] == {"xmin": 10, "ymin": 20, "xmax": 110, "ymax": 220}
    assert meta["method"] == "rsvg_zeroov"
    # ZeroOV-only signals are preserved under extras.
    assert meta["extras"]["clip_box_similarity"] == 0.81
    assert meta["extras"]["mask_coverage"] == 12.3
    # CLIP box similarity is surfaced as the confidence proxy.
    assert isinstance(result.confidence, float)


def test_zeroov_failure_emits_canonical_schema(stage1_record, monkeypatch):
    result = _zeroov_result(
        stage1_record,
        monkeypatch,
        {"success": False, "failure_reason": "zeroov_failed"},
    )

    meta = result.to_metadata()
    assert set(meta.keys()) == set(GROUNDING_METADATA_KEYS)
    assert result.success is False
    assert meta["bbox"] is None
    assert meta["failure_reason"] == "zeroov_failed"


def test_zeroov_handles_missing_metadata(stage1_record, monkeypatch):
    result = _zeroov_result(stage1_record, monkeypatch, None)
    meta = result.to_metadata()
    assert result.success is False
    assert meta["bbox"] is None
    assert meta["failure_reason"]
