"""Stage 2 -> Stage 3 handoff: the adapter must not drop keys Stage 3 needs."""

from __future__ import annotations

import pytest

from stvg.propagation.stage3_adapter import build_propagation_input
from stvg.schemas import BBox, GroundingResult, PropagationInput


def _success_result(vid: str = "2400171624_1") -> GroundingResult:
    return GroundingResult(
        vid=vid,
        text_query="a man wearing a red shirt",
        target_object="man",
        success=True,
        method="rsvg_zeroov",
        bbox=BBox(10, 20, 110, 220),
        answer="man",
    )


def test_adapter_produces_complete_propagation_input(stage1_record):
    grounding = _success_result()
    prop = build_propagation_input(stage1_record, grounding)

    assert isinstance(prop, PropagationInput)
    # All fields Stage 3 strictly requires must be populated.
    assert prop.video_path == stage1_record.video_path
    assert prop.key_frame_idx == stage1_record.original_frame_idx  # prefers original idx
    assert prop.key_frame_path == stage1_record.key_frame_path
    assert prop.bbox.to_dict() == {"xmin": 10, "ymin": 20, "xmax": 110, "ymax": 220}
    # validate() must not raise.
    prop.validate()


def test_adapter_prompt_prefers_answer_then_target_then_query(stage1_record):
    grounding = _success_result()
    grounding.answer = "boy"
    prop = build_propagation_input(stage1_record, grounding)
    assert prop.prompt == "boy"

    grounding.answer = None
    prop = build_propagation_input(stage1_record, grounding)
    assert prop.prompt == "man"  # target_object


def test_adapter_raises_clear_error_when_video_missing(stage1_record_missing_video):
    grounding = _success_result(vid="2400171624_2")
    with pytest.raises(KeyError) as exc:
        build_propagation_input(stage1_record_missing_video, grounding)
    assert "video_path" in str(exc.value)


def test_adapter_requires_bbox_for_propagation(stage1_record):
    grounding = GroundingResult(
        vid="2400171624_1",
        text_query="a man",
        target_object="man",
        success=False,
        method="rsvg_zeroov",
        bbox=None,
        failure_reason="subject_not_found",
    )
    with pytest.raises(KeyError) as exc:
        build_propagation_input(stage1_record, grounding, require_bbox=True)
    assert "bbox" in str(exc.value)

    # When the caller does not require a bbox (Case 3/4), no exception.
    prop = build_propagation_input(stage1_record, grounding, require_bbox=False)
    assert prop.bbox is None
    assert prop.grounding_success is False


def test_propagation_input_to_dict_is_serializable(stage1_record):
    prop = build_propagation_input(stage1_record, _success_result())
    data = prop.to_dict()
    assert data["bbox"] == {"xmin": 10, "ymin": 20, "xmax": 110, "ymax": 220}
    assert set(("vid", "video_path", "key_frame_idx", "bbox", "key_frame_path")).issubset(data)
