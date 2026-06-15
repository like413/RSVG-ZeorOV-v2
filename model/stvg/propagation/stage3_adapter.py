"""Adapter from unified Stage 2 output to Stage 3 input.

Bridges :class:`~stvg.schemas.GroundingResult` (+ the originating
:class:`~stvg.schemas.Stage1Record`) into a validated
:class:`~stvg.schemas.PropagationInput`. This is the single choke point that
guarantees Stage 3 never crashes on a missing key: :func:`build_propagation_input`
validates up front and raises a clear ``KeyError`` naming the missing field.
"""

from __future__ import annotations

import os
from typing import Optional

from stvg.io.filesystem import vid_output_dir
from stvg.io.grounding import load_grounding_result
from stvg.io.stage1 import load_stage1_record
from stvg.schemas import GroundingResult, PropagationInput, Stage1Record


def build_propagation_input(
    stage1: Stage1Record,
    grounding: GroundingResult,
    require_bbox: bool = True,
) -> PropagationInput:
    """Combine a Stage 1 record and a grounding result into Stage 3 input.

    Raises ``KeyError`` if a field Stage 3 requires is missing (e.g. video path,
    frame index, or bbox when ``require_bbox`` is set).
    """
    prop = PropagationInput.from_stages(stage1, grounding)
    if require_bbox:
        prop.validate()
    return prop


def build_propagation_input_from_disk(
    output_dir: str,
    grounding_subdir: str,
    vid: str,
    require_bbox: bool = True,
) -> Optional[PropagationInput]:
    """Load Stage 1 + unified Stage 2 metadata from disk and adapt them.

    Returns ``None`` if either the Stage 1 record or grounding metadata is absent.
    """
    base_vid, idx = _split(vid)
    stage1_clip = os.path.join(output_dir, "stage1", base_vid, *(idx,) if idx else ())
    stage1 = load_stage1_record(stage1_clip, vid)
    if stage1 is None:
        return None
    grounding = load_grounding_result(output_dir, grounding_subdir, vid)
    if grounding is None:
        return None
    return build_propagation_input(stage1, grounding, require_bbox=require_bbox)


def _split(vid: str):
    if "_" in vid:
        base_vid, idx = vid.rsplit("_", 1)
        return base_vid, idx
    return vid, None
