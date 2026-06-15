"""Pytest configuration for the STVG test suite.

Adds the ``model/`` directory to ``sys.path`` so ``import stvg`` works, and
provides lightweight fixtures (fake Stage 1 records, fake parsed outputs) so the
contract tests never need to load Qwen, SAM, or Stable Diffusion.
"""

from __future__ import annotations

import os
import sys

import pytest

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from stvg.schemas import Stage1Record  # noqa: E402


@pytest.fixture
def stage1_record() -> Stage1Record:
    """A complete Stage 1 record with all fields Stage 3 will need."""
    return Stage1Record(
        vid="2400171624_1",
        key_frame_path="/tmp/fake/key_frame.png",
        text_query="a man wearing a red shirt",
        target_object="man",
        video_path="/tmp/fake/video.mp4",
        key_frame_idx=42,
        original_frame_idx=128,
        raw_metadata={"text_query": "a man wearing a red shirt"},
    )


@pytest.fixture
def stage1_record_missing_video() -> Stage1Record:
    """A Stage 1 record missing the source video path (Stage 3 cannot run)."""
    return Stage1Record(
        vid="2400171624_2",
        key_frame_path="/tmp/fake/key_frame.png",
        text_query="a dog",
        target_object="dog",
        video_path=None,
        key_frame_idx=None,
        original_frame_idx=None,
    )
