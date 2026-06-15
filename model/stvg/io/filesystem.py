"""Filesystem path conventions shared by every stage.

The on-disk layout (kept identical to the legacy scripts) is::

    {output_dir}/
      stage1/{base_vid}/{idx}/metadata.json, key_frame.png
      {grounding_subdir}/{base_vid}/{idx}/metadata.json   <- unified Stage 2
      {stage3_subdir}/{base_vid}/{idx}/masks/mask_*.png   <- Stage 3
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


def split_vid(vid: str) -> Tuple[str, Optional[str]]:
    """Split ``"2400171624_1"`` -> ``("2400171624", "1")``.

    A vid without a trailing ``_<idx>`` returns ``(vid, None)``.
    """
    if "_" in vid:
        base_vid, idx = vid.rsplit("_", 1)
        return base_vid, idx
    return vid, None


def vid_output_dir(output_dir: str, subdir: str, vid: str) -> str:
    """Build ``{output_dir}/{subdir}/{base_vid}[/{idx}]`` for a given vid."""
    base_vid, idx = split_vid(vid)
    parts = [output_dir, subdir, base_vid]
    if idx is not None:
        parts.append(idx)
    return os.path.join(*parts)
