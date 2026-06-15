"""Read/write the unified Stage 2 grounding ``metadata.json``.

Both grounding strategies persist through these helpers, guaranteeing one
on-disk schema regardless of model. Stage 3 then reads the same files.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from stvg.io.filesystem import vid_output_dir
from stvg.schemas import GroundingResult


def save_grounding_result(
    output_dir: str,
    subdir: str,
    result: GroundingResult,
    key_frame_path: Optional[str] = None,
) -> str:
    """Persist a result to ``{output_dir}/{subdir}/{base_vid}[/{idx}]/metadata.json``.

    Optionally copies the key frame next to the metadata for inspection. Returns
    the metadata path.
    """
    clip_dir = vid_output_dir(output_dir, subdir, result.vid)
    os.makedirs(clip_dir, exist_ok=True)
    if key_frame_path and os.path.isfile(key_frame_path):
        try:
            shutil.copyfile(key_frame_path, os.path.join(clip_dir, "key_frame.png"))
        except OSError:
            pass
    return result.save(clip_dir)


def load_grounding_result(output_dir: str, subdir: str, vid: str) -> Optional[GroundingResult]:
    clip_dir = vid_output_dir(output_dir, subdir, vid)
    meta_path = os.path.join(clip_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    return GroundingResult.load(meta_path)


def grounding_completed(output_dir: str, subdir: str, vid: str) -> bool:
    clip_dir = vid_output_dir(output_dir, subdir, vid)
    return os.path.isfile(os.path.join(clip_dir, "metadata.json"))
