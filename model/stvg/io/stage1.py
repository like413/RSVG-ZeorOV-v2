"""Scan Stage 1 outputs into :class:`~stvg.schemas.Stage1Record` objects.

This mirrors the directory-scanning logic in the legacy ``stage2_qwen_grounding``
main, including key-frame materialization from the source video when
``key_frame.png`` is missing.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

from stvg.schemas import Stage1Record


def _resolve_key_frame(stage1_dir: str) -> Optional[str]:
    """Return ``key_frame.png`` path, materializing it from the video if needed.

    Delegates to the legacy ``resolve_key_frame_path`` when available (so behavior
    matches Stage 2 exactly); falls back to a plain existence check otherwise.
    """
    direct = os.path.join(stage1_dir, "key_frame.png")
    if os.path.isfile(direct):
        return direct
    try:
        from stvg._legacy import load_stage2

        return load_stage2().resolve_key_frame_path(stage1_dir)
    except Exception:
        return None


def load_stage1_record(stage1_dir: str, vid: str) -> Optional[Stage1Record]:
    """Build a :class:`Stage1Record` from one Stage 1 clip directory."""
    meta_path = os.path.join(stage1_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    key_frame = _resolve_key_frame(stage1_dir)
    if not key_frame:
        return None
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    if "text_query" not in metadata:
        return None
    return Stage1Record.from_metadata(vid, metadata, key_frame)


def iter_stage1_records(output_dir: str, limit: Optional[int] = None) -> Iterator[Stage1Record]:
    """Yield every Stage 1 record under ``{output_dir}/stage1``.

    Handles both the per-sentence layout (``stage1/{vid}/{idx}/``) and the flat
    layout (``stage1/{vid}/``).
    """
    stage1_base = os.path.join(output_dir, "stage1")
    if not os.path.isdir(stage1_base):
        return

    count = 0
    for vid in sorted(os.listdir(stage1_base)):
        clip_dir = os.path.join(stage1_base, vid)
        if not os.path.isdir(clip_dir):
            continue
        sub_idxs = sorted((s for s in os.listdir(clip_dir) if s.isdigit()), key=int)
        if sub_idxs:
            for idx in sub_idxs:
                record = load_stage1_record(os.path.join(clip_dir, idx), f"{vid}_{idx}")
                if record is not None:
                    yield record
                    count += 1
                    if limit is not None and count >= limit:
                        return
        else:
            record = load_stage1_record(clip_dir, vid)
            if record is not None:
                yield record
                count += 1
                if limit is not None and count >= limit:
                    return


def collect_stage1_records(output_dir: str, limit: Optional[int] = None) -> List[Stage1Record]:
    return list(iter_stage1_records(output_dir, limit=limit))
