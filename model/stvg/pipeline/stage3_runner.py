"""Stage 3 runner helpers.

For full-dataset, production runs the project keeps using the battle-tested
``stage3_sam3_qwen.py`` entrypoint (see :mod:`stvg.cli.stage3_propagation`, which
selects the backend via flags). This module provides a small in-process helper
that propagates a single clip through the unified backend factory -- handy for
demos, tests, and notebooks.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from stvg.propagation import create_propagation_backend
from stvg.propagation.stage3_adapter import build_propagation_input_from_disk


def run_stage3_clip(
    output_dir: str,
    vid: str,
    config: Dict[str, Any],
    backend: str = "sam3",
    grounding_subdir: str = "grounding",
    stage3_subdir: str = "stage3",
    num_gpus: int = 1,
    save_masks: bool = True,
) -> Optional[Dict[int, Any]]:
    """Propagate one clip's grounding bbox into a per-frame mask tube.

    Returns ``{frame_idx: mask}`` or ``None`` if inputs are missing.
    """
    prop = build_propagation_input_from_disk(output_dir, grounding_subdir, vid)
    if prop is None:
        print(f"Stage 3: missing stage1/grounding inputs for vid={vid!r}")
        return None
    if not prop.grounding_success or prop.bbox is None:
        print(f"Stage 3: no usable bbox for vid={vid!r}, skipping propagation")
        return None

    engine = create_propagation_backend(backend, config, num_gpus=num_gpus)
    masks = engine.propagate(prop)
    if save_masks and masks:
        _save_masks(output_dir, stage3_subdir, vid, masks)
    return masks


def _save_masks(output_dir: str, stage3_subdir: str, vid: str, masks: Dict[int, Any]) -> None:
    import numpy as np
    from PIL import Image

    from stvg.io.filesystem import vid_output_dir

    masks_dir = os.path.join(vid_output_dir(output_dir, stage3_subdir, vid), "masks")
    os.makedirs(masks_dir, exist_ok=True)
    for frame_idx, mask in masks.items():
        arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
        Image.fromarray(arr).save(os.path.join(masks_dir, f"mask_{int(frame_idx):05d}.png"))
