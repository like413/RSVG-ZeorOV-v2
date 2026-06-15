"""RSVG-ZeroOV spatial grounding strategy (wraps ``stage4_zeroov.py``).

ZeroOV is mask-first: it runs Qwen cross-attention + Stable Diffusion
self-attention + fusion + SAM2.1 to produce a segmentation mask, then derives a
bbox from the mask extent and CLIP similarity scores. This strategy maps that
output onto the *same* :class:`~stvg.schemas.GroundingResult` contract used by the
Qwen strategy, so Stage 3 sees an identical schema either way.

ZeroOV is inherently filesystem-oriented (it shells out to the external
RSVG-ZeroOV repo and writes intermediate ``.npy`` files), so the strategy needs a
``work_dir``. The heavy call (:func:`process_zeroov`) is isolated in ``_infer`` and
the pure mapping in ``_normalize`` for testability.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from stvg.grounding.base import BaseGroundingModel
from stvg.schemas import BBox, GroundingResult, Stage1Record

DEFAULT_RSVG_DIR = "/mnt/data/disk2/zyu/videoVG/RSVG-ZeorOV"


class ZeroOVGroundingStrategy(BaseGroundingModel):
    """Zero-shot open-vocabulary grounding via the RSVG-ZeroOV pipeline."""

    method_name = "zeroov"

    def __init__(
        self,
        config: Dict[str, Any],
        work_dir: str,
        rsvg_dir: str = DEFAULT_RSVG_DIR,
        stage_dir_name: str = "stage4",
        qwen_model: Optional[str] = None,
        gpu_id: int = 0,
    ) -> None:
        self.config = config or {}
        self.work_dir = work_dir
        self.rsvg_dir = rsvg_dir
        self.stage_dir_name = stage_dir_name
        self.qwen_model = qwen_model
        self.gpu_id = gpu_id
        self.method = "rsvg_zeroov"

    # ------------------------------------------------------------------ #
    # Inference (heavy)
    # ------------------------------------------------------------------ #
    def _infer(self, record: Stage1Record) -> Optional[Dict[str, Any]]:
        """Run the RSVG-ZeroOV pipeline. Returns the stage4 metadata dict."""
        from stvg._legacy import load_stage4

        s4 = load_stage4()
        return s4.process_zeroov(
            key_frame_path=record.key_frame_path,
            text_query=record.text_query,
            output_dir=self.work_dir,
            vid_id=record.vid,
            config=self.config,
            rsvg_dir=self.rsvg_dir,
            gpu_id=self.gpu_id,
            stage4_dir_name=self.stage_dir_name,
            qwen_model=self.qwen_model,
        )

    def _mask_path_for(self, vid: str) -> Optional[str]:
        if "_" in vid:
            base_vid, idx = vid.rsplit("_", 1)
        else:
            base_vid, idx = vid, None
        parts = [self.work_dir, self.stage_dir_name, base_vid]
        if idx:
            parts.append(idx)
        mask_path = os.path.join(*parts, "mask.npy")
        return mask_path if os.path.exists(mask_path) else None

    # ------------------------------------------------------------------ #
    # Normalization (pure)
    # ------------------------------------------------------------------ #
    def _normalize(
        self,
        record: Stage1Record,
        metadata: Optional[Dict[str, Any]],
        mask_path: Optional[str] = None,
    ) -> GroundingResult:
        if not metadata or not metadata.get("success"):
            return GroundingResult(
                vid=record.vid,
                text_query=record.text_query,
                target_object=record.target_object,
                success=False,
                method=self.method,
                bbox=None,
                confidence=None,
                answer=None,
                mask_path=None,
                raw_output=None,
                failure_reason=(metadata or {}).get("failure_reason", "zeroov_failed"),
                extras={},
            )

        bbox = BBox.from_dict(metadata.get("bbox"))
        box_sim = metadata.get("clip_box_similarity")
        mask_sim = metadata.get("clip_mask_similarity")
        # Use the CLIP box similarity as the primary confidence proxy; expose both.
        confidence = box_sim if box_sim is not None else mask_sim
        return GroundingResult(
            vid=record.vid,
            text_query=record.text_query,
            target_object=record.target_object,
            success=True,
            method=self.method,
            bbox=bbox,
            confidence=None if confidence is None else float(confidence),
            answer=None,  # ZeroOV does not answer interrogative queries
            mask_path=mask_path,
            raw_output=None,
            failure_reason=None,
            extras={
                "clip_box_similarity": box_sim,
                "clip_mask_similarity": mask_sim,
                "mask_coverage": metadata.get("mask_coverage"),
            },
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ground(self, record: Stage1Record) -> GroundingResult:
        metadata = self._infer(record)
        mask_path = self._mask_path_for(record.vid) if metadata else None
        return self._normalize(record, metadata, mask_path)
