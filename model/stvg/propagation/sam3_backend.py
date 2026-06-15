"""SAM3 propagation backend.

Wraps ``stage3_sam3_qwen.propagate_mask_with_sam3_tracker``, which performs a
SAM3 image box prompt on the key frame and then propagates with the SAM3 video
tracker. Same call signature as the SAM2 backend, so the two are interchangeable.
"""

from __future__ import annotations

from typing import Any, Dict

from stvg.propagation.base import BasePropagationBackend
from stvg.schemas import PropagationInput


class SAM3PropagationBackend(BasePropagationBackend):
    """Bbox -> SAM3 image box prompt -> SAM3 tracker propagation."""

    backend_name = "sam3"

    def propagate(self, propagation_input: PropagationInput) -> Dict[int, Any]:
        from stvg._legacy import load_stage3

        s3 = load_stage3()
        initial_mask = self._build_initial_mask(propagation_input)
        masks = s3.propagate_mask_with_sam3_tracker(
            propagation_input.video_path,
            propagation_input.key_frame_idx,
            initial_mask,
            self.config,
            bbox=propagation_input.bbox.to_dict(),
            num_gpus=self.num_gpus,
            stage1_meta=None,
            qwen_key_frame_path=propagation_input.key_frame_path,
        )
        return masks or {}
