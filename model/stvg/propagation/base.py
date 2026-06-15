"""Abstract base class for Stage 3 mask-propagation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from stvg.schemas import PropagationInput


class BasePropagationBackend(ABC):
    """Common interface for propagating a key-frame bbox into a per-frame mask tube.

    Implementations wrap the legacy SAM2 / SAM3 propagation functions but accept
    the unified :class:`~stvg.schemas.PropagationInput` so Stage 3 is decoupled
    from how the bbox was produced.
    """

    backend_name: str = "base"

    def __init__(self, config: Dict[str, Any], num_gpus: int = 1) -> None:
        self.config = config or {}
        self.num_gpus = num_gpus

    @abstractmethod
    def propagate(self, propagation_input: PropagationInput) -> Dict[int, "Any"]:
        """Return ``{frame_idx: binary_mask}`` for the whole clip.

        Returns an empty dict if propagation could not be performed.
        """
        raise NotImplementedError

    def _build_initial_mask(self, propagation_input: PropagationInput):
        """Filled-rectangle initial mask from the bbox, sized to the key frame."""
        import numpy as np
        from PIL import Image

        propagation_input.validate()
        bbox = propagation_input.bbox
        with Image.open(propagation_input.key_frame_path) as img:
            width, height = img.size
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[bbox.ymin : bbox.ymax, bbox.xmin : bbox.xmax] = 1
        return mask
