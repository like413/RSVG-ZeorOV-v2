"""STVG: a high-cohesion, low-coupling toolkit for zero-shot Spatio-Temporal Video Grounding.

This package re-organizes the original ``stageN_*.py`` research scripts behind a
small set of stable interfaces:

- :mod:`stvg.schemas`      - dataclasses describing the data contract between stages.
- :mod:`stvg.grounding`    - Stage 2 grounding strategies (Qwen / ZeroOV) behind a factory.
- :mod:`stvg.propagation`  - Stage 3 mask-propagation backends (SAM2 / SAM3) behind a factory.
- :mod:`stvg.pipeline`     - thin orchestrators that glue stages together.
- :mod:`stvg.cli`          - command line entrypoints.

The original scripts are *not* modified; the strategy/backend wrappers import and
reuse their functions lazily so importing :mod:`stvg` never pulls in heavy model
dependencies (vLLM, torch, SAM, Stable Diffusion).
"""

from stvg.schemas import (
    BBox,
    GroundingResult,
    PropagationInput,
    Stage1Record,
)

__all__ = [
    "BBox",
    "GroundingResult",
    "PropagationInput",
    "Stage1Record",
]

__version__ = "0.1.0"
