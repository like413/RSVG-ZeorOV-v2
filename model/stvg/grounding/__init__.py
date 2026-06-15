"""Stage 2 spatial-grounding strategies behind a common interface + factory."""

from stvg.grounding.base import BaseGroundingModel
from stvg.grounding.factory import (
    GROUNDING_METHODS,
    create_grounding_model,
    register_grounding_model,
)

__all__ = [
    "BaseGroundingModel",
    "create_grounding_model",
    "register_grounding_model",
    "GROUNDING_METHODS",
]
