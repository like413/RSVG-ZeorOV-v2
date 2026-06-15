"""Stage 3 mask-propagation backends (SAM2 / SAM3) behind a common interface."""

from stvg.propagation.base import BasePropagationBackend
from stvg.propagation.factory import (
    PROPAGATION_BACKENDS,
    create_propagation_backend,
    register_propagation_backend,
)

__all__ = [
    "BasePropagationBackend",
    "create_propagation_backend",
    "register_propagation_backend",
    "PROPAGATION_BACKENDS",
]
