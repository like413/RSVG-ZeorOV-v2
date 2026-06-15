"""Factory for Stage 3 propagation backends.

Stage 3 uses a single propagation backend: **SAM3**.

Usage::

    from stvg.propagation import create_propagation_backend
    backend = create_propagation_backend("sam3", config, num_gpus=1)
    masks = backend.propagate(propagation_input)
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from stvg.propagation.base import BasePropagationBackend

_REGISTRY: Dict[str, Callable[..., BasePropagationBackend]] = {}


def register_propagation_backend(name: str, builder: Callable[..., BasePropagationBackend]) -> None:
    _REGISTRY[name] = builder


def _build_sam3(config: Dict[str, Any], **kwargs: Any) -> BasePropagationBackend:
    from stvg.propagation.sam3_backend import SAM3PropagationBackend

    return SAM3PropagationBackend(config, **kwargs)


register_propagation_backend("sam3", _build_sam3)

#: Names of all registered propagation backends.
PROPAGATION_BACKENDS = tuple(_REGISTRY.keys())


def create_propagation_backend(backend: str, config: Dict[str, Any], **kwargs: Any) -> BasePropagationBackend:
    try:
        builder = _REGISTRY[backend]
    except KeyError:
        raise ValueError(
            f"Unknown propagation backend {backend!r}. "
            f"Available: {', '.join(sorted(_REGISTRY))}"
        )
    return builder(config, **kwargs)
