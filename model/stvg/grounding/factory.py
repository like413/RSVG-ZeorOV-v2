"""Factory for Stage 2 grounding strategies.

Stage 2 uses a single grounding strategy: **RSVG-ZeroOV**.

Usage::

    from stvg.grounding import create_grounding_model
    model = create_grounding_model("zeroov", config, work_dir="/path/out")

Additional strategies can still be plugged in via
:func:`register_grounding_model` without touching this module.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from stvg.grounding.base import BaseGroundingModel

# method name -> builder(config, **kwargs) -> BaseGroundingModel
_REGISTRY: Dict[str, Callable[..., BaseGroundingModel]] = {}


def register_grounding_model(name: str, builder: Callable[..., BaseGroundingModel]) -> None:
    _REGISTRY[name] = builder


def _build_zeroov(config: Dict[str, Any], **kwargs: Any) -> BaseGroundingModel:
    from stvg.grounding.zeroov_strategy import ZeroOVGroundingStrategy

    return ZeroOVGroundingStrategy(config, **kwargs)


register_grounding_model("zeroov", _build_zeroov)

#: Names of all registered grounding methods.
GROUNDING_METHODS = tuple(_REGISTRY.keys())


def create_grounding_model(method: str, config: Dict[str, Any], **kwargs: Any) -> BaseGroundingModel:
    """Instantiate a grounding strategy by name.

    Construction is cheap and lazy: heavy model weights are only loaded on the
    first :meth:`~stvg.grounding.base.BaseGroundingModel.ground` call.
    """
    try:
        builder = _REGISTRY[method]
    except KeyError:
        raise ValueError(
            f"Unknown grounding method {method!r}. "
            f"Available: {', '.join(sorted(_REGISTRY))}"
        )
    return builder(config, **kwargs)
