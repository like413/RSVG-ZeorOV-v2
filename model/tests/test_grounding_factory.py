"""Factory tests: method/backend names resolve to the right class, lazily."""

from __future__ import annotations

import pytest

from stvg.grounding import GROUNDING_METHODS, create_grounding_model
from stvg.grounding.base import BaseGroundingModel
from stvg.grounding.zeroov_strategy import ZeroOVGroundingStrategy
from stvg.propagation import PROPAGATION_BACKENDS, create_propagation_backend
from stvg.propagation.base import BasePropagationBackend
from stvg.propagation.sam3_backend import SAM3PropagationBackend


def test_only_zeroov_grounding_registered():
    assert set(GROUNDING_METHODS) == {"zeroov"}


def test_create_zeroov_strategy(tmp_path):
    model = create_grounding_model("zeroov", {}, work_dir=str(tmp_path))
    assert isinstance(model, ZeroOVGroundingStrategy)
    assert isinstance(model, BaseGroundingModel)
    assert model.method_name == "zeroov"


def test_unknown_grounding_method_raises():
    with pytest.raises(ValueError):
        create_grounding_model("qwen", {})


def test_only_sam3_propagation_registered():
    assert set(PROPAGATION_BACKENDS) == {"sam3"}


def test_create_sam3_backend():
    backend = create_propagation_backend("sam3", {})
    assert isinstance(backend, SAM3PropagationBackend)
    assert isinstance(backend, BasePropagationBackend)
    assert backend.backend_name == "sam3"


def test_unknown_propagation_backend_raises():
    with pytest.raises(ValueError):
        create_propagation_backend("sam2", {})
