"""Lazy importers for the original ``stageN_*.py`` research scripts.

The legacy scripts live in the ``model/`` directory (next to the ``stvg``
package) and have heavy top-level imports (vLLM, torch, ...). They are imported
*lazily and by file path* so that:

1. Importing :mod:`stvg` (and running the unit tests) never triggers those heavy
   imports.
2. The ``model/`` directory is added to ``sys.path`` first, so the legacy scripts'
   own ``import stage3_sam3_qwen`` / ``from modules.config_manager import ...``
   statements keep working unchanged.
"""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType

# model/  (directory that contains the legacy stageN_*.py scripts and this package)
MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# repo root (contains the `modules` package used by the legacy scripts)
REPO_ROOT = os.path.dirname(MODEL_DIR)


def _ensure_paths() -> None:
    for p in (MODEL_DIR, REPO_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)


def import_legacy(module_name: str) -> ModuleType:
    """Import a legacy top-level module (e.g. ``stage2_qwen_grounding``)."""
    _ensure_paths()
    return importlib.import_module(module_name)


def load_stage2() -> ModuleType:
    return import_legacy("stage2_qwen_grounding")


def load_stage4() -> ModuleType:
    return import_legacy("stage4_zeroov")


def load_stage3() -> ModuleType:
    return import_legacy("stage3_sam3_qwen")
