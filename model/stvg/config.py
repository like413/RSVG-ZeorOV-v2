"""Configuration loading for STVG.

Thin wrapper over the existing :class:`modules.config_manager.ConfigManager` so
the new package does not duplicate config logic. Falls back to a plain YAML/dict
load if the legacy module is unavailable (e.g. in a minimal test environment).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from stvg._legacy import REPO_ROOT, _ensure_paths

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the pipeline config as a plain dict."""
    path = config_path or DEFAULT_CONFIG_PATH
    _ensure_paths()
    try:
        from modules.config_manager import ConfigManager

        return ConfigManager(path).config
    except Exception:
        import yaml

        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}


def dataset_output_dir(config: Dict[str, Any], dataset: str) -> Optional[str]:
    return config.get("datasets", {}).get(dataset, {}).get("output_dir")
