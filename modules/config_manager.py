"""
Configuration Manager for the referring video segmentation pipeline.
Handles loading, saving, and accessing configuration parameters.
"""

import yaml
import os
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigManager:
    """Manages configuration parameters for the pipeline."""
    
    DEFAULT_CONFIG = {
        'tfvtg': {
            'model_name': 'blip2_image_text_matching',
            'model_type': 'coco',
            'fps': 3.0,
            'batch_size': 128
        },
        'rsvg': {
            'model_path': '/home/xdu/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct',
            'attention_layers': [16, 17, 18, 19],
            'layer_weights': [0.1, 0.1, 0.3, 0.5],
            'mask_threshold': 0.3
        },
        'sam2': {
            'model_path': '/home/xdu/.cache/modelscope/hub/models/facebook/sam2.1-hiera-base-plus',
            'points_per_side': 32
        },
        'output': {
            'save_masks': True,
            'save_similarity_plot': True,
            'overlay_alpha': 0.5,
            'overlay_color': [0, 255, 0]
        },
        'device': 'cuda',
        'log_level': 'INFO'
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Optional path to YAML configuration file.
                        If None, uses default configuration.
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path is not None:
            self.load_from_file(config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        Supports nested keys using dot notation (e.g., 'tfvtg.fps').
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        Supports nested keys using dot notation (e.g., 'tfvtg.fps').
        
        Args:
            key: Configuration key (supports dot notation for nested keys)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        Merges with existing configuration.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        if loaded_config is not None:
            self._merge_configs(self.config, loaded_config)
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            config_path: Path to save YAML configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def _merge_configs(self, base: Dict, update: Dict) -> None:
        """
        Recursively merge update dictionary into base dictionary.
        
        Args:
            base: Base configuration dictionary (modified in place)
            update: Update configuration dictionary
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """
        Get complete configuration dictionary.
        
        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation of ConfigManager."""
        return f"ConfigManager(config={self.config})"
