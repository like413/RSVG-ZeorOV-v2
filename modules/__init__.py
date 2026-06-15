"""
Modules package for the referring video segmentation pipeline.
Contains TFVTG, RSVG, SAM2, and output generation modules.
"""
from .config_manager import ConfigManager
from .data_models import VideoData, ProcessingResult, AnnotationData
from .tfvtg_module import TFVTGModule
from .rsvg_module import RSVGModule
from .sam2_module import SAM2Module

__version__ = "0.1.0"

__all__ = [
    'ConfigManager',
    'VideoData',
    'ProcessingResult',
    'AnnotationData',
    'TFVTGModule',
    'RSVGModule',
    'SAM2Module'
]
