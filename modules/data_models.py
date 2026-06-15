"""
Data models for the referring video segmentation pipeline.
"""
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
import numpy as np
import json


@dataclass
class VideoData:
    """Data model for video information."""
    video_id: str
    video_path: str
    text_query: str
    fps: float
    num_frames: int
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoData':
        """Create VideoData from dictionary."""
        return cls(**data)


@dataclass
class ProcessingResult:
    """Data model for pipeline processing results."""
    video_id: str
    text_query: str
    key_frame_idx: int
    similarity_scores: np.ndarray
    masks: Dict[int, np.ndarray]
    output_dir: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Args:
            include_arrays: If True, convert numpy arrays to lists for JSON serialization.
                          If False, exclude large array data.
        """
        result = {
            'video_id': self.video_id,
            'text_query': self.text_query,
            'key_frame_idx': self.key_frame_idx,
            'output_dir': self.output_dir,
            'processing_time': self.processing_time,
            'success': self.success,
            'error_message': self.error_message
        }
        
        if include_arrays:
            result['similarity_scores'] = self.similarity_scores.tolist() if isinstance(self.similarity_scores, np.ndarray) else self.similarity_scores
            result['num_masks'] = len(self.masks)
        else:
            result['num_frames'] = len(self.similarity_scores) if isinstance(self.similarity_scores, np.ndarray) else 0
            result['num_masks'] = len(self.masks)
        
        return result
    
    def to_json(self, include_arrays: bool = False) -> str:
        """
        Convert to JSON string.
        
        Args:
            include_arrays: If True, include array data in JSON output.
        """
        return json.dumps(self.to_dict(include_arrays=include_arrays), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingResult':
        """
        Create ProcessingResult from dictionary.
        Note: This creates a minimal result without masks and similarity scores.
        """
        # Convert similarity_scores back to numpy array if present
        if 'similarity_scores' in data and isinstance(data['similarity_scores'], list):
            data['similarity_scores'] = np.array(data['similarity_scores'])
        else:
            data['similarity_scores'] = np.array([])
        
        # Set empty masks dict if not present
        if 'masks' not in data:
            data['masks'] = {}
        
        return cls(**data)


@dataclass
class AnnotationData:
    """Data model for VidSTG annotation information."""
    vid: str
    captions: List[Dict[str, Any]]
    temporal_gt: Dict[str, int]
    fps: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnnotationData':
        """Create AnnotationData from dictionary."""
        return cls(**data)
