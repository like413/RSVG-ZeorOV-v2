"""
Output generation and visualization module for the referring video segmentation pipeline.
Handles creation of annotated videos, mask saving, similarity plots, and metadata generation.
"""
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import logging

from utils.video_utils import load_video, get_video_metadata
from utils.mask_utils import overlay_mask_on_frame, save_mask_png


logger = logging.getLogger(__name__)


class OutputGenerator:
    """
    Generates visualization outputs and saves results to disk.
    
    Handles:
    - Creating output directory structure
    - Generating annotated videos with mask overlays
    - Saving individual frame masks
    - Creating similarity curve plots
    - Generating metadata JSON files
    """
    
    def __init__(self, output_root: str):
        """
        Initialize OutputGenerator with output root directory.
        
        Args:
            output_root: Root directory for all output files
        """
        self.output_root = output_root
        logger.info(f"OutputGenerator initialized with output root: {output_root}")
    
    def create_output_dir(self, video_id: str) -> str:
        """
        Create output directory structure for a video.
        
        Args:
            video_id: Unique identifier for the video
            
        Returns:
            Path to the created output directory
        """
        output_dir = os.path.join(self.output_root, video_id)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def save_annotated_video(
        self,
        video_path: str,
        masks: Dict[int, np.ndarray],
        output_path: str,
        text_query: str,
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
        overlay_alpha: float = 0.5
    ) -> None:
        """
        Generate annotated video with mask overlays and text query.
        
        Overlays masks on video frames with configurable color and transparency,
        adds text query as overlay, and encodes with H264 codec at original FPS using mediapy.
        
        Args:
            video_path: Path to the original video file
            masks: Dictionary mapping frame indices to binary masks
            output_path: Path to save the annotated video
            text_query: Text query to display on each frame
            overlay_color: RGB color tuple for mask overlay (default: green)
            overlay_alpha: Transparency value in [0, 1] (default: 0.5)
            
        Raises:
            FileNotFoundError: If video file does not exist
            ValueError: If video cannot be opened or processed
        """
        logger.info(f"Generating annotated video: {output_path}")
        
        # Load video and get metadata
        cap = load_video(video_path)
        metadata = get_video_metadata(video_path)
        
        fps = metadata['fps']
        num_frames = metadata['num_frames']
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Collect all annotated frames
        annotated_frames = []
        
        try:
            frame_idx = 0
            while frame_idx < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB for mediapy
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Apply mask overlay if available for this frame
                if frame_idx in masks:
                    mask = masks[frame_idx]
                    # overlay_mask_on_frame expects BGR, so convert back temporarily
                    frame_bgr = overlay_mask_on_frame(
                        frame, mask, color=overlay_color, alpha=overlay_alpha
                    )
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Add text query overlay
                # Position text at top-left with background for readability
                text = text_query
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_color = (255, 255, 255)  # White text (RGB)
                bg_color = (0, 0, 0)  # Black background (RGB)
                
                # Convert back to BGR for OpenCV text operations
                frame_bgr_for_text = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )
                
                # Draw background rectangle
                padding = 10
                cv2.rectangle(
                    frame_bgr_for_text,
                    (padding, padding),
                    (padding + text_width + padding, padding + text_height + padding),
                    bg_color,
                    -1  # Filled rectangle
                )
                
                # Draw text
                cv2.putText(
                    frame_bgr_for_text,
                    text,
                    (padding + padding // 2, padding + text_height + padding // 2),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA
                )
                
                # Convert back to RGB for mediapy
                frame_rgb_final = cv2.cvtColor(frame_bgr_for_text, cv2.COLOR_BGR2RGB)
                annotated_frames.append(frame_rgb_final)
                
                frame_idx += 1
            
            # Write video using mediapy with ffmpeg
            logger.info(f"Writing {len(annotated_frames)} frames to video using mediapy...")
            media.write_video(output_path, annotated_frames, fps=fps, codec='h264')
            
            logger.info(f"Annotated video saved: {output_path} ({len(annotated_frames)} frames)")
        
        finally:
            cap.release()

    def save_frame_masks(
        self,
        masks: Dict[int, np.ndarray],
        output_dir: str
    ) -> None:
        """
        Save individual frame masks as PNG files.
        
        Creates a masks subdirectory and saves each frame mask with zero-padded naming.
        
        Args:
            masks: Dictionary mapping frame indices to binary masks
            output_dir: Output directory where masks subdirectory will be created
        """
        logger.info(f"Saving {len(masks)} frame masks to: {output_dir}")
        
        # Create masks subdirectory
        masks_dir = os.path.join(output_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
        
        # Save each mask with zero-padded naming
        for frame_idx, mask in masks.items():
            # Format: frame_000000.png, frame_000001.png, etc.
            mask_filename = f"frame_{frame_idx:06d}.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            save_mask_png(mask, mask_path)
        
        logger.info(f"Saved {len(masks)} masks to: {masks_dir}")

    def save_similarity_plot(
        self,
        similarity_scores: np.ndarray,
        key_frame_idx: int,
        output_path: str,
        text_query: Optional[str] = None
    ) -> None:
        """
        Create and save similarity curve plot.
        
        Creates a matplotlib figure showing similarity scores over time,
        marks the key frame with a red vertical line, and saves as PNG.
        
        Args:
            similarity_scores: Array of similarity scores for each frame
            key_frame_idx: Index of the key frame to mark
            output_path: Path to save the plot image
            text_query: Optional text query to include in title
        """
        logger.info(f"Generating similarity plot: {output_path}")
        
        # Create figure and axis
        plt.figure(figsize=(12, 6))
        
        # Plot similarity scores
        frame_indices = np.arange(len(similarity_scores))
        plt.plot(frame_indices, similarity_scores, 'b-', linewidth=2, label='Similarity Score')
        
        # Mark key frame with red vertical line
        plt.axvline(
            x=key_frame_idx,
            color='r',
            linestyle='--',
            linewidth=2,
            label=f'Key Frame (idx={key_frame_idx})'
        )
        
        # Add labels and title
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Similarity Score', fontsize=12)
        
        if text_query:
            plt.title(f'Frame-Text Similarity Curve\nQuery: "{text_query}"', fontsize=14)
        else:
            plt.title('Frame-Text Similarity Curve', fontsize=14)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(loc='best', fontsize=10)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save plot
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Similarity plot saved: {output_path}")

    def save_metadata(
        self,
        metadata: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Save processing metadata as formatted JSON file.
        
        Collects all processing information including video ID, text query,
        key frame index, similarity scores, and timestamps.
        
        Args:
            metadata: Dictionary containing processing metadata
            output_path: Path to save the JSON file
            
        Expected metadata keys:
            - video_id: Video identifier
            - text_query: Text query used for grounding
            - key_frame_idx: Index of the selected key frame
            - similarity_scores: List of similarity scores (optional)
            - processing_start: Start timestamp
            - processing_end: End timestamp
            - processing_time: Total processing time in seconds
            - num_frames: Total number of frames
            - num_masks: Number of masks generated
            - success: Whether processing succeeded
            - error_message: Error message if processing failed (optional)
        """
        logger.info(f"Saving metadata: {output_path}")
        
        # Add timestamp if not present
        if 'processing_end' not in metadata:
            metadata['processing_end'] = datetime.now().isoformat()
        
        # Convert numpy arrays to lists for JSON serialization
        metadata_serializable = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                metadata_serializable[key] = value.item()
            else:
                metadata_serializable[key] = value
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as formatted JSON
        with open(output_path, 'w') as f:
            json.dump(metadata_serializable, f, indent=2)
        
        logger.info(f"Metadata saved: {output_path}")
    
    def generate_complete_output(
        self,
        video_id: str,
        video_path: str,
        text_query: str,
        key_frame_idx: int,
        similarity_scores: np.ndarray,
        masks: Dict[int, np.ndarray],
        processing_start: str,
        processing_end: str,
        processing_time: float,
        success: bool = True,
        error_message: Optional[str] = None,
        save_masks: bool = True,
        save_similarity_plot: bool = True,
        overlay_color: Tuple[int, int, int] = (0, 255, 0),
        overlay_alpha: float = 0.5
    ) -> str:
        """
        Generate all output files for a processed video.
        
        This is a convenience method that creates the output directory and
        generates all output files (annotated video, masks, plot, metadata).
        
        Args:
            video_id: Unique identifier for the video
            video_path: Path to the original video file
            text_query: Text query used for grounding
            key_frame_idx: Index of the selected key frame
            similarity_scores: Array of similarity scores
            masks: Dictionary mapping frame indices to masks
            processing_start: Start timestamp (ISO format)
            processing_end: End timestamp (ISO format)
            processing_time: Total processing time in seconds
            success: Whether processing succeeded
            error_message: Error message if processing failed
            save_masks: Whether to save individual frame masks
            save_similarity_plot: Whether to save similarity plot
            overlay_color: RGB color for mask overlay
            overlay_alpha: Transparency for mask overlay
            
        Returns:
            Path to the output directory
        """
        logger.info(f"Generating complete output for video: {video_id}")
        
        # Create output directory
        output_dir = self.create_output_dir(video_id)
        
        # Define output paths
        original_video_path = os.path.join(output_dir, f"{video_id}_original.mp4")
        annotated_video_path = os.path.join(output_dir, f"{video_id}_annotated.mp4")
        similarity_plot_path = os.path.join(output_dir, "similarity_curve.png")
        metadata_path = os.path.join(output_dir, "metadata.json")
        
        try:
            # Copy original video (optional - can be skipped if disk space is a concern)
            # For now, we'll skip copying to save space
            
            # Generate annotated video
            if success and masks:
                self.save_annotated_video(
                    video_path=video_path,
                    masks=masks,
                    output_path=annotated_video_path,
                    text_query=text_query,
                    overlay_color=overlay_color,
                    overlay_alpha=overlay_alpha
                )
            
            # Save individual frame masks
            if save_masks and success and masks:
                self.save_frame_masks(masks=masks, output_dir=output_dir)
            
            # Save similarity plot
            if save_similarity_plot and success:
                self.save_similarity_plot(
                    similarity_scores=similarity_scores,
                    key_frame_idx=key_frame_idx,
                    output_path=similarity_plot_path,
                    text_query=text_query
                )
            
            # Prepare metadata
            metadata = {
                'video_id': video_id,
                'text_query': text_query,
                'key_frame_idx': int(key_frame_idx),
                'num_frames': len(similarity_scores),
                'num_masks': len(masks),
                'processing_start': processing_start,
                'processing_end': processing_end,
                'processing_time': processing_time,
                'success': success,
                'similarity_scores': similarity_scores,
                'output_files': {
                    'annotated_video': annotated_video_path if success else None,
                    'similarity_plot': similarity_plot_path if success else None,
                    'masks_directory': os.path.join(output_dir, 'masks') if save_masks and success else None,
                    'metadata': metadata_path
                }
            }
            
            if error_message:
                metadata['error_message'] = error_message
            
            # Save metadata
            self.save_metadata(metadata=metadata, output_path=metadata_path)
            
            logger.info(f"Complete output generated in: {output_dir}")
            return output_dir
        
        except Exception as e:
            logger.error(f"Error generating output for {video_id}: {e}")
            
            # Save error metadata
            error_metadata = {
                'video_id': video_id,
                'text_query': text_query,
                'processing_start': processing_start,
                'processing_end': datetime.now().isoformat(),
                'success': False,
                'error_message': str(e)
            }
            self.save_metadata(metadata=error_metadata, output_path=metadata_path)
            
            raise
