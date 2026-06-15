"""
TFVTG Module for temporal grounding using BLIP-2 image-text matching.
Computes frame-level similarity scores and identifies key frames.
"""
import torch
import numpy as np
from typing import Tuple, Optional, List
from torchvision import transforms
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# Set logging level to DEBUG for detailed output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TFVTGModule:
    """
    Training-Free Video Temporal Grounding module.
    Uses BLIP-2 image-text matching to compute frame-level similarity scores.
    """
    
    def __init__(
        self,
        model_name: str = 'blip2_image_text_matching',
        model_type: str = 'coco',
        device: str = 'cuda',
        batch_size: int = 128
    ):
        """
        Initialize TFVTG module with BLIP-2 model.
        
        Args:
            model_name: Name of the BLIP-2 model to load
            model_type: Type/variant of the model (e.g., 'coco')
            device: Device to run the model on ('cuda' or 'cpu')
            batch_size: Batch size for processing frames
            
        Raises:
            RuntimeError: If model initialization fails
            ValueError: If device is invalid
        """
        self.model_name = model_name
        self.model_type = model_type
        self.batch_size = batch_size
        
        # Validate device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = device
        
        # Initialize model
        try:
            logger.info(f"Loading BLIP-2 model: {model_name} ({model_type}) on {device}")
            self.model, self.vis_processors, self.text_processors = self._load_model()
            logger.info("BLIP-2 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BLIP-2 model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e
    
    def _load_model(self):
        """
        Load BLIP-2 model and preprocessors from transformers library.
        
        Returns:
            Tuple of (model, visual_processors, text_processors)
            
        Raises:
            ImportError: If transformers library is not installed
            RuntimeError: If model loading fails
        """
        try:
            from transformers import Blip2Processor, Blip2ForImageTextRetrieval
        except ImportError as e:
            raise ImportError(
                "transformers library not found. Please install it with: "
                "pip install transformers"
            ) from e
        
        try:
            # Load BLIP-2 model for image-text matching
            # Using the pretrained model from Salesforce
            model_id = "Salesforce/blip2-itm-vit-g"
            
            logger.info(f"Loading BLIP-2 from transformers: {model_id}")
            
            # Load processor and model
            processor = Blip2Processor.from_pretrained(model_id)
            model = Blip2ForImageTextRetrieval.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            model = model.to(self.device)
            model.eval()
            
            # Create visual preprocessor
            # BLIP-2 expects images of size 224x224
            vis_processors = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                )
            ])
            
            # Store processor for text tokenization
            self.processor = processor
            
            return model, vis_processors, processor
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e
    
    def __repr__(self) -> str:
        """String representation of TFVTGModule."""
        return (
            f"TFVTGModule(model={self.model_name}, "
            f"type={self.model_type}, device={self.device}, "
            f"batch_size={self.batch_size})"
        )


    
    def _load_video_frames(self, video_path: str, fps: float, video_fps: Optional[float] = None) -> torch.Tensor:
        """
        Load video frames at specified FPS using decord, or from image directory.
        
        Args:
            video_path: Path to the video file or directory of images
            fps: Target frames per second for sampling
            video_fps: Source FPS (used when video_path is image directory; default 30.0)
            
        Returns:
            Tensor of shape (num_frames, C, H, W) with values in [0, 1]
            
        Raises:
            ImportError: If decord library is not installed
            FileNotFoundError: If video file/dir does not exist
        """
        import os
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path not found: {video_path}")
        if os.path.isdir(video_path):
            return self._load_frames_from_directory(
                video_path, fps, video_fps=video_fps or 30.0
            )
        try:
            from decord import VideoReader, cpu
        except ImportError as e:
            raise ImportError(
                "decord library not found. Please install it with: "
                "pip install decord"
            ) from e
        
        try:
            # Load video with decord
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            
            # Get video length - ensure it's a valid integer
            try:
                video_length = len(vr)
            except Exception as e:
                raise RuntimeError(f"Failed to get video length: {str(e)}") from e
            
            # Validate and convert video_length
            if video_length is None:
                raise RuntimeError(f"Video length is None")
            try:
                video_length = int(video_length)
            except (TypeError, ValueError) as e:
                raise RuntimeError(f"Invalid video length type: {type(video_length)}, value: {video_length}") from e
            
            if video_length <= 0:
                raise RuntimeError(f"Invalid video length: {video_length}")
            
            # Get FPS, handle case where it might be None
            try:
                video_fps = vr.get_avg_fps()
            except Exception as e:
                logger.warning(f"Failed to get video FPS: {str(e)}, using default 30.0 fps")
                video_fps = 30.0
            
            if video_fps is None:
                logger.warning(f"Video FPS is None, using default 30.0 fps")
                video_fps = 30.0
            
            # Ensure video_fps is a valid number
            try:
                video_fps = float(video_fps)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid video FPS type: {type(video_fps)}, value: {video_fps}, using default 30.0 fps")
                video_fps = 30.0
            
            if video_fps <= 0:
                logger.warning(f"Invalid video FPS value: {video_fps}, using default 30.0 fps")
                video_fps = 30.0
            
            # Validate fps parameter
            if fps is None:
                logger.warning(f"fps parameter is None, using default 3.0 fps")
                fps = 3.0
            
            try:
                fps = float(fps)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid fps parameter type: {type(fps)}, value: {fps}, using default 3.0 fps")
                fps = 3.0
            
            if fps <= 0:
                logger.warning(f"Invalid fps parameter value: {fps}, using default 3.0 fps")
                fps = 3.0
            
            # Calculate duration
            duration = float(video_length) / float(video_fps)
            
            # Calculate frame indices to sample
            num_sampled_frames = round(duration * fps)
            if num_sampled_frames is None:
                num_sampled_frames = 1
            try:
                num_sampled_frames = int(num_sampled_frames)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid num_sampled_frames: {num_sampled_frames}, using 1")
                num_sampled_frames = 1
            
            if num_sampled_frames <= 0:
                num_sampled_frames = 1
            
            # Ensure we don't sample more frames than available
            if num_sampled_frames > video_length:
                num_sampled_frames = video_length
            
            # Calculate frame indices - ensure all values are valid integers
            if num_sampled_frames == 1:
                all_index = np.array([max(0, video_length - 1)], dtype=np.int32)
            else:
                all_index = np.linspace(0, max(0, video_length - 1), num=num_sampled_frames).round().astype(np.int32)
            
            # Seek to start and get frames
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()  # Get as numpy array
            
            # Convert to torch tensor and normalize
            # buffer shape: (T, H, W, C) -> (T, C, H, W)
            buffer = torch.from_numpy(buffer).permute(0, 3, 1, 2).float() / 255.0
            
            return buffer
            
        except Exception as e:
            raise RuntimeError(f"Failed to load video frames: {str(e)}") from e

    def _load_frames_from_directory(
        self,
        dir_path: str,
        fps: float,
        video_fps: float = 30.0,
        target_size: Tuple[int, int] = (224, 224)
    ) -> torch.Tensor:
        """
        Load frames from an image directory (e.g. SAVG img folder). Resize each frame
        to target_size so that torch.stack works when resolutions differ.
        
        Args:
            dir_path: Path to directory containing image files
            fps: Target frames per second for sampling
            video_fps: FPS of the source (frame count / duration)
            target_size: (H, W) to resize every frame to (default 224,224 for BLIP)
            
        Returns:
            Tensor of shape (num_frames, C, H, W) with values in [0, 1]
        """
        import os
        from PIL import Image
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
        frame_files = []
        for f in os.listdir(dir_path):
            if f.endswith(image_extensions):
                frame_files.append(os.path.join(dir_path, f))
        frame_files = sorted(
            frame_files,
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            if os.path.splitext(os.path.basename(x))[0].isdigit() else float('inf')
        )
        if not frame_files:
            raise RuntimeError(f"No image frames found in directory: {dir_path}")
        video_length = len(frame_files)
        duration = float(video_length) / float(video_fps)
        num_sampled_frames = max(1, int(round(duration * fps)))
        if num_sampled_frames > video_length:
            num_sampled_frames = video_length
        if num_sampled_frames == 1:
            all_index = np.array([max(0, video_length - 1)], dtype=np.int32)
        else:
            all_index = np.linspace(0, max(0, video_length - 1), num=num_sampled_frames).round().astype(np.int32)
        frames = []
        for idx in all_index:
            img_path = frame_files[int(idx)]
            try:
                img = Image.open(img_path).convert('RGB')
                arr = np.array(img)
            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}, skipping")
                continue
            # (H, W, C) -> (C, H, W), float [0,1]
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            # Resize to target_size so all frames have same shape for stack
            if t.shape[1] != target_size[0] or t.shape[2] != target_size[1]:
                t = torch.nn.functional.interpolate(
                    t.unsqueeze(0),
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            frames.append(t)
        if not frames:
            raise RuntimeError(f"Could not load any frame from directory: {dir_path}")
        return torch.stack(frames)

    def compute_similarity(
        self,
        video_path: str,
        text_query: str,
        fps: float = 3.0,
        video_fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute ITM (Image-Text Matching) similarity scores between video frames and text query.
        
        Args:
            video_path: Path to the video file
            text_query: Text query string
            fps: Target frames per second for sampling
            
        Returns:
            Similarity scores array of shape (num_frames,) normalized to [0, 1]
            
        Raises:
            ValueError: If text query is empty
            RuntimeError: If similarity computation fails
        """
        if not text_query or not text_query.strip():
            raise ValueError("Text query cannot be empty")
        
        try:
            logger.info(f"Computing ITM similarity for query: '{text_query}'")
            
            # Load video frames (video_fps used when video_path is image directory)
            video_frames = self._load_video_frames(video_path, fps, video_fps=video_fps)
            num_frames = video_frames.size(0)
            
            logger.info(f"Computing ITM scores for {num_frames} frames")
            logger.debug(f"batch_size: {self.batch_size}, type: {type(self.batch_size)}")
            
            # Tokenize text once
            # Note: Don't pass max_length to avoid issues with num_query_tokens
            text_inputs = self.processor(
                text=[text_query],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            logger.debug(f"Text inputs keys: {text_inputs.keys()}")
            
            # Move text inputs to device
            text_input_ids = text_inputs.input_ids.to(self.device)
            text_attention_mask = text_inputs.attention_mask.to(self.device)
            
            logger.debug(f"Text input IDs shape: {text_input_ids.shape}")
            logger.debug(f"Text attention mask shape: {text_attention_mask.shape}")
            
            itm_scores = []
            
            logger.debug(f"Starting batch processing: num_frames={num_frames}, batch_size={self.batch_size}")
            
            # Process frames in batches
            for bid in range(0, num_frames, self.batch_size):
                logger.debug(f"Processing batch starting at frame {bid}")
                batch_end = min(bid + self.batch_size, num_frames)
                batch_frames = video_frames[bid:batch_end]  # (B, C, H, W)
                
                # Preprocess images - manually apply batch-compatible transforms
                # transforms.Compose doesn't work on batches, so we use functional transforms
                # Resize: (B, C, H, W) -> (B, C, 224, 224)
                batch_img = torch.nn.functional.interpolate(
                    batch_frames,
                    size=(224, 224),
                    mode='bilinear',
                    align_corners=False
                )
                
                # Normalize: apply normalization to batch
                # BLIP-2 normalization parameters
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], 
                                  device=batch_img.device, dtype=batch_img.dtype).view(1, 3, 1, 1)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711], 
                                 device=batch_img.device, dtype=batch_img.dtype).view(1, 3, 1, 1)
                batch_img = (batch_img - mean) / std
                
                batch_img = batch_img.to(self.device)
                
                # Convert to half precision if model uses it
                if next(self.model.parameters()).dtype == torch.float16:
                    batch_img = batch_img.half()
                
                batch_size_actual = batch_img.size(0)
                
                # Repeat text inputs for batch
                batch_text_ids = text_input_ids.repeat(batch_size_actual, 1)
                batch_text_mask = text_attention_mask.repeat(batch_size_actual, 1)
                
                with torch.no_grad():
                    # Use BLIP-2 for ITM
                    # Note: Blip2ForImageTextRetrieval automatically uses ITM head
                    outputs = self.model(
                        pixel_values=batch_img,
                        input_ids=batch_text_ids,
                        attention_mask=batch_text_mask,
                        return_dict=True
                    )
                    
                    # Get ITM scores (logits for [not_match, match])
                    # Shape: (batch_size, 2)
                    logger.debug(f"Output attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                    
                    if hasattr(outputs, 'itm_score') and outputs.itm_score is not None:
                        itm_logits = outputs.itm_score
                        logger.debug(f"Using itm_score, shape: {itm_logits.shape}")
                    elif hasattr(outputs, 'logits') and outputs.logits is not None:
                        itm_logits = outputs.logits
                        logger.debug(f"Using logits, shape: {itm_logits.shape}")
                    else:
                        # Fallback: use similarity score
                        logger.warning("ITM score not found in outputs")
                        logger.warning(f"Available attributes: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
                        
                        if hasattr(outputs, 'logits_per_image') and outputs.logits_per_image is not None:
                            logger.warning("Using logits_per_image as fallback")
                            itm_probs = outputs.logits_per_image[:, 0]
                        else:
                            logger.warning("No valid ITM scores found, using default value 0.5")
                            itm_probs = torch.ones(batch_size_actual, device=self.device) * 0.5
                        
                        itm_scores.append(itm_probs.cpu())
                        continue
                    
                    # Check if itm_logits is None
                    if itm_logits is None:
                        logger.error("itm_logits is None!")
                        raise RuntimeError("ITM logits is None")
                    
                    logger.debug(f"ITM logits shape: {itm_logits.shape}, dtype: {itm_logits.dtype}")
                    
                    # Apply softmax and take the "match" probability (index 1)
                    itm_probs = torch.nn.functional.softmax(itm_logits, dim=1)[:, 1]
                    logger.debug(f"ITM probs shape: {itm_probs.shape}, values: {itm_probs}")
                    
                    itm_scores.append(itm_probs.cpu())
                
                # Clear GPU cache after each batch
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Concatenate all batch scores
            if not itm_scores:
                raise RuntimeError("No ITM scores computed")
            
            itm_scores = torch.cat(itm_scores, dim=0).numpy()
            
            logger.info(f"Raw ITM scores - shape: {itm_scores.shape}, dtype: {itm_scores.dtype}")
            logger.info(f"Raw ITM scores - min: {itm_scores.min()}, max: {itm_scores.max()}, mean: {itm_scores.mean()}")
            
            # Check for NaN or inf values
            if np.isnan(itm_scores).any():
                logger.error("ITM scores contain NaN values!")
                itm_scores = np.nan_to_num(itm_scores, nan=0.0)
            
            if np.isinf(itm_scores).any():
                logger.error("ITM scores contain inf values!")
                itm_scores = np.nan_to_num(itm_scores, posinf=1.0, neginf=0.0)
            
            # Return raw scores without normalization
            logger.info(
                f"Computed ITM similarity scores (raw, not normalized) - "
                f"min: {itm_scores.min():.4f}, "
                f"max: {itm_scores.max():.4f}, "
                f"mean: {itm_scores.mean():.4f}"
            )
            
            return itm_scores
            
        except Exception as e:
            logger.error(f"ITM similarity computation failed: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise RuntimeError(f"ITM similarity computation failed: {str(e)}") from e
    

    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize similarity scores to [0, 1] range.
        
        Args:
            scores: Raw similarity scores
            
        Returns:
            Normalized scores in [0, 1] range
        """
        logger.debug(f"_normalize_scores input - type: {type(scores)}, shape: {scores.shape if hasattr(scores, 'shape') else 'N/A'}")
        
        if scores.size == 0:
            logger.warning("Empty scores array")
            return scores
        
        try:
            min_val = scores.min()
            max_val = scores.max()
            
            logger.debug(f"Min value: {min_val} (type: {type(min_val)})")
            logger.debug(f"Max value: {max_val} (type: {type(max_val)})")
            
            # Ensure they are not None
            if min_val is None or max_val is None:
                logger.error(f"min or max is None! min={min_val}, max={max_val}")
                raise ValueError(f"min or max is None! min={min_val}, max={max_val}")
            
            min_score = float(min_val)
            max_score = float(max_val)
            
            logger.debug(f"Converted - min_score: {min_score}, max_score: {max_score}")
            
            if max_score - min_score < 1e-8:
                # All scores are the same, return uniform distribution
                logger.warning("All similarity scores are the same, returning uniform distribution")
                return np.ones_like(scores) * 0.5
            
            # Min-max normalization
            normalized = (scores - min_score) / (max_score - min_score)
            return normalized
            
        except Exception as e:
            logger.error(f"Error in _normalize_scores: {e}")
            logger.error(f"Scores: {scores}")
            raise

    def get_key_frame(
        self,
        video_path: str,
        text_query: str,
        fps: float = 3.0,
        video_fps: Optional[float] = None
    ) -> Tuple[int, np.ndarray, np.ndarray, int]:
        """
        Extract key frame with highest ITM similarity to text query.
        
        This is the main entry point that uses BLIP-2's ITM head to find
        the frame that best matches the text query.
        
        Args:
            video_path: Path to the video file or directory of images
            text_query: Text query string
            fps: Target frames per second for sampling (default: 3.0)
            video_fps: Source FPS when video_path is image directory (default: 30.0)
            
        Returns:
            Tuple of (key_frame_index, key_frame_image, similarity_scores, original_frame_idx):
                - key_frame_index: Index of the key frame in sampled frames
                - key_frame_image: Key frame as numpy array (H, W, 3) in RGB format
                - similarity_scores: ITM similarity scores for all frames
                - original_frame_idx: Index of the key frame in the original video
                
        Raises:
            FileNotFoundError: If video file does not exist
            ValueError: If text query is empty
            RuntimeError: If processing fails
        """
        if not text_query or not text_query.strip():
            raise ValueError("Text query cannot be empty")
        
        try:
            logger.info(f"Processing video: {video_path}")
            logger.info(f"Text query: '{text_query}'")
            
            # Compute ITM similarity scores directly
            similarity_scores = self.compute_similarity(video_path, text_query, fps, video_fps=video_fps)
            
            # Select key frame
            key_frame_idx, key_frame_image, original_frame_idx = self._select_key_frame(
                video_path,
                similarity_scores,
                fps,
                video_fps=video_fps
            )
            
            logger.info(
                f"Selected key frame {key_frame_idx} (original frame {original_frame_idx}) "
                f"with ITM similarity score {similarity_scores[key_frame_idx]:.4f}"
            )
            
            return key_frame_idx, key_frame_image, similarity_scores, original_frame_idx
            
        except Exception as e:
            logger.error(f"Key frame extraction failed: {str(e)}")
            raise RuntimeError(f"Key frame extraction failed: {str(e)}") from e
    
    def _select_key_frame(
        self,
        video_path: str,
        similarity_scores: np.ndarray,
        fps: float,
        video_fps: Optional[float] = None
    ) -> Tuple[int, np.ndarray, int]:
        """
        Select key frame with maximum similarity score.
        
        Args:
            video_path: Path to the video file or directory of images
            similarity_scores: Similarity scores for all frames
            fps: FPS used for frame sampling
            video_fps: Source FPS when video_path is image directory (default: 30.0)
            
        Returns:
            Tuple of (key_frame_index, key_frame_image, original_frame_idx)
            
        Raises:
            RuntimeError: If key frame extraction fails
        """
        # Find frame with maximum similarity
        # If there are ties, argmax returns the first occurrence (earliest frame)
        key_frame_idx = int(np.argmax(similarity_scores))
        
        # Extract the key frame image and get original frame index
        key_frame_image, original_frame_idx = self._extract_key_frame_image(
            video_path,
            key_frame_idx,
            fps,
            video_fps=video_fps
        )
        
        return key_frame_idx, key_frame_image, original_frame_idx
    
    def _extract_key_frame_image(
        self,
        video_path: str,
        sampled_frame_idx: int,
        fps: float,
        video_fps: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Extract key frame image from video or image directory.
        
        Args:
            video_path: Path to the video file or directory of images
            sampled_frame_idx: Index in the sampled frames
            fps: FPS used for frame sampling
            video_fps: Source FPS when video_path is image directory (default: 30.0)
            
        Returns:
            Tuple of (frame_image, original_frame_idx):
                - frame_image: Frame image as numpy array (H, W, 3) in RGB format
                - original_frame_idx: Index of the frame in the original video
            
        Raises:
            RuntimeError: If frame extraction fails
        """
        import os
        if os.path.isdir(video_path):
            # Load from image directory (same sampling as _load_frames_from_directory)
            from PIL import Image
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
            frame_files = []
            for f in os.listdir(video_path):
                if f.endswith(image_extensions):
                    frame_files.append(os.path.join(video_path, f))
            frame_files = sorted(
                frame_files,
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                if os.path.splitext(os.path.basename(x))[0].isdigit() else float('inf')
            )
            if not frame_files:
                raise RuntimeError(f"No image frames found in directory: {video_path}")
            video_length = len(frame_files)
            vfps = float(video_fps or 30.0)
            duration = float(video_length) / vfps
            num_sampled_frames = max(1, int(round(duration * fps)))
            if num_sampled_frames > video_length:
                num_sampled_frames = video_length
            if num_sampled_frames == 1:
                all_index = np.array([max(0, video_length - 1)], dtype=np.int32)
            else:
                all_index = np.linspace(0, max(0, video_length - 1), num=num_sampled_frames).round().astype(np.int32)
            if sampled_frame_idx < 0:
                sampled_frame_idx = 0
            elif sampled_frame_idx >= len(all_index):
                sampled_frame_idx = len(all_index) - 1
            actual_frame_idx = int(all_index[sampled_frame_idx])
            img = Image.open(frame_files[actual_frame_idx]).convert('RGB')
            frame = np.array(img)
            return frame, actual_frame_idx
        try:
            from decord import VideoReader, cpu
            
            # Load video
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            
            # Get video length - ensure it's a valid integer
            try:
                video_length = len(vr)
            except Exception as e:
                raise RuntimeError(f"Failed to get video length: {str(e)}") from e
            
            # Validate and convert video_length
            if video_length is None:
                raise RuntimeError(f"Video length is None")
            try:
                video_length = int(video_length)
            except (TypeError, ValueError) as e:
                raise RuntimeError(f"Invalid video length type: {type(video_length)}, value: {video_length}") from e
            
            if video_length <= 0:
                raise RuntimeError(f"Invalid video length: {video_length}")
            
            # Get FPS, handle case where it might be None
            try:
                video_fps = vr.get_avg_fps()
            except Exception as e:
                logger.warning(f"Failed to get video FPS: {str(e)}, using default 30.0 fps")
                video_fps = 30.0
            
            if video_fps is None:
                logger.warning(f"Video FPS is None, using default 30.0 fps")
                video_fps = 30.0
            
            # Ensure video_fps is a valid number
            try:
                video_fps = float(video_fps)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid video FPS type: {type(video_fps)}, value: {video_fps}, using default 30.0 fps")
                video_fps = 30.0
            
            if video_fps <= 0:
                logger.warning(f"Invalid video FPS value: {video_fps}, using default 30.0 fps")
                video_fps = 30.0
            
            # Validate fps parameter
            if fps is None:
                logger.warning(f"fps parameter is None, using default 3.0 fps")
                fps = 3.0
            
            try:
                fps = float(fps)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid fps parameter type: {type(fps)}, value: {fps}, using default 3.0 fps")
                fps = 3.0
            
            if fps <= 0:
                logger.warning(f"Invalid fps parameter value: {fps}, using default 3.0 fps")
                fps = 3.0
            
            # Calculate duration
            duration = float(video_length) / float(video_fps)
            
            # Calculate the actual frame index in the original video
            num_sampled_frames = round(duration * fps)
            if num_sampled_frames is None:
                num_sampled_frames = 1
            try:
                num_sampled_frames = int(num_sampled_frames)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid num_sampled_frames: {num_sampled_frames}, using 1")
                num_sampled_frames = 1
            
            if num_sampled_frames <= 0:
                num_sampled_frames = 1
            
            # Ensure we don't sample more frames than available
            if num_sampled_frames > video_length:
                num_sampled_frames = video_length
            
            # Calculate frame indices - ensure all values are valid integers
            if num_sampled_frames == 1:
                all_index = np.array([max(0, video_length - 1)], dtype=np.int32)
            else:
                all_index = np.linspace(0, max(0, video_length - 1), num=num_sampled_frames).round().astype(np.int32)
            
            # Ensure sampled_frame_idx is within bounds
            if sampled_frame_idx < 0:
                sampled_frame_idx = 0
            elif sampled_frame_idx >= len(all_index):
                sampled_frame_idx = len(all_index) - 1
            
            # Get the actual frame index
            actual_frame_idx = int(all_index[sampled_frame_idx])
            
            # Extract the frame
            vr.seek(0)
            frame = vr[actual_frame_idx].asnumpy()  # Returns RGB format
            
            return frame, actual_frame_idx
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract key frame image: {str(e)}") from e

    def compute_itm_match_probs_for_images(
        self,
        images_rgb: List[np.ndarray],
        text_query: str,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        对一批 RGB 图像（如 tube ROI crop）计算 BLIP-2 ITM 的 P(match)，
        与 compute_similarity 中 batch 前向逻辑一致，供 Stage6 主语基线对比等场景使用。

        Args:
            images_rgb: 每张为 (H, W, 3) uint8 RGB；可为变分辨率。
            text_query: 单条文本（action 或 subject）。

        Returns:
            shape (N,) 的 float32，每帧 softmax(ITM logits) 的 match 概率（与 compute_similarity 一致）。
        """
        if not text_query or not str(text_query).strip():
            raise ValueError("Text query cannot be empty")
        if not images_rgb:
            return np.array([], dtype=np.float32)

        text_inputs = self.processor(
            text=[text_query],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        text_attention_mask = text_inputs.attention_mask.to(self.device)

        tensors: List[torch.Tensor] = []
        for arr in images_rgb:
            a = np.asarray(arr)
            if a.ndim != 3 or a.shape[2] != 3:
                raise ValueError(f"Expected RGB image (H,W,3), got shape {a.shape}")
            if a.dtype != np.uint8:
                a = np.clip(a, 0, 255).astype(np.uint8)
            t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
            # Tube ROI crops 宽高随 bbox 变化，必须先统一到 224 再 stack（与 compute_similarity 中 batch 一致）
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            tensors.append(t)
        video_frames = torch.stack(tensors, dim=0)
        num_frames = int(video_frames.size(0))

        itm_scores: List[torch.Tensor] = []
        bs = int(batch_size) if batch_size is not None else int(self.batch_size)
        bs = max(1, bs)

        for bid in range(0, num_frames, bs):
            batch_end = min(bid + bs, num_frames)
            batch_frames = video_frames[bid:batch_end]

            batch_img = torch.nn.functional.interpolate(
                batch_frames,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073],
                device=batch_img.device,
                dtype=batch_img.dtype,
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711],
                device=batch_img.device,
                dtype=batch_img.dtype,
            ).view(1, 3, 1, 1)
            batch_img = (batch_img - mean) / std
            batch_img = batch_img.to(self.device)

            if next(self.model.parameters()).dtype == torch.float16:
                batch_img = batch_img.half()

            bsz = batch_img.size(0)
            batch_text_ids = text_input_ids.repeat(bsz, 1)
            batch_text_mask = text_attention_mask.repeat(bsz, 1)

            with torch.no_grad():
                outputs = self.model(
                    pixel_values=batch_img,
                    input_ids=batch_text_ids,
                    attention_mask=batch_text_mask,
                    return_dict=True,
                )

                if hasattr(outputs, "itm_score") and outputs.itm_score is not None:
                    itm_logits = outputs.itm_score
                elif hasattr(outputs, "logits") and outputs.logits is not None:
                    itm_logits = outputs.logits
                else:
                    if hasattr(outputs, "logits_per_image") and outputs.logits_per_image is not None:
                        itm_probs = outputs.logits_per_image[:, 0]
                    else:
                        itm_probs = torch.ones(bsz, device=self.device) * 0.5
                    itm_scores.append(itm_probs.cpu().float())
                    continue

                if itm_logits is None:
                    raise RuntimeError("ITM logits is None")
                itm_probs = torch.nn.functional.softmax(itm_logits, dim=1)[:, 1]
                itm_scores.append(itm_probs.cpu().float())

            if self.device == "cuda":
                torch.cuda.empty_cache()

        if not itm_scores:
            raise RuntimeError("No ITM scores computed")
        out = torch.cat(itm_scores, dim=0).numpy().astype(np.float32)
        if np.isnan(out).any():
            out = np.nan_to_num(out, nan=0.0)
        if np.isinf(out).any():
            out = np.nan_to_num(out, posinf=1.0, neginf=0.0)
        return out

    def cleanup(self):
        """
        Release GPU memory and cleanup resources.
        """
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            logger.info("Cleaned up GPU memory")
