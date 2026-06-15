"""
RSVG Module for spatial grounding using Qwen-VL attention maps.
Generates initial segmentation masks on key frames using vision-language models.
"""
import torch
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import logging
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class RSVGModule:
    """
    Zero-Shot Open-Vocabulary Visual Grounding module using Qwen-VL.
    Generates segmentation masks by extracting and fusing cross-attention maps.
    """
    
    def __init__(
        self,
        model_path: str = './Qwen/Qwen2.5-VL-7B-Instruct',
        attention_layers: List[int] = [16, 17, 18, 19],
        layer_weights: List[float] = [0.1, 0.1, 0.3, 0.5],
        mask_threshold: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize RSVG module with Qwen-VL model.
        
        Args:
            model_path: Path to Qwen2.5-VL model
            attention_layers: List of transformer layer indices to extract attention from
            layer_weights: Weights for fusing attention from different layers
            mask_threshold: Threshold for binary mask generation (0.0 to 1.0)
            device: Device to run the model on ('cuda' or 'cpu')
            
        Raises:
            RuntimeError: If model initialization fails
            ValueError: If device is invalid or layer_weights don't match attention_layers
        """
        self.model_path = model_path
        self.attention_layers = attention_layers
        self.layer_weights = layer_weights
        self.mask_threshold = mask_threshold
        
        # Validate layer weights
        if len(layer_weights) != len(attention_layers):
            raise ValueError(
                f"Number of layer_weights ({len(layer_weights)}) must match "
                f"number of attention_layers ({len(attention_layers)})"
            )
        
        # Normalize layer weights to sum to 1
        weight_sum = sum(layer_weights)
        self.layer_weights = [w / weight_sum for w in layer_weights]
        
        # Validate device
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        self.device = device
        
        # Initialize model
        try:
            logger.info(f"Loading Qwen-VL model from {model_path} on {device}")
            self.model, self.processor = self._load_model()
            logger.info("Qwen-VL model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qwen-VL model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e
    
    def _load_model(self):
        """
        Load Qwen2.5-VL model with AutoProcessor.
        
        Returns:
            Tuple of (model, processor)
            
        Raises:
            ImportError: If transformers library is not installed
            RuntimeError: If model loading fails
        """
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers library not found. Please install it with: "
                "pip install transformers>=4.35.0"
            ) from e
        
        try:
            # Load processor (supports both local paths and HuggingFace repo IDs)
            processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=False  # Allow fallback to download if needed
            )
            
            # Load model with bfloat16 and eager attention
            # Use AutoModelForVision2Seq to automatically detect the correct model class
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map=self.device,
                trust_remote_code=True,
                local_files_only=False  # Allow fallback to download if needed
            )
            
            # Set model to evaluation mode
            model.eval()
            
            return model, processor
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen-VL model: {str(e)}") from e
    
    def __repr__(self) -> str:
        """String representation of RSVGModule."""
        return (
            f"RSVGModule(model_path={self.model_path}, "
            f"layers={self.attention_layers}, weights={self.layer_weights}, "
            f"threshold={self.mask_threshold}, device={self.device})"
        )

    def generate_attention_maps(
        self,
        image: np.ndarray,
        text_query: str
    ) -> np.ndarray:
        """
        Extract cross-attention maps from Qwen-VL transformer layers.
        
        Args:
            image: Image array (H, W, 3) in RGB format
            text_query: Text query string
            
        Returns:
            Attention maps array of shape (num_layers, H', W') where H' and W'
            are the spatial dimensions of the attention maps
            
        Raises:
            ValueError: If text query is empty or image is invalid
            RuntimeError: If attention extraction fails
        """
        if not text_query or not text_query.strip():
            raise ValueError("Text query cannot be empty")
        
        if image.size == 0 or image.ndim != 3:
            raise ValueError("Image must be a 3D array (H, W, 3)")
        
        try:
            logger.info(f"Extracting attention maps for query: '{text_query}'")
            
            # Convert numpy array to PIL Image
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Assume values are in [0, 1]
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
            else:
                image_pil = Image.fromarray(image.astype(np.uint8))
            
            # Prepare input messages with vision info
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_pil,
                        },
                        {"type": "text", "text": text_query},
                    ],
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = self._process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            
            # Extract attention maps from specified layers
            attention_maps = self._extract_cross_attention(inputs)
            
            logger.info(f"Extracted attention maps with shape: {attention_maps.shape}")
            return attention_maps
            
        except Exception as e:
            logger.error(f"Attention map extraction failed: {str(e)}")
            raise RuntimeError(f"Attention map extraction failed: {str(e)}") from e
    
    def _process_vision_info(self, messages: List[Dict[str, Any]]) -> Tuple[Optional[List], Optional[List]]:
        """
        Process vision information from messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Tuple of (image_inputs, video_inputs)
        """
        image_inputs = []
        video_inputs = []
        
        for message in messages:
            if isinstance(message.get("content"), list):
                for content_item in message["content"]:
                    if content_item.get("type") == "image":
                        image_inputs.append(content_item["image"])
                    elif content_item.get("type") == "video":
                        video_inputs.append(content_item["video"])
        
        return (image_inputs if image_inputs else None,
                video_inputs if video_inputs else None)
    
    def _extract_cross_attention(self, inputs: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Extract cross-attention from transformer layers.
        
        Args:
            inputs: Processed inputs dictionary
            
        Returns:
            Attention maps array of shape (num_layers, H', W')
            
        Raises:
            RuntimeError: If attention extraction fails
        """
        try:
            with torch.no_grad():
                # Forward pass with output_attentions=True
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True
                )
            
            # Extract attention from specified layers
            all_attentions = outputs.attentions  # Tuple of attention tensors
            
            attention_maps = []
            for layer_idx in self.attention_layers:
                if layer_idx >= len(all_attentions):
                    logger.warning(
                        f"Layer {layer_idx} not found in model "
                        f"(total layers: {len(all_attentions)}), skipping"
                    )
                    continue
                
                # Get attention for this layer
                # Shape: (batch_size, num_heads, seq_len, seq_len)
                layer_attention = all_attentions[layer_idx]
                
                # Process attention to get spatial map
                spatial_map = self._process_layer_attention(
                    layer_attention,
                    inputs
                )
                
                attention_maps.append(spatial_map)
            
            if not attention_maps:
                raise RuntimeError("No attention maps extracted from any layer")
            
            # Stack attention maps: (num_layers, H', W')
            attention_maps = np.stack(attention_maps, axis=0)
            
            return attention_maps
            
        except Exception as e:
            raise RuntimeError(f"Cross-attention extraction failed: {str(e)}") from e
    
    def _process_layer_attention(
        self,
        layer_attention: torch.Tensor,
        inputs: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """
        Process layer attention to extract spatial attention map.
        
        Computes mean attention across heads and generated tokens,
        then reshapes to spatial dimensions.
        
        Args:
            layer_attention: Attention tensor of shape (batch, num_heads, seq_len, seq_len)
            inputs: Input dictionary containing pixel_values and other info
            
        Returns:
            Spatial attention map of shape (H', W')
        """
        # Mean across attention heads: (batch, seq_len, seq_len)
        attention = layer_attention.mean(dim=1)
        
        # Get the attention from text tokens to image tokens
        # For Qwen-VL, we need to identify which tokens correspond to the image
        # Typically, image tokens come first in the sequence
        
        # Get image grid size from pixel_values
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            # pixel_values shape: (batch, channels, height, width)
            pixel_values = inputs['pixel_values']
            
            # Qwen-VL uses a vision encoder that produces spatial features
            # The spatial size depends on the model's vision encoder configuration
            # For Qwen2.5-VL, the vision encoder typically produces a grid
            
            # Calculate the number of image tokens
            # This is model-specific; for Qwen-VL it's typically (H/patch_size) * (W/patch_size)
            # We'll use a heuristic: find the first significant block of attention
            
            # Take attention from the last generated token (query) to all tokens (keys)
            # Shape: (batch, seq_len)
            query_attention = attention[0, -1, :]  # Last token's attention to all tokens
            
            # Estimate image token count (heuristic: first N tokens with high attention)
            # For Qwen-VL, image tokens are typically at the beginning
            # We'll reshape based on the model's vision encoder output
            
            # Get the spatial dimensions from the model's vision encoder
            # For Qwen2.5-VL-7B, the vision encoder produces features at a certain resolution
            # We'll use a common grid size (e.g., 24x24 or 32x32)
            
            # Extract attention to image tokens only
            # Assuming image tokens are the first N tokens
            # We need to determine N based on the model architecture
            
            # For simplicity, we'll take the mean attention across all generated tokens
            # to all image tokens, then reshape to spatial grid
            
            # Mean across generated tokens (queries): (batch, seq_len)
            mean_attention = attention[0].mean(dim=0)  # Mean over query positions
            
            # Determine spatial grid size
            # For Qwen-VL, this is typically sqrt(num_image_tokens)
            # We'll estimate this from the attention pattern
            
            # Find the likely number of image tokens
            # (heuristic: tokens with consistently high attention)
            num_image_tokens = self._estimate_image_token_count(mean_attention)
            
            # Extract attention to image tokens
            image_attention = mean_attention[:num_image_tokens]
            
            # Reshape to spatial grid
            grid_size = int(np.sqrt(num_image_tokens))
            if grid_size * grid_size != num_image_tokens:
                # Adjust to nearest square
                grid_size = int(np.ceil(np.sqrt(num_image_tokens)))
                # Pad if necessary
                if grid_size * grid_size > num_image_tokens:
                    padding = grid_size * grid_size - num_image_tokens
                    image_attention = torch.cat([
                        image_attention,
                        torch.zeros(padding, device=image_attention.device)
                    ])
            
            # Reshape to spatial grid: (grid_size, grid_size)
            spatial_map = image_attention[:grid_size * grid_size].reshape(grid_size, grid_size)
            
            # Convert to float32 before numpy conversion
            return spatial_map.float().cpu().numpy()
        
        else:
            raise RuntimeError("No pixel_values found in inputs")
    
    def _estimate_image_token_count(self, attention: torch.Tensor) -> int:
        """
        Estimate the number of image tokens from attention pattern.
        
        Args:
            attention: Attention tensor of shape (seq_len,)
            
        Returns:
            Estimated number of image tokens
        """
        # Heuristic: image tokens typically have higher attention values
        # and form a contiguous block at the beginning
        
        # Convert to float32 first to avoid bfloat16 numpy conversion issues
        # Find the first significant drop in attention
        attention_np = attention.float().cpu().numpy()
        
        # Use a threshold-based approach
        threshold = attention_np.mean()
        high_attention_mask = attention_np > threshold
        
        # Find the longest contiguous sequence from the start
        count = 0
        for val in high_attention_mask:
            if val:
                count += 1
            else:
                break
        
        # Ensure it's a reasonable number (at least 256 for a 16x16 grid)
        if count < 256:
            count = min(256, len(attention))
        
        # Round to nearest perfect square
        grid_size = int(np.sqrt(count))
        count = grid_size * grid_size
        
        return count

    def fuse_attention_maps(self, attention_maps: np.ndarray) -> np.ndarray:
        """
        Apply weighted fusion of attention maps from multiple layers.
        
        Normalizes each attention map before fusion and computes weighted sum.
        
        Args:
            attention_maps: Attention maps array of shape (num_layers, H', W')
            
        Returns:
            Fused attention map of shape (H', W')
            
        Raises:
            ValueError: If attention_maps shape is invalid
        """
        if attention_maps.ndim != 3:
            raise ValueError(
                f"attention_maps must be 3D (num_layers, H, W), "
                f"got shape {attention_maps.shape}"
            )
        
        num_layers = attention_maps.shape[0]
        if num_layers != len(self.layer_weights):
            raise ValueError(
                f"Number of attention maps ({num_layers}) does not match "
                f"number of layer weights ({len(self.layer_weights)})"
            )
        
        logger.info(f"Fusing {num_layers} attention maps with weights {self.layer_weights}")
        
        # Normalize each attention map before fusion
        normalized_maps = []
        for i, attention_map in enumerate(attention_maps):
            normalized = self._normalize_attention_map(attention_map)
            normalized_maps.append(normalized)
        
        normalized_maps = np.stack(normalized_maps, axis=0)
        
        # Compute weighted sum
        # Reshape weights for broadcasting: (num_layers, 1, 1)
        weights = np.array(self.layer_weights).reshape(-1, 1, 1)
        
        # Weighted sum: (num_layers, H, W) * (num_layers, 1, 1) -> (H, W)
        fused_map = np.sum(normalized_maps * weights, axis=0)
        
        logger.info(
            f"Fused attention map - "
            f"min: {fused_map.min():.4f}, "
            f"max: {fused_map.max():.4f}, "
            f"mean: {fused_map.mean():.4f}"
        )
        
        return fused_map
    
    def _normalize_attention_map(self, attention_map: np.ndarray) -> np.ndarray:
        """
        Normalize attention map to [0, 1] range.
        
        Args:
            attention_map: Attention map array of shape (H, W)
            
        Returns:
            Normalized attention map with values in [0, 1]
        """
        min_val = attention_map.min()
        max_val = attention_map.max()
        
        if max_val - min_val > 1e-8:
            normalized = (attention_map - min_val) / (max_val - min_val)
        else:
            # All values are the same, return zeros
            normalized = np.zeros_like(attention_map)
        
        return normalized

    def generate_mask(
        self,
        image: np.ndarray,
        text_query: str,
        apply_region_growing: bool = False
    ) -> np.ndarray:
        """
        Generate binary segmentation mask from image and text query.
        
        This is the main entry point that combines attention extraction,
        fusion, and mask generation.
        
        Args:
            image: Image array (H, W, 3) in RGB format
            text_query: Text query string
            apply_region_growing: Whether to apply region growing for mask refinement
            
        Returns:
            Binary mask array of shape (H, W) with values 0 or 1
            
        Raises:
            ValueError: If text query is empty or image is invalid
            RuntimeError: If mask generation fails
        """
        if not text_query or not text_query.strip():
            raise ValueError("Text query cannot be empty")
        
        if image.size == 0 or image.ndim != 3:
            raise ValueError("Image must be a 3D array (H, W, 3)")
        
        try:
            logger.info(f"Generating mask for query: '{text_query}'")
            original_height, original_width = image.shape[:2]
            
            # Extract attention maps from multiple layers
            attention_maps = self.generate_attention_maps(image, text_query)
            
            # Fuse attention maps with layer-specific weights
            fused_attention = self.fuse_attention_maps(attention_maps)
            
            # Apply threshold to generate binary mask
            binary_mask = self._apply_threshold(fused_attention)
            
            # Optionally apply region growing for refinement
            if apply_region_growing:
                binary_mask = self._apply_region_growing(binary_mask)
            
            # Resize mask to original image resolution
            mask_resized = self._resize_mask_to_image(
                binary_mask,
                (original_width, original_height)
            )
            
            # Calculate mask statistics
            mask_area = np.sum(mask_resized)
            total_area = mask_resized.size
            coverage = (mask_area / total_area) * 100
            
            logger.info(
                f"Generated mask - "
                f"size: {mask_resized.shape}, "
                f"coverage: {coverage:.2f}%"
            )
            
            return mask_resized
            
        except Exception as e:
            logger.error(f"Mask generation failed: {str(e)}")
            raise RuntimeError(f"Mask generation failed: {str(e)}") from e
    
    def _apply_threshold(self, attention_map: np.ndarray) -> np.ndarray:
        """
        Apply threshold to attention map to generate binary mask.
        
        Args:
            attention_map: Attention map array of shape (H, W) with values in [0, 1]
            
        Returns:
            Binary mask array of shape (H, W) with values 0 or 1
        """
        # Normalize attention map to [0, 1] if not already
        normalized = self._normalize_attention_map(attention_map)
        
        # Apply threshold
        binary_mask = (normalized >= self.mask_threshold).astype(np.uint8)
        
        logger.debug(
            f"Applied threshold {self.mask_threshold} - "
            f"mask pixels: {np.sum(binary_mask)}/{binary_mask.size}"
        )
        
        return binary_mask
    
    def _apply_region_growing(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply region growing algorithm for mask refinement.
        
        This helps to fill holes and smooth the mask boundaries.
        
        Args:
            mask: Binary mask array of shape (H, W)
            
        Returns:
            Refined binary mask array of shape (H, W)
        """
        # Apply morphological operations for refinement
        # Close small holes
        kernel_close = np.ones((5, 5), np.uint8)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Remove small noise
        kernel_open = np.ones((3, 3), np.uint8)
        mask_refined = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
        
        logger.debug("Applied region growing refinement")
        
        return mask_refined
    
    def _resize_mask_to_image(
        self,
        mask: np.ndarray,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize mask to original image resolution.
        
        Args:
            mask: Binary mask array of shape (H', W')
            target_size: Target size as (width, height)
            
        Returns:
            Resized binary mask array
        """
        # Use nearest neighbor interpolation to preserve binary values
        mask_resized = cv2.resize(
            mask,
            target_size,
            interpolation=cv2.INTER_NEAREST
        )
        
        # Ensure binary values
        mask_resized = (mask_resized > 0).astype(np.uint8)
        
        return mask_resized
    
    def cleanup(self):
        """
        Release GPU memory and cleanup resources.
        """
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            logger.info("Cleaned up GPU memory")
