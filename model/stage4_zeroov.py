#!/usr/bin/env python3
"""
Stage 4: Spatial Grounding using RSVG-ZeroOV (Qwen2.5-VL + Stable Diffusion + SAM2.1)

This stage uses RSVG-ZeroOV pipeline to generate segmentation masks:
1. Qwen2.5-VL cross-attention
2. Stable Diffusion self-attention
3. Fusion + Region growing + SAM2.1

Usage (VidSTG dataset):
    python model/stage4_zeroov.py --output-dir model/vidstg_declarative

Usage (SAVG dataset - test set only):
    python model/stage4_zeroov.py --dataset savg --output-dir model/output/savg --num-gpus 4

Usage (HC dataset):
    python model/stage4_zeroov.py --dataset hc --output-dir model/hc_v1

Usage (HCSTVG-v1 dataset):
    python model/stage4_zeroov.py --dataset hcstvg-v1 --output-dir model/hcstvg_v1_test

Usage (HCSTVG-v2 dataset):
    python model/stage4_zeroov.py --dataset hcstvg-v2 --output-dir model/hcstvg_v2_val

Note: Stage 4 reads from Stage 1 output directory, so make sure Stage 1 has been run first.
"""

import os
import sys
import json
import argparse
import subprocess
import multiprocessing
import time
from multiprocessing import Process, Queue, Manager
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from PIL import Image
import cv2
import torch
import gc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import ConfigManager

SUPPORTED_DATASETS = ("vidstg_declarative", "hcstvg_v1", "hcstvg_v2", "savg")
DATASET_ALIASES = {
    "vidstg": "vidstg_declarative",
    "vidstg_declarative": "vidstg_declarative",
    "hcstvg-v1": "hcstvg_v1",
    "hcstvg_v1": "hcstvg_v1",
    "hcstvg-v2": "hcstvg_v2",
    "hcstvg_v2": "hcstvg_v2",
    "savg": "savg",
}
DATASET_TO_SENTENCE_TYPE = {
    "vidstg_declarative": "declarative",
    "hcstvg_v1": "hcstvg-v1",
    "hcstvg_v2": "hcstvg-v2",
    "savg": "savg",
}


def normalize_dataset_key(dataset: str) -> str:
    key = DATASET_ALIASES.get((dataset or "").strip())
    if key is None:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            f"Expected one of: {', '.join(SUPPORTED_DATASETS)}"
        )
    return key


def clear_gpu_memory():
    """Clear GPU memory thoroughly"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset peak memory stats to help release fragmented memory
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
        # Give GPU some time to release memory
        time.sleep(0.1)


def is_task_completed(
    output_dir: str,
    vid_id: str,
    stage4_dir_name: str = "stage4",
) -> bool:
    """
    Check if a task has already been completed.
    
    Args:
        output_dir: Output directory
        vid_id: Video ID
        stage4_dir_name: Stage 4 output subdirectory name (default: stage4)
        
    Returns:
        True if task is completed, False otherwise
    """
    # Parse vid to get base_vid and idx
    if "_" in vid_id:
        base_vid, idx = vid_id.rsplit("_", 1)
    else:
        base_vid = vid_id
        idx = None
    
    # Check Stage 4 output directory
    stage4_base = os.path.join(output_dir, stage4_dir_name)
    if idx:
        final_output_dir = os.path.join(stage4_base, base_vid, idx)
    else:
        final_output_dir = os.path.join(stage4_base, base_vid)
    
    metadata_path = os.path.join(final_output_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return False
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if task was successful
        if metadata.get('success', False):
            # Also check if essential output files exist
            mask_path = os.path.join(final_output_dir, "mask.png")
            if os.path.exists(mask_path):
                return True
    except Exception:
        pass
    
    return False


def load_stage1_data(
    base_dir: str,
    vid: str,
    sentence_type: str
) -> Optional[Dict]:
    """
    Load stage1 data for a given video.
    
    Args:
        base_dir: Base directory containing vidstg_declarative, vidstg_interrogative, savg, or hcstvg datasets
        vid: Video ID (e.g., "2400171624_1" or "irrigation_system_5_1" or "1_gEI9qBdVt5I_1")
        sentence_type: "declarative", "interrogative", "savg", "hcstvg-v1", or "hcstvg-v2"
        
    Returns:
        Stage1 metadata dictionary or None if not found
    """
    # Parse vid to get base_vid and idx
    if "_" in vid:
        base_vid, idx = vid.rsplit("_", 1)
    else:
        base_vid = vid
        idx = None
    
    # Preferred: dataset-root layout base_dir/stage1/{base_vid}/{idx}/
    if idx:
        stage1_path = os.path.join(base_dir, "stage1", base_vid, str(idx), "metadata.json")
        key_frame_path = os.path.join(base_dir, "stage1", base_vid, str(idx), "key_frame.png")
    else:
        stage1_path = os.path.join(base_dir, "stage1", base_vid, "metadata.json")
        key_frame_path = os.path.join(base_dir, "stage1", base_vid, "key_frame.png")

    # Backward compatibility: parent-root layout base_dir/<prefix>/stage1/...
    if not os.path.exists(stage1_path):
        prefix = None
        if sentence_type == "declarative":
            prefix = "vidstg_declarative"
        elif sentence_type == "interrogative":
            prefix = "vidstg_interrogative"
        elif sentence_type == "savg":
            prefix = "savg"
        elif sentence_type in ["hcstvg-v1", "hcstvg-v2"]:
            prefix = None
        if prefix:
            if idx:
                stage1_path = os.path.join(base_dir, prefix, "stage1", base_vid, str(idx), "metadata.json")
                key_frame_path = os.path.join(base_dir, prefix, "stage1", base_vid, str(idx), "key_frame.png")
            else:
                stage1_path = os.path.join(base_dir, prefix, "stage1", base_vid, "metadata.json")
                key_frame_path = os.path.join(base_dir, prefix, "stage1", base_vid, "key_frame.png")
    
    if not os.path.exists(stage1_path):
        print(f"✗ Stage 1 metadata not found: {stage1_path}")
        return None
    
    if not os.path.exists(key_frame_path):
        print(f"✗ Stage 1 key frame not found: {key_frame_path}")
        return None
    
    with open(stage1_path, 'r') as f:
        stage1_meta = json.load(f)
    
    # Add paths to metadata
    stage1_meta['key_frame_path'] = key_frame_path
    
    return stage1_meta




def step_4_1_qwen_attention(
    key_frame_path: str,
    text_query: str,
    output_dir: str,
    vid_id: str,
    config: Dict,
    rsvg_dir: str = "/mnt/data/disk2/zyu/videoVG/RSVG-ZeorOV",
    gpu_id: int = 0,
    stage4_dir_name: str = "stage4",
    qwen_model: Optional[str] = None,
) -> Optional[str]:
    """
    Step 4.1: Qwen-VL Cross Attention (Qwen2.5-VL or Qwen3-VL)
    
    Args:
        key_frame_path: Path to key frame image
        text_query: Text query
        output_dir: Output directory
        vid_id: Video ID
        config: Configuration dictionary
        rsvg_dir: RSVG-ZeroOV directory
        gpu_id: GPU ID
        stage4_dir_name: Stage 4 output subdirectory name (default: stage4)
        qwen_model: Override Qwen model path (default: from config rsvg.qwen_model)
        
    Returns:
        Path to cross attention file or None if failed
    """
    # Convert key_frame_path to absolute path
    if not os.path.isabs(key_frame_path):
        key_frame_path = os.path.abspath(key_frame_path)
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    # Parse vid to get base_vid and idx
    if "_" in vid_id:
        base_vid, idx = vid_id.rsplit("_", 1)
    else:
        base_vid = vid_id
        idx = None
    
    # Create Stage 4 output directory
    stage4_base = os.path.join(output_dir, stage4_dir_name)
    if idx:
        final_output_dir = os.path.join(stage4_base, base_vid, idx)
    else:
        final_output_dir = os.path.join(stage4_base, base_vid)
    
    os.makedirs(final_output_dir, exist_ok=True)
    
    cross_attn_path = os.path.join(final_output_dir, "step1_cross_attn.npy")
    
    # Check if already completed
    if os.path.exists(cross_attn_path):
        print(f"  [Step 4.1] {vid_id}: Already completed, skipping")
        return cross_attn_path
    
    try:
        print(f"  [Step 4.1] {vid_id}: Qwen2.5-VL Cross Attention...")
        clear_gpu_memory()
        
        # Create a temporary working directory inside rsvg_dir to avoid file conflicts
        # This ensures any relative paths in llmattn.py still work
        import tempfile
        import shutil
        import uuid
        
        # Create temp directory inside rsvg_dir
        temp_work_dir = os.path.join(rsvg_dir, f"temp_qwen_{vid_id}_{uuid.uuid4().hex[:8]}")
        os.makedirs(temp_work_dir, exist_ok=True)
        
        try:
            rsvg_config = config.get('rsvg', {})
            qwen_model_path = qwen_model if qwen_model else rsvg_config.get('qwen_model', '/home/xdu/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct')
            
            # Convert key_frame_path to absolute path
            abs_key_frame_path = os.path.abspath(key_frame_path)
            
            # Use absolute path to llmattn.py and run from temp directory
            # This way output goes to temp directory, avoiding conflicts
            llmattn_script = os.path.join(rsvg_dir, "llmattn.py")
            if not os.path.exists(llmattn_script):
                print(f"  ✗ [Step 4.1] {vid_id}: llmattn.py not found at {llmattn_script}")
                shutil.rmtree(temp_work_dir, ignore_errors=True)
                return None
            
            cmd = [
                "python", os.path.abspath(llmattn_script),
                "--image_path", abs_key_frame_path,
                "--question", text_query,
                "--model_id", qwen_model_path
            ]
            
            env = os.environ.copy()
            if "CUDA_VISIBLE_DEVICES" not in env:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # Get timeout from config (default: 600 seconds = 10 minutes for model loading)
            rsvg_config = config.get('rsvg', {})
            timeout = rsvg_config.get('qwen_timeout', 600)
            
            # Run from temp directory so output goes there
            try:
                result = subprocess.run(
                    cmd, cwd=temp_work_dir, capture_output=True, text=True, 
                    env=env, timeout=timeout
                )
            except subprocess.TimeoutExpired:
                print(f"  ✗ [Step 4.1] {vid_id}: Timeout after {timeout}s")
                print(f"    Model loading may take too long or process hung")
                shutil.rmtree(temp_work_dir, ignore_errors=True)
                clear_gpu_memory()
                return None
            except KeyboardInterrupt:
                print(f"  ✗ [Step 4.1] {vid_id}: Interrupted by user")
                shutil.rmtree(temp_work_dir, ignore_errors=True)
                clear_gpu_memory()
                return None
            
            clear_gpu_memory()
            
            if result.returncode != 0:
                print(f"  ✗ [Step 4.1] {vid_id}: Failed (returncode: {result.returncode})")
                
                # Print both stderr and stdout for better debugging
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')
                    # Show last 10 lines of stderr
                    print(f"    Stderr (last 10 lines):")
                    for line in stderr_lines[-10:]:
                        if line.strip():
                            print(f"      {line}")
                elif result.stdout:
                    # If no stderr, check stdout for errors
                    stdout_lines = result.stdout.split('\n')
                    error_lines = [l for l in stdout_lines if any(keyword in l.lower() for keyword in ['error', 'exception', 'failed', 'traceback'])]
                    if error_lines:
                        print(f"    Errors in stdout (last 10):")
                        for line in error_lines[-10:]:
                            if line.strip():
                                print(f"      {line}")
                    else:
                        # Show last 10 lines of stdout if no obvious errors found
                        print(f"    Stdout (last 10 lines):")
                        for line in stdout_lines[-10:]:
                            if line.strip():
                                print(f"      {line}")
                else:
                    print(f"    No error output captured (process may have been killed)")
                
                shutil.rmtree(temp_work_dir, ignore_errors=True)
                return None
            
            # Look for heatmap in temp directory
            heatmap_src = os.path.join(temp_work_dir, "heatmap_16171819.npy")
            if os.path.exists(heatmap_src):
                shutil.move(heatmap_src, cross_attn_path)
                print(f"  ✓ [Step 4.1] {vid_id}: Completed")
                # Clean up temp directory
                shutil.rmtree(temp_work_dir, ignore_errors=True)
                return cross_attn_path
            else:
                print(f"  ✗ [Step 4.1] {vid_id}: Cross attention heatmap not found at {heatmap_src}")
                if result.stdout:
                    # Show last part of stdout for debugging
                    stdout_lines = result.stdout.split('\n')
                    print(f"    Last stdout lines: {stdout_lines[-5:]}")
                shutil.rmtree(temp_work_dir, ignore_errors=True)
                return None
                
        except Exception as e:
            # Clean up temp directory on error
            shutil.rmtree(temp_work_dir, ignore_errors=True)
            raise e
        
    except Exception as e:
        print(f"  ✗ [Step 4.1] {vid_id}: Exception - {e}")
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return None


def step_4_2_sd_attention(
    key_frame_path: str,
    text_query: str,
    output_dir: str,
    vid_id: str,
    config: Dict,
    rsvg_dir: str = "/mnt/data/disk2/zyu/videoVG/RSVG-ZeorOV",
    gpu_id: int = 0,
    stage4_dir_name: str = "stage4",
) -> Optional[str]:
    """
    Step 4.2: Stable Diffusion Self Attention
    
    Args:
        key_frame_path: Path to key frame image
        text_query: Text query
        output_dir: Output directory
        vid_id: Video ID
        config: Configuration dictionary
        rsvg_dir: RSVG-ZeroOV directory
        gpu_id: GPU ID
        
    Returns:
        Path to self attention directory or None if failed
    """
    # Convert key_frame_path to absolute path
    if not os.path.isabs(key_frame_path):
        key_frame_path = os.path.abspath(key_frame_path)
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    # Parse vid to get base_vid and idx
    if "_" in vid_id:
        base_vid, idx = vid_id.rsplit("_", 1)
    else:
        base_vid = vid_id
        idx = None
    
    # Create Stage 4 output directory
    stage4_base = os.path.join(output_dir, stage4_dir_name)
    if idx:
        final_output_dir = os.path.join(stage4_base, base_vid, idx)
    else:
        final_output_dir = os.path.join(stage4_base, base_vid)
    
    step2_attn_dir = os.path.join(final_output_dir, "step2_attn")
    os.makedirs(step2_attn_dir, exist_ok=True)
    
    self_attn_dir = os.path.join(step2_attn_dir, "pipeline")
    
    # Check if already completed
    if os.path.exists(self_attn_dir) and len(os.listdir(self_attn_dir)) > 0:
        print(f"  [Step 4.2] {vid_id}: Already completed, skipping")
        return self_attn_dir
    
    try:
        print(f"  [Step 4.2] {vid_id}: Stable Diffusion Self Attention...")
        clear_gpu_memory()
        
        rsvg_config = config.get('rsvg', {})
        sd_model_path = rsvg_config.get('sd_model', '/home/xdu/.cache/modelscope/hub/models/CompVis/stable-diffusion-v1-4')
        
        cmd = [
            "python", "generate_single.py",
            "--image_path", key_frame_path,
            "--prompt", text_query,
            "--tag_id", "pipeline",
            "--output_dir", step2_attn_dir,
            "--model_path", sd_model_path
        ]
        
        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        result = subprocess.run(cmd, cwd=rsvg_dir, capture_output=True, text=True, env=env)
        
        clear_gpu_memory()
        
        if result.returncode != 0:
            print(f"  ✗ [Step 4.2] {vid_id}: Failed")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            return None
        
        if not os.path.exists(self_attn_dir):
            print(f"  ✗ [Step 4.2] {vid_id}: Self attention directory not found at {self_attn_dir}")
            return None
        
        print(f"  ✓ [Step 4.2] {vid_id}: Completed")
        return self_attn_dir
        
    except Exception as e:
        print(f"  ✗ [Step 4.2] {vid_id}: Exception - {e}")
        clear_gpu_memory()
        return None


def step_4_3_fusion_sam(
    key_frame_path: str,
    text_query: str,
    output_dir: str,
    vid_id: str,
    config: Dict,
    cross_attn_path: str,
    self_attn_dir: str,
    rsvg_dir: str = "/mnt/data/disk2/zyu/videoVG/RSVG-ZeorOV",
    gpu_id: int = 0,
    stage4_dir_name: str = "stage4",
) -> Optional[Dict]:
    """
    Step 4.3: Fusion + Region Growing + SAM2.1 and final processing
    
    Args:
        key_frame_path: Path to key frame image
        text_query: Text query
        output_dir: Output directory
        vid_id: Video ID
        config: Configuration dictionary
        cross_attn_path: Path to cross attention file from step 4.1
        self_attn_dir: Path to self attention directory from step 4.2
        rsvg_dir: RSVG-ZeroOV directory
        gpu_id: GPU ID
        
    Returns:
        Metadata dictionary or None if failed
    """
    # Convert key_frame_path to absolute path
    if not os.path.isabs(key_frame_path):
        key_frame_path = os.path.abspath(key_frame_path)
    
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    
    # Parse vid to get base_vid and idx
    if "_" in vid_id:
        base_vid, idx = vid_id.rsplit("_", 1)
    else:
        base_vid = vid_id
        idx = None
    
    # Create Stage 4 output directory
    stage4_base = os.path.join(output_dir, stage4_dir_name)
    if idx:
        final_output_dir = os.path.join(stage4_base, base_vid, idx)
    else:
        final_output_dir = os.path.join(stage4_base, base_vid)
    
    os.makedirs(final_output_dir, exist_ok=True)
    
    try:
        print(f"  [Step 4.3] {vid_id}: Fusion + Region Growing + SAM2.1...")
        clear_gpu_memory()
        
        step3_output = os.path.join(final_output_dir, "step3_final")
        os.makedirs(step3_output, exist_ok=True)
        
        rsvg_config = config.get('rsvg', {})
        sam_checkpoint = rsvg_config.get('sam_checkpoint', '/home/xdu/.cache/modelscope/hub/models/facebook/sam2.1-hiera-base-plus')
        threshold = rsvg_config.get('threshold', 0.4)
        
        cmd = [
            "python", "rs_evolve_single.py",
            "--image_path", key_frame_path,
            "--cross_attn_path", cross_attn_path,
            "--self_attn_dir", self_attn_dir,
            "--output_dir", step3_output,
            "--sam_checkpoint", sam_checkpoint,
            "--use_sam",
            "--threshold", str(threshold)
        ]
        
        env = os.environ.copy()
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        result = subprocess.run(cmd, cwd=rsvg_dir, capture_output=True, text=True, env=env)
        
        clear_gpu_memory()
        
        if result.returncode != 0:
            print(f"  ✗ [Step 4.3] {vid_id}: Failed")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            return None
        
        # Load final mask
        final_mask_path = os.path.join(step3_output, "final_mask.npy")
        if not os.path.exists(final_mask_path):
            print(f"  ✗ [Step 4.3] {vid_id}: Final mask not found: {final_mask_path}")
            return None
        
        initial_mask = np.load(final_mask_path)
        
        # Resize mask to match key frame
        key_frame_image = np.array(Image.open(key_frame_path))
        if initial_mask.shape != key_frame_image.shape[:2]:
            initial_mask = cv2.resize(
                initial_mask,
                (key_frame_image.shape[1], key_frame_image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        mask_coverage = (np.sum(initial_mask) / initial_mask.size) * 100
        print(f"  ✓ [Step 4.3] {vid_id}: Mask generated, coverage: {mask_coverage:.2f}%")
        
        # Save mask files
        mask_png_path = os.path.join(final_output_dir, "mask.png")
        mask_npy_path = os.path.join(final_output_dir, "mask.npy")
        Image.fromarray((initial_mask * 255).astype(np.uint8)).save(mask_png_path)
        np.save(mask_npy_path, initial_mask)
        
        # Create visualization overlay
        overlay_image = key_frame_image.copy()
        mask_overlay = np.zeros_like(overlay_image)
        mask_overlay[initial_mask > 0] = [0, 255, 0]
        overlay_image = cv2.addWeighted(overlay_image, 0.7, mask_overlay, 0.3, 0)
        overlay_path = os.path.join(final_output_dir, "mask_overlay.png")
        Image.fromarray(overlay_image).save(overlay_path)
        
        # Copy key frame
        from shutil import copyfile
        copyfile(key_frame_path, os.path.join(final_output_dir, "key_frame.png"))
        
        # Calculate bbox from mask
        mask_coords = np.where(initial_mask > 0)
        if len(mask_coords[0]) > 0:
            y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
            x_min, x_max = mask_coords[1].min(), mask_coords[1].max()
            bbox = {
                'xmin': int(x_min),
                'ymin': int(y_min),
                'xmax': int(x_max),
                'ymax': int(y_max)
            }
        else:
            bbox = None
        
        # Calculate CLIP similarity
        print(f"  [Step 4.3] {vid_id}: Calculating CLIP similarity...")
        box_similarity = 0.0
        mask_similarity = 0.0
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            clip_config = config.get('clip', {})
            clip_model_path = clip_config.get('model_path', '/home/xdu/.cache/modelscope/hub/models/openai-mirror/clip-vit-base-patch16')
            
            clip_model = CLIPModel.from_pretrained(clip_model_path)
            clip_processor = CLIPProcessor.from_pretrained(clip_model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model = clip_model.to(device)
            
            target_text = text_query
            
            # 1. Calculate Box similarity
            if bbox:
                box_crop = key_frame_image[y_min:y_max, x_min:x_max]
                box_crop_pil = Image.fromarray(box_crop)
                
                inputs = clip_processor(
                    text=[target_text],
                    images=box_crop_pil,
                    return_tensors="pt",
                    padding=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    box_similarity = logits_per_image.cpu().numpy()[0][0]
                
                box_crop_pil.save(os.path.join(final_output_dir, "box_crop.png"))
            
            # 2. Calculate Mask similarity
            masked_image = key_frame_image.copy()
            masked_image[initial_mask == 0] = 0
            masked_image_pil = Image.fromarray(masked_image)
            
            inputs = clip_processor(
                text=[target_text],
                images=masked_image_pil,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                mask_similarity = logits_per_image.cpu().numpy()[0][0]
            
            masked_image_pil.save(os.path.join(final_output_dir, "masked_crop.png"))
            
            del clip_model
            del clip_processor
            clear_gpu_memory()
            
        except Exception as e:
            print(f"    ✗ CLIP similarity calculation failed: {e}")
            clear_gpu_memory()
        
        # Save metadata
        metadata = {
            'vid': vid_id,
            'text_query': text_query,
            'method': 'rsvg_zeroov',
            'success': True,
            'mask_coverage': float(mask_coverage),
            'clip_box_similarity': float(box_similarity),
            'clip_mask_similarity': float(mask_similarity),
            'bbox': bbox,
            'key_frame_path': key_frame_path
        }
        
        metadata_path = os.path.join(final_output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ [Step 4.3] {vid_id}: Completed! Output: {final_output_dir}")
        clear_gpu_memory()
        
        return metadata
        
    except Exception as e:
        print(f"  ✗ [Step 4.3] {vid_id}: Exception - {e}")
        import traceback
        traceback.print_exc()
        
        failure_metadata = {
            'vid': vid_id,
            'text_query': text_query,
            'method': 'rsvg_zeroov',
            'success': False,
            'failure_reason': str(e)
        }
        metadata_path = os.path.join(final_output_dir, "metadata.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump(failure_metadata, f, indent=2, ensure_ascii=False)
        except:
            pass
        
        clear_gpu_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            time.sleep(0.3)
        
        return None


def process_zeroov(
    key_frame_path: str,
    text_query: str,
    output_dir: str,
    vid_id: str,
    config: Dict,
    rsvg_dir: str = "/mnt/data/disk2/zyu/videoVG/RSVG-ZeorOV",
    gpu_id: int = 0,
    stage4_dir_name: str = "stage4",
    qwen_model: Optional[str] = None,
) -> Optional[Dict]:
    """
    Process RSVG-ZeroOV pipeline to generate segmentation mask.
    This function is kept for backward compatibility but now delegates to the three step functions.
    
    Args:
        key_frame_path: Path to key frame image
        text_query: Text query
        output_dir: Output directory
        vid_id: Video ID
        config: Configuration dictionary
        rsvg_dir: RSVG-ZeroOV directory
        gpu_id: GPU ID
        
    Returns:
        Metadata dictionary or None if failed
    """
    print(f"\n{'='*80}")
    print(f"Stage 4: {vid_id} - RSVG-ZeroOV Spatial Grounding")
    print(f"{'='*80}\n")
    print(f"Text query: '{text_query}'")
    print(f"Key frame: {key_frame_path}")
    
    # Step 4.1
    cross_attn_path = step_4_1_qwen_attention(
        key_frame_path, text_query, output_dir, vid_id, config, rsvg_dir, gpu_id,
        stage4_dir_name=stage4_dir_name, qwen_model=qwen_model
    )
    if not cross_attn_path:
        return None
    
    # Step 4.2
    self_attn_dir = step_4_2_sd_attention(
        key_frame_path, text_query, output_dir, vid_id, config, rsvg_dir, gpu_id,
        stage4_dir_name=stage4_dir_name
    )
    if not self_attn_dir:
        return None
    
    # Step 4.3
    metadata = step_4_3_fusion_sam(
        key_frame_path, text_query, output_dir, vid_id, config,
        cross_attn_path, self_attn_dir, rsvg_dir, gpu_id,
        stage4_dir_name=stage4_dir_name
    )
    
    return metadata


def task_producer(tasks: List[Tuple], task_queue: Queue, num_gpus: int):
    """
    Producer process that loads tasks into the queue.
    
    Args:
        tasks: List of (key_frame_path, text_query, vid_id) tuples
        task_queue: Queue to put tasks into
        num_gpus: Number of GPUs (for end signals)
    """
    try:
        for task in tasks:
            task_queue.put(task)
        
        # Put end signals for each worker
        for _ in range(num_gpus):
            task_queue.put(None)
    except Exception as e:
        print(f"✗ Task producer error: {e}")
        import traceback
        traceback.print_exc()


def process_zeroov_worker(
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    output_dir: str,
    config: Dict,
    rsvg_dir: str,
    gpu_delay: float = 2.0,
    stage4_dir_name: str = "stage4",
    qwen_model: Optional[str] = None,
):
    """
    Worker process for processing tasks on a specific GPU.
    
    Args:
        gpu_id: GPU ID to use
        task_queue: Queue containing tasks (key_frame_path, text_query, vid_id)
        result_queue: Queue for results
        output_dir: Output directory
        config: Configuration dictionary
        rsvg_dir: RSVG-ZeroOV directory
        gpu_delay: Delay before starting (to stagger GPU model loading)
        stage4_dir_name: Stage 4 output subdirectory name (default: stage4)
        qwen_model: Override Qwen model path (default: from config)
    """
    # Set CUDA_VISIBLE_DEVICES to isolate this worker to one GPU
    # Each worker should use a different physical GPU
    parent_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if parent_cuda_visible:
        try:
            parent_visible_gpus = [int(x.strip()) for x in parent_cuda_visible.split(',') if x.strip()]
            if gpu_id < len(parent_visible_gpus):
                physical_gpu_id = parent_visible_gpus[gpu_id]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_gpu_id)
            else:
                # Fallback: use gpu_id directly
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        except (ValueError, AttributeError):
            # Fallback: use gpu_id directly
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        # No parent restriction, use gpu_id directly
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Verify CUDA_VISIBLE_DEVICES is set correctly
    print(f"[GPU {gpu_id}] CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    
    # Import after setting CUDA_VISIBLE_DEVICES
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    current_file_path = Path(__file__)
    if not current_file_path.exists():
        import inspect
        current_file_path = Path(inspect.getfile(process_zeroov_worker)).parent
    
    sys.path.insert(0, str(current_file_path.parent))
    
    from modules.config_manager import ConfigManager
    
    # Delay to stagger model loading across GPUs
    if gpu_delay > 0:
        delay = gpu_id * gpu_delay
        print(f"[GPU {gpu_id}] Waiting {delay:.1f}s before starting (to stagger model loading)...")
        time.sleep(delay)
    
    print(f"[GPU {gpu_id}] Worker started")
    
    # Load config in worker
    config_path = config.get('config_path', 'model/config.yaml')
    try:
        config_manager = ConfigManager(config_path)
        worker_config = config_manager.config
    except Exception as e:
        print(f"[GPU {gpu_id}] ⚠️  Failed to load config from {config_path}: {e}, using provided config")
        worker_config = config
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    processed_count = 0
    
    try:
        while True:
            try:
                # Get task from queue
                task = task_queue.get(timeout=5)
                
                if task is None:  # End signal
                    break
                
                key_frame_path, text_query, vid_id = task
                processed_count += 1
                
                # Check if task is already completed
                if is_task_completed(output_dir, vid_id, stage4_dir_name=stage4_dir_name):
                    skip_count += 1
                    result_queue.put({
                        'vid': vid_id,
                        'gpu_id': gpu_id,
                        'success': True,
                        'status': 'skipped'
                    })
                    print(f"[GPU {gpu_id}] Skipping {vid_id} (already completed)")
                    continue
                
                print(f"[GPU {gpu_id}] Processing {vid_id} ({processed_count} tasks)")
                
                try:
                    result = process_zeroov(
                        key_frame_path=key_frame_path,
                        text_query=text_query,
                        output_dir=output_dir,
                        vid_id=vid_id,
                        config=worker_config,
                        rsvg_dir=rsvg_dir,
                        gpu_id=0,  # In worker, CUDA_VISIBLE_DEVICES is set, so use 0
                        stage4_dir_name=stage4_dir_name,
                        qwen_model=qwen_model,
                    )
                    
                    if result:
                        success_count += 1
                        result_queue.put({
                            'vid': vid_id,
                            'gpu_id': gpu_id,
                            'success': True,
                            'status': 'completed'
                        })
                        print(f"[GPU {gpu_id}] ✓ {vid_id} -> completed")
                    else:
                        fail_count += 1
                        result_queue.put({
                            'vid': vid_id,
                            'gpu_id': gpu_id,
                            'success': False,
                            'status': 'failed'
                        })
                        print(f"[GPU {gpu_id}] ✗ {vid_id} -> failed")
                    
                except Exception as e:
                    fail_count += 1
                    error_msg = str(e)
                    result_queue.put({
                        'vid': vid_id,
                        'gpu_id': gpu_id,
                        'success': False,
                        'status': 'failed',
                        'error': error_msg[:100] if error_msg else None
                    })
                    print(f"[GPU {gpu_id}] ✗ {vid_id} -> error: {error_msg[:100]}")
                    # Clear GPU memory on exception
                    clear_gpu_memory()
                
                # Clear GPU memory after each task (thoroughly)
                clear_gpu_memory()
                # Additional cleanup: ensure subprocess memory is released
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    # Wait a bit longer for GPU to release memory from subprocesses
                    time.sleep(0.2)
                
            except Exception as e:
                if "Empty" not in str(e):  # Ignore queue timeout
                    print(f"[GPU {gpu_id}] Error in worker loop: {e}")
                continue
    finally:
        print(f"[GPU {gpu_id}] Worker finished: {success_count} success, {fail_count} failed, {skip_count} skipped, {processed_count} total")
        result_queue.put({'worker_done': True, 'gpu_id': gpu_id})


def batch_step1_producer(valid_tasks, task_queue, num_gpus):
    """Producer for step 4.1"""
    try:
        for task in valid_tasks:
            task_queue.put(('step1', task))
        for _ in range(num_gpus):
            task_queue.put(None)
    except Exception as e:
        print(f"✗ Step 4.1 producer error: {e}")


def batch_step1_worker(gpu_id, task_queue, result_queue, output_dir, config, rsvg_dir, gpu_delay, stage4_dir_name="stage4", qwen_model=None):
    """Worker for step 4.1"""
    parent_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if parent_cuda_visible:
        try:
            parent_visible_gpus = [int(x.strip()) for x in parent_cuda_visible.split(',') if x.strip()]
            if gpu_id < len(parent_visible_gpus):
                physical_gpu_id = parent_visible_gpus[gpu_id]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_gpu_id)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        except:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    if gpu_delay > 0:
        time.sleep(gpu_id * gpu_delay)
    
    config_path = config.get('config_path', 'model/config.yaml')
    try:
        from modules.config_manager import ConfigManager
        config_manager = ConfigManager(config_path)
        worker_config = config_manager.config
    except:
        worker_config = config
    
    while True:
        try:
            item = task_queue.get(timeout=5)
            if item is None:
                break
            
            step_name, (key_frame_path, text_query, vid_id) = item
            if step_name == 'step1':
                cross_attn_path = step_4_1_qwen_attention(
                    key_frame_path, text_query, output_dir, vid_id,
                    worker_config, rsvg_dir, gpu_id=0,
                    stage4_dir_name=stage4_dir_name, qwen_model=qwen_model
                )
                result_queue.put({
                    'vid': vid_id,
                    'step': 'step1',
                    'result': cross_attn_path,
                    'gpu_id': gpu_id
                })
                clear_gpu_memory()
        except:
            continue


def batch_step2_producer(step2_tasks, task_queue, num_gpus):
    """Producer for step 4.2"""
    try:
        for task in step2_tasks:
            task_queue.put(('step2', task))
        for _ in range(num_gpus):
            task_queue.put(None)
    except Exception as e:
        print(f"✗ Step 4.2 producer error: {e}")


def batch_step2_worker(gpu_id, task_queue, result_queue, output_dir, config, rsvg_dir, gpu_delay, stage4_dir_name="stage4"):
    """Worker for step 4.2"""
    parent_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if parent_cuda_visible:
        try:
            parent_visible_gpus = [int(x.strip()) for x in parent_cuda_visible.split(',') if x.strip()]
            if gpu_id < len(parent_visible_gpus):
                physical_gpu_id = parent_visible_gpus[gpu_id]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_gpu_id)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        except:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    if gpu_delay > 0:
        time.sleep(gpu_id * gpu_delay)
    
    config_path = config.get('config_path', 'model/config.yaml')
    try:
        from modules.config_manager import ConfigManager
        config_manager = ConfigManager(config_path)
        worker_config = config_manager.config
    except:
        worker_config = config
    
    while True:
        try:
            item = task_queue.get(timeout=5)
            if item is None:
                break
            
            step_name, (key_frame_path, text_query, vid_id) = item
            if step_name == 'step2':
                self_attn_dir = step_4_2_sd_attention(
                    key_frame_path, text_query, output_dir, vid_id,
                    worker_config, rsvg_dir, gpu_id=0,
                    stage4_dir_name=stage4_dir_name
                )
                result_queue.put({
                    'vid': vid_id,
                    'step': 'step2',
                    'result': self_attn_dir,
                    'gpu_id': gpu_id
                })
                clear_gpu_memory()
        except:
            continue


def batch_step3_producer(step3_task_data, task_queue, num_gpus):
    """Producer for step 4.3
    step3_task_data: List of (key_frame_path, text_query, vid_id, cross_attn_path, self_attn_dir) tuples
    """
    try:
        for task_data in step3_task_data:
            task_queue.put(('step3', task_data))
        for _ in range(num_gpus):
            task_queue.put(None)
    except Exception as e:
        print(f"✗ Step 4.3 producer error: {e}")


def batch_step3_worker(gpu_id, task_queue, result_queue, output_dir, config, rsvg_dir, gpu_delay, stage4_dir_name="stage4"):
    """Worker for step 4.3"""
    parent_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if parent_cuda_visible:
        try:
            parent_visible_gpus = [int(x.strip()) for x in parent_cuda_visible.split(',') if x.strip()]
            if gpu_id < len(parent_visible_gpus):
                physical_gpu_id = parent_visible_gpus[gpu_id]
                os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_gpu_id)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        except:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    if gpu_delay > 0:
        time.sleep(gpu_id * gpu_delay)
    
    config_path = config.get('config_path', 'model/config.yaml')
    try:
        from modules.config_manager import ConfigManager
        config_manager = ConfigManager(config_path)
        worker_config = config_manager.config
    except:
        worker_config = config
    
    while True:
        try:
            item = task_queue.get(timeout=5)
            if item is None:
                break
            
            step_name, task_data = item
            if step_name == 'step3':
                key_frame_path, text_query, vid_id, cross_attn_path, self_attn_dir = task_data
                metadata = step_4_3_fusion_sam(
                    key_frame_path, text_query, output_dir, vid_id,
                    worker_config, cross_attn_path, self_attn_dir,
                    rsvg_dir, gpu_id=0,
                    stage4_dir_name=stage4_dir_name
                )
                result_queue.put({
                    'vid': vid_id,
                    'step': 'step3',
                    'success': metadata is not None,
                    'gpu_id': gpu_id
                })
                clear_gpu_memory()
        except:
            continue


def process_videos_batch_steps(
    tasks: List[Tuple],
    output_dir: str,
    config: Dict,
    rsvg_dir: str,
    num_gpus: int = 4,
    gpu_delay: float = 10.0,
    stage4_dir_name: str = "stage4",
    qwen_model: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Process videos in batch mode: all tasks complete step 4.1, then all tasks complete step 4.2, then all tasks complete step 4.3.
    This avoids repeated model loading and reduces GPU memory fragmentation.
    
    Args:
        tasks: List of (key_frame_path, text_query, vid_id) tuples
        output_dir: Output directory
        config: Configuration dictionary
        rsvg_dir: RSVG-ZeroOV directory
        num_gpus: Number of GPUs to use
        gpu_delay: Delay between GPU model loading (seconds)
        stage4_dir_name: Stage 4 output subdirectory name (default: stage4)
        qwen_model: Override Qwen model path (default: from config)
    
    Returns:
        Tuple of (success_count, fail_count)
    """
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return 0, len(tasks)
    
    available_gpus = torch.cuda.device_count()
    if num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(num_gpus, available_gpus)
    
    if num_gpus <= 1:
        print("⚠️  Only 1 GPU available or requested, using sequential processing")
        num_gpus = 1
    
    print(f"\n{'='*80}")
    print(f"Batch Step Processing Mode")
    print(f"{'='*80}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Using {num_gpus} GPU(s)")
    print(f"Step order: 4.1 (Qwen) -> 4.2 (SD) -> 4.3 (Fusion+SAM)")
    print(f"{'='*80}\n")
    
    # Filter out already completed tasks
    valid_tasks = []
    skip_count = 0
    for task in tasks:
        key_frame_path, text_query, vid_id = task
        if is_task_completed(output_dir, vid_id, stage4_dir_name=stage4_dir_name):
            skip_count += 1
            print(f"⊘ Skipping {vid_id} (already completed)")
        else:
            valid_tasks.append(task)
    
    if skip_count > 0:
        print(f"⊘ Skipped {skip_count} already completed tasks\n")
    
    if not valid_tasks:
        print("No tasks to process.")
        return 0, 0
    
    print(f"Processing {len(valid_tasks)} tasks in batch step mode...\n")
    
    # Store intermediate results: vid_id -> (cross_attn_path, self_attn_dir)
    step_results = {}
    
    # ==========================================
    # Step 4.1: Qwen2.5-VL Cross Attention (all tasks)
    # ==========================================
    print(f"\n{'='*80}")
    print(f"Step 4.1: Qwen2.5-VL Cross Attention (Processing all {len(valid_tasks)} tasks)")
    print(f"{'='*80}\n")
    
    if num_gpus > 1:
        # Multi-GPU processing for step 4.1
        manager = Manager()
        task_queue = manager.Queue(maxsize=num_gpus * 3)
        result_queue = manager.Queue()
        
        producer = Process(
            target=batch_step1_producer,
            args=(valid_tasks, task_queue, num_gpus)
        )
        producer.start()
        
        workers = []
        for gpu_id in range(num_gpus):
            worker = Process(
                target=batch_step1_worker,
                args=(gpu_id, task_queue, result_queue, output_dir, config, rsvg_dir, gpu_delay, stage4_dir_name, qwen_model)
            )
            worker.start()
            workers.append(worker)
            if gpu_id < num_gpus - 1:
                time.sleep(2.0)
        
        completed = 0
        while completed < len(valid_tasks):
            try:
                result = result_queue.get(timeout=30)
                vid_id = result['vid']
                cross_attn_path = result['result']
                if cross_attn_path:
                    step_results[vid_id] = {'cross_attn_path': cross_attn_path}
                completed += 1
            except:
                if all(not w.is_alive() for w in workers):
                    break
        
        producer.join(timeout=60)
        for worker in workers:
            worker.join(timeout=10)
    else:
        # Single GPU sequential processing for step 4.1
        for idx, (key_frame_path, text_query, vid_id) in enumerate(valid_tasks, 1):
            print(f"[{idx}/{len(valid_tasks)}] Processing step 4.1: {vid_id}")
            cross_attn_path = step_4_1_qwen_attention(
                key_frame_path, text_query, output_dir, vid_id,
                config, rsvg_dir, gpu_id=0,
                stage4_dir_name=stage4_dir_name, qwen_model=qwen_model
            )
            if cross_attn_path:
                step_results[vid_id] = {'cross_attn_path': cross_attn_path}
            clear_gpu_memory()
    
    print(f"\n✓ Step 4.1 completed: {len(step_results)}/{len(valid_tasks)} tasks succeeded")
    
    # Filter tasks that passed step 4.1
    step2_tasks = [(kf, tq, vid) for kf, tq, vid in valid_tasks if vid in step_results]
    
    if not step2_tasks:
        print("✗ No tasks passed step 4.1, aborting")
        return 0, len(valid_tasks)
    
    # ==========================================
    # Step 4.2: Stable Diffusion Self Attention (all remaining tasks)
    # ==========================================
    print(f"\n{'='*80}")
    print(f"Step 4.2: Stable Diffusion Self Attention (Processing {len(step2_tasks)} tasks)")
    print(f"{'='*80}\n")
    
    if num_gpus > 1:
        # Multi-GPU processing for step 4.2
        manager = Manager()
        task_queue = manager.Queue(maxsize=num_gpus * 3)
        result_queue = manager.Queue()
        
        producer = Process(
            target=batch_step2_producer,
            args=(step2_tasks, task_queue, num_gpus)
        )
        producer.start()
        
        workers = []
        for gpu_id in range(num_gpus):
            worker = Process(
                target=batch_step2_worker,
                args=(gpu_id, task_queue, result_queue, output_dir, config, rsvg_dir, gpu_delay, stage4_dir_name)
            )
            worker.start()
            workers.append(worker)
            if gpu_id < num_gpus - 1:
                time.sleep(2.0)
        
        completed = 0
        while completed < len(step2_tasks):
            try:
                result = result_queue.get(timeout=30)
                vid_id = result['vid']
                self_attn_dir = result['result']
                if self_attn_dir and vid_id in step_results:
                    step_results[vid_id]['self_attn_dir'] = self_attn_dir
                completed += 1
            except:
                if all(not w.is_alive() for w in workers):
                    break
        
        producer.join(timeout=60)
        for worker in workers:
            worker.join(timeout=10)
    else:
        # Single GPU sequential processing for step 4.2
        for idx, (key_frame_path, text_query, vid_id) in enumerate(step2_tasks, 1):
            print(f"[{idx}/{len(step2_tasks)}] Processing step 4.2: {vid_id}")
            self_attn_dir = step_4_2_sd_attention(
                key_frame_path, text_query, output_dir, vid_id,
                config, rsvg_dir, gpu_id=0,
                stage4_dir_name=stage4_dir_name
            )
            if self_attn_dir and vid_id in step_results:
                step_results[vid_id]['self_attn_dir'] = self_attn_dir
            clear_gpu_memory()
    
    step2_success = sum(1 for v in step_results.values() if 'self_attn_dir' in v)
    print(f"\n✓ Step 4.2 completed: {step2_success}/{len(step2_tasks)} tasks succeeded")
    
    # Filter tasks that passed step 4.2
    step3_tasks = []
    for kf, tq, vid in step2_tasks:
        if vid in step_results and 'self_attn_dir' in step_results[vid]:
            step3_tasks.append((kf, tq, vid))
    
    if not step3_tasks:
        print("✗ No tasks passed step 4.2, aborting")
        return 0, len(valid_tasks)
    
    # ==========================================
    # Step 4.3: Fusion + Region Growing + SAM2.1 (all remaining tasks)
    # ==========================================
    print(f"\n{'='*80}")
    print(f"Step 4.3: Fusion + Region Growing + SAM2.1 (Processing {len(step3_tasks)} tasks)")
    print(f"{'='*80}\n")
    
    success_count = 0
    fail_count = 0
    
    if num_gpus > 1:
        # Multi-GPU processing for step 4.3
        # Prepare task data with step results
        step3_task_data = []
        for kf, tq, vid in step3_tasks:
            if vid in step_results:
                cross_attn_path = step_results[vid]['cross_attn_path']
                self_attn_dir = step_results[vid]['self_attn_dir']
                step3_task_data.append((kf, tq, vid, cross_attn_path, self_attn_dir))
        
        manager = Manager()
        task_queue = manager.Queue(maxsize=num_gpus * 3)
        result_queue = manager.Queue()
        
        producer = Process(
            target=batch_step3_producer,
            args=(step3_task_data, task_queue, num_gpus)
        )
        producer.start()
        
        workers = []
        for gpu_id in range(num_gpus):
            worker = Process(
                target=batch_step3_worker,
                args=(gpu_id, task_queue, result_queue, output_dir, config, rsvg_dir, gpu_delay, stage4_dir_name)
            )
            worker.start()
            workers.append(worker)
            if gpu_id < num_gpus - 1:
                time.sleep(2.0)
        
        completed = 0
        while completed < len(step3_tasks):
            try:
                result = result_queue.get(timeout=30)
                if result.get('success'):
                    success_count += 1
                else:
                    fail_count += 1
                completed += 1
            except:
                if all(not w.is_alive() for w in workers):
                    break
        
        producer.join(timeout=60)
        for worker in workers:
            worker.join(timeout=10)
    else:
        # Single GPU sequential processing for step 4.3
        for idx, (key_frame_path, text_query, vid_id) in enumerate(step3_tasks, 1):
            print(f"[{idx}/{len(step3_tasks)}] Processing step 4.3: {vid_id}")
            if vid_id in step_results:
                cross_attn_path = step_results[vid_id]['cross_attn_path']
                self_attn_dir = step_results[vid_id]['self_attn_dir']
                metadata = step_4_3_fusion_sam(
                    key_frame_path, text_query, output_dir, vid_id,
                    config, cross_attn_path, self_attn_dir,
                    rsvg_dir, gpu_id=0,
                    stage4_dir_name=stage4_dir_name
                )
                if metadata:
                    success_count += 1
                else:
                    fail_count += 1
            clear_gpu_memory()
    
    # Count failures from earlier steps
    fail_count += len(valid_tasks) - len(step3_tasks)
    
    print(f"\n{'='*80}")
    print("Batch Step Processing Complete")
    print(f"{'='*80}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Skipped (already completed): {skip_count}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    processed_count = len(valid_tasks)
    if processed_count > 0:
        print(f"Success rate: {success_count/processed_count*100:.2f}%")
    print(f"{'='*80}")
    
    return success_count, fail_count


def process_videos_multi_gpu(
    tasks: List[Tuple],
    output_dir: str,
    config: Dict,
    rsvg_dir: str,
    num_gpus: int = 4,
    gpu_delay: float = 10.0
) -> Tuple[int, int]:
    """
    Process videos using multiple GPUs in parallel.
    
    Args:
        tasks: List of (key_frame_path, text_query, vid_id) tuples
        output_dir: Output directory
        config: Configuration dictionary
        rsvg_dir: RSVG-ZeroOV directory
        num_gpus: Number of GPUs to use
        gpu_delay: Delay between GPU model loading (seconds)
    
    Returns:
        Tuple of (success_count, fail_count)
    """
    # Set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available, falling back to single GPU processing")
        return 0, len(tasks)
    
    available_gpus = torch.cuda.device_count()
    if num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(num_gpus, available_gpus)
    
    if num_gpus <= 1:
        print("⚠️  Only 1 GPU available or requested, falling back to sequential processing")
        return 0, len(tasks)
    
    print(f"Using {num_gpus} GPUs (available: {available_gpus})")
    print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")
    
    # Create queues
    manager = Manager()
    task_queue = manager.Queue(maxsize=num_gpus * 3)
    result_queue = manager.Queue()
    
    # Start producer process
    producer = Process(
        target=task_producer,
        args=(tasks, task_queue, num_gpus)
    )
    producer.start()
    print(f"Task producer started (queue buffer: {num_gpus * 3})")
    
    # Start worker processes with staggered delays
    workers = []
    for gpu_id in range(num_gpus):
        worker = Process(
            target=process_zeroov_worker,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                output_dir,
                config,
                rsvg_dir,
                gpu_delay
            )
        )
        worker.start()
        workers.append(worker)
        print(f"Started worker on GPU {gpu_id} (will start after {gpu_id * gpu_delay:.1f}s delay)")
        # Stagger worker startup
        if gpu_id < num_gpus - 1:
            time.sleep(2.0)
    
    # Collect results
    success_count = 0
    fail_count = 0
    skip_count = 0
    completed_workers = 0
    
    print(f"\nProcessing tasks on {num_gpus} GPUs in parallel...")
    print("=" * 80)
    
    while completed_workers < num_gpus:
        try:
            result = result_queue.get(timeout=30)
            
            if result.get('worker_done'):
                completed_workers += 1
                print(f"Worker on GPU {result['gpu_id']} completed ({completed_workers}/{num_gpus})")
            else:
                vid = result.get('vid', 'unknown')
                status = result.get('status', 'unknown')
                
                if status == 'skipped':
                    skip_count += 1
                    print(f"⊘ [GPU {result.get('gpu_id', '?')}] {vid} -> skipped (already completed)")
                elif result.get('success'):
                    success_count += 1
                    print(f"✓ [GPU {result.get('gpu_id', '?')}] {vid} -> {status}")
                else:
                    fail_count += 1
                    print(f"✗ [GPU {result.get('gpu_id', '?')}] {vid} -> {status}")
                
        except Exception as e:
            if "Empty" not in str(e):
                print(f"Error collecting results: {e}")
            # Check if workers are still alive
            alive_workers = sum(1 for w in workers if w.is_alive())
            if alive_workers == 0 and completed_workers < num_gpus:
                print("⚠️  All workers died unexpectedly")
                break
    
    # Wait for producer to finish
    producer.join(timeout=60)
    if producer.is_alive():
        print(f"⚠️  Producer did not terminate, forcing...")
        producer.terminate()
        producer.join()
    
    # Wait for all workers to finish
    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            print(f"⚠️  Worker {worker.pid} did not terminate, forcing...")
            worker.terminate()
            worker.join()
    
    print(f"\n{'='*80}")
    print("Multi-GPU Processing Complete")
    print(f"{'='*80}")
    print(f"Total: {len(tasks)}")
    print(f"Skipped (already completed): {skip_count}")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    processed_count = len(tasks) - skip_count
    if processed_count > 0:
        print(f"Success rate: {success_count/processed_count*100:.2f}%")
    print(f"{'='*80}")
    
    return success_count, fail_count


def determine_sentence_type(output_dir: str) -> Optional[str]:
    """Determine sentence type from output_dir path."""
    output_dir = os.path.abspath(output_dir)
    if 'vidstg_declarative' in output_dir:
        return 'declarative'
    elif 'vidstg_interrogative' in output_dir:
        return 'interrogative'
    elif 'savg' in output_dir:
        return 'savg'
    elif 'hcstvg_v1' in output_dir or 'hcstvg-v1' in output_dir:
        return 'hcstvg-v1'
    elif 'hcstvg_v2' in output_dir or 'hcstvg-v2' in output_dir:
        return 'hcstvg-v2'
    return None


def main():
    parser = argparse.ArgumentParser(description='Stage 4: RSVG-ZeroOV Spatial Grounding')
    parser.add_argument('--video-id', type=str, help='Video ID from test set')
    parser.add_argument('--image-path', type=str, help='Direct path to image file')
    parser.add_argument('--text-query', type=str, help='Text query')
    parser.add_argument('--num', type=int, default=None, help='Number of videos to process')
    parser.add_argument('--dataset', type=str, default='vidstg_declarative', 
                        choices=['vidstg_declarative', 'hcstvg_v1', 'hcstvg_v2', 'savg', 'vidstg', 'hcstvg-v1', 'hcstvg-v2'],
                        help='Dataset key (recommended): vidstg_declarative / hcstvg_v1 / hcstvg_v2 / savg')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: from config.yaml based on dataset)')
    parser.add_argument('--config', type=str, default='model/config.yaml', help='Config file')
    parser.add_argument('--rsvg-dir', type=str, default='/mnt/data/disk2/zyu/videoVG/RSVG-ZeorOV',
                        help='RSVG-ZeroOV directory')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs to use for parallel processing (default: all available)')
    parser.add_argument('--gpu-delay', type=float, default=10.0,
                        help='Delay between GPU model loading in seconds (default: 10.0)')
    parser.add_argument('--stage4-dir-name', type=str, default='stage4',
                        help='Stage 4 output subdirectory name under output_dir (default: stage4). E.g. stage4_zeroov_qwen3_8b for Qwen3-VL runs.')
    parser.add_argument('--qwen-model', type=str, default=None,
                        help='Override Qwen VL model path (default: from config rsvg.qwen_model). E.g. /path/to/Qwen3-VL-8B-Instruct for Qwen3-VL.')
    
    args = parser.parse_args()
    
    args.dataset = normalize_dataset_key(args.dataset)
    sentence_type = DATASET_TO_SENTENCE_TYPE[args.dataset]
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Get output_dir from config if not provided
    if args.output_dir is None:
        if args.dataset:
            dataset_config = config.get('datasets', {}).get(args.dataset, {})
            if 'output_dir' in dataset_config:
                args.output_dir = dataset_config['output_dir']
                print(f"Using output_dir from config: {args.output_dir}")
            else:
                args.output_dir = 'model/vidstg_declarative'
        else:
            dataset_config = config.get('datasets', {}).get('vidstg_declarative', {})
            args.output_dir = dataset_config.get('output_dir', 'model/vidstg_declarative')
    elif args.dataset:
        if args.output_dir == 'model/vidstg_declarative':
            dataset_config = config.get('datasets', {}).get(args.dataset, {})
            if 'output_dir' in dataset_config:
                args.output_dir = dataset_config['output_dir']
                print(f"Using output_dir from config: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tasks = []  # List of (key_frame_path, text_query, vid_id)
    
    # ==========================================
    # Mode 1: Single Image (CLI)
    # ==========================================
    if args.image_path:
        if not args.text_query:
            parser.error("--text-query is required with --image-path")
        vid_id = args.video_id if args.video_id else Path(args.image_path).stem
        tasks.append((args.image_path, args.text_query, vid_id))
    
    # ==========================================
    # Mode 2: Specific Video ID (Folder Scan)
    # ==========================================
    elif args.video_id:
        # Dataset-first: output_dir is the dataset root.
        base_dir = args.output_dir
        
        stage1_meta = load_stage1_data(base_dir, args.video_id, sentence_type)
        if stage1_meta:
            # Convert to absolute path
            key_frame_path = os.path.abspath(stage1_meta['key_frame_path'])
            tasks.append((key_frame_path, stage1_meta['text_query'], args.video_id))
    
    # ==========================================
    # Mode 3: Batch Process (Folder Scan)
    # ==========================================
    else:
        dataset_info = f" ({args.dataset} dataset)" if args.dataset else ""
        print(f"Scanning for Stage 1 outputs in {args.output_dir}{dataset_info}...")
        
        sentence_type = DATASET_TO_SENTENCE_TYPE[args.dataset]
        
        # For savg dataset, read from test_ann.json to get exact task list
        if args.dataset == 'savg':
            # Try to get base_dir from config, fallback to default
            savg_config = config.get('datasets', {}).get('savg', {})
            savg_base_dir = savg_config.get('base_dir', '/mnt/data/disk2/zyu/videoVG/data/savg')
            annotation_file = os.path.join(savg_base_dir, 'test_ann.json')
            
            if os.path.exists(annotation_file):
                print(f"Loading SAVG test annotations from {annotation_file}...")
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)
                
                print(f"✓ Loaded {len(annotations)} annotations")
                
                # For savg, directly scan stage1 outputs (same as stage2, no deduplication)
                # This ensures stage4 processes the same number of tasks as stage2
                stage1_base_dir = os.path.join(args.output_dir, "stage1")
                
                if os.path.exists(stage1_base_dir):
                    video_dirs = os.listdir(stage1_base_dir)
                    for vid in video_dirs:
                        stage1_dir = os.path.join(stage1_base_dir, vid)
                        if not os.path.isdir(stage1_dir):
                            continue
                        
                        # Check sub-folders
                        subitems = [s for s in os.listdir(stage1_dir) if s.isdigit()]
                        if subitems:
                            for s in sorted(subitems, key=int):
                                vid_id = f"{vid}_{s}"
                                stage1_meta_path = os.path.join(stage1_base_dir, vid, s, "metadata.json")
                                key_frame_path = os.path.join(stage1_base_dir, vid, s, "key_frame.png")
                                if os.path.exists(stage1_meta_path) and os.path.exists(key_frame_path):
                                    # Convert to absolute path
                                    key_frame_path = os.path.abspath(key_frame_path)
                                    
                                    with open(stage1_meta_path, 'r') as f:
                                        stage1_meta = json.load(f)
                                    
                                    text_query = stage1_meta.get('text_query', '')
                                    
                                    # Add all tasks (no deduplication, same as stage2)
                                    tasks.append((key_frame_path, text_query, vid_id))
                        else:
                            # Root folder
                            stage1_meta_path = os.path.join(stage1_base_dir, vid, "metadata.json")
                            key_frame_path = os.path.join(stage1_base_dir, vid, "key_frame.png")
                            if os.path.exists(stage1_meta_path) and os.path.exists(key_frame_path):
                                # Convert to absolute path
                                key_frame_path = os.path.abspath(key_frame_path)
                                
                                with open(stage1_meta_path, 'r') as f:
                                    stage1_meta = json.load(f)
                                
                                text_query = stage1_meta.get('text_query', '')
                                
                                # Add all tasks (no deduplication, same as stage2)
                                tasks.append((key_frame_path, text_query, vid))
                
                print(f"✓ Found {len(tasks)} tasks from stage1 outputs (same as stage2)")
            else:
                print(f"⚠️  Annotation file not found: {annotation_file}, falling back to folder scan")
                # Fall back to folder scan (same as stage2, no deduplication)
                base_dir = args.output_dir
                stage1_base_dir = os.path.join(args.output_dir, "stage1")
                
                if os.path.exists(stage1_base_dir):
                    video_dirs = os.listdir(stage1_base_dir)
                    for vid in video_dirs:
                        stage1_dir = os.path.join(stage1_base_dir, vid)
                        if not os.path.isdir(stage1_dir):
                            continue
                        
                        # Check sub-folders
                        subitems = [s for s in os.listdir(stage1_dir) if s.isdigit()]
                        if subitems:
                            for s in sorted(subitems, key=int):
                                vid_id = f"{vid}_{s}"
                                # For savg, directly construct path (no prefix needed)
                                stage1_meta_path = os.path.join(stage1_base_dir, vid, s, "metadata.json")
                                key_frame_path = os.path.join(stage1_base_dir, vid, s, "key_frame.png")
                                if os.path.exists(stage1_meta_path) and os.path.exists(key_frame_path):
                                    # Convert to absolute path
                                    key_frame_path = os.path.abspath(key_frame_path)
                                    
                                    with open(stage1_meta_path, 'r') as f:
                                        stage1_meta = json.load(f)
                                    
                                    text_query = stage1_meta['text_query']
                                    
                                    # Add all tasks (no deduplication, same as stage2)
                                    tasks.append((key_frame_path, text_query, vid_id))
                        else:
                            # Root folder
                            stage1_meta_path = os.path.join(stage1_base_dir, vid, "metadata.json")
                            key_frame_path = os.path.join(stage1_base_dir, vid, "key_frame.png")
                            if os.path.exists(stage1_meta_path) and os.path.exists(key_frame_path):
                                # Convert to absolute path
                                key_frame_path = os.path.abspath(key_frame_path)
                                
                                with open(stage1_meta_path, 'r') as f:
                                    stage1_meta = json.load(f)
                                
                                text_query = stage1_meta['text_query']
                                
                                # Add all tasks (no deduplication, same as stage2)
                                tasks.append((key_frame_path, text_query, vid))
        else:
            # For all four datasets, output_dir is treated as dataset root
            base_dir = args.output_dir
            
            stage1_base_dir = os.path.join(args.output_dir, "stage1")
            
            if os.path.exists(stage1_base_dir):
                video_dirs = os.listdir(stage1_base_dir)
                for vid in video_dirs:
                    stage1_dir = os.path.join(stage1_base_dir, vid)
                    if not os.path.isdir(stage1_dir):
                        continue
                    
                    # Check sub-folders
                    subitems = [s for s in os.listdir(stage1_dir) if s.isdigit()]
                    if subitems:
                        for s in sorted(subitems, key=int):
                            vid_id = f"{vid}_{s}"
                            stage1_meta = load_stage1_data(base_dir, vid_id, sentence_type)
                            if stage1_meta:
                                # Convert to absolute path
                                key_frame_path = os.path.abspath(stage1_meta['key_frame_path'])
                                tasks.append((key_frame_path, stage1_meta['text_query'], vid_id))
                    else:
                        # Root folder
                        stage1_meta = load_stage1_data(base_dir, vid, sentence_type)
                        if stage1_meta:
                            # Convert to absolute path
                            key_frame_path = os.path.abspath(stage1_meta['key_frame_path'])
                            tasks.append((key_frame_path, stage1_meta['text_query'], vid))
        
        if args.num:
            tasks = tasks[:args.num]
    
    # Execute
    if tasks:
        print(f"✓ Found {len(tasks)} tasks.")
        
        # Check if multi-GPU processing is requested
        num_gpus = args.num_gpus
        if num_gpus and num_gpus > 1:
            # Multi-GPU batch step processing (all tasks complete 4.1, then all complete 4.2, then all complete 4.3)
            config_with_path = config.copy()
            config_with_path['config_path'] = args.config
            
            success_count, fail_count = process_videos_batch_steps(
                tasks=tasks,
                output_dir=args.output_dir,
                config=config_with_path,
                rsvg_dir=args.rsvg_dir,
                num_gpus=num_gpus,
                gpu_delay=args.gpu_delay,
                stage4_dir_name=args.stage4_dir_name,
                qwen_model=args.qwen_model,
            )
        else:
            # Single GPU sequential processing
            success_count = 0
            fail_count = 0
            skip_count = 0
            
            for idx, (key_frame_path, text_query, vid_id) in enumerate(tasks, 1):
                # Check if task is already completed
                if is_task_completed(args.output_dir, vid_id, stage4_dir_name=args.stage4_dir_name):
                    skip_count += 1
                    print(f"\n{'='*80}")
                    print(f"Skipping {idx}/{len(tasks)}: {vid_id} (already completed)")
                    print(f"{'='*80}")
                    continue
                
                print(f"\n{'='*80}")
                print(f"Processing {idx}/{len(tasks)}: {vid_id}")
                print(f"{'='*80}")
                
                result = process_zeroov(
                    key_frame_path=key_frame_path,
                    text_query=text_query,
                    output_dir=args.output_dir,
                    vid_id=vid_id,
                    config=config,
                    rsvg_dir=args.rsvg_dir,
                    gpu_id=0,
                    stage4_dir_name=args.stage4_dir_name,
                    qwen_model=args.qwen_model,
                )
                
                if result:
                    success_count += 1
                else:
                    fail_count += 1
            
            print(f"\n{'='*80}")
            print("Batch Processing Complete")
            print(f"{'='*80}")
            print(f"Total: {len(tasks)}")
            print(f"Skipped (already completed): {skip_count}")
            print(f"Success: {success_count}")
            print(f"Failed: {fail_count}")
            processed_count = len(tasks) - skip_count
            if processed_count > 0:
                print(f"Success rate: {success_count/processed_count*100:.2f}%")
            print(f"{'='*80}")
    else:
        print("No tasks found.")


if __name__ == '__main__':
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    main()
