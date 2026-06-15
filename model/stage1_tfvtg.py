#!/usr/bin/env python3
"""
Stage 1: Temporal Localization using TFVTG (Text-Frame Video Temporal Grounding)

This stage identifies the key frame that best matches the text query using BLIP-2 ITM.

Usage (VidSTG dataset):
    # Process all declarative sentences (all types, default)
    python model/stage1_tfvtg.py
    
    # Process only 'object' type captions
    python model/stage1_tfvtg.py --caption-type object
    
    # Process first 10 videos (all types)
    python model/stage1_tfvtg.py --num 10
    
    # Process specific video
    python model/stage1_tfvtg.py --video-id 7771650716
    
    # Process custom video
    python model/stage1_tfvtg.py --video-path path/to/video.mp4 --text-query "a dog walks"

Usage (SAVG dataset):
    # Process all captions from SAVG test set
    python model/stage1_tfvtg.py --dataset savg --output-dir model/savg_test
    
    # Process first 10 videos from SAVG test set
    python model/stage1_tfvtg.py --dataset savg --num 10 --output-dir model/savg_test
    
    # Process specific video from SAVG test set
    python model/stage1_tfvtg.py --dataset savg --video-id all-terrain_vehicle_13 --output-dir model/savg_test

Usage (HC dataset):
    # Process all videos from HC test set (skips corrupted videos automatically)
    python model/stage1_tfvtg.py --dataset hc --output-dir model/hc_test
    
    # Process first 10 videos from HC test set
    python model/stage1_tfvtg.py --dataset hc --num 10 --output-dir model/hc_test
    
    # Process specific video from HC test set (use video filename without extension)
    python model/stage1_tfvtg.py --dataset hc --video-id 55_vfjywN5CN0Y --output-dir model/hc_test

Usage (HCSTVG-v1 dataset - test split):
    # Process all videos from HCSTVG-v1 test set
    python model/stage1_tfvtg.py --dataset hcstvg-v1 --output-dir model/output/hcstvg_v1
    
    # Process first 10 videos
    python model/stage1_tfvtg.py --dataset hcstvg-v1 --num 10 --output-dir model/hcstvg_v2_test

Usage (HCSTVG-v2 dataset - val split):
    # Process all videos from HCSTVG-v2 val set
    python model/stage1_tfvtg.py --dataset hcstvg-v2 --output-dir model/hcstvg-v2
    
    # Process first 10 videos
    python model/stage1_tfvtg.py --dataset hcstvg-v2 --num 10 --output-dir model/hcstvg_v2_val
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
from PIL import Image
import torch
import gc
import multiprocessing
from multiprocessing import Process, Queue
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import cv2
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import ConfigManager
from modules.tfvtg_module import TFVTGModule

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
DATASET_RUNTIME_NAME = {
    "vidstg_declarative": "vidstg",
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
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def create_video_from_frames(frame_dir: str, fps: float = 30.0, temp_dir: Optional[str] = None) -> str:
    """
    Create a temporary video file from a directory of image frames.
    If the video already exists, return it directly to avoid recreating.
    
    Args:
        frame_dir: Directory containing image frames (e.g., /path/to/vid/img/)
        fps: Frames per second for the output video
        temp_dir: Temporary directory to store the video (if None, uses system temp)
        
    Returns:
        Path to the created temporary video file
    """
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
    
    # Create temporary video file path
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    
    video_id = Path(frame_dir).parent.name
    temp_video_path = os.path.join(temp_dir, f"{video_id}_temp.mp4")
    
    # Check if video already exists and is valid
    if os.path.exists(temp_video_path) and os.path.getsize(temp_video_path) > 0:
        return temp_video_path
    
    # Get all image files sorted by name
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    frame_files = []
    for ext in image_extensions:
        frame_files.extend(Path(frame_dir).glob(f'*{ext}'))
        frame_files.extend(Path(frame_dir).glob(f'*{ext.upper()}'))
    
    if not frame_files:
        raise ValueError(f"No image frames found in {frame_dir}")
    
    # Sort by filename (assuming they are numbered)
    frame_files = sorted(frame_files, key=lambda x: int(x.stem) if x.stem.isdigit() else float('inf'))
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise ValueError(f"Failed to read first frame: {frame_files[0]}")
    height, width, _ = first_frame.shape
    
    # Use H.264 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    try:
        num_frames = len(frame_files)
        for idx, frame_file in enumerate(frame_files):
            frame = cv2.imread(str(frame_file))
            if frame is not None:
                out.write(frame)
            # Progress indicator for large videos
            if num_frames > 100 and (idx + 1) % 100 == 0:
                print(f"    Creating video: {idx + 1}/{num_frames} frames...")
    finally:
        out.release()
    
    if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
        raise RuntimeError(f"Failed to create video file: {temp_video_path}")
    
    return temp_video_path


def get_savg_video_path(vid: str, savg_base_dir: str = '/mnt/data/disk2/zyu/videoVG/data/savg') -> Optional[str]:
    """
    Get video path for SAVG dataset. If video file exists, return it. Otherwise, 
    create a temporary video from image frames.
    
    Args:
        vid: Video ID
        savg_base_dir: Base directory of SAVG dataset
        
    Returns:
        Path to video file (temporary if created from frames)
    """
    # Try to find video file first
    video_dir = os.path.join(savg_base_dir, 'Test', 'Test', vid)
    if not os.path.exists(video_dir):
        return None
    
    # Check for video file
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    for ext in video_extensions:
        video_path = os.path.join(video_dir, f"{vid}{ext}")
        if os.path.exists(video_path):
            return video_path
    
    # If no video file, try to create from frames
    frame_dir = os.path.join(video_dir, 'img')
    if os.path.exists(frame_dir):
        try:
            # Load annotation to get fps
            ann_file = os.path.join(savg_base_dir, 'test_ann.json')
            fps = 30.0  # default
            if os.path.exists(ann_file):
                with open(ann_file, 'r') as f:
                    annotations = json.load(f)
                    for ann in annotations:
                        if ann.get('vid') == vid:
                            fps = ann.get('fps', 30.0)
                            break
            
            temp_video = create_video_from_frames(frame_dir, fps=fps)
            return temp_video
        except Exception as e:
            print(f"  Warning: Failed to create video from frames: {e}")
            return None
    
    return None


def get_savg_video_path_cached(vid: str, savg_base_dir: str, fps: float = 30.0) -> Optional[str]:
    """
    Get video path for SAVG dataset with cached fps (optimized version to avoid reloading JSON).
    
    Args:
        vid: Video ID
        savg_base_dir: Base directory of SAVG dataset
        fps: Frames per second (cached from annotation)
        
    Returns:
        Path to video file (temporary if created from frames)
    """
    # Try to find video file first
    video_dir = os.path.join(savg_base_dir, 'Test', 'Test', vid)
    if not os.path.exists(video_dir):
        return None
    
    # Check for video file
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    for ext in video_extensions:
        video_path = os.path.join(video_dir, f"{vid}{ext}")
        if os.path.exists(video_path):
            return video_path
    
    # If no video file, return frame directory directly (TFVTGModule now supports it!)
    frame_dir = os.path.join(video_dir, 'img')
    if os.path.exists(frame_dir):
        return frame_dir
    
    return None


def is_video_file_valid(video_path: str) -> bool:
    """
    Check if a video file is valid and can be opened.
    Supports mp4, mkv, avi, mov and other formats supported by OpenCV.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video is valid, False otherwise
    """
    if not os.path.exists(video_path):
        return False
    
    # Check file extension - if it's a known video format, try to open it
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v']
    file_ext = os.path.splitext(video_path)[1].lower()
    
    # If it's not a known video extension, still try to open it (might be valid)
    # But if it's clearly not a video (like .txt, .json), skip
    non_video_extensions = ['.txt', '.json', '.csv', '.yaml', '.yml', '.py', '.sh']
    if file_ext in non_video_extensions:
        return False
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        # For mkv files, sometimes the first read might fail but the file is still valid
        # So we check if we can at least open the file
        if not ret or frame is None:
            # Try one more time with a different approach
            cap2 = cv2.VideoCapture(video_path)
            if cap2.isOpened():
                # Get frame count - if > 0, file is likely valid
                frame_count = cap2.get(cv2.CAP_PROP_FRAME_COUNT)
                cap2.release()
                # If we can open it and it has frames, consider it valid
                # (even if first read failed, it might work during actual processing)
                if frame_count > 0:
                    return True
            return False
        
        return True
    except Exception as e:
        print(f"    Warning: Error checking video {video_path}: {e}")
        # For mkv files, if we can't verify but file exists, still try to process it
        # (OpenCV might handle it during actual processing)
        if file_ext == '.mkv':
            print(f"    Note: MKV file validation failed, but will attempt to process anyway")
            return True  # Give mkv files the benefit of the doubt
        return False


def get_hc_video_path(video_filename: str, hc_base_dir: str = '/mnt/data/disk2/zyu/videoVG/data/hc') -> Optional[str]:
    """
    Get video path for HC dataset. Checks if video file exists and is valid.
    Supports both mp4 and mkv files.
    
    Args:
        video_filename: Video filename (e.g., "55_vfjywN5CN0Y.mp4" or "10_A9WSiEDeu0I.mkv")
        hc_base_dir: Base directory of HC dataset
        
    Returns:
        Path to video file if valid, None otherwise
    """
    video_dir = os.path.join(hc_base_dir, 'v1')
    video_path = os.path.join(video_dir, video_filename)
    
    if not os.path.exists(video_path):
        return None
    
    # Check file extension
    file_ext = os.path.splitext(video_filename)[1].lower()
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v']
    
    if file_ext not in video_extensions:
        print(f"    Warning: Unknown video format: {file_ext} for {video_filename}")
        return None
    
    # For mkv files, be more lenient with validation
    # decord (used by TFVTGModule) supports mkv files better than OpenCV
    if file_ext == '.mkv':
        # Try validation, but if it fails, still return the path
        if is_video_file_valid(video_path):
            return video_path
        else:
            print(f"    Note: MKV file validation failed for {video_filename}, but will attempt to process with decord")
            # Still return the path - decord might handle it during actual processing
            return video_path
    else:
        # For other formats, use strict validation
        if is_video_file_valid(video_path):
            return video_path
        else:
            print(f"    Warning: Video file appears corrupted: {video_path}")
            return None


def get_hcstvg_video_path(video_filename: str, base_dir: str, video_dir_name: str = 'video') -> Optional[str]:
    """
    Get video path for HCSTVG dataset. Checks if video file exists and is valid.
    Supports both mp4 and mkv files. For mkv files, uses lenient validation
    since decord (used by TFVTGModule) supports mkv better than OpenCV.
    
    Args:
        video_filename: Video filename (e.g., "55_vfjywN5CN0Y.mp4" or "10_A9WSiEDeu0I.mkv")
        base_dir: Base directory of HCSTVG dataset
        video_dir_name: Name of video directory (default: 'video')
        
    Returns:
        Path to video file if valid, None otherwise
    """
    video_dir = os.path.join(base_dir, video_dir_name)
    video_path = os.path.join(video_dir, video_filename)
    
    if not os.path.exists(video_path):
        return None
    
    # Check file extension
    file_ext = os.path.splitext(video_filename)[1].lower()
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v']
    
    if file_ext not in video_extensions:
        print(f"    Warning: Unknown video format: {file_ext} for {video_filename}")
        return None
    
    # For mkv files, be more lenient with validation
    # decord (used by TFVTGModule) supports mkv files better than OpenCV
    # So even if OpenCV validation fails, the file might still work with decord
    if file_ext == '.mkv':
        # Try validation, but if it fails, still return the path
        # (decord will handle it during actual processing)
        if is_video_file_valid(video_path):
            return video_path
        else:
            print(f"    Note: MKV file validation failed for {video_filename}, but will attempt to process with decord")
            # Still return the path - decord might handle it during actual processing
            return video_path
    else:
        # For other formats (mp4, etc.), use strict validation
        if is_video_file_valid(video_path):
            return video_path
        else:
            print(f"    Warning: Video file appears corrupted: {video_path}")
    return None


def refine_target_object(target_object: str) -> str:
    """Convert 'adult' to 'person' for better generalization"""
    if target_object.lower() in ['adult', 'adults']:
        return 'person'
    return target_object


def _qwen_generate_text(model_path: str, prompt: str, max_new_tokens: int = 50) -> str:
    """Run a text-only chat completion with a Qwen model.

    Works with both Qwen text LLMs (Qwen2/Qwen3) and Qwen-VL checkpoints
    (Qwen2.5-VL / Qwen3-VL) -- the same model family Stage 2 (RSVG-ZeroOV) uses.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]

    # Preferred path: plain causal LM (Qwen2/Qwen3 text models).
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text
    except Exception:
        # Fallback: Qwen-VL conditional generation, text-only message.
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
        except Exception:
            from transformers import AutoModelForVision2Seq as _VLModel

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = _VLModel.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        vl_messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        chat_text = processor.apply_chat_template(
            vl_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[chat_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text


def extract_target_object_with_qwen(text_query: str, config: Optional[Dict] = None) -> str:
    """Extract the primary subject noun from a query using Qwen.

    Uses the same Qwen model as Stage 2 (RSVG-ZeroOV), resolved from
    ``config['rsvg']['qwen_model']``. Falls back to a rule-based extractor.
    """
    import re

    # 预处理：标准化输入
    text_query = text_query.strip()
    if text_query.endswith('.'):
        text_query = text_query[:-1]

    try:
        import torch  # noqa: F401
        import json as json_module

        # Same model as Stage 2 (RSVG-ZeroOV): rsvg.qwen_model
        model_path = None
        if config:
            model_path = config.get('rsvg', {}).get('qwen_model')
        if not model_path:
            model_path = "/home/xdu/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct"

        print(f"    Loading Qwen model for target extraction: {model_path}")

        # --- 核心修改：针对句法结构的精准 Prompt ---
        prompt = f"""You are a Linguistic Parser for Object Detection. 
Your task is to identify the **Primary Subject** of the sentence. The Primary Subject is the entity whose existence or location is being asserted, or the entity performing an action.

**CRITICAL REQUIREMENT:**
- Output ONLY ONE SINGLE WORD (the core noun, no adjectives, no modifiers)
- Examples: "woman" (not "adult woman"), "seat" (not "baby seat"), "adult" (not "female adult")

**ANALYSIS RULES:**
1. **"There is/are X..." structure**: The target is ALWAYS the core noun in **X**.
   - "There is a [chair] behind the man." -> Target: chair
   - "There is a [toy] near the baby." -> Target: toy
   - "There is a [seat] beneath a baby." -> Target: seat (NOT "baby seat")
2. **"X is [preposition] Y..." structure**: The target is ALWAYS the core noun in **X**.
   - "The [lamp] is above the table." -> Target: lamp
3. **Action structure (X does Y)**: The target is the core noun of agent **X**.
   - "A [dog] chases a cat." -> Target: dog
   - "An [adult] woman in pink..." -> Target: woman (NOT "adult woman")
   - "A [female] adult caresses..." -> Target: adult (NOT "female adult")
4. **Vocabulary Mapping**:
   - If the core noun is 'adult', output 'person'
   - If the core noun is 'man' or 'woman', keep it as is (do NOT convert to 'person')

**OUTPUT FORMAT:**
Return a JSON object with a single key "target_prompt". Value must be EXACTLY ONE WORD (the core noun only).

**FEW-SHOT EXAMPLES:**
Input: 'there is a blue chair behind a child outdoors.'
Output: {{"target_prompt": "chair"}}

Input: 'there is a sofa beneath an adult man in blue in a room.'
Output: {{"target_prompt": "sofa"}}

Input: 'an adult opens an oven in the kitchen.'
Output: {{"target_prompt": "person"}}

Input: 'there is an orange toy towards a baby on a sofa.'
Output: {{"target_prompt": "toy"}}

Input: 'the toy is next to the child in black.'
Output: {{"target_prompt": "toy"}}

Input: 'an adult woman in pink is away another adult woman in green.'
Output: {{"target_prompt": "woman"}}

Input: 'a female adult caresses a brown dog.'
Output: {{"target_prompt": "person"}}

Input: 'there is a baby seat beneath a baby.'
Output: {{"target_prompt": "seat"}}

---
**TASK:**
Sentence: '{text_query}'
Output:"""
        
        generated_text = _qwen_generate_text(model_path, prompt, max_new_tokens=50)
        print(f"    LLM raw output: {generated_text}")
        
        # Parse JSON
        target_prompt = ""
        try:
            # 尝试直接解析
            if "{" in generated_text:
                json_str = generated_text[generated_text.find("{"):generated_text.rfind("}")+1]
                result = json_module.loads(json_str)
                target_prompt = result.get('target_prompt', '').strip()
        except Exception as e:
            print(f"    JSON Parse error: {e}")
        
        # 如果 JSON 解析失败，尝试从文本提取（容错）
        if not target_prompt:
             # 简单的启发式清理
             clean = generated_text.replace('Output:', '').replace('"', '').replace('}', '').replace('{', '').strip()
             if clean:
                 target_prompt = clean

        if not target_prompt:
            raise ValueError("Empty target_prompt from LLM")
        
        # 后处理：确保只提取一个单词
        # 1. 处理下划线：将下划线替换为空格
        target_prompt = target_prompt.replace('_', ' ').strip()
        
        # 2. 如果包含多个词，只保留最后一个词（核心名词）
        words = target_prompt.split()
        if len(words) > 1:
            # 取最后一个词作为核心名词
            target_prompt = words[-1]
            print(f"    ⚠️  Multiple words detected, using last word: '{target_prompt}'")
        
        # 3. 确保 adult -> person 映射（在提取单个词之后）
        if target_prompt.lower() in ['adult', 'adults']:
            target_prompt = 'person'
        # 注意：man 和 woman 不再转换为 person，保持原样
            
        print(f"    ✓ Qwen extraction: '{target_prompt}'")

        return target_prompt

    except Exception as e:
        print(f"    Warning: Qwen target extraction failed: {e}, using enhanced rule-based fallback...")
        return extract_target_fallback(text_query)

def extract_target_fallback(text: str) -> str:
    """Enhanced Rule-based extraction logic"""
    import re
    text = text.lower().strip()
    
    # 规则 1: 处理 "There is/are [Target] [Preposition] [Reference]"
    # 捕获 "there is a" 后面的核心名词，直到遇到介词
    match_exist = re.search(r'there\s+(?:is|are)\s+(?:a|an|the)?\s*([a-z\s]+?)\s+(?:in|on|at|under|beneath|behind|above|next|near|with|towards|watching|biting|facing)', text)
    if match_exist:
        # 获取捕获组，例如 "blue chair"
        raw_target = match_exist.group(1).strip()
        # 取最后一个词作为核心词 (blue chair -> chair)
        target = raw_target.split()[-1]
        return normalize_target(target)

    # 规则 2: 处理 "[Target] is [Preposition]..."
    match_is = re.search(r'^(?:a|an|the)?\s*([a-z\s]+?)\s+is\s+', text)
    if match_is:
        raw_target = match_is.group(1).strip()
        target = raw_target.split()[-1]
        return normalize_target(target)
        
    # 规则 3: 处理 "[Target] [Action]..." (通用主语)
    # 简单提取第一个名词短语，但只取最后一个词（核心名词）
    match_simple = re.search(r'^(?:a|an|the)?\s*([a-z]+(?:\s+[a-z]+)?)', text)
    if match_simple:
        raw_target = match_simple.group(1).strip()
        # 取最后一个词作为核心词（例如 "adult woman" -> "woman"）
        target = raw_target.split()[-1]
        return normalize_target(target)

    return "object"

def normalize_target(word: str) -> str:
    """Normalize specific words and ensure single word output"""
    # 处理下划线
    word = word.replace('_', ' ').strip()
    
    # 如果包含多个词，只保留最后一个词（核心名词）
    words = word.split()
    if len(words) > 1:
        word = words[-1]
    
    # 映射 adult -> person
    if word.lower() in ['adult', 'adults']:
        return 'person'
    return word


def check_stage1_completed(output_dir: str, vid: str, sentence_index: Optional[int] = None) -> Optional[Dict]:
    """
    Check if Stage 1 is already completed for a given video and sentence.
    
    Args:
        output_dir: Base output directory
        vid: Video ID
        sentence_index: Optional sentence index (for videos with multiple sentences)
        
    Returns:
        Metadata dictionary if completed, None otherwise
    """
    # Build output path
    stage1_base = os.path.join(output_dir, "stage1", vid)
    if sentence_index is not None:
        stage1_output = os.path.join(stage1_base, str(sentence_index))
    else:
        stage1_output = stage1_base
    
    # Check if metadata.json exists
    metadata_path = os.path.join(stage1_output, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    
    # Check if key output files exist
    key_frame_path = os.path.join(stage1_output, "key_frame.png")
    similarity_scores_path = os.path.join(stage1_output, "similarity_scores.npy")
    
    if not os.path.exists(key_frame_path) or not os.path.exists(similarity_scores_path):
        return None
    
    # Load and return metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"  ⚠️  Warning: Failed to load existing metadata: {e}")
        return None


def visualize_similarity_scores(similarity_scores: np.ndarray, fps: float, output_dir: str, vid: str, key_frame_idx: int):
    """
    Visualize similarity scores over time (frame index).
    
    Args:
        similarity_scores: Array of similarity scores for each frame
        fps: Frames per second for sampling
        output_dir: Directory to save the visualization
        vid: Video ID
        key_frame_idx: Index of the key frame
    """
    try:
        num_frames = len(similarity_scores)
        frame_indices = np.arange(num_frames)
        time_seconds = frame_indices / fps  # Convert frame index to time in seconds
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot similarity scores
        ax.plot(time_seconds, similarity_scores, 'b-', linewidth=1.5, label='Similarity Score')
        
        # Mark the key frame
        key_frame_time = key_frame_idx / fps
        key_frame_score = similarity_scores[key_frame_idx]
        ax.plot(key_frame_time, key_frame_score, 'ro', markersize=10, label=f'Key Frame (idx={key_frame_idx})')
        ax.axvline(x=key_frame_time, color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        # Labels and title
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Similarity Score', fontsize=12)
        ax.set_title(f'Similarity Score vs Time - {vid}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add text annotation for key frame
        ax.annotate(
            f'Key Frame\nScore: {key_frame_score:.4f}',
            xy=(key_frame_time, key_frame_score),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
        
        # Save figure
        output_path = os.path.join(output_dir, "similarity_scores_plot.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Similarity scores visualization saved: {output_path}")
        
    except Exception as e:
        print(f"  ⚠️ Failed to create similarity scores visualization: {e}")
        import traceback
        traceback.print_exc()


def process_stage1(video_path: str, text_query: str, output_dir: str, config: Dict, vid: Optional[str] = None, use_mistral: bool = True, sentence_index: Optional[int] = None, device: str = 'cuda', video_fps: float = 30.0, ann_gt_info: Optional[Dict] = None) -> Optional[Dict]:
    """Process Stage 1: Temporal Localization"""
    if vid is None:
        vid = Path(video_path).stem
    
    print(f"\n{'='*80}")
    print(f"Stage 1: {vid} - Temporal Localization")
    print(f"{'='*80}\n")
    print(f"Query: '{text_query}'")
    
    # Check if already completed
    existing_metadata = check_stage1_completed(output_dir, vid, sentence_index)
    if existing_metadata is not None:
        print(f"✓ Stage 1 already completed! Skipping. Output: {os.path.join(output_dir, 'stage1', vid, str(sentence_index) if sentence_index else '')}")
        return existing_metadata
    
    try:
        clear_gpu_memory()
        
        tfvtg_config = config.get('tfvtg', {})
        # Use provided device or fall back to config
        actual_device = device if device != 'cuda' or torch.cuda.is_available() else 'cpu'
        tfvtg = TFVTGModule(
            model_name=tfvtg_config.get('model_name', 'blip2_image_text_matching'),
            model_type=tfvtg_config.get('model_type', 'coco'),
            device=actual_device,
            batch_size=tfvtg_config.get('batch_size', 128)
        )
        
        fps = tfvtg_config.get('fps', 3.0)
        # video_fps is now passed as parameter (from annotation for SAVG dataset)
        
        key_frame_idx, key_frame_image, similarity_scores, original_frame_idx = tfvtg.get_key_frame(
            video_path=video_path, text_query=text_query, fps=fps, video_fps=video_fps
        )
        
        print(f"✓ Key frame: {key_frame_idx} (original: {original_frame_idx}), score: {similarity_scores[key_frame_idx]:.4f}")
        
        # Extract target (skip for interrogative sentences)
        if use_mistral:
            print(f"\nExtracting target object...")
            target_object = extract_target_object_with_qwen(text_query, config)
            print(f"  Target: {target_object}")
        else:
            # For interrogative sentences, use text_query as target_object
            target_object = text_query
            print(f"\nSkipping target extraction for interrogative sentence")
            print(f"  Using text_query as target: {target_object[:50]}...")
        
        # Save results
        # Organize output by stage: output_dir/stage1/{vid}/
        stage1_base = os.path.join(output_dir, "stage1", vid)
        # If sentence_index is provided, create a subfolder for this sentence
        if sentence_index is not None:
            stage1_output = os.path.join(stage1_base, str(sentence_index))
        else:
            stage1_output = stage1_base
        os.makedirs(stage1_output, exist_ok=True)
        
        Image.fromarray(key_frame_image).save(os.path.join(stage1_output, "key_frame.png"))
        np.save(os.path.join(stage1_output, "similarity_scores.npy"), similarity_scores)
        
        # Visualize similarity score with time
        visualize_similarity_scores(similarity_scores, fps, stage1_output, vid, key_frame_idx)
        
        metadata = {
            'vid': vid,
            'video_path': video_path,
            'text_query': text_query,
            'target_object': target_object,
            'key_frame_idx': int(key_frame_idx),
            'original_frame_idx': int(original_frame_idx),
            'num_sampled_frames': len(similarity_scores),
            'max_similarity': float(similarity_scores[key_frame_idx]),
            'sampling_fps': fps
        }
        
        # Add sentence_index to metadata if provided
        if sentence_index is not None:
            metadata['sentence_index'] = sentence_index
        
        if ann_gt_info and isinstance(ann_gt_info, dict):
            if ann_gt_info.get('temporal_gt'):
                metadata['ann_temporal_gt'] = ann_gt_info['temporal_gt']
            if ann_gt_info.get('target_id') is not None:
                metadata['ann_target_id'] = ann_gt_info['target_id']
            if ann_gt_info.get('used_relation'):
                metadata['ann_used_relation'] = ann_gt_info['used_relation']
            if ann_gt_info.get('used_segment'):
                metadata['ann_used_segment'] = ann_gt_info['used_segment']
        
        with open(os.path.join(stage1_output, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Stage 1 completed! Output: {stage1_output}")
        
        tfvtg.cleanup()
        del tfvtg
        clear_gpu_memory()
        
        return metadata
        
    except Exception as e:
        print(f"✗ Stage 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Temporal Localization (TFVTG)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sentences (both declarative and interrogative, all types)
  python model/stage1_tfvtg.py
  
  # Process only declarative sentences (captions, all types)
  python model/stage1_tfvtg.py --sentence-type declarative
  
  # Process only interrogative sentences (questions, all types)
  python model/stage1_tfvtg.py --sentence-type interrogative
  
  # Process only 'object' type declarative sentences
  python model/stage1_tfvtg.py --sentence-type declarative --caption-type object
  
  # Process first 10 videos (both types)
  python model/stage1_tfvtg.py --num 10
  
  # Process first 10 interrogative sentences
  python model/stage1_tfvtg.py --num 10 --sentence-type interrogative
  
  # Process specific video
  python model/stage1_tfvtg.py --video-id 7771650716
  
  # Process custom video
  python model/stage1_tfvtg.py --video-path video.mp4 --text-query "a dog walks"
        """
    )
    
    parser.add_argument('--video-id', type=str, help='Video ID from test set')
    parser.add_argument('--video-path', type=str, help='Direct path to video file')
    parser.add_argument('--text-query', type=str, help='Text query (required with --video-path)')
    parser.add_argument('--num', type=int, default=None, help='Number of videos (default: all)')
    parser.add_argument('--caption-type', type=str, default=None, 
                        help='Filter captions by type (e.g., "object", "person"). If None, process all types (default: None)')
    parser.add_argument('--sentence-type', type=str, default='declarative', choices=['declarative', 'interrogative', 'both'],
                        help='Deprecated for dataset-mode runs. For the 4 datasets this is forced to declarative.')
    parser.add_argument('--dataset', type=str, default='vidstg_declarative', 
                        choices=['vidstg_declarative', 'hcstvg_v1', 'hcstvg_v2', 'savg', 'vidstg', 'hcstvg-v1', 'hcstvg-v2'],
                        help='Dataset key (recommended): vidstg_declarative / hcstvg_v1 / hcstvg_v2 / savg')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: from config.yaml based on dataset)')
    parser.add_argument('--config', type=str, default='/mnt/data/disk2/zyu/videoVG/model/config.yaml', help='Config file')
    parser.add_argument('--num-gpus', type=int, default=None, help='Number of GPUs to use for parallel processing (default: all available)')
    parser.add_argument('--savg-base-dir', type=str, default='/mnt/data/disk2/zyu/videoVG/data/savg',
                        help='Base directory for SAVG dataset (default: /mnt/data/disk2/zyu/videoVG/data/savg)')
    parser.add_argument('--hc-base-dir', type=str, default='/mnt/data/disk2/zyu/videoVG/data/hc',
                        help='Base directory for HC dataset (default: /mnt/data/disk2/zyu/videoVG/data/hc)')
    parser.add_argument('--hcstvg-v1-base-dir', type=str, default='/mnt/data/disk2/zyu/videoVG/data/hcstvg-v1',
                        help='Base directory for HCSTVG-v1 dataset (default: /mnt/data/disk2/zyu/videoVG/data/hcstvg-v1)')
    parser.add_argument('--hcstvg-v2-base-dir', type=str, default='/mnt/data/disk2/zyu/videoVG/data/hcstvg-v2',
                        help='Base directory for HCSTVG-v2 dataset (default: /mnt/data/disk2/zyu/videoVG/data/hcstvg-v2)')
    
    args = parser.parse_args()
    
    args.dataset_key = normalize_dataset_key(args.dataset)
    args.dataset = DATASET_RUNTIME_NAME[args.dataset_key]
    if args.sentence_type != 'declarative':
        print(
            f"⚠️  --sentence-type={args.sentence_type} ignored in dataset-mode; "
            "forcing declarative."
        )
        args.sentence_type = 'declarative'

    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Get output_dir from config if not provided
    if args.output_dir is None:
        # Try to get output_dir from config based on dataset
        if args.dataset_key:
            dataset_config = config.get('datasets', {}).get(args.dataset_key, {})
            if 'output_dir' in dataset_config:
                args.output_dir = dataset_config['output_dir']
                print(f"Using output_dir from config: {args.output_dir}")
            else:
                # Fallback to default
                args.output_dir = 'model/vidstg_declarative'
        else:
            # Default for vidstg
            dataset_config = config.get('datasets', {}).get('vidstg_declarative', {})
            args.output_dir = dataset_config.get('output_dir', 'model/vidstg_declarative')
    elif args.dataset:
        # If output_dir is provided but dataset is specified, check if we should override
        # Only override if output_dir is the default value
        if args.output_dir == 'model/vidstg_declarative':
            dataset_config = config.get('datasets', {}).get(args.dataset_key, {})
            if 'output_dir' in dataset_config:
                args.output_dir = dataset_config['output_dir']
                print(f"Using output_dir from config: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.video_path:
        # Single video mode
        if not args.text_query:
            parser.error("--text-query is required with --video-path")
        if not os.path.exists(args.video_path):
            print(f"✗ Video not found: {args.video_path}")
            return
        
        # Default to use mistral for custom video
        result = process_stage1(args.video_path, args.text_query, args.output_dir, config, use_mistral=True)
        print(f"\n{'='*80}")
        print("Stage 1 Completed!" if result else "Stage 1 Failed!")
        print(f"{'='*80}")
    
    elif args.video_id:
        # Single video from test set
        if args.dataset == 'hc':
            annotation_file = os.path.join(args.hc_base_dir, 'test.json')
            if not os.path.exists(annotation_file):
                print(f"✗ Annotation file not found: {annotation_file}")
                return
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            found = False
            # HC dataset uses video filename as key
            video_filename = None
            for key in annotations.keys():
                if key.startswith(args.video_id) or key == args.video_id:
                    video_filename = key
                    ann = annotations[key]
                    video_path = get_hc_video_path(video_filename, args.hc_base_dir)
                    if video_path:
                        caption = ann.get('caption', '')
                        if caption:
                            result = process_stage1(video_path, caption, args.output_dir, config, 
                                                  video_filename, use_mistral=True, sentence_index=1)
                            found = True
                    break
            
            if not found:
                print(f"✗ Video ID {args.video_id} not found in HC dataset or video file is corrupted")
        elif args.dataset == 'hcstvg-v1':
            # Get config for hcstvg-v1
            dataset_config = config.get('datasets', {}).get('hcstvg_v1', {})
            annotation_file = os.path.join(args.hcstvg_v1_base_dir, dataset_config.get('annotation_file', 'anno/test.json'))
            if not os.path.exists(annotation_file):
                print(f"✗ Annotation file not found: {annotation_file}")
                return
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            found = False
            caption_field = dataset_config.get('caption_field', 'caption')
            for video_filename, ann in annotations.items():
                if video_filename.startswith(args.video_id) or video_filename == args.video_id:
                    video_path = get_hcstvg_video_path(video_filename, args.hcstvg_v1_base_dir, 
                                                      dataset_config.get('video_dir', 'video'))
                    if video_path:
                        caption = ann.get(caption_field, '')
                        if caption:
                            vid = os.path.splitext(video_filename)[0]
                            result = process_stage1(video_path, caption, args.output_dir, config, 
                                                  vid, use_mistral=True, sentence_index=1)
                            found = True
                    break
            
            if not found:
                print(f"✗ Video ID {args.video_id} not found in HCSTVG-v1 dataset or video file is corrupted")
        elif args.dataset == 'hcstvg-v2':
            # Get config for hcstvg-v2
            dataset_config = config.get('datasets', {}).get('hcstvg_v2', {})
            annotation_file = os.path.join(args.hcstvg_v2_base_dir, dataset_config.get('annotation_file', 'anno/val_v2.json'))
            if not os.path.exists(annotation_file):
                print(f"✗ Annotation file not found: {annotation_file}")
                return
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            found = False
            caption_field = dataset_config.get('caption_field', 'English')
            for video_filename, ann in annotations.items():
                if video_filename.startswith(args.video_id) or video_filename == args.video_id:
                    video_path = get_hcstvg_video_path(video_filename, args.hcstvg_v2_base_dir, 
                                                      dataset_config.get('video_dir', 'video'))
                    if video_path:
                        caption = ann.get(caption_field, '')
                        if caption:
                            vid = os.path.splitext(video_filename)[0]
                            result = process_stage1(video_path, caption, args.output_dir, config, 
                                                  vid, use_mistral=True, sentence_index=1)
                            found = True
                    break
            
            if not found:
                print(f"✗ Video ID {args.video_id} not found in HCSTVG-v2 dataset or video file is corrupted")
        elif args.dataset == 'savg':
            annotation_file = os.path.join(args.savg_base_dir, 'test_ann.json')
            if not os.path.exists(annotation_file):
                print(f"✗ Annotation file not found: {annotation_file}")
                return
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            found = False
            sentence_count = 0
            for ann in annotations:
                if ann.get('vid') == args.video_id:
                    # Get fps from annotation
                    fps = ann.get('fps', 30.0)
                    video_path = get_savg_video_path_cached(args.video_id, args.savg_base_dir, fps)
                    if video_path:
                        # Process all captions (SAVG has captions as list of strings)
                        if ann.get('captions'):
                            for idx, caption in enumerate(ann['captions'], 1):
                                sentence_count += 1
                                result = process_stage1(video_path, caption, args.output_dir, config, 
                                                      ann['vid'], use_mistral=True, sentence_index=sentence_count)
                            found = True
                    break
            
            if not found:
                print(f"✗ Video ID {args.video_id} not found in SAVG dataset")
        else:
            # vidstg dataset
            annotation_file = 'data/vidstg/sent_annos/test_annotations.json'
            video_dir = 'data/vidstg/videos'
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            found = False
            sentence_count = 0
            for ann in annotations:
                if ann['vid'] == args.video_id:
                    video_path = os.path.join(video_dir, f"{ann['vid']}.mp4")
                    if os.path.exists(video_path):
                        # Process all captions
                        if ann.get('captions'):
                            for idx, caption in enumerate(ann['captions'], 1):
                                sentence_count += 1
                                result = process_stage1(
                                    video_path, caption['description'], args.output_dir, config,
                                    ann['vid'], use_mistral=True, sentence_index=sentence_count,
                                    ann_gt_info={
                                        'temporal_gt': ann.get('temporal_gt'),
                                        'target_id': caption.get('target_id'),
                                        'used_relation': ann.get('used_relation'),
                                        'used_segment': ann.get('used_segment'),
                                    },
                                )
                            found = True
                        
                        # Process all questions
                        if ann.get('questions'):
                            for idx, question in enumerate(ann['questions'], 1):
                                sentence_count += 1
                                result = process_stage1(
                                    video_path, question['description'], args.output_dir, config,
                                    ann['vid'], use_mistral=False, sentence_index=sentence_count,
                                    ann_gt_info={
                                        'temporal_gt': ann.get('temporal_gt'),
                                        'target_id': question.get('target_id'),
                                        'used_relation': ann.get('used_relation'),
                                        'used_segment': ann.get('used_segment'),
                                    },
                                )
                            found = True
                        break
            
            if not found:
                print(f"✗ Video ID {args.video_id} not found")
    
    else:
        # Batch mode (default)
        if args.dataset == 'hc':
            annotation_file = os.path.join(args.hc_base_dir, 'test.json')
            if not os.path.exists(annotation_file):
                print(f"✗ Annotation file not found: {annotation_file}")
                return
            
            print(f"Loading HC test annotations...")
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            print(f"✓ Loaded {len(annotations)} annotations")
            print(f"Collecting samples...")
            
            # Collect samples from HC dataset
            samples = []
            skipped_corrupted = 0
            skipped_completed = 0
            
            for idx, (video_filename, ann) in enumerate(annotations.items()):
                caption = ann.get('caption', '')
                if not caption:
                    continue
                
                # Get video path and check if valid
                video_path = get_hc_video_path(video_filename, args.hc_base_dir)
                if not video_path:
                    skipped_corrupted += 1
                    if skipped_corrupted % 10 == 0:
                        print(f"  Skipped {skipped_corrupted} corrupted/missing videos...")
                    continue
                
                # Extract video ID (filename without extension)
                vid = os.path.splitext(video_filename)[0]
                
                # Check if already completed (sentence_index is 1 for HC dataset)
                if check_stage1_completed(args.output_dir, vid, sentence_index=1) is not None:
                    skipped_completed += 1
                    if skipped_completed % 100 == 0:
                        print(f"  Skipped {skipped_completed} already completed samples...")
                    continue
                
                samples.append({
                    'vid': vid,
                    'video_path': video_path,
                    'text_query': caption,
                    'is_interrogative': False,
                    'video_fps': 30.0  # Default fps for HC dataset
                })
                
                if args.num and len(samples) >= args.num:
                    break
                
                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(annotations)} videos, collected {len(samples)} samples, skipped {skipped_corrupted} corrupted, {skipped_completed} completed...")
            
            if skipped_corrupted > 0:
                print(f"  ⚠️  Skipped {skipped_corrupted} corrupted or missing video files")
            if skipped_completed > 0:
                print(f"  ✓ Skipped {skipped_completed} already completed samples")
        elif args.dataset == 'hcstvg-v1':
            # Get config for hcstvg-v1
            dataset_config = config.get('datasets', {}).get('hcstvg_v1', {})
            annotation_file = os.path.join(args.hcstvg_v1_base_dir, dataset_config.get('annotation_file', 'anno/test.json'))
            if not os.path.exists(annotation_file):
                print(f"✗ Annotation file not found: {annotation_file}")
                return
            
            print(f"Loading HCSTVG-v1 test annotations...")
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            print(f"✓ Loaded {len(annotations)} annotations")
            print(f"Collecting samples...")
            
            caption_field = dataset_config.get('caption_field', 'caption')
            video_dir_name = dataset_config.get('video_dir', 'video')
            samples = []
            skipped_corrupted = 0
            skipped_completed = 0
            
            for idx, (video_filename, ann) in enumerate(annotations.items()):
                caption = ann.get(caption_field, '')
                if not caption:
                    continue
                
                # Get video path and check if valid
                video_path = get_hcstvg_video_path(video_filename, args.hcstvg_v1_base_dir, video_dir_name)
                if not video_path:
                    skipped_corrupted += 1
                    if skipped_corrupted % 10 == 0:
                        print(f"  Skipped {skipped_corrupted} corrupted/missing videos...")
                    continue
                
                # Extract video ID (filename without extension)
                vid = os.path.splitext(video_filename)[0]
                
                # Check if already completed (sentence_index is 1 for hcstvg-v1)
                if check_stage1_completed(args.output_dir, vid, sentence_index=1) is not None:
                    skipped_completed += 1
                    if skipped_completed % 100 == 0:
                        print(f"  Skipped {skipped_completed} already completed samples...")
                    continue
                
                samples.append({
                    'vid': vid,
                    'video_path': video_path,
                    'text_query': caption,
                    'is_interrogative': False,
                    'video_fps': 30.0  # Default fps
                })
                
                if args.num and len(samples) >= args.num:
                    break
                
                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(annotations)} videos, collected {len(samples)} samples, skipped {skipped_corrupted} corrupted, {skipped_completed} completed...")
            
            if skipped_corrupted > 0:
                print(f"  ⚠️  Skipped {skipped_corrupted} corrupted or missing video files")
            if skipped_completed > 0:
                print(f"  ✓ Skipped {skipped_completed} already completed samples")
        elif args.dataset == 'hcstvg-v2':
            # Get config for hcstvg-v2
            dataset_config = config.get('datasets', {}).get('hcstvg_v2', {})
            annotation_file = os.path.join(args.hcstvg_v2_base_dir, dataset_config.get('annotation_file', 'anno/val_v2.json'))
            if not os.path.exists(annotation_file):
                print(f"✗ Annotation file not found: {annotation_file}")
                return
            
            print(f"Loading HCSTVG-v2 val annotations...")
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            print(f"✓ Loaded {len(annotations)} annotations")
            print(f"Collecting samples...")
            
            caption_field = dataset_config.get('caption_field', 'English')
            video_dir_name = dataset_config.get('video_dir', 'video')
            samples = []
            skipped_corrupted = 0
            skipped_completed = 0
            
            for idx, (video_filename, ann) in enumerate(annotations.items()):
                caption = ann.get(caption_field, '')
                if not caption:
                    continue
                
                # Get video path and check if valid
                video_path = get_hcstvg_video_path(video_filename, args.hcstvg_v2_base_dir, video_dir_name)
                if not video_path:
                    skipped_corrupted += 1
                    if skipped_corrupted % 10 == 0:
                        print(f"  Skipped {skipped_corrupted} corrupted/missing videos...")
                    continue
                
                # Extract video ID (filename without extension)
                vid = os.path.splitext(video_filename)[0]
                
                # Check if already completed (sentence_index is 1 for hcstvg-v2)
                if check_stage1_completed(args.output_dir, vid, sentence_index=1) is not None:
                    skipped_completed += 1
                    if skipped_completed % 100 == 0:
                        print(f"  Skipped {skipped_completed} already completed samples...")
                    continue
                
                samples.append({
                    'vid': vid,
                    'video_path': video_path,
                    'text_query': caption,
                    'is_interrogative': False,
                    'video_fps': 30.0  # Default fps
                })
                
                if args.num and len(samples) >= args.num:
                    break
                
                # Progress indicator
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{len(annotations)} videos, collected {len(samples)} samples, skipped {skipped_corrupted} corrupted, {skipped_completed} completed...")
            
            if skipped_corrupted > 0:
                print(f"  ⚠️  Skipped {skipped_corrupted} corrupted or missing video files")
            if skipped_completed > 0:
                print(f"  ✓ Skipped {skipped_completed} already completed samples")
        elif args.dataset == 'savg':
            annotation_file = os.path.join(args.savg_base_dir, 'test_ann.json')
            if not os.path.exists(annotation_file):
                print(f"✗ Annotation file not found: {annotation_file}")
                return
            
            print(f"Loading SAVG test annotations...")
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            # Create fps cache dictionary to avoid reloading JSON for each video
            print(f"✓ Loaded {len(annotations)} annotations")
            print(f"Building fps cache...")
            fps_cache = {}
            for ann in annotations:
                if ann.get('vid'):
                    fps_cache[ann['vid']] = ann.get('fps', 30.0)
            
            print(f"Collecting samples...")
            # Collect sentences (SAVG only has declarative captions)
            samples = []
            video_path_cache = {}  # Cache video paths to avoid duplicate lookups
            skipped_completed = 0
            
            for idx, ann in enumerate(annotations):
                if ann.get('captions'):
                    vid = ann.get('vid')
                    if not vid:
                        continue
                    
                    # Get video path once per video and cache it
                    if vid not in video_path_cache:
                        fps = fps_cache.get(vid, 30.0)
                        video_path = get_savg_video_path_cached(vid, args.savg_base_dir, fps)
                        video_path_cache[vid] = (video_path, fps)  # Cache both path and fps
                    else:
                        video_path, fps = video_path_cache[vid]
                    
                    if video_path:
                        # Track sentence index for this video
                        sentence_idx = 0
                        for caption in ann['captions']:
                            sentence_idx += 1
                            
                            # Check if already completed
                            if check_stage1_completed(args.output_dir, vid, sentence_index=sentence_idx) is not None:
                                skipped_completed += 1
                                if skipped_completed % 100 == 0:
                                    print(f"  Skipped {skipped_completed} already completed samples...")
                                continue
                            
                            samples.append({
                                'vid': vid, 
                                'video_path': video_path, 
                                'text_query': caption,
                                'is_interrogative': False,
                                'video_fps': fps  # Store fps for later use
                            })
                            if args.num and len(samples) >= args.num:
                                break
                    
                    # Progress indicator (more frequent for better feedback)
                    if (idx + 1) % 50 == 0:
                        print(f"  Processed {idx + 1}/{len(annotations)} videos, collected {len(samples)} samples, cached {len(video_path_cache)} video paths, skipped {skipped_completed} completed...")
                
                if args.num and len(samples) >= args.num:
                    break
            
            if skipped_completed > 0:
                print(f"  ✓ Skipped {skipped_completed} already completed samples")
        else:
            # vidstg dataset
            annotation_file = 'data/vidstg/sent_annos/test_annotations.json'
            video_dir = 'data/vidstg/videos'
            
            print(f"Loading test annotations...")
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            # Collect sentences based on sentence-type argument
            samples = []
            available_types = set()  # Track available types for error message
            skipped_completed = 0
            # Track sentence index per video for checking completion
            video_sentence_idx = {}
            
            for ann in annotations:
                vid = ann['vid']
                if vid not in video_sentence_idx:
                    video_sentence_idx[vid] = 0
                
                # Process declarative sentences (captions)
                if args.sentence_type in ['declarative', 'both'] and ann.get('captions'):
                    for caption in ann['captions']:
                        caption_type = caption.get('type')
                        available_types.add(caption_type if caption_type is not None else 'None')
                        
                        # Filter by caption type if specified, otherwise process all
                        if args.caption_type is None or caption_type == args.caption_type:
                            video_path = os.path.join(video_dir, f"{ann['vid']}.mp4")
                            if os.path.exists(video_path):
                                video_sentence_idx[vid] += 1
                                sentence_idx = video_sentence_idx[vid]
                                
                                # Check if already completed
                                if check_stage1_completed(args.output_dir, vid, sentence_index=sentence_idx) is not None:
                                    skipped_completed += 1
                                    if skipped_completed % 100 == 0:
                                        print(f"  Skipped {skipped_completed} already completed samples...")
                                    continue
                                
                                samples.append({
                                    'vid': ann['vid'], 
                                    'video_path': video_path, 
                                    'text_query': caption['description'],
                                    'is_interrogative': False,
                                    'ann_temporal_gt': ann.get('temporal_gt'),
                                    'ann_target_id': caption.get('target_id'),
                                    'ann_used_relation': ann.get('used_relation'),
                                    'ann_used_segment': ann.get('used_segment'),
                                })
                                if args.num and len(samples) >= args.num:
                                    break
                
                # Process interrogative sentences (questions)
                if args.sentence_type in ['interrogative', 'both'] and ann.get('questions'):
                    for question in ann['questions']:
                        question_type = question.get('type')
                        available_types.add(question_type if question_type is not None else 'None')
                        
                        # Filter by type if specified (for questions)
                        if args.caption_type is None or question_type == args.caption_type:
                            video_path = os.path.join(video_dir, f"{ann['vid']}.mp4")
                            if os.path.exists(video_path):
                                video_sentence_idx[vid] += 1
                                sentence_idx = video_sentence_idx[vid]
                                
                                # Check if already completed
                                if check_stage1_completed(args.output_dir, vid, sentence_index=sentence_idx) is not None:
                                    skipped_completed += 1
                                    if skipped_completed % 100 == 0:
                                        print(f"  Skipped {skipped_completed} already completed samples...")
                                    continue
                                
                                samples.append({
                                    'vid': ann['vid'], 
                                    'video_path': video_path, 
                                    'text_query': question['description'],
                                    'is_interrogative': True,
                                    'ann_temporal_gt': ann.get('temporal_gt'),
                                    'ann_target_id': question.get('target_id'),
                                    'ann_used_relation': ann.get('used_relation'),
                                    'ann_used_segment': ann.get('used_segment'),
                                })
                                if args.num and len(samples) >= args.num:
                                    break
                
                if args.num and len(samples) >= args.num:
                    break
            
            if skipped_completed > 0:
                print(f"  ✓ Skipped {skipped_completed} already completed samples")
        
        # Assign sentence indices for each video
        # Group samples by video ID and assign indices
        video_sentence_counts = {}
        for sample in samples:
            vid = sample['vid']
            if vid not in video_sentence_counts:
                video_sentence_counts[vid] = 0
            video_sentence_counts[vid] += 1
            sample['sentence_index'] = video_sentence_counts[vid]
        
        # Build filter description string
        if args.dataset == 'savg':
            print(f"✓ Processing {len(samples)} sentences from SAVG dataset")
        elif args.dataset == 'hcstvg-v1':
            print(f"✓ Processing {len(samples)} sentences from HCSTVG-v1 (test) dataset")
        elif args.dataset == 'hcstvg-v2':
            print(f"✓ Processing {len(samples)} sentences from HCSTVG-v2 (val) dataset")
        else:
            filter_parts = []
            if args.sentence_type == 'declarative':
                filter_parts.append('declarative')
            elif args.sentence_type == 'interrogative':
                filter_parts.append('interrogative')
            else:
                filter_parts.append('both types')
            
            if args.caption_type:
                filter_parts.append(f"type='{args.caption_type}'")
            
            filter_str = f" ({', '.join(filter_parts)})" if filter_parts else ""
            print(f"✓ Processing {len(samples)} sentences from {len(video_sentence_counts)} videos{filter_str}")
        
        if len(samples) == 0:
            print(f"\n{'='*80}")
            print("No videos found matching the criteria!")
            if args.dataset == 'vidstg' and args.caption_type:
                print(f"  - Specified caption type: '{args.caption_type}'")
                # Format available types
                type_list = []
                for t in sorted(available_types):
                    if t == 'None':
                        type_list.append('None (no type field)')
                    else:
                        type_list.append(f"'{t}'")
                print(f"  - Available types in dataset: {', '.join(type_list)}")
                print(f"  - Tip: Try running without --caption-type to process all captions")
            print(f"{'='*80}")
            return
        
        # Use multi-GPU processing if available
        process_samples_multi_gpu(samples, args.output_dir, config, num_gpus=args.num_gpus)


def worker_process(gpu_id: int, samples: List[Dict], output_dir: str, config: Dict, result_queue: Queue):
    """
    Worker process that processes samples on a specific GPU.
    
    Args:
        gpu_id: GPU ID to use (0, 1, 2, ...)
        samples: List of samples to process
        output_dir: Output directory
        config: Configuration dictionary
        result_queue: Queue to put results
    """
    # Set CUDA_VISIBLE_DEVICES to only see this GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Re-import torch after setting CUDA_VISIBLE_DEVICES
    import torch
    torch.cuda.set_device(0)  # Now GPU 0 is the actual GPU we want
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    print(f"[GPU {gpu_id}] Worker started, processing {len(samples)} samples on device {device}")
    
    success_count = 0
    for idx, sample in enumerate(samples, 1):
        try:
            sentence_type_str = "interrogative" if sample['is_interrogative'] else "declarative"
            sentence_index = sample.get('sentence_index', None)
            
            print(f"[GPU {gpu_id}] Processing {idx}/{len(samples)}: {sample['vid']} ({sentence_type_str}, sentence #{sentence_index if sentence_index else 'N/A'})")
            
            # Use mistral only for declarative sentences
            use_mistral = not sample['is_interrogative']
            
            ann_gt = None
            if sample.get('ann_temporal_gt') or sample.get('ann_target_id') is not None:
                ann_gt = {
                    'temporal_gt': sample.get('ann_temporal_gt'),
                    'target_id': sample.get('ann_target_id'),
                    'used_relation': sample.get('ann_used_relation'),
                    'used_segment': sample.get('ann_used_segment'),
                }
            result = process_stage1(
                sample['video_path'], 
                sample['text_query'], 
                output_dir, 
                config, 
                sample['vid'], 
                use_mistral=use_mistral, 
                sentence_index=sentence_index,
                device=device,
                video_fps=sample.get('video_fps', 30.0),
                ann_gt_info=ann_gt,
            )
            
            if result:
                success_count += 1
                result_queue.put({
                    'vid': sample['vid'],
                    'sentence_index': sentence_index,
                    'success': True,
                    'gpu_id': gpu_id
                })
            else:
                result_queue.put({
                    'vid': sample['vid'],
                    'sentence_index': sentence_index,
                    'success': False,
                    'gpu_id': gpu_id
                })
                
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {sample['vid']}: {e}")
            import traceback
            traceback.print_exc()
            result_queue.put({
                'vid': sample['vid'],
                'sentence_index': sample.get('sentence_index'),
                'success': False,
                'gpu_id': gpu_id,
                'error': str(e)
            })
    
    print(f"[GPU {gpu_id}] Worker finished: {success_count}/{len(samples)} succeeded")
    result_queue.put({'worker_done': True, 'gpu_id': gpu_id})


def process_samples_multi_gpu(samples: List[Dict], output_dir: str, config: Dict, num_gpus: Optional[int] = None):
    """
    Process samples using multiple GPUs in parallel.
    
    Args:
        samples: List of samples to process
        output_dir: Output directory
        config: Configuration dictionary
        num_gpus: Number of GPUs to use (None = use all available)
    """
    # Detect available GPUs
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to single CPU processing")
        # Fall back to single process
        success_count = 0
        for idx, sample in enumerate(samples, 1):
            sentence_type_str = "interrogative" if sample['is_interrogative'] else "declarative"
            sentence_index = sample.get('sentence_index', None)
            print(f"Processing {idx}/{len(samples)}: {sample['vid']} ({sentence_type_str}, sentence #{sentence_index if sentence_index else 'N/A'})")
            
            use_mistral = not sample['is_interrogative']
            ann_gt = None
            if sample.get('ann_temporal_gt') or sample.get('ann_target_id') is not None:
                ann_gt = {
                    'temporal_gt': sample.get('ann_temporal_gt'),
                    'target_id': sample.get('ann_target_id'),
                    'used_relation': sample.get('ann_used_relation'),
                    'used_segment': sample.get('ann_used_segment'),
                }
            if process_stage1(sample['video_path'], sample['text_query'], output_dir, config, 
                            sample['vid'], use_mistral=use_mistral, sentence_index=sentence_index, device='cpu',
                            video_fps=sample.get('video_fps', 30.0), ann_gt_info=ann_gt):
                success_count += 1
        
        print(f"\n{'='*80}")
        print("Batch Processing Complete")
        print(f"{'='*80}")
        print(f"Total: {len(samples)}, Success: {success_count}, Failed: {len(samples)-success_count}")
        return
    
    available_gpus = torch.cuda.device_count()
    if num_gpus is None:
        num_gpus = available_gpus
    else:
        num_gpus = min(num_gpus, available_gpus)
    
    if num_gpus <= 0:
        print("⚠️  No GPUs available, falling back to CPU")
        num_gpus = 1
        device = 'cpu'
    else:
        print(f"✓ Using {num_gpus} GPU(s) out of {available_gpus} available")
    
    if num_gpus == 1:
        # Single GPU, no need for multiprocessing
        print("Processing on single GPU...")
        success_count = 0
        for idx, sample in enumerate(samples, 1):
            sentence_type_str = "interrogative" if sample['is_interrogative'] else "declarative"
            sentence_index = sample.get('sentence_index', None)
            print(f"Processing {idx}/{len(samples)}: {sample['vid']} ({sentence_type_str}, sentence #{sentence_index if sentence_index else 'N/A'})")
            
            use_mistral = not sample['is_interrogative']
            ann_gt = None
            if sample.get('ann_temporal_gt') or sample.get('ann_target_id') is not None:
                ann_gt = {
                    'temporal_gt': sample.get('ann_temporal_gt'),
                    'target_id': sample.get('ann_target_id'),
                    'used_relation': sample.get('ann_used_relation'),
                    'used_segment': sample.get('ann_used_segment'),
                }
            if process_stage1(sample['video_path'], sample['text_query'], output_dir, config, 
                            sample['vid'], use_mistral=use_mistral, sentence_index=sentence_index, device='cuda',
                            video_fps=sample.get('video_fps', 30.0), ann_gt_info=ann_gt):
                success_count += 1
        
        print(f"\n{'='*80}")
        print("Batch Processing Complete")
        print(f"{'='*80}")
        print(f"Total: {len(samples)}, Success: {success_count}, Failed: {len(samples)-success_count}")
        if len(samples) > 0:
            print(f"Success rate: {success_count/len(samples)*100:.2f}%")
        print(f"{'='*80}")
        return
    
    # Split samples across GPUs
    samples_per_gpu = len(samples) // num_gpus
    remainder = len(samples) % num_gpus
    
    gpu_samples = []
    start_idx = 0
    for gpu_id in range(num_gpus):
        # Distribute remainder samples to first few GPUs
        num_samples = samples_per_gpu + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + num_samples
        gpu_samples.append((gpu_id, samples[start_idx:end_idx]))
        start_idx = end_idx
    
    print(f"\n{'='*80}")
    print("Multi-GPU Processing Distribution:")
    for gpu_id, gpu_sample_list in gpu_samples:
        print(f"  GPU {gpu_id}: {len(gpu_sample_list)} samples")
    print(f"{'='*80}\n")
    
    # Create processes and queues
    processes = []
    result_queue = Queue()
    
    # Start worker processes
    for gpu_id, gpu_sample_list in gpu_samples:
        if len(gpu_sample_list) > 0:  # Only start process if there are samples
            p = Process(target=worker_process, args=(gpu_id, gpu_sample_list, output_dir, config, result_queue))
            p.start()
            processes.append(p)
    
    # Collect results
    results = []
    workers_done = set()
    
    print("Waiting for workers to complete...")
    while len(workers_done) < len(processes):
        try:
            result = result_queue.get(timeout=5)
            if result.get('worker_done'):
                gpu_id = result['gpu_id']
                workers_done.add(gpu_id)
                print(f"[GPU {gpu_id}] Worker completed")
            else:
                results.append(result)
        except:
            # Timeout, check if processes are still alive
            for i, p in enumerate(processes):
                if not p.is_alive() and i not in workers_done:
                    workers_done.add(i)
                    print(f"[GPU {i}] Worker process died, marking as done")
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Count successes
    success_count = sum(1 for r in results if r.get('success', False))
    
    print(f"\n{'='*80}")
    print("Multi-GPU Batch Processing Complete")
    print(f"{'='*80}")
    print(f"Total: {len(samples)}, Success: {success_count}, Failed: {len(samples)-success_count}")
    if len(samples) > 0:
        print(f"Success rate: {success_count/len(samples)*100:.2f}%")
    print(f"{'='*80}")


if __name__ == '__main__':
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    main()
