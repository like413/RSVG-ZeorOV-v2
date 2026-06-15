#!/usr/bin/env python3
"""
Stage 3 (Qwen bbox only): Video Segmentation using tubes + Qwen bbox.

- Bbox source: stage2 (Qwen detection only).
- Case 1: tubes exist AND bbox exist → tube select; if tube select fails → Case 2 (bbox propagation).
- Case 2: tubes do not exist AND bbox exist → bbox bidirectional propagation.
- Other: failure (no bbox or no tubes and no bbox).

Usage:
    python model/stage3_sam3_qwen.py --sentence-type savg --num-gpus 4 --retry-failed
    python model/stage3_sam3_qwen.py --sentence-type savg --retry-failed --no-visualization
    python model/stage3_sam3_qwen.py --video-id 2400171624_1 --sentence-type declarative
"""

import os
import sys
import json
import copy
import argparse
import re
import logging
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2
import torch
import gc
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from queue import Queue

# #region agent log
_DEBUG_LOG_PATH = "/mnt/disk2/zyu/videoVG/model/.cursor/debug-753d55.log"


def _debug_ndjson(
    message: str,
    data: dict,
    location: str = "stage3_sam3_qwen",
    hypothesis_id: str = "",
) -> None:
    try:
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "sessionId": "753d55",
                        "location": location,
                        "message": message,
                        "data": data,
                        "timestamp": int(time.time() * 1000),
                        "hypothesisId": hypothesis_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except OSError:
        pass


# #endregion

# Disable PIL/PngImagePlugin DEBUG logging
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config_manager import ConfigManager
# Delay import SAM2Module to avoid CUDA initialization before CUDA_VISIBLE_DEVICES is set in worker processes
# SAM2Module will be imported inside propagate_mask_with_sam2 function


SUPPORTED_DATASETS = (
    "vidstg_declarative",
    "hcstvg_v1",
    "hcstvg_v2",
    "savg",
)

DATASET_TO_SENTENCE_TYPE = {
    "vidstg_declarative": "declarative",
    "hcstvg_v1": "declarative",
    "hcstvg_v2": "declarative",
    "savg": "savg",
}

LEGACY_SENTENCE_TO_DATASET = {
    "declarative": "vidstg_declarative",
    "interrogative": "vidstg_declarative",
    "savg": "savg",
}


def _resolve_dataset_and_sentence_type(
    dataset: Optional[str],
    legacy_sentence_type: Optional[str],
) -> Tuple[str, str]:
    """
    Resolve dataset + sentence_type for backward compatibility.
    New behavior is dataset-first; sentence-type is only a legacy fallback.
    """
    if dataset:
        if dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Invalid dataset '{dataset}'. "
                f"Expected one of: {', '.join(SUPPORTED_DATASETS)}"
            )
        if legacy_sentence_type:
            print(
                f"⚠️  Both --dataset and --sentence-type provided. "
                f"Using --dataset={dataset}."
            )
        return dataset, DATASET_TO_SENTENCE_TYPE[dataset]

    if legacy_sentence_type:
        mapped = LEGACY_SENTENCE_TO_DATASET.get(legacy_sentence_type)
        if mapped is None:
            raise ValueError(
                f"Invalid sentence_type '{legacy_sentence_type}'. "
                "Use --dataset with one of: "
                + ", ".join(SUPPORTED_DATASETS)
            )
        print(
            f"⚠️  --sentence-type is deprecated. "
            f"Mapped '{legacy_sentence_type}' -> dataset '{mapped}'."
        )
        return mapped, DATASET_TO_SENTENCE_TYPE[mapped]

    raise ValueError(
        "Either --dataset or --sentence-type must be provided."
    )


def _default_video_search_roots() -> List[str]:
    env_roots = os.environ.get("STAGE3_VIDEO_SEARCH_ROOTS", "").strip()
    if env_roots:
        return [p.strip() for p in env_roots.split(":") if p.strip()]
    return [
        "/mnt/data/disk2/zyu/videoVG/data",
        "/mnt/disk2/zyu/videoVG/data",
    ]


_VIDEO_PATH_CACHE: Dict[str, str] = {}
_VIDEO_BASENAME_INDEX: Optional[Dict[str, str]] = None


def _build_video_basename_index(roots: List[str]) -> Dict[str, str]:
    """Build basename -> absolute path index once to avoid repeated full-tree scans."""
    index: Dict[str, str] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name not in index:
                    index[name] = os.path.join(dirpath, name)
    return index


def _maybe_replace_mnt_data_prefix(video_path: str) -> Optional[str]:
    if not video_path.startswith("/mnt/data/"):
        return None
    candidate = "/mnt/" + video_path[len("/mnt/data/") :]
    return candidate if os.path.exists(candidate) else None


def _search_under_data_roots(video_path: str) -> Optional[str]:
    global _VIDEO_BASENAME_INDEX
    roots = _default_video_search_roots()
    # First, reconstruct relative path after "/data/".
    if "/data/" in video_path:
        rel = video_path.split("/data/", 1)[1]
        for root in roots:
            candidate = os.path.join(root, rel)
            if os.path.exists(candidate):
                return candidate

    # Fallback: search by filename under the two data roots.
    basename = os.path.basename(video_path)
    if not basename:
        return None
    if _VIDEO_BASENAME_INDEX is None:
        _VIDEO_BASENAME_INDEX = _build_video_basename_index(roots)
    return _VIDEO_BASENAME_INDEX.get(basename)


def _resolve_video_path_from_stage1(video_path: str) -> str:
    """Try to recover a valid video path when metadata stores another machine's absolute path."""
    if not video_path:
        return video_path
    cached = _VIDEO_PATH_CACHE.get(video_path)
    if cached:
        return cached
    if os.path.exists(video_path):
        _VIDEO_PATH_CACHE[video_path] = video_path
        return video_path

    searched = _search_under_data_roots(video_path)
    if searched:
        print(f"  (video_path recovered by data root search: {searched})")
        _VIDEO_PATH_CACHE[video_path] = searched
        return searched

    replaced = _maybe_replace_mnt_data_prefix(video_path)
    if replaced:
        print(f"  (video_path remap /mnt/data -> /mnt: {replaced})")
        _VIDEO_PATH_CACHE[video_path] = replaced
        return replaced

    _VIDEO_PATH_CACHE[video_path] = video_path
    return video_path


def _prepare_sam3_tracker_video_input(video_path: str) -> Tuple[str, Optional[str]]:
    """
    SAM3 tracker currently supports MP4 file or JPEG folder.
    If input is a non-MP4 video (e.g., MKV), transcode to a temp MP4.
    Returns (usable_path, temp_path_to_cleanup).
    """
    if os.path.isdir(video_path):
        return video_path, None
    ext = os.path.splitext(video_path)[1].lower()
    if ext == ".mp4":
        return video_path, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return video_path, None

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return video_path, None

    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"sam3_tracker_input_{uuid.uuid4().hex[:10]}.mp4",
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        writer.release()
        return video_path, None

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    if frame_count <= 0 or not os.path.exists(temp_path):
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return video_path, None

    print(f"  (tracker input converted: {video_path} -> {temp_path}, frames={frame_count})")
    return temp_path, temp_path


def resolve_io_dirs(
    dataset: str,
    base_dir: str,
    output_dir: str,
    config: Optional[Dict] = None,
    input_root_override: Optional[str] = None,
    output_root_override: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Resolve where to read stage1/stage2 from, and where to write stage3 to.

    Priority:
    - If input_root_override / output_root_override are set, those paths are used (absolute).
    - Else if config.yaml has datasets.<key>.output_dir: use that as BOTH input_root and output_root.
      (This makes stage3 paths fully controllable from config.)
    - Else fall back to legacy layout:
        input_root = os.path.join(base_dir, prefix)
        output_root = os.path.join(output_dir, prefix)

    Returns:
      (prefix, input_root, output_root)
    """
    prefix = dataset

    cfg_out = None
    if isinstance(config, dict):
        datasets_cfg = config.get("datasets", {}) if isinstance(config.get("datasets", {}), dict) else {}
        if isinstance(datasets_cfg.get(dataset), dict):
            cfg_out = datasets_cfg.get(dataset, {}).get("output_dir")

    if cfg_out:
        input_root = cfg_out
        output_root = cfg_out
    else:
        input_root = os.path.join(base_dir, prefix)
        output_root = os.path.join(output_dir, prefix)

    if input_root_override:
        input_root = os.path.abspath(os.path.expanduser(input_root_override))
    if output_root_override:
        output_root = os.path.abspath(os.path.expanduser(output_root_override))

    return prefix, input_root, output_root


def clear_gpu_memory():
    """Clear GPU memory"""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except RuntimeError as e:
        # Ignore CUDA initialization errors in subprocesses
        # This can happen if CUDA_VISIBLE_DEVICES is set after torch import
        if "re-initialize" not in str(e).lower():
            raise


def _get_sam2_module_with_image_sequence():
    """
    延迟导入并返回 SAM2ModuleWithImageSequence 类
    这样可以避免在 worker 进程启动前初始化 CUDA
    """
    from modules.sam2_module import SAM2Module
    
    class SAM2ModuleWithImageSequence(SAM2Module):
        """
        SAM2Module 的扩展版本，支持图片序列目录
        """
        def _load_video_frames(self, video_path: str):
            """
            加载视频帧，支持视频文件和图片序列目录
            
            Args:
                video_path: 视频文件路径或图片序列目录路径
                
            Returns:
                List of PIL Image objects
            """
            import cv2
            from PIL import Image
            
            # 检查是否是目录
            if os.path.isdir(video_path):
                # 图片序列目录
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
                frame_files = []
                for ext in image_extensions:
                    frame_files.extend(Path(video_path).glob(f'*{ext}'))
                
                if not frame_files:
                    raise RuntimeError(f"No image frames found in {video_path}")
                
                # 按文件名排序（假设文件名是数字）
                def sort_key(x):
                    try:
                        # 尝试提取文件名中的数字
                        name = x.stem
                        # 如果文件名是纯数字，直接转换
                        if name.isdigit():
                            return int(name)
                        # 否则尝试提取数字部分
                        numbers = re.findall(r'\d+', name)
                        if numbers:
                            return int(numbers[0])
                        return float('inf')
                    except:
                        return float('inf')
                
                frame_files = sorted(frame_files, key=sort_key)
                
                # 加载所有图片
                frames = []
                for img_path in frame_files:
                    img = Image.open(img_path)
                    # 确保是RGB模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    frames.append(img)
                
                if len(frames) == 0:
                    raise RuntimeError("No frames extracted from image sequence")
                
                return frames
            else:
                # 视频文件，使用父类方法
                return super()._load_video_frames(video_path)
    
    return SAM2ModuleWithImageSequence


def propagate_mask_with_sam2(
    video_path: str,
    key_frame_idx: int,
    initial_mask: np.ndarray,
    config: Dict,
    bbox: Optional[Dict] = None,
    num_gpus: int = 1,
    stage1_meta: Optional[Dict] = None,
    qwen_key_frame_path: Optional[str] = None,
) -> Optional[Dict[int, np.ndarray]]:
    """
    使用 SAM2 传播 mask 通过视频（双向传播）
    
    这个方法从 stage3_old_version.py 中提取，使用 SAM2 而不是 SAM3 进行 mask 传播。
    
    Args:
        video_path: 视频文件路径或图片序列目录路径
        key_frame_idx: 关键帧索引（原始视频帧索引）
        initial_mask: 初始 mask (H, W)，可以是全零（如果提供 bbox 会从 bbox 创建）
        config: 配置字典
        bbox: 可选的边界框字典，包含 'xmin', 'ymin', 'xmax', 'ymax'
              如果提供，将从 bbox 创建初始 mask
        num_gpus: GPU 数量（未使用，保留以兼容接口）
        
    Returns:
        字典映射 frame_idx -> mask array (H, W)，值为 0 或 1
        如果失败返回 None
    """
    # qwen_key_frame_path: unused; kept for API parity with propagate_mask_with_sam3_tracker.
    sam2_module = None
    
    try:
        # 清除 GPU 内存
        clear_gpu_memory()
        
        # 获取视频信息用于判断是否需要 chunking
        if os.path.isdir(video_path):
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
            frame_files = [f for f in os.listdir(video_path) 
                         if any(f.lower().endswith(ext) for ext in image_extensions)]
            num_frames = len(sorted(frame_files))
            if frame_files:
                first_img_path = os.path.join(video_path, sorted(frame_files)[0])
                first_img = Image.open(first_img_path)
                frame_width, frame_height = first_img.size
            else:
                print(f"  ✗ 无法从图片序列目录获取尺寸")
                return None
        else:
            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        
        # 估算内存需求：每帧约 width * height * 4 bytes (考虑 PIL 对象开销)
        estimated_memory_mb = (num_frames * frame_width * frame_height * 4) / (1024 * 1024)
        print(f"  [INFO] Video has {num_frames} frames, estimated memory: {estimated_memory_mb:.1f} MB")
        
        # 获取配置中的阈值
        sam2_config = config.get('sam2', {})
        max_frames_for_sam2 = sam2_config.get('max_frames', config.get('sam2_max_frames', 1500))
        max_memory_mb = sam2_config.get('max_memory_mb', 6 * 1024)  # 默认 6GB 阈值
        
        print(f"  [INFO] Chunking thresholds: max_frames={max_frames_for_sam2}, max_memory={max_memory_mb} MB")
        
        # 判断是否需要 chunking
        use_chunking = False
        reason = []
        if num_frames > max_frames_for_sam2:
            reason.append(f"frames ({num_frames} > {max_frames_for_sam2})")
            use_chunking = True
        if estimated_memory_mb > max_memory_mb:
            reason.append(f"memory ({estimated_memory_mb:.1f} MB > {max_memory_mb} MB)")
            use_chunking = True
        
        # 如果提供了 bbox，从 bbox 创建初始 mask（在 chunking 检测之前，因为 chunking 也需要）
        if bbox is not None:
            # 从 bbox 创建 mask（使用已经获取的 frame_width 和 frame_height）
            initial_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            xmin = max(0, bbox['xmin'])
            ymin = max(0, bbox['ymin'])
            xmax = min(frame_width, bbox['xmax'])
            ymax = min(frame_height, bbox['ymax'])
            initial_mask[ymin:ymax, xmin:xmax] = 1
            
            mask_coverage = (np.sum(initial_mask) / initial_mask.size) * 100
            print(f"  ✓ 从 bbox 创建初始 mask，覆盖率: {mask_coverage:.2f}%")
        
        if use_chunking:
            print(f"  [INFO] Video exceeds limits: {', '.join(reason)}, using chunking strategy")
            # 使用 chunking 策略
            return _propagate_mask_with_sam2_chunked(
                video_path=video_path,
                key_frame_idx=key_frame_idx,
                initial_mask=initial_mask,
                config=config,
                bbox=bbox,
                num_frames=num_frames,
                frame_width=frame_width,
                frame_height=frame_height,
                num_gpus=num_gpus
            )
        else:
            print(f"  [INFO] Video within limits, processing without chunking")
        
        # 初始化 SAM2
        # 延迟导入 SAM2Module 和类定义，确保在 worker 进程中 CUDA_VISIBLE_DEVICES 已经设置后再导入
        SAM2ModuleWithImageSequence = _get_sam2_module_with_image_sequence()
        
        # 在 worker 进程中，CUDA_VISIBLE_DEVICES 已经设置，PyTorch 会将可见的 GPU 映射为 cuda:0
        # 所以应该使用 'cuda:0' 而不是 'cuda'
        sam2_config = config.get('sam2', {})
        # 检查是否在 worker 进程中（通过 CUDA_VISIBLE_DEVICES 判断）
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if visible_devices:
            # Worker 进程：使用 cuda:0（因为 CUDA_VISIBLE_DEVICES 已经将 GPU 映射为 0）
            device = 'cuda:0'
            # 添加调试信息：显示实际使用的GPU
            import torch
            if torch.cuda.is_available():
                actual_gpu_id = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(actual_gpu_id)
                print(f"  [DEBUG] Worker process: CUDA_VISIBLE_DEVICES={visible_devices}, using device={device}, actual GPU={actual_gpu_id} ({gpu_name})")
        else:
            # 主进程：使用配置中的设备或默认 cuda
            device = config.get('device', 'cuda')
            print(f"  [DEBUG] Main process: using device={device}")
        
        sam2_module = SAM2ModuleWithImageSequence(
            model_path=sam2_config.get('model_path', '/home/xdu/.cache/modelscope/hub/models/facebook/sam2.1-hiera-base-plus'),
            device=device
        )
        
        # 初始化视频
        num_frames = sam2_module.initialize_video(video_path)
        print(f"  ✓ 视频初始化: {num_frames} 帧")
        
        # 添加 prompt
        sam2_module.add_prompt(frame_idx=key_frame_idx, mask=initial_mask)
        
        # 传播 mask（双向传播）
        print(f"  ✓ 开始双向传播 mask...")
        masks_dict = sam2_module.propagate_masks()
        print(f"  ✓ 生成 {len(masks_dict)} 个 mask")
        
        # 清理
        sam2_module.cleanup()
        del sam2_module
        clear_gpu_memory()
        
        return masks_dict
        
    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ SAM2 传播失败: {error_msg}")
        
        # 清理资源
        if sam2_module is not None:
            try:
                sam2_module.cleanup()
            except:
                pass
            del sam2_module
        
        clear_gpu_memory()
        
        # 检查是否是 OOM 错误
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower() or "killed" in error_msg.lower():
            print(f"  ⊘ OOM 错误，跳过此任务...")
            return None
        else:
            # 其他错误，返回 None 但不打印 OOM 信息
            return None


def clear_old_frames_safely(
    predictor,
    session_id: str,
    frames_to_clear: list,
    keep_last_n_frames: int = 15
) -> None:
    """
    Safely clear old frame data from inference state while preserving memory bank.
    
    This function removes old frame outputs from GPU memory while keeping:
    - tracker_metadata (for obj_id continuity)
    - Last keep_last_n_frames frames in output_dict (for memory bank connection)
    - tracker_states_local (for object state continuity)
    
    Args:
        predictor: SAM3 video predictor instance
        session_id: Session ID
        frames_to_clear: List of frame indices to clear
        keep_last_n_frames: Number of recent frames to keep for memory bank (default: 15)
    """
    if not frames_to_clear or len(frames_to_clear) <= keep_last_n_frames:
        return  # No need to clear
    
    try:
        session = predictor._get_session(session_id)
        inference_state = session["state"]
        
        # Sort frames to determine which ones to keep
        sorted_frames = sorted(frames_to_clear)
        frames_to_keep = sorted_frames[-keep_last_n_frames:]
        frames_to_actually_clear = sorted_frames[:-keep_last_n_frames]
        
        print(f"  Clearing {len(frames_to_actually_clear)} old frames (keeping last {len(frames_to_keep)} frames for memory bank)...")
        
        # Clear cached_frame_outputs (main storage in SAM3VideoInference)
        # This is where SAM3VideoInference stores frame outputs
        if "cached_frame_outputs" in inference_state:
            for frame_idx in frames_to_actually_clear:
                inference_state["cached_frame_outputs"].pop(frame_idx, None)
        
        # Clear output_dict if it exists (for tracker-based inference)
        # Note: In SAM3VideoInference, output_dict may not exist directly in inference_state
        # It might be in tracker_inference_states
        if "output_dict" in inference_state:
            output_dict = inference_state["output_dict"]
            for frame_idx in frames_to_actually_clear:
                if isinstance(output_dict, dict):
                    if "non_cond_frame_outputs" in output_dict:
                        output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
                    if "cond_frame_outputs" in output_dict:
                        output_dict["cond_frame_outputs"].pop(frame_idx, None)
            
            # Clear per-object outputs
            if "output_dict_per_obj" in inference_state:
                for obj_output_dict in inference_state["output_dict_per_obj"]:
                    if isinstance(obj_output_dict, dict):
                        for frame_idx in frames_to_actually_clear:
                            if "non_cond_frame_outputs" in obj_output_dict:
                                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
                            if "cond_frame_outputs" in obj_output_dict:
                                obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
            
            # Clear temp outputs
            if "temp_output_dict_per_obj" in inference_state:
                for obj_temp_output_dict in inference_state["temp_output_dict_per_obj"]:
                    if isinstance(obj_temp_output_dict, dict):
                        for frame_idx in frames_to_actually_clear:
                            if "non_cond_frame_outputs" in obj_temp_output_dict:
                                obj_temp_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
                            if "cond_frame_outputs" in obj_temp_output_dict:
                                obj_temp_output_dict["cond_frame_outputs"].pop(frame_idx, None)
        
        # Also check tracker_inference_states (for tracker-based inference)
        # Tracker states may contain output_dict
        if "tracker_inference_states" in inference_state:
            for tracker_state in inference_state["tracker_inference_states"]:
                if isinstance(tracker_state, dict):
                    if "output_dict" in tracker_state:
                        output_dict = tracker_state["output_dict"]
                        for frame_idx in frames_to_actually_clear:
                            if isinstance(output_dict, dict):
                                if "non_cond_frame_outputs" in output_dict:
                                    output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
                                if "cond_frame_outputs" in output_dict:
                                    output_dict["cond_frame_outputs"].pop(frame_idx, None)
                    
                    # Clear per-object outputs in tracker state
                    if "output_dict_per_obj" in tracker_state:
                        for obj_output_dict in tracker_state["output_dict_per_obj"]:
                            if isinstance(obj_output_dict, dict):
                                for frame_idx in frames_to_actually_clear:
                                    if "non_cond_frame_outputs" in obj_output_dict:
                                        obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
                                    if "cond_frame_outputs" in obj_output_dict:
                                        obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
        
        # Clear feature_cache for old frames (optional, can be more aggressive)
        if "feature_cache" in inference_state:
            # Keep recent features for memory bank
            feature_cache = inference_state["feature_cache"]
            if "frame_features" in feature_cache:
                for frame_idx in frames_to_actually_clear:
                    feature_cache["frame_features"].pop(frame_idx, None)
        
        # Clear cached_frame_outputs for old frames
        if "cached_frame_outputs" in inference_state:
            for frame_idx in frames_to_actually_clear:
                inference_state["cached_frame_outputs"].pop(frame_idx, None)
        
        # Important: DO NOT clear tracker_metadata - it contains obj_id information
        # Important: DO NOT clear tracker_states_local - it contains object states
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        print(f"  ✓ Cleared {len(frames_to_actually_clear)} frames, kept {len(frames_to_keep)} frames for memory bank")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Error clearing old frames: {e}")
        import traceback
        traceback.print_exc()


def _sorted_image_filenames_in_dir(video_path: str) -> list:
    """Image filenames under a directory, same order as SAM3 / keyframe loading."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    frame_files = []
    for f in os.listdir(video_path):
        if any(f.lower().endswith(ext.lower()) for ext in image_extensions):
            frame_files.append(f)

    def sort_key(x):
        try:
            name = os.path.splitext(x)[0]
            if name.isdigit():
                return int(name)
            numbers = re.findall(r'\d+', name)
            if numbers:
                return int(numbers[0])
            return float('inf')
        except Exception:
            return float('inf')

    return sorted(frame_files, key=sort_key)


def resolve_sam3_keyframe_read_index(
    key_frame_idx: int, stage1_meta: Optional[Dict] = None
) -> int:
    """
    For HCSTVG, stage1's ``key_frame_idx`` is the index into the linspace-sampled
    keyframe list (e.g. 0..59 for 60 samples), *not* the opencv / ``CAP_PROP_POS_FRAMES``
    index. The actual video frame index is ``original_frame_idx``. Qwen's bbox and
    ``key_frame.png`` are aligned to that frame; SAM3 image must read the same frame.
    When ``original_frame_idx`` is absent, ``key_frame_idx`` is used as a raw index
    (SAVG/VidSTG or legacy behavior).
    """
    if isinstance(stage1_meta, dict) and stage1_meta.get("original_frame_idx", None) is not None:
        return int(stage1_meta["original_frame_idx"])
    return int(key_frame_idx)


def _load_keyframe_pil(video_path: str, key_frame_idx: int) -> Optional[Image.Image]:
    """Load one frame as RGB PIL Image from a video file or sorted image directory."""
    if os.path.isdir(video_path):
        frame_files = _sorted_image_filenames_in_dir(video_path)
        if not frame_files:
            print(f"  ✗ No image frames in directory: {video_path}")
            return None
        ki = int(key_frame_idx)
        if ki < 0 or ki >= len(frame_files):
            print(f"  ✗ key_frame_idx {ki} out of range [0, {len(frame_files) - 1}]")
            return None
        img_path = os.path.join(video_path, frame_files[ki])
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ Failed to open video for key frame extraction: {video_path}")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(key_frame_idx))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"  ✗ Failed to read key frame {key_frame_idx} from video")
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def _mask_box_iou_sam3(m: np.ndarray, box_pix: list) -> float:
    """Mask vs axis-aligned box IoU (same convention as stage3_validate.compute_mask_rect_iou)."""
    m = np.squeeze(m)
    if m.ndim != 2:
        return 0.0
    if m.max() > 1:
        m = (m > 0).astype(np.uint8)
    m = m > 0
    H, W = m.shape
    x1, y1, x2, y2 = int(box_pix[0]), int(box_pix[1]), int(box_pix[2]), int(box_pix[3])
    x1 = int(max(0, min(x1, W - 1)))
    x2 = int(max(0, min(x2, W)))
    y1 = int(max(0, min(y1, H - 1)))
    y2 = int(max(0, min(y2, H)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    a = float(m.sum())
    br = float((x2 - x1) * (y2 - y1))
    inter = float(m[y1:y2, x1:x2].sum())
    u = a + br - inter
    return (inter / u) if u > 0 else 0.0


def _select_sam3_mask_by_box_iou(
    masks: np.ndarray, box_list: list
) -> Tuple[np.ndarray, int, int, list]:
    """
    When SAM3 returns N candidate masks, pick the one with max mask–box IoU to Qwen bbox.
    Returns (2D mask, n_candidates, best_i, iou_list_per_candidate).
    """
    def flat(mi: np.ndarray) -> np.ndarray:
        if mi.ndim == 3 and mi.shape[0] == 1:
            return mi[0]
        return np.squeeze(mi)

    iou_list: list = []
    if masks.ndim == 4:
        n = masks.shape[0]
        best_i, best_s = 0, -1.0
        for i in range(n):
            s = _mask_box_iou_sam3(flat(masks[i]), box_list)
            iou_list.append(s)
            if s > best_s:
                best_s, best_i = s, i
        m2 = flat(masks[best_i])
        return m2, n, best_i, iou_list

    if masks.ndim == 3 and masks.shape[0] > 1:
        n = masks.shape[0]
        best_i, best_s = 0, -1.0
        for i in range(n):
            s = _mask_box_iou_sam3(masks[i], box_list)
            iou_list.append(s)
            if s > best_s:
                best_s, best_i = s, i
        return np.squeeze(masks[best_i]), n, best_i, iou_list

    if masks.ndim == 3:
        m2 = flat(masks[0])
        s0 = _mask_box_iou_sam3(m2, box_list)
        return m2, 1, 0, [s0]

    if masks.ndim == 2:
        s0 = _mask_box_iou_sam3(masks, box_list)
        return masks, 1, 0, [s0]

    raise ValueError(f"bad masks ndim: {masks.ndim}")


def get_mask_from_bbox_with_sam3(
    video_path: str,
    key_frame_idx: int,
    bbox: Dict,
    config: Dict,
    stage1_meta: Optional[Dict] = None,
    qwen_key_frame_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    Use SAM3 **image** model with a bbox prompt on a single key frame.

    相比之前直接用 SAM3 video predictor 在整段视频上做 box prompt，
    这里只在 key frame 图像上跑一次 SAM3 image 模型，大幅降低显存占用，
    然后仍然可以把这个 mask 交给后面的 tracker 做视频传播。

    Args:
        video_path: Path to video file
        key_frame_idx: When stage1_meta is None, the frame index for ``_load_keyframe_pil``.
            If ``stage1_meta`` includes ``original_frame_idx`` (HCSTVG), that overrides
            the read index so SAM3 matches ``key_frame.png`` / Qwen.
        bbox: Bounding box dict with keys 'xmin', 'ymin', 'xmax', 'ymax'
        config: Configuration dictionary
        stage1_meta: Optional stage1 ``metadata.json``; see ``resolve_sam3_keyframe_read_index``.
        qwen_key_frame_path: If set and the file exists, load this image (same pixels as Qwen
            / ``stage2/.../key_frame.png``) instead of re-decoding the video, so the bbox
            and SAM3 see identical coordinates.

    Returns:
        Binary mask (H, W) with values 0 or 1, or None if failed
    """
    predictor = None
    processor = None

    try:
        # ------------------------------------------------------------------
        # 1. 从视频或图片序列目录抽取 key frame 图像
        # ------------------------------------------------------------------
        read_idx = resolve_sam3_keyframe_read_index(key_frame_idx, stage1_meta)
        if read_idx != int(key_frame_idx) and isinstance(stage1_meta, dict):
            if stage1_meta.get("original_frame_idx", None) is not None:
                print(
                    f"  (SAM3 image: keyframe read index {read_idx} = original_frame_idx, "
                    f"not key_frame_idx={key_frame_idx})"
                )
        image: Optional[Image.Image] = None
        kfp = (qwen_key_frame_path or "").strip()
        key_frame_file_used = False
        if kfp and os.path.isfile(kfp):
            try:
                image = Image.open(kfp).convert("RGB")
                key_frame_file_used = True
                print(
                    f"  (SAM3 image: keyframe from Qwen file {kfp}, {image.size[0]}x{image.size[1]})"
                )
            except OSError as e:
                print(f"  ⚠️  Open qwen_key_frame_path failed, using video frame: {e}")
        if image is None:
            image = _load_keyframe_pil(video_path, read_idx)
        if image is None:
            return None
        width, height = image.size

        # ------------------------------------------------------------------
        # 2. 加载 SAM3 image 模型
        # ------------------------------------------------------------------
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sam3_package_dir = os.path.join(current_dir, '..', 'sam', 'sam3')
        if sam3_package_dir not in sys.path:
            sys.path.insert(0, sam3_package_dir)

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        sam3_config = config.get('sam3', {})
        model_path = sam3_config.get('model_path', '/home/xdu/.cache/modelscope/hub/models/facebook/sam3')

        # Find checkpoint (same逻辑，支持目录/单文件)
        def find_checkpoint_file(model_path_):
            if os.path.isfile(model_path_):
                return model_path_
            if os.path.isdir(model_path_):
                for name in ['sam3.pt', 'checkpoint.pt', 'model.pt']:
                    checkpoint_path_ = os.path.join(model_path_, name)
                    if os.path.isfile(checkpoint_path_):
                        return checkpoint_path_
                for file in os.listdir(model_path_):
                    if file.endswith('.pt'):
                        return os.path.join(model_path_, file)
            return model_path_

        checkpoint_path = find_checkpoint_file(model_path)

        print(f"  Loading SAM3 image model for box prompt segmentation...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictor = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        processor = Sam3Processor(predictor, device=device)

        # ------------------------------------------------------------------
        # 3. 把 bbox 转成归一化的 (cx, cy, w, h)，在 key frame 上做一次分割
        # ------------------------------------------------------------------
        xmin, ymin, xmax, ymax = (
            bbox['xmin'],
            bbox['ymin'],
            bbox['xmax'],
            bbox['ymax'],
        )
        cx = (xmin + xmax) / 2.0 / width
        cy = (ymin + ymax) / 2.0 / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height

        print(f"  Keyframe size: {width}x{height}")
        print(f"  Normalized box (cx, cy, w, h): {cx:.4f}, {cy:.4f}, {bw:.4f}, {bh:.4f}")

        state = processor.set_image(image)
        state = processor.add_geometric_prompt(
            box=[cx, cy, bw, bh],
            label=True,
            state=state,
        )

        masks = state.get('masks')
        if masks is None:
            print("  ✗ No masks returned from SAM3 image model")
            return None

        # masks: [N, 1, H, W] bool / uint8
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        if masks.ndim == 4:
            if masks.shape[0] == 0:
                print("  ✗ Empty masks from SAM3 image model")
                return None
        elif masks.ndim not in (2, 3):
            print(f"  ✗ Unexpected mask shape from SAM3 image model: {masks.shape}")
            return None

        # #region agent log
        # H-B: 坐标系一致 — 若 width/height 与 bbox 单位一致，应看到合理 IoU/候选
        # H-A: 多枚候选时选非 0 号会显著改变与 bbox 重叠
        _debug_ndjson(
            "sam3_keyframe_input",
            {
                "W": int(width),
                "H": int(height),
                "xmin": float(xmin),
                "ymin": float(ymin),
                "xmax": float(xmax),
                "ymax": float(ymax),
                "read_idx": int(read_idx),
                "key_frame_idx": int(key_frame_idx),
                "masks_shape": list(masks.shape),
                "image_source": (
                    "qwen_key_frame" if key_frame_file_used else "video"
                ),
                "qwen_key_frame_path": kfp if kfp else None,
            },
            location="get_mask_from_bbox_with_sam3:pre_select",
            hypothesis_id="B",
        )
        # #endregion

        # SAM3 常对同一 box 返回多枚候选，取 [0] 易与 Qwen 框不重合；按与「bbox 矩形」的
        # mask–box IoU 选最优（与 stage3_validate.compute_mask_rect_iou 同口径）
        box_list = [xmin, ymin, xmax, ymax]
        try:
            mask, n_cand, best_i, iou_list = _select_sam3_mask_by_box_iou(
                masks, box_list
            )
        except ValueError as e:
            print(f"  ✗ Unexpected mask layout: {e}")
            return None

        # #region agent log
        _debug_ndjson(
            "sam3_mask_select",
            {
                "n_candidates": n_cand,
                "best_i": best_i,
                "iou_per_candidate": [round(float(x), 6) for x in iou_list],
            },
            location="get_mask_from_bbox_with_sam3:post_select",
            hypothesis_id="A",
        )
        # #endregion

        if n_cand > 1:
            best_score = iou_list[best_i] if iou_list else 0.0
            print(
                f"  (selected mask index {best_i}/{n_cand - 1} by mask–box IoU = {float(best_score):.4f})"
            )

        mask = mask.astype(np.uint8)
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        print(
            f"  ✓ Got mask from SAM3 image model (shape: {mask.shape}, "
            f"coverage: {mask.sum() / mask.size * 100:.2f}%)"
        )

        return mask

    except RuntimeError as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "cuda" in msg.lower():
            print(f"  ✗ CUDA OOM error during SAM3 image box prompt: {e}")
        else:
            print(f"  ✗ SAM3 image box prompt segmentation failed (RuntimeError): {e}")
        import traceback
        traceback.print_exc()
        return None

    except Exception as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "cuda" in msg.lower():
            print(f"  ✗ CUDA OOM error during SAM3 image box prompt: {e}")
        else:
            print(f"  ✗ SAM3 image box prompt segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        try:
            if processor is not None:
                del processor
        except Exception:
            pass
        try:
            if predictor is not None:
                del predictor
        except Exception:
            pass
        clear_gpu_memory()


def extend_tube_with_bidirectional_propagation(
    video_path: str,
    existing_tube: Dict[int, np.ndarray],
    config: Dict
) -> Dict[int, np.ndarray]:
    """
    Extend an existing tube using bidirectional propagation.
    参考 batch_staged_pipeline.py 中 process_stage9 的实现方式。
    
    IMPORTANT: Preserves all existing masks from the original tube.
    Only adds new frames that don't exist in the original tube.
    
    Args:
        video_path: Path to video file
        existing_tube: Dictionary mapping frame_idx -> mask array (H, W) from text prompt segmentation
        config: Configuration dictionary
        
    Returns:
        Extended dictionary mapping frame_idx -> mask array (H, W)
        Original masks are preserved, only new frames are added
    """
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 获取tube的起始和结束帧
    sorted_frames = sorted(existing_tube.keys())
    if not sorted_frames:
        return existing_tube
    
    tube_start = sorted_frames[0]
    tube_end = sorted_frames[-1]
    
    # 初始化：从existing_tube开始
    extended_masks_dict = dict(existing_tube)
    
    # 辅助函数：查找checkpoint文件
    def find_checkpoint_file(model_path):
        if os.path.isfile(model_path):
            return model_path
        if os.path.isdir(model_path):
            for name in ['sam3.pt', 'checkpoint.pt', 'model.pt']:
                checkpoint_path = os.path.join(model_path, name)
                if os.path.isfile(checkpoint_path):
                    return checkpoint_path
            for file in os.listdir(model_path):
                if file.endswith('.pt'):
                    return os.path.join(model_path, file)
        return model_path
    
    # 辅助函数：获取GPU配置
    def get_gpu_config(sam3_config):
        gpus_to_use = sam3_config.get('gpus_to_use')
        if gpus_to_use is None:
            visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            if visible_devices:
                device_list = [int(d.strip()) for d in visible_devices.split(',') if d.strip()]
                if len(device_list) > 1:
                    gpus_to_use = list(range(len(device_list)))
                else:
                    gpus_to_use = [0] if torch.cuda.is_available() else None
            else:
                if torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                    gpus_to_use = list(range(num_gpus)) if num_gpus > 0 else [0]
                else:
                    gpus_to_use = None
        return gpus_to_use
    
    # 前向传播：从 tube_end 到视频末尾
    if tube_end < total_frames - 1:
        print(f"  前向传播: 从 tube_end ({tube_end}) 到视频末尾 ({total_frames - 1})")
        
        try:
            clear_gpu_memory()
            
            # 创建新的predictor和session
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sam3_package_dir = os.path.join(current_dir, '..', 'sam', 'sam3')
            if sam3_package_dir not in sys.path:
                sys.path.insert(0, sam3_package_dir)
            
            from sam3.model_builder import build_sam3_video_predictor
            
            sam3_config = config.get('sam3', {})
            model_path = sam3_config.get('model_path', '/home/xdu/.cache/modelscope/hub/models/facebook/sam3')
            checkpoint_path = find_checkpoint_file(model_path)
            gpus_to_use = get_gpu_config(sam3_config)
            
            predictor_forward = build_sam3_video_predictor(
                checkpoint_path=checkpoint_path,
                gpus_to_use=gpus_to_use
            )
            
            session_result = predictor_forward.start_session(resource_path=video_path)
            session_id_forward = session_result['session_id']
            
            # 获取 inference_state
            session = predictor_forward._get_session(session_id_forward)
            inference_state = session["state"]
            
            # 在 tube_end 添加 mask prompt（只添加一个mask作为起点）
            mask_end = existing_tube[tube_end]
            if mask_end.max() > 1:
                mask_end = (mask_end > 0).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask_end).float()
            mask_coverage = mask_end.sum() / mask_end.size * 100
            
            print(f"  ✓ 在帧 {tube_end} 添加 mask prompt (覆盖率: {mask_coverage:.2f}%)")
            prop_obj_id = 1
            predictor_forward.model.tracker.add_new_mask(
                inference_state=inference_state,
                frame_idx=tube_end,
                obj_id=prop_obj_id,
                mask=mask_tensor,
                add_mask_to_memory=True
            )
            
            # 关键：调用 preflight 准备传播
            predictor_forward.model.tracker.propagate_in_video_preflight(
                inference_state, run_mem_encoder=True
            )
            
            # 前向传播到视频末尾
            forward_frames = total_frames - tube_end
            print(f"  ▶ 前向传播: 从帧 {tube_end} 到 {total_frames - 1} (共 {forward_frames} 帧)")
            
            for result in predictor_forward.propagate_in_video(
                session_id=session_id_forward,
                propagation_direction="forward",  # 只前向
                start_frame_idx=tube_end,
                max_frame_num_to_track=forward_frames
            ):
                frame_idx = result['frame_index']
                outputs = result['outputs']
                
                if isinstance(outputs, dict) and 'out_binary_masks' in outputs:
                    binary_masks = outputs['out_binary_masks']
                    raw_obj_ids = outputs.get('out_obj_ids', list(range(len(binary_masks))))
                    obj_ids = np.array(raw_obj_ids) if raw_obj_ids is not None else np.array([])
                    
                    if obj_ids.size > 0 and len(binary_masks) > 0:
                        # 选择匹配 prop_obj_id 的对象
                        prop_mask_idx = 0
                        if prop_obj_id in obj_ids:
                            prop_mask_idx = int(np.where(obj_ids == prop_obj_id)[0][0])
                        mask = binary_masks[prop_mask_idx]
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        mask = mask.astype(np.uint8)
                        if mask.ndim > 2:
                            mask = mask.squeeze()
                        if mask.max() > 1:
                            mask = (mask > 0).astype(np.uint8)
                        
                        if mask.shape != (frame_height, frame_width):
                            mask = cv2.resize(
                                mask,
                                (frame_width, frame_height),
                                interpolation=cv2.INTER_NEAREST
                            )
                        
                        # 只添加新帧，保留原始masks
                        if frame_idx not in existing_tube:
                            extended_masks_dict[frame_idx] = mask
            
            predictor_forward.close_session(session_id_forward)
            del predictor_forward
            clear_gpu_memory()
            
        except Exception as e:
            print(f"  ✗ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 后向传播：从 tube_start 到视频开头
    if tube_start > 0:
        print(f"  后向传播: 从 tube_start ({tube_start}) 到视频开头 (0)")
        
        try:
            clear_gpu_memory()
            
            # 创建新的predictor和session
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sam3_package_dir = os.path.join(current_dir, '..', 'sam', 'sam3')
            if sam3_package_dir not in sys.path:
                sys.path.insert(0, sam3_package_dir)
            
            from sam3.model_builder import build_sam3_video_predictor
            
            sam3_config = config.get('sam3', {})
            model_path = sam3_config.get('model_path', '/home/xdu/.cache/modelscope/hub/models/facebook/sam3')
            checkpoint_path = find_checkpoint_file(model_path)
            gpus_to_use = get_gpu_config(sam3_config)
            
            predictor_backward = build_sam3_video_predictor(
                checkpoint_path=checkpoint_path,
                gpus_to_use=gpus_to_use
            )
            
            session_result = predictor_backward.start_session(resource_path=video_path)
            session_id_backward = session_result['session_id']
            
            # 获取 inference_state
            session = predictor_backward._get_session(session_id_backward)
            inference_state = session["state"]
            
            # 在 tube_start 添加 mask prompt（只添加一个mask作为起点）
            mask_start = existing_tube[tube_start]
            if mask_start.max() > 1:
                mask_start = (mask_start > 0).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask_start).float()
            mask_coverage = mask_start.sum() / mask_start.size * 100
            
            print(f"  ✓ 在帧 {tube_start} 添加 mask prompt (覆盖率: {mask_coverage:.2f}%)")
            prop_obj_id = 1
            predictor_backward.model.tracker.add_new_mask(
                inference_state=inference_state,
                frame_idx=tube_start,
                obj_id=prop_obj_id,
                mask=mask_tensor,
                add_mask_to_memory=True
            )
            
            # 关键：调用 preflight 准备传播
            predictor_backward.model.tracker.propagate_in_video_preflight(
                inference_state, run_mem_encoder=True
            )
            
            # 后向传播到视频开头
            backward_frames = tube_start + 1
            print(f"  ▶ 后向传播: 从帧 {tube_start} 到 0 (共 {backward_frames} 帧)")
            
            for result in predictor_backward.propagate_in_video(
                session_id=session_id_backward,
                propagation_direction="backward",  # 只后向
                start_frame_idx=tube_start,
                max_frame_num_to_track=backward_frames
            ):
                frame_idx = result['frame_index']
                outputs = result['outputs']
                
                if isinstance(outputs, dict) and 'out_binary_masks' in outputs:
                    binary_masks = outputs['out_binary_masks']
                    raw_obj_ids = outputs.get('out_obj_ids', list(range(len(binary_masks))))
                    obj_ids = np.array(raw_obj_ids) if raw_obj_ids is not None else np.array([])
                    
                    if obj_ids.size > 0 and len(binary_masks) > 0:
                        # 选择匹配 prop_obj_id 的对象
                        prop_mask_idx = 0
                        if prop_obj_id in obj_ids:
                            prop_mask_idx = int(np.where(obj_ids == prop_obj_id)[0][0])
                        mask = binary_masks[prop_mask_idx]
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        mask = mask.astype(np.uint8)
                        if mask.ndim > 2:
                            mask = mask.squeeze()
                        if mask.max() > 1:
                            mask = (mask > 0).astype(np.uint8)
                        
                        if mask.shape != (frame_height, frame_width):
                            mask = cv2.resize(
                                mask,
                                (frame_width, frame_height),
                                interpolation=cv2.INTER_NEAREST
                            )
                        
                        # 只添加新帧，保留原始masks
                        if frame_idx not in existing_tube:
                            extended_masks_dict[frame_idx] = mask
            
            predictor_backward.close_session(session_id_backward)
            del predictor_backward
            clear_gpu_memory()
            
        except Exception as e:
            print(f"  ✗ 后向传播失败: {e}")
            import traceback
            traceback.print_exc()
    
    new_frames_count = len(extended_masks_dict) - len(existing_tube)
    print(f"  ✓ Extended tube: {len(existing_tube)} original frames (preserved) + {new_frames_count} new frames = {len(extended_masks_dict)} total")
    
    return extended_masks_dict


def expand_all_tubes_bidirectionally(
    video_path: str,
    all_tubes: Dict[int, Dict[int, np.ndarray]],
    config: Dict
) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Expand all tubes using bidirectional propagation.
    
    Args:
        video_path: Path to video file
        all_tubes: Dictionary of tubes {tube_id: {frame_idx: mask}}
        config: Configuration dictionary
        
    Returns:
        Dictionary of expanded tubes {tube_id: {frame_idx: mask}}
        Original masks are preserved for each tube
    """
    if not all_tubes or len(all_tubes) == 0:
        return all_tubes
    
    print(f"\n  Expanding {len(all_tubes)} tubes using bidirectional propagation...")
    expanded_tubes = {}
    
    for tube_id, tube_masks in all_tubes.items():
        if len(tube_masks) == 0:
            print(f"    Skipping tube {tube_id} (empty)")
            expanded_tubes[tube_id] = tube_masks
            continue
        
        print(f"    Expanding tube {tube_id} ({len(tube_masks)} frames)...")
        try:
            expanded_tube = extend_tube_with_bidirectional_propagation(
                video_path=video_path,
                existing_tube=tube_masks,
                config=config
            )
            expanded_tubes[tube_id] = expanded_tube
            print(f"      ✓ Tube {tube_id}: {len(tube_masks)} -> {len(expanded_tube)} frames")
        except Exception as e:
            print(f"      ✗ Tube {tube_id} expansion failed: {e}")
            # Keep original tube if expansion fails
            expanded_tubes[tube_id] = tube_masks
    
    print(f"  ✓ Expanded {len(expanded_tubes)} tubes")
    return expanded_tubes


def propagate_mask_with_sam3_tracker(
    video_path: str,
    key_frame_idx: int,
    initial_mask: np.ndarray,
    config: Dict,
    bbox: Optional[Dict] = None,
    num_gpus: int = 1,
    stage1_meta: Optional[Dict] = None,
    qwen_key_frame_path: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """
    Propagate mask through video using SAM3 Tracker with bidirectional propagation.
    
    Follows tracker-first propagation (similar to CVPRW26-MeViSv2TrackSolution):
    - If bbox is provided: first segment key frame to an initial mask, then add mask to tracker
    - If no bbox: use provided initial_mask directly
    
    Args:
        video_path: Path to video file
        key_frame_idx: Frame index where the initial mask is provided
        initial_mask: Binary mask (H, W) with values 0 or 1
        config: Configuration dictionary
        bbox: Optional bounding box dict with keys 'xmin', 'ymin', 'xmax', 'ymax'
              If provided, will use box instead of mask
        
    Returns:
        Dictionary mapping frame_idx -> mask array (H, W) with values 0 or 1
    """
    predictor = None
    tracker_video_path = video_path
    tracker_temp_video_path = None

    try:
        clear_gpu_memory()

        # Add SAM3 to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sam3_package_dir = os.path.join(current_dir, '..', 'sam', 'sam3')
        if sam3_package_dir not in sys.path:
            sys.path.insert(0, sam3_package_dir)

        from sam3.model_builder import build_sam3_video_predictor

        sam3_config = config.get('sam3', {})
        model_path = sam3_config.get('model_path', '/home/xdu/.cache/modelscope/hub/models/facebook/sam3')

        # Find checkpoint
        def find_checkpoint_file(model_path):
            if os.path.isfile(model_path):
                return model_path
            if os.path.isdir(model_path):
                for name in ['sam3.pt', 'checkpoint.pt', 'model.pt']:
                    checkpoint_path = os.path.join(model_path, name)
                    if os.path.isfile(checkpoint_path):
                        return checkpoint_path
                for file in os.listdir(model_path):
                    if file.endswith('.pt'):
                        return os.path.join(model_path, file)
            return model_path

        checkpoint_path = find_checkpoint_file(model_path)

        tracker_video_path, tracker_temp_video_path = _prepare_sam3_tracker_video_input(video_path)

        # Determine frame metadata once
        cap = cv2.VideoCapture(tracker_video_path)
        total_frames_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height_cv2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        max_frames_for_tracker = sam3_config.get(
            'max_frames_for_tracker',
            config.get('sam3_max_frames_for_tracker', 2000),
        )
        if total_frames_cv2 > max_frames_for_tracker:
            print(
                f"  ⚠️  Video too long ({total_frames_cv2} frames > {max_frames_for_tracker}), "
                "skipping to avoid OOM"
            )
            print(f"  ⊘ Will be processed later with larger GPU memory")
            return {}

        # Build keyframe prompt mask BEFORE loading video tracker heavy state.
        ann_frame_idx = key_frame_idx
        ann_obj_id = 1

        prompt_mask = None
        if bbox:
            xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            print(f"  BBox prompt (pixels): ({xmin}, {ymin}, {xmax}, {ymax})")
            print("  Converting bbox to keyframe mask using SAM3 image model...")
            prompt_mask = get_mask_from_bbox_with_sam3(
                video_path=video_path,
                key_frame_idx=ann_frame_idx,
                bbox=bbox,
                config=config,
                stage1_meta=stage1_meta,
                qwen_key_frame_path=qwen_key_frame_path,
            )

            if prompt_mask is None:
                print("  ⚠️  SAM3 image box-to-mask failed, falling back to rectangular bbox mask")
                prompt_mask = np.zeros(
                    (max(1, video_height_cv2), max(1, video_width_cv2)), dtype=np.uint8
                )
                x0 = max(0, min(prompt_mask.shape[1] - 1, int(round(xmin))))
                y0 = max(0, min(prompt_mask.shape[0] - 1, int(round(ymin))))
                x1 = max(x0 + 1, min(prompt_mask.shape[1], int(round(xmax))))
                y1 = max(y0 + 1, min(prompt_mask.shape[0], int(round(ymax))))
                prompt_mask[y0:y1, x0:x1] = 1
        else:
            prompt_mask = initial_mask

        if prompt_mask is None:
            print("  ✗ No valid prompt mask available for tracker propagation")
            return {}

        # Get GPU configuration
        gpus_to_use = None
        if num_gpus is not None and num_gpus > 0:
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                gpus_to_use = list(range(min(num_gpus, available_gpus)))
            else:
                gpus_to_use = None
        else:
            gpus_to_use = sam3_config.get('gpus_to_use')
            if gpus_to_use is None:
                visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                if visible_devices:
                    device_list = [int(d.strip()) for d in visible_devices.split(',') if d.strip()]
                    if len(device_list) > 1:
                        gpus_to_use = list(range(len(device_list)))
                    else:
                        gpus_to_use = [0] if torch.cuda.is_available() else None
                else:
                    if torch.cuda.is_available():
                        available_gpus = torch.cuda.device_count()
                        gpus_to_use = list(range(available_gpus)) if available_gpus > 0 else [0]
                    else:
                        gpus_to_use = None

        print(f"  Loading SAM3 Tracker...")
        print(f"  Using GPUs: {gpus_to_use}")
        predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            gpus_to_use=gpus_to_use
        )

        tracker = predictor.model.tracker
        if getattr(tracker, "backbone", None) is None and hasattr(predictor.model, "detector"):
            tracker.backbone = predictor.model.detector.backbone

        print(f"  Initializing tracker state...")
        try:
            inference_state = tracker.init_state(
                video_path=tracker_video_path,
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
                async_loading_frames=False,
            )
        except TypeError:
            inference_state = tracker.init_state(video_path=tracker_video_path)

        # Resolve frame metadata from inference_state
        total_frames = int(inference_state.get("num_frames", total_frames_cv2))
        if total_frames <= 0:
            total_frames = total_frames_cv2
        print(f"  Using frame count: {total_frames} frames")

        video_height = int(inference_state.get("video_height", inference_state.get("orig_height", video_height_cv2)))
        video_width = int(inference_state.get("video_width", inference_state.get("orig_width", video_width_cv2)))
        if video_height <= 0 or video_width <= 0:
            video_height = prompt_mask.shape[0]
            video_width = prompt_mask.shape[1]

        # Validate frame idx
        if ann_frame_idx < 0:
            print(f"  ✗ Invalid frame index: {ann_frame_idx} < 0")
            return {}
        if total_frames > 0 and ann_frame_idx >= total_frames:
            print(f"  ✗ Invalid frame index: {ann_frame_idx} >= {total_frames}")
            return {}

        # Normalize mask shape/type
        if prompt_mask.ndim > 2:
            prompt_mask = prompt_mask.squeeze()
        prompt_mask = prompt_mask.astype(np.uint8)
        if prompt_mask.max() > 1:
            prompt_mask = (prompt_mask > 0).astype(np.uint8)
        if prompt_mask.shape != (video_height, video_width):
            print(
                f"  ℹ️  Resizing prompt mask from {prompt_mask.shape} "
                f"to ({video_height}, {video_width})"
            )
            prompt_mask = cv2.resize(
                prompt_mask,
                (video_width, video_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
            if prompt_mask.max() > 1:
                prompt_mask = (prompt_mask > 0).astype(np.uint8)

        if total_frames > 0:
            forward_frames = max(0, total_frames - ann_frame_idx)
            backward_frames = max(1, ann_frame_idx + 1)
        else:
            forward_frames = 0
            backward_frames = max(1, ann_frame_idx + 1)

        print(f"  Propagating masks through video (bidirectional)...")
        print(f"    Total frames: {total_frames}")
        print(f"    Start frame: {ann_frame_idx}")
        print(f"    Forward frames: {forward_frames}")
        print(f"    Backward frames: {backward_frames}")

        masks_dict = {ann_frame_idx: prompt_mask}

        def _collect_tracker_outputs(reverse: bool, propagate_preflight: bool):
            kwargs = dict(
                inference_state=inference_state,
                start_frame_idx=int(ann_frame_idx),
                max_frame_num_to_track=None,
                reverse=bool(reverse),
                tqdm_disable=True,
                propagate_preflight=bool(propagate_preflight),
            )
            try:
                iterator = tracker.propagate_in_video(**kwargs)
            except TypeError:
                kwargs.pop("propagate_preflight", None)
                iterator = tracker.propagate_in_video(**kwargs)

            for item in iterator:
                if not isinstance(item, tuple) or len(item) < 4:
                    continue
                frame_idx = int(item[0])
                out_obj_ids = item[1]
                video_res_masks = item[3]

                if isinstance(video_res_masks, torch.Tensor):
                    logits = video_res_masks.detach().cpu().numpy()
                else:
                    logits = np.asarray(video_res_masks)
                if logits.size == 0:
                    continue

                if logits.ndim == 4 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                elif logits.ndim == 2:
                    logits = logits[None, ...]

                obj_idx = 0
                if out_obj_ids is not None:
                    out_obj_ids_arr = np.array(out_obj_ids)
                    if out_obj_ids_arr.size > 0 and ann_obj_id in out_obj_ids_arr:
                        obj_idx = int(np.where(out_obj_ids_arr == ann_obj_id)[0][0])
                if obj_idx >= logits.shape[0]:
                    obj_idx = 0

                mask = (logits[obj_idx] > 0).astype(np.uint8)
                if mask.shape != (video_height, video_width):
                    mask = cv2.resize(
                        mask,
                        (video_width, video_height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.uint8)
                    if mask.max() > 1:
                        mask = (mask > 0).astype(np.uint8)
                masks_dict[frame_idx] = mask

        # Keep tracker path entirely outside autograd to avoid inference/autograd
        # mode conflicts inside SAM3 internals.
        with torch.no_grad():
            if hasattr(tracker, "clear_all_points_in_video"):
                tracker.clear_all_points_in_video(inference_state)

            mask_tensor = torch.from_numpy(prompt_mask.astype(np.uint8))
            print(f"  Adding mask to tracker at frame {key_frame_idx}...")
            tracker.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                mask=mask_tensor,
            )
            print(
                f"  ✓ Added mask prompt at frame {key_frame_idx} "
                f"(coverage: {prompt_mask.sum() / prompt_mask.size * 100:.2f}%)"
            )

            # Align with reference workflow: preflight at first direction only.
            _collect_tracker_outputs(reverse=False, propagate_preflight=True)
            if ann_frame_idx > 0:
                _collect_tracker_outputs(reverse=True, propagate_preflight=False)

        print(f"  ✓ Generated {len(masks_dict)} masks")

        if len(masks_dict) == 1:
            print(f"  ⚠️  Warning: Only 1 mask generated, propagation likely failed")
            print(f"  ⊘ Returning empty dict to mark as failure")
            return {}

        return masks_dict
        
    except RuntimeError as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower() or "killed" in error_msg.lower():
            print(f"  ✗ CUDA OOM error during SAM3 Tracker propagation: {e}")
            print(f"  ⊘ Recording OOM error and skipping this task...")
            # Return empty dict instead of raising, so the calling function can handle it
            return {}
        else:
            print(f"  ✗ SAM3 Tracker failed (RuntimeError): {e}")
        import traceback
        traceback.print_exc()
        raise
        
    except Exception as e:
        error_msg = str(e)
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower() or "killed" in error_msg.lower():
            print(f"  ✗ CUDA OOM error during SAM3 Tracker propagation: {e}")
            print(f"  ⊘ Recording OOM error and skipping this task...")
            # Return empty dict instead of raising, so the calling function can handle it
            return {}
        else:
            print(f"  ✗ SAM3 Tracker failed: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        if tracker_temp_video_path is not None:
            try:
                if os.path.exists(tracker_temp_video_path):
                    os.remove(tracker_temp_video_path)
            except Exception as e:
                print(f"  ⚠️  Failed to cleanup temp tracker mp4: {e}")
        if predictor is not None:
            try:
                del predictor
                predictor = None
            except Exception:
                pass
        clear_gpu_memory()


def _get_bbox_propagation_method_name() -> str:
    """
    Return method label based on the currently bound bbox propagation function.
    This keeps metadata accurate when wrappers monkey-patch SAM2 -> SAM3.
    """
    func_name = getattr(propagate_mask_with_sam2, "__name__", "")
    if func_name == "propagate_mask_with_sam3_tracker":
        return "qwen_bbox_sam3_propagation"
    return "qwen_bbox_sam2_propagation"


def process_video_worker(
    gpu_id: int,
    video_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    base_dir: str,
    sentence_type: str,
    dataset: str,
    output_dir: str,
    config: Dict,
    selection_mode: str,
    each_tube_visualization: bool,
    gpu_delay: float,
    tube_bidirectional_expand: bool = False,
    retry_failed: bool = False,
    skip_visualization: bool = False,
    stage2_subdir: str = "stage2",
    stage3_subdir: str = "stage3",
    sam3_prompt_override: Optional[str] = None,
):
    """
    Worker process for processing videos on a specific GPU.
    
    Args:
        gpu_id: GPU ID to use
        video_queue: Queue containing video IDs to process
        result_queue: Queue to put results
        base_dir: Base directory
        sentence_type: Sentence type
        output_dir: Output directory
        config: Configuration dictionary
        selection_mode: Tube selection mode
        each_tube_visualization: Whether to generate individual tube visualizations
        gpu_delay: Delay before starting (to stagger GPU loading)
    """
    try:
        # Set CUDA_VISIBLE_DEVICES to only see this GPU
        # IMPORTANT: Must be set BEFORE any torch imports
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"[GPU {gpu_id}] Set CUDA_VISIBLE_DEVICES={gpu_id}", flush=True)
        
        # Import after setting CUDA_VISIBLE_DEVICES
        import torch
        import gc
    
        # Verify GPU is available
        if torch.cuda.is_available():
            actual_device_count = torch.cuda.device_count()
            print(f"[GPU {gpu_id}] CUDA available: {actual_device_count} device(s) visible", flush=True)
            if actual_device_count > 0:
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                print(f"[GPU {gpu_id}] Using device {current_device}: {device_name}", flush=True)
        else:
            print(f"[GPU {gpu_id}] ⚠️  WARNING: CUDA not available!", flush=True)
        
        def clear_gpu_memory_worker():
            """Clear GPU memory for worker process"""
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except RuntimeError as e:
                # Ignore CUDA initialization errors
                if "re-initialize" not in str(e).lower():
                    raise
        
        # Stagger model loading to avoid channel congestion
        if gpu_delay > 0:
            time.sleep(gpu_id * gpu_delay)
        
        print(f"[GPU {gpu_id}] Worker started (delayed {gpu_id * gpu_delay:.1f}s to avoid channel congestion)", flush=True)
        
        # Create a separate visualization manager for this worker (skip if --no-visualization)
        worker_viz_manager = None if skip_visualization else VisualizationManager(max_workers=1)
        
        processed_count = 0
        success_count = 0
        fail_count = 0
        
        print(f"[GPU {gpu_id}] Entering main processing loop...", flush=True)
        while True:
            try:
                # Get task from queue (with timeout to check for None)
                try:
                    task = video_queue.get(timeout=1)
                except Exception as queue_error:
                    # Timeout is normal, continue waiting
                    continue
                
                if task is None:  # End signal
                    print(f"[GPU {gpu_id}] Received end signal, exiting...", flush=True)
                    break
                
                vid = task
                processed_count += 1
                print(f"[GPU {gpu_id}] Got task #{processed_count}: {vid}", flush=True)
                
                print(f"\n[GPU {gpu_id}] Processing {processed_count}: {vid}", flush=True)
                print(f"{'='*80}", flush=True)
                
                try:
                    result = process_stage3(
                        base_dir=base_dir,
                        vid=vid,
                        sentence_type=sentence_type,
                        dataset=dataset,
                        output_dir=output_dir,
                        config=config,
                        selection_mode=selection_mode,
                        each_tube_visualization=each_tube_visualization,
                        viz_manager=worker_viz_manager,
                        tube_bidirectional_expand=tube_bidirectional_expand,
                        retry_failed=retry_failed,
                        num_gpus=1,  # Worker process uses single GPU (CUDA_VISIBLE_DEVICES is set)
                        skip_visualization=skip_visualization,
                        stage2_subdir=stage2_subdir,
                        stage3_subdir=stage3_subdir,
                        sam3_prompt_override=sam3_prompt_override,
                    )
                    
                    if result:
                        success_count += 1
                        result_queue.put({
                            'video_id': vid,
                            'gpu_id': gpu_id,
                            'success': True
                        })
                        print(f"[GPU {gpu_id}] ✓ Success: {vid}", flush=True)
                    else:
                        fail_count += 1
                        result_queue.put({
                            'video_id': vid,
                            'gpu_id': gpu_id,
                            'success': False
                        })
                        print(f"[GPU {gpu_id}] ✗ Failed: {vid}", flush=True)
                    
                except RuntimeError as e:
                    error_msg = str(e)
                    if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                        print(f"[GPU {gpu_id}] ✗ CUDA OOM: {vid} - {e}", flush=True)
                    elif "re-initialize" in error_msg.lower():
                        print(f"[GPU {gpu_id}] ✗ CUDA initialization error: {vid} - {e}", flush=True)
                        print(f"[GPU {gpu_id}] This should not happen with 'spawn' method", flush=True)
                    else:
                        print(f"[GPU {gpu_id}] ✗ RuntimeError: {vid} - {e}", flush=True)
                    fail_count += 1
                    result_queue.put({
                        'video_id': vid,
                        'gpu_id': gpu_id,
                        'success': False,
                        'error': str(e)
                    })
                    clear_gpu_memory_worker()
                    
                except Exception as e:
                    print(f"[GPU {gpu_id}] ✗ Exception: {vid} - {e}", flush=True)
                    fail_count += 1
                    result_queue.put({
                        'video_id': vid,
                        'gpu_id': gpu_id,
                        'success': False,
                        'error': str(e)
                    })
                    clear_gpu_memory_worker()
                
                # Cleanup after each video
                clear_gpu_memory_worker()
                
            except KeyboardInterrupt:
                print(f"[GPU {gpu_id}] Received KeyboardInterrupt, exiting...", flush=True)
                break
            except Exception as e:
                print(f"[GPU {gpu_id}] ⚠️  Worker error in main loop: {e}", flush=True)
                import traceback
                traceback.print_exc()
                # Don't exit on error, continue processing
                continue
        
        # Wait for visualization tasks
        if worker_viz_manager:
            print(f"[GPU {gpu_id}] Shutting down visualization manager...", flush=True)
            worker_viz_manager.shutdown(wait=True)
        
        print(f"[GPU {gpu_id}] ========== Worker finished ==========", flush=True)
        print(f"[GPU {gpu_id}] Success: {success_count}, Failed: {fail_count}, Total: {processed_count}", flush=True)
        print(f"[GPU {gpu_id}] =====================================", flush=True)
    except Exception as e:
        # Catch any unhandled exceptions during worker initialization or execution
        print(f"[GPU {gpu_id}] ⚠️  FATAL ERROR in worker process: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Put error result in queue so main process knows this worker failed
        try:
            result_queue.put({
                'video_id': None,
                'gpu_id': gpu_id,
                'success': False,
                'error': f'Worker fatal error: {str(e)}'
            })
        except:
            pass
        raise  # Re-raise to ensure process exits with error code


def process_videos_multi_gpu(
    video_ids: list,
    base_dir: str,
    sentence_type: str,
    dataset: str,
    output_dir: str,
    config: Dict,
    selection_mode: str,
    each_tube_visualization: bool,
    num_gpus: int,
    gpu_delay: float,
    tube_bidirectional_expand: bool = False,
    retry_failed: bool = False,
    skip_visualization: bool = False,
    stage2_subdir: str = "stage2",
    stage3_subdir: str = "stage3",
    sam3_prompt_override: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Process videos using multiple GPUs in parallel.
    
    Args:
        video_ids: List of video IDs to process
        base_dir: Base directory
        sentence_type: Sentence type
        output_dir: Output directory
        config: Configuration dictionary
        selection_mode: Tube selection mode
        each_tube_visualization: Whether to generate individual tube visualizations
        num_gpus: Number of GPUs to use
        gpu_delay: Delay between GPU model loading (seconds)
    
    Returns:
        Tuple of (success_count, fail_count)
    """
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before creating any processes
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("✗ CUDA not available, falling back to single GPU processing")
        return 0, len(video_ids)
    
    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        print(f"⚠️  Requested {num_gpus} GPUs but only {available_gpus} available, using {available_gpus}")
        num_gpus = available_gpus
    
    if num_gpus <= 1:
        print("⚠️  Only 1 GPU available or requested, falling back to sequential processing")
        return 0, len(video_ids)
    
    print(f"Using {num_gpus} GPUs (available: {available_gpus})")
    print(f"Multiprocessing start method: {multiprocessing.get_start_method()}")
    
    # Create queues
    manager = multiprocessing.Manager()
    video_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # Add all videos to queue
    for vid in video_ids:
        video_queue.put(vid)
    
    # Add end signals (one per GPU)
    for _ in range(num_gpus):
        video_queue.put(None)
    
    # Start worker processes
    workers = []
    for gpu_id in range(num_gpus):
        p = multiprocessing.Process(
            target=process_video_worker,
                args=(
                gpu_id,
                video_queue,
                result_queue,
                base_dir,
                sentence_type,
                dataset,
                output_dir,
                config,
                selection_mode,
                each_tube_visualization,
                gpu_delay,
                tube_bidirectional_expand,
                retry_failed,
                skip_visualization,
                stage2_subdir,
                stage3_subdir,
                sam3_prompt_override,
            )
        )
        p.start()
        workers.append(p)
        print(f"  Started worker for GPU {gpu_id}")
    
    print(f"\n✓ Started {num_gpus} worker processes")
    print(f"  Workers will load models with {gpu_delay}s delay between each GPU")
    print(f"  This helps avoid PCIe channel congestion\n")
    
    # Wait for all workers to finish
    print(f"\nWaiting for {len(workers)} worker processes to finish...")
    for idx, p in enumerate(workers):
        print(f"  Waiting for worker {idx} (GPU {idx})...")
        p.join()
        exit_code = p.exitcode
        if exit_code != 0:
            print(f"  ⚠️  Worker {idx} (GPU {idx}) exited with code {exit_code}")
        else:
            print(f"  ✓ Worker {idx} (GPU {idx}) finished normally")
    
    # Collect results
    print(f"\nCollecting results from result queue...")
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    print(f"  Collected {len(results)} results")
    
    # Calculate statistics
    success_count = sum(1 for r in results if r.get('success', False))
    fail_count = len(results) - success_count
    
    # Print per-GPU statistics
    gpu_stats = {}
    for gpu_id in range(num_gpus):
        gpu_results = [r for r in results if r.get('gpu_id') == gpu_id]
        gpu_stats[gpu_id] = {
            'total': len(gpu_results),
            'success': sum(1 for r in gpu_results if r.get('success', False)),
            'failed': sum(1 for r in gpu_results if not r.get('success', False))
        }
    
    print(f"\n{'='*80}")
    print("Multi-GPU Processing Statistics")
    print(f"{'='*80}")
    for gpu_id in range(num_gpus):
        stats = gpu_stats[gpu_id]
        print(f"GPU {gpu_id}: {stats['success']} success, {stats['failed']} failed, {stats['total']} total")
    print(f"{'='*80}")
    
    return success_count, fail_count


class VisualizationManager:
    """Manage asynchronous visualization tasks in background CPU processes"""
    def __init__(self, max_workers=None):
        if max_workers is None:
            max_workers = min(2, multiprocessing.cpu_count())
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.task_info = []
    
    def submit(self, func, *args, **kwargs):
        """Submit a visualization task to background process"""
        future = self.executor.submit(func, *args, **kwargs)
        self.futures.append(future)
        # Store task info for logging
        task_name = func.__name__ if hasattr(func, '__name__') else 'unknown'
        self.task_info.append((task_name, future))
        return future
    
    def wait_all(self, timeout=None):
        """Wait for all tasks to complete and return results"""
        results = []
        for i, future in enumerate(as_completed(self.futures, timeout=timeout)):
            task_name = self.task_info[i][0] if i < len(self.task_info) else 'unknown'
            try:
                result = future.result()
                results.append((True, task_name, result))
            except Exception as e:
                results.append((False, task_name, str(e)))
        self.futures.clear()
        self.task_info.clear()
        return results
    
    def shutdown(self, wait=True):
        """Shutdown the executor"""
        self.executor.shutdown(wait=wait)


def generate_mask_video(video_path, masks_dict, output_path, mask_color=(128, 0, 128), mask_alpha=0.5):
    """Generate video with transparent mask overlay"""
    from moviepy.editor import ImageSequenceClip
    
    print(f"  Generating mask video...")
    
    processed_frames = []
    
    if os.path.isdir(video_path):
        image_fps_default = 24.0
        names = _sorted_image_filenames_in_dir(video_path)
        frame_count = len(names)
        fps = image_fps_default
        if frame_count == 0:
            print(f"  ✗ No frames in image directory: {video_path}")
            return
        for frame_idx, name in enumerate(names):
            fp = os.path.join(video_path, name)
            frame = cv2.imread(fp)
            if frame is None:
                print(f"  ⚠️  Could not read {fp}, using black frame {frame_idx}")
                if processed_frames:
                    h, w = processed_frames[-1].shape[:2]
                    frame_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    frame_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_idx in masks_dict:
                mask = masks_dict[frame_idx]
                if mask.ndim > 2:
                    mask = mask.squeeze()
                if mask.max() > 1:
                    mask = (mask > 127).astype(np.uint8)
                overlay = np.zeros_like(frame_rgb)
                overlay[mask > 0] = mask_color
                frame_rgb = cv2.addWeighted(frame_rgb, 1 - mask_alpha, overlay, mask_alpha, 0)
            processed_frames.append(frame_rgb)
            if (frame_idx + 1) % 100 == 0:
                print(f"    Progress: {frame_idx + 1}/{frame_count} frames")
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ✗ Could not open video: {video_path}")
            return
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if frame_idx in masks_dict:
                mask = masks_dict[frame_idx]
                
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                if mask.max() > 1:
                    mask = (mask > 127).astype(np.uint8)
                
                overlay = np.zeros_like(frame_rgb)
                overlay[mask > 0] = mask_color
                
                frame_rgb = cv2.addWeighted(frame_rgb, 1 - mask_alpha, overlay, mask_alpha, 0)
            
            processed_frames.append(frame_rgb)
            
            if (frame_idx + 1) % 100 == 0:
                print(f"    Progress: {frame_idx + 1}/{frame_count} frames")
        
        cap.release()
    
    if not processed_frames:
        print(f"  ✗ No frames to encode (empty sequence)")
        return
    
    print(f"  Generating video file...")
    
    clip = ImageSequenceClip(processed_frames, fps=fps)
    clip.write_videofile(output_path, codec='libx264', audio=False,
                         preset='medium', bitrate='5000k',
                         logger=None)
    
    print(f"  ✓ Video saved: {output_path}")


def generate_sam3_tubes_visualization(video_path, all_tubes, output_path, key_frame_idx):
    """Generate visualization video showing all SAM3 tubes with different colors"""
    from moviepy.editor import ImageSequenceClip
    
    print(f"  Generating SAM3 tubes visualization...")
    
    # Generate distinct colors for each tube
    def generate_colors(n):
        colors = []
        for i in range(n):
            hue = int(180 * i / max(n, 1)) % 180
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
            colors.append(tuple(int(c) for c in color))
        return colors
    
    tube_ids = sorted(all_tubes.keys())
    tube_colors = generate_colors(len(tube_ids))
    color_map = {tube_id: color for tube_id, color in zip(tube_ids, tube_colors)}
    
    processed_frames = []
    
    if os.path.isdir(video_path):
        fps = 24.0
        names = _sorted_image_filenames_in_dir(video_path)
        frame_count = len(names)
        if frame_count == 0:
            print(f"  ✗ No frames in image directory: {video_path}")
            return
        for frame_idx, name in enumerate(names):
            fp = os.path.join(video_path, name)
            frame = cv2.imread(fp)
            if frame is None:
                if processed_frames:
                    h, w = processed_frames[-1].shape[:2]
                    frame_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    frame_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for tube_id, color in color_map.items():
                if frame_idx in all_tubes[tube_id]:
                    mask = all_tubes[tube_id][frame_idx]
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    if mask.max() > 1:
                        mask = (mask > 0).astype(np.uint8)
                    overlay = np.zeros_like(frame_rgb)
                    overlay[mask > 0] = color
                    frame_rgb = cv2.addWeighted(frame_rgb, 0.7, overlay, 0.3, 0)
            processed_frames.append(frame_rgb)
            if (frame_idx + 1) % 100 == 0:
                print(f"    Progress: {frame_idx + 1}/{frame_count} frames")
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ✗ Could not open video: {video_path}")
            return
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_idx in range(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Overlay all tubes with different colors
            for tube_id, color in color_map.items():
                if frame_idx in all_tubes[tube_id]:
                    mask = all_tubes[tube_id][frame_idx]
                    
                    if mask.ndim > 2:
                        mask = mask.squeeze()
                    
                    if mask.max() > 1:
                        mask = (mask > 0).astype(np.uint8)
                    
                    overlay = np.zeros_like(frame_rgb)
                    overlay[mask > 0] = color
                    
                    frame_rgb = cv2.addWeighted(frame_rgb, 0.7, overlay, 0.3, 0)
            
            processed_frames.append(frame_rgb)
            
            if (frame_idx + 1) % 100 == 0:
                print(f"    Progress: {frame_idx + 1}/{frame_count} frames")
        
        cap.release()
    
    if not processed_frames:
        print(f"  ✗ No frames to encode (empty sequence)")
        return
    
    print(f"  Generating video file...")
    
    clip = ImageSequenceClip(processed_frames, fps=fps)
    clip.write_videofile(output_path, codec='libx264', audio=False,
                         preset='medium', bitrate='5000k',
                         logger=None)
    
    print(f"  ✓ SAM3 tubes visualization saved: {output_path}")


def generate_each_tube_visualization(video_path, all_tubes, output_dir, base_vid):
    """Generate individual visualization video for each tube"""
    print(f"  Generating individual visualizations for each tube...")
    
    tube_ids = sorted(all_tubes.keys())
    
    for tube_id in tube_ids:
        tube_masks = all_tubes[tube_id]
        
        if len(tube_masks) == 0:
            continue
        
        # Generate distinct color for this tube
        hue = int(180 * tube_id / max(len(tube_ids), 1)) % 180
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
        mask_color = tuple(int(c) for c in color)
        
        output_path = os.path.join(output_dir, f"{base_vid}_tube_{tube_id}_visualization.mp4")
        
        print(f"    Generating visualization for tube {tube_id}...")
        generate_mask_video(video_path, tube_masks, output_path, 
                          mask_color=mask_color, mask_alpha=0.5)
    
    print(f"  ✓ Generated {len(tube_ids)} individual tube visualizations")


def save_sam3_tubes(all_tubes: Dict, sam3_seg_dir: str, video_path: str, key_frame_idx: int, prompt: str, object_name: Optional[str] = None):
    """Save SAM3 tubes to disk
    
    Args:
        all_tubes: Dictionary of tubes to save
        sam3_seg_dir: Base directory for SAM3 segmentation output
        video_path: Path to video file
        key_frame_idx: Key frame index
        prompt: SAM3 prompt used
        object_name: Name of the object (e.g., 'person', 'table'). If None, uses prompt.
                    Used to create {object_name}_tubes directory.
    """
    # Generate object name from prompt if not provided
    if object_name is None:
        # Use prompt as object name, sanitize for filesystem
        object_name = prompt.lower().strip()
        # Replace spaces and special characters with underscores
        object_name = re.sub(r'[^\w\-]', '_', object_name)
        # Remove multiple consecutive underscores
        object_name = re.sub(r'_+', '_', object_name)
        # Remove leading/trailing underscores
        object_name = object_name.strip('_')
    
    print(f"  Saving SAM3 tubes for '{object_name}' to {sam3_seg_dir}...")
    
    os.makedirs(sam3_seg_dir, exist_ok=True)
    
    # Save each tube in {object_name}_tubes directory
    tubes_dir = os.path.join(sam3_seg_dir, f"{object_name}_tubes")
    os.makedirs(tubes_dir, exist_ok=True)
    
    for tube_id, frames in all_tubes.items():
        tube_dir = os.path.join(tubes_dir, f"tube_{tube_id}")
        os.makedirs(tube_dir, exist_ok=True)
        
        for frame_idx, mask in frames.items():
            if mask.ndim > 2:
                mask = mask.squeeze()
            frame_idx_int = int(frame_idx)  # Ensure Python int type
            mask_file = os.path.join(tube_dir, f"mask_{frame_idx_int:05d}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_file)
    
    # Save metadata in the object-specific tubes directory
    # Convert all integers to Python native int to avoid JSON serialization issues
    metadata = {
        'num_tubes': int(len(all_tubes)),
        'tube_ids': [int(tube_id) for tube_id in sorted(all_tubes.keys())],
        'key_frame_idx': int(key_frame_idx),  # Keep for tube selection logic
        'start_frame_idx': 0,  # Text prompt segmentation starts from frame 0
        'prompt': prompt,
        'object_name': object_name,
        'video_path': video_path,
        'frames_per_tube': {int(tube_id): [int(frame_idx) for frame_idx in sorted(frames.keys())] 
                            for tube_id, frames in all_tubes.items()}
    }
    
    with open(os.path.join(tubes_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved {len(all_tubes)} tubes for '{object_name}' in {tubes_dir}")


def load_tubes_from_stage3_tubes(
    base_dir: str,
    prefix: str,
    base_vid: str,
    prompt: str,
    video_path: str
) -> Tuple[Optional[Dict], Optional[bool]]:
    """Load SAM3 tubes from stage3_tubes directory (priority source)
    
    Args:
        base_dir: Base directory containing vidstg_declarative/interrogative/savg
        prefix: "vidstg_declarative", "vidstg_interrogative", or "savg"
        base_vid: Base video ID (without idx)
        prompt: SAM3 prompt used
        video_path: Video path for validation
    
    Returns:
        Tuple of (tubes_dict, has_zero_tubes_flag)
        - tubes_dict: Dictionary of tubes or None if not found
        - has_zero_tubes_flag: True if metadata exists and num_tubes=0, None otherwise
    """
    # Sanitize prompt for use in directory name (same as stage3_collect_tube.py)
    sanitized_prompt = re.sub(r'[^\w\s-]', '', prompt).strip().replace(' ', '_')
    if not sanitized_prompt:
        sanitized_prompt = 'unknown'
    
    # Stage3_tubes directory structure: {base_dir}/{prefix}/stage3_tubes/{base_vid}/{sanitized_prompt}/
    stage3_tubes_dir = os.path.join(base_dir, prefix, "stage3_tubes", base_vid, sanitized_prompt)
    metadata_path = os.path.join(stage3_tubes_dir, "metadata.json")
    tubes_dir = os.path.join(stage3_tubes_dir, "tubes")
    
    if not os.path.exists(metadata_path):
        return None, None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Validate that prompt and video_path match
        if metadata.get('prompt') != prompt or metadata.get('video_path') != video_path:
            return None, None
        
        # Check if num_tubes is 0 (from CSV or metadata)
        num_tubes = metadata.get('num_tubes', 0)
        if num_tubes == 0:
            print(f"  ⊘ Found metadata with num_tubes=0 for '{prompt}' in stage3_tubes/{base_vid}/{sanitized_prompt}")
            print(f"    SAM3 text prompt segmentation detected no tubes, skipping to avoid redundant processing...")
            return {}, True  # Return empty dict and flag indicating zero tubes
        
        if not os.path.exists(tubes_dir):
            return None, None
        
        all_tubes = {}
        
        for tube_id in metadata.get('tube_ids', []):
            tube_dir = os.path.join(tubes_dir, f"tube_{tube_id}")
            if not os.path.exists(tube_dir):
                continue
            
            all_tubes[tube_id] = {}
            
            for mask_file in sorted(os.listdir(tube_dir)):
                if mask_file.startswith('mask_') and mask_file.endswith('.png'):
                    # Extract frame index from "mask_XXXXX.png" (5 characters after "mask_")
                    frame_idx = int(mask_file[5:-4])
                    mask_path = os.path.join(tube_dir, mask_file)
                    mask = np.array(Image.open(mask_path))
                    mask = (mask > 127).astype(np.uint8)
                    all_tubes[tube_id][frame_idx] = mask
        
        loaded_object = metadata.get('prompt', prompt)
        print(f"  ✓ Loaded {len(all_tubes)} tubes for '{loaded_object}' from stage3_tubes/{base_vid}/{sanitized_prompt}")
        return all_tubes, False  # Return tubes and flag indicating non-zero tubes
    except Exception as e:
        print(f"  ⚠️  Error loading tubes from stage3_tubes: {e}")
        return None, None


def load_sam3_tubes(sam3_seg_dir: str, object_name: Optional[str] = None) -> Optional[Dict]:
    """Load SAM3 tubes from disk (fallback to sam3_seg directory)
    
    Args:
        sam3_seg_dir: Base directory for SAM3 segmentation output
        object_name: Name of the object (e.g., 'person', 'table'). If None, tries to find any tubes directory.
    
    Returns:
        Dictionary of tubes or None if not found
    """
    # If object_name is provided, load from specific directory
    if object_name is not None:
        tubes_dir = os.path.join(sam3_seg_dir, f"{object_name}_tubes")
        metadata_path = os.path.join(tubes_dir, "metadata.json")
    else:
        # Try to find any {object_name}_tubes directory
        tubes_dir = None
        metadata_path = None
        
        if os.path.exists(sam3_seg_dir):
            for item in os.listdir(sam3_seg_dir):
                if item.endswith('_tubes') and os.path.isdir(os.path.join(sam3_seg_dir, item)):
                    potential_metadata = os.path.join(sam3_seg_dir, item, "metadata.json")
                    if os.path.exists(potential_metadata):
                        tubes_dir = os.path.join(sam3_seg_dir, item)
                        metadata_path = potential_metadata
                        break
        
        # Fallback to old format (tubes directory)
        if tubes_dir is None:
            tubes_dir = os.path.join(sam3_seg_dir, "tubes")
            metadata_path = os.path.join(sam3_seg_dir, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if not os.path.exists(tubes_dir):
        return None
    
    all_tubes = {}
    
    for tube_id in metadata['tube_ids']:
        tube_dir = os.path.join(tubes_dir, f"tube_{tube_id}")
        if not os.path.exists(tube_dir):
            continue
        
        all_tubes[tube_id] = {}
        
        for mask_file in sorted(os.listdir(tube_dir)):
            if mask_file.startswith('mask_') and mask_file.endswith('.png'):
                frame_idx = int(mask_file[5:-4])  # Extract frame index from "mask_XXXXX.png"
                mask_path = os.path.join(tube_dir, mask_file)
                mask = np.array(Image.open(mask_path))
                mask = (mask > 127).astype(np.uint8)
                all_tubes[tube_id][frame_idx] = mask
    
    loaded_object = metadata.get('object_name', object_name or 'unknown')
    print(f"  ✓ Loaded {len(all_tubes)} tubes for '{loaded_object}' from {tubes_dir}")
    return all_tubes


def check_stage3_completed(
    base_dir: str,
    vid: str,
    sentence_type: str,
    retry_failed: bool = False,
    stage3_subdir: str = "stage3",
) -> Tuple[bool, Optional[str]]:
    """
    Check if Stage 3 is already completed or failed for a given video.
    
    Args:
        base_dir: Resolved output_root directory (already contains prefix, e.g., model/output/savg)
        vid: Video ID (e.g., "24001787725_1" or "all-terrain_vehicle_8_3")
        sentence_type: "declarative", "interrogative", or "savg"
        retry_failed: If True, retry failed tasks (especially no_masks_generated). Default: False
        stage3_subdir: Subfolder under base_dir for stage3 outputs (default: stage3).

    Returns:
        Tuple of (is_completed_or_failed, output_path)
        - is_completed_or_failed: True if completed successfully OR failed (should skip)
        - output_path: Path to stage3 output directory
    """
    # base_dir is treated as the resolved output_root (already contains prefix)
    # No need to add prefix again
    
    # Parse vid to get base_vid and idx
    if "_" in vid:
        base_vid, idx = vid.rsplit("_", 1)
    else:
        base_vid = vid
        idx = None
    
    # Check stage3 output directory
    if idx:
        stage3_output = os.path.join(base_dir, stage3_subdir, base_vid, str(idx))
    else:
        stage3_output = os.path.join(base_dir, stage3_subdir, base_vid)
    
    # Check if metadata.json exists
    metadata_path = os.path.join(stage3_output, "metadata.json")
    if not os.path.exists(metadata_path):
        return False, stage3_output
    
    # Check metadata for status
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check if it's marked as failed
        status = metadata.get('status', 'unknown')
        if status == 'failed':
            failure_reason = metadata.get('failure_reason', 'unknown')
            if retry_failed:
                # If retry_failed is True, retry failed tasks (especially no_masks_generated)
                print(f"  ↻ Retrying {vid} (previously failed: {failure_reason})")
                return False, stage3_output  # Return False to retry
            else:
                print(f"  ⊘ Skipping {vid} (marked as failed: {failure_reason})")
                return True, stage3_output  # Return True to skip
        
        # Check if metadata indicates success (only check metadata.json, not masks/ directory)
        # This allows running on a server with only metadata.json synced (no mask files)
        if metadata.get('num_masks', 0) > 0:
            # metadata.json exists and num_masks > 0, consider as completed
            return True, stage3_output
        
        # num_masks == 0 or not present, not completed
        return False, stage3_output
    except Exception as e:
        print(f"  ⚠️  Error reading metadata: {e}")
        return False, stage3_output


def _create_chunk_video(video_path: str, start_frame: int, end_frame: int, total_frames: int) -> str:
    """
    创建视频片段（只包含指定范围的帧）
    
    对于图片序列目录，创建临时目录并复制/链接文件
    对于视频文件，返回原路径（由 LazyFrameSequence 处理范围）
    
    Args:
        video_path: 原始视频路径
        start_frame: 起始帧索引
        end_frame: 结束帧索引（不包含）
        total_frames: 总帧数
    
    Returns:
        临时视频/目录路径
    """
    if os.path.isdir(video_path):
        # 图片序列目录：创建临时目录并复制/链接文件
        temp_dir = tempfile.mkdtemp(prefix='sam2_chunk_')
        
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
        frame_files = []
        for f in os.listdir(video_path):
            if any(f.lower().endswith(ext.lower()) for ext in image_extensions):
                frame_files.append(f)
        
        # 使用与 _load_video_frames 相同的排序逻辑
        def sort_key(x):
            try:
                name = os.path.splitext(x)[0]
                if name.isdigit():
                    return int(name)
                numbers = re.findall(r'\d+', name)
                if numbers:
                    return int(numbers[0])
                return float('inf')
            except:
                return float('inf')
        
        frame_files = sorted(frame_files, key=sort_key)
        
        # 只复制/链接指定范围的帧
        chunk_files = frame_files[start_frame:end_frame]
        
        print(f"  [Chunking] Creating temporary directory with {len(chunk_files)} frames (range: {start_frame}-{end_frame-1} of {len(frame_files)})")
        for i, frame_file in enumerate(chunk_files):
            src_path = os.path.join(video_path, frame_file)
            # 使用新的索引命名，确保顺序
            dst_file = f"{i:06d}{os.path.splitext(frame_file)[1]}"
            dst_path = os.path.join(temp_dir, dst_file)
            # 使用符号链接节省空间（如果支持）
            try:
                os.symlink(src_path, dst_path)
            except:
                shutil.copy2(src_path, dst_path)
        
        # 验证临时目录中的文件数量
        actual_files = len([f for f in os.listdir(temp_dir) 
                           if any(f.lower().endswith(ext.lower()) for ext in image_extensions)])
        if actual_files != len(chunk_files):
            print(f"  ⚠️  Warning: Temporary directory has {actual_files} files, expected {len(chunk_files)}")
        else:
            print(f"  ✓ Temporary directory created successfully with {actual_files} files")
        
        return temp_dir
    else:
        # 视频文件：返回原路径（由 LazyFrameSequence 处理范围，或需要提取视频片段）
        return video_path


def _propagate_mask_with_sam2_chunked(
    video_path: str,
    key_frame_idx: int,
    initial_mask: np.ndarray,
    config: Dict,
    bbox: Optional[Dict],
    num_frames: int,
    frame_width: int,
    frame_height: int,
    num_gpus: int = 1
) -> Optional[Dict[int, np.ndarray]]:
    """
    分块处理长视频，尽量保持 SAM2 双向传播效果
    
    策略：
    1. 将视频分成多个重叠的片段（chunks）
    2. 每个片段独立运行 SAM2 双向传播
    3. 在重叠区域合并结果，保持连续性
    
    Args:
        video_path: 视频路径
        key_frame_idx: 关键帧索引
        initial_mask: 初始 mask
        config: 配置字典
        bbox: 可选的边界框字典
        num_frames: 总帧数
        frame_width: 帧宽度
        frame_height: 帧高度
        num_gpus: GPU 数量
    
    Returns:
        合并后的 mask 字典
    """
    sam2_config = config.get('sam2', {})
    chunk_size = sam2_config.get('chunk_size', 1500)  # 每个 chunk 的帧数
    overlap_size = sam2_config.get('chunk_overlap', 200)  # 重叠区域大小
    
    print(f"  [Chunking] Splitting video into chunks: chunk_size={chunk_size}, overlap={overlap_size}")
    
    # 计算 chunks
    chunks = []
    chunk_start = 0
    while chunk_start < num_frames:
        chunk_end = min(chunk_start + chunk_size, num_frames)
        chunks.append((chunk_start, chunk_end))
        if chunk_end >= num_frames:
            break
        chunk_start = chunk_end - overlap_size  # 重叠
    
    print(f"  [Chunking] Total {len(chunks)} chunks: {chunks}")
    
    # 找到包含 key_frame 的 chunk（主 chunk）
    main_chunk_idx = None
    for i, (start, end) in enumerate(chunks):
        if start <= key_frame_idx < end:
            main_chunk_idx = i
            break
    
    if main_chunk_idx is None:
        print(f"  ✗ Key frame {key_frame_idx} not found in any chunk")
        return None
    
    print(f"  [Chunking] Main chunk: {main_chunk_idx} (contains key_frame {key_frame_idx})")
    
    all_masks = {}  # 最终合并的 mask 字典
    
    # 只创建一次 SAM2 模型实例，避免重复加载导致内存碎片化
    print(f"  [Memory Optimization] Loading SAM2 model once for all chunks...")
    SAM2ModuleWithImageSequence = _get_sam2_module_with_image_sequence()
    
    # 确定设备
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible_devices:
        device = 'cuda:0'
    else:
        device = config.get('device', 'cuda')
    
    sam2_module = None
    try:
        sam2_module = SAM2ModuleWithImageSequence(
            model_path=sam2_config.get('model_path', '/home/xdu/.cache/modelscope/hub/models/facebook/sam2.1-hiera-base-plus'),
            device=device
        )
        print(f"  ✓ SAM2 model loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load SAM2 model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 处理主 chunk（包含 key_frame）
    main_start, main_end = chunks[main_chunk_idx]
    print(f"\n  [Chunk {main_chunk_idx}] Processing main chunk: frames {main_start}-{main_end-1}")
    
    try:
        # 创建临时视频片段
        chunk_video_path = _create_chunk_video(video_path, main_start, main_end, num_frames)
        
        # 初始化视频（复用同一个模型实例）
        chunk_frames = main_end - main_start
        sam2_module.initialize_video(chunk_video_path)
        print(f"  ✓ Chunk {main_chunk_idx} initialized: {chunk_frames} frames")
        
        # 添加 prompt（key_frame 在 chunk 中的相对索引）
        chunk_key_frame_idx = key_frame_idx - main_start
        sam2_module.add_prompt(frame_idx=chunk_key_frame_idx, mask=initial_mask)
        
        # 双向传播
        chunk_masks = sam2_module.propagate_masks()
        print(f"  ✓ Chunk {main_chunk_idx} generated {len(chunk_masks)} masks")
        
        # 将 chunk 的 mask 转换为全局索引
        for chunk_frame_idx, mask in chunk_masks.items():
            global_frame_idx = main_start + chunk_frame_idx
            # 确保 mask 是 2D 数组
            if mask.ndim > 2:
                mask = mask.squeeze()
            all_masks[global_frame_idx] = mask
        
        # 重置状态（清理视频会话，但保留模型）
        sam2_module.reset()
        if hasattr(sam2_module, 'video_frames') and hasattr(sam2_module.video_frames, 'clear_cache'):
            sam2_module.video_frames.clear_cache()
        
        # 清理临时文件
        if chunk_video_path != video_path and os.path.exists(chunk_video_path):
            if os.path.isdir(chunk_video_path):
                shutil.rmtree(chunk_video_path)
            else:
                os.remove(chunk_video_path)
        
    except Exception as e:
        print(f"  ✗ Chunk {main_chunk_idx} failed: {e}")
        import traceback
        traceback.print_exc()
        if sam2_module is not None:
            try:
                sam2_module.cleanup()
            except:
                pass
        return None
    
    # 处理主 chunk 之前的 chunks（反向传播）
    for i in range(main_chunk_idx - 1, -1, -1):
        chunk_start, chunk_end = chunks[i]
        print(f"\n  [Chunk {i}] Processing backward chunk: frames {chunk_start}-{chunk_end-1}")
        
        # 从下一个 chunk 的边界获取初始 mask
        next_chunk_start = chunks[i + 1][0]
        if next_chunk_start not in all_masks:
            print(f"  ⚠️  Boundary mask not found, skipping chunk {i}")
            continue
        
        boundary_mask = all_masks[next_chunk_start]
        
        # 确保 mask 是 2D 数组 (H, W)
        if boundary_mask.ndim > 2:
            boundary_mask = boundary_mask.squeeze()
        if boundary_mask.ndim != 2:
            print(f"  ⚠️  Invalid mask shape: {boundary_mask.shape}, skipping chunk {i}")
            continue
        
        try:
            # 复用同一个模型实例，只重置视频会话
            chunk_video_path = _create_chunk_video(video_path, chunk_start, chunk_end, num_frames)
            sam2_module.initialize_video(chunk_video_path)
            
            # 在 chunk 的最后一个帧（与下一个 chunk 重叠的边界）添加 prompt
            chunk_prompt_idx = next_chunk_start - chunk_start
            sam2_module.add_prompt(frame_idx=chunk_prompt_idx, mask=boundary_mask)
            
            # 双向传播
            chunk_masks = sam2_module.propagate_masks()
            print(f"  ✓ Chunk {i} generated {len(chunk_masks)} masks")
            
            # 合并结果：在重叠区域使用加权平均
            for chunk_frame_idx, mask in chunk_masks.items():
                global_frame_idx = chunk_start + chunk_frame_idx
                
                # 确保 mask 是 2D 数组
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                if global_frame_idx in all_masks:
                    # 重叠区域：加权平均（距离边界越近，权重越小）
                    overlap_start = next_chunk_start
                    if global_frame_idx < overlap_start:
                        # 非重叠区域，直接使用
                        all_masks[global_frame_idx] = mask
                    else:
                        # 重叠区域：加权平均
                        weight = (global_frame_idx - overlap_start) / overlap_size
                        weight = max(0.0, min(1.0, weight))  # 限制在 [0, 1]
                        existing_mask = all_masks[global_frame_idx]
                        if existing_mask.ndim > 2:
                            existing_mask = existing_mask.squeeze()
                        existing_mask = existing_mask.astype(np.float32)
                        new_mask = mask.astype(np.float32)
                        merged_mask = (1 - weight) * existing_mask + weight * new_mask
                        all_masks[global_frame_idx] = (merged_mask > 0.5).astype(np.uint8)
                else:
                    # 非重叠区域，直接使用
                    all_masks[global_frame_idx] = mask
            
            # 重置状态（清理视频会话，但保留模型）
            sam2_module.reset()
            if hasattr(sam2_module, 'video_frames') and hasattr(sam2_module.video_frames, 'clear_cache'):
                sam2_module.video_frames.clear_cache()
            
            if chunk_video_path != video_path and os.path.exists(chunk_video_path):
                if os.path.isdir(chunk_video_path):
                    shutil.rmtree(chunk_video_path)
                else:
                    os.remove(chunk_video_path)
                    
        except Exception as e:
            print(f"  ✗ Chunk {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 处理主 chunk 之后的 chunks（正向传播）
    for i in range(main_chunk_idx + 1, len(chunks)):
        chunk_start, chunk_end = chunks[i]
        print(f"\n  [Chunk {i}] Processing forward chunk: frames {chunk_start}-{chunk_end-1}")
        
        # 从前一个 chunk 的边界获取初始 mask
        prev_chunk_end = chunks[i - 1][1]
        boundary_frame = prev_chunk_end - 1
        if boundary_frame not in all_masks:
            print(f"  ⚠️  Boundary mask not found, skipping chunk {i}")
            continue
        
        boundary_mask = all_masks[boundary_frame]
        
        # 确保 mask 是 2D 数组 (H, W)
        if boundary_mask.ndim > 2:
            boundary_mask = boundary_mask.squeeze()
        if boundary_mask.ndim != 2:
            print(f"  ⚠️  Invalid mask shape: {boundary_mask.shape}, skipping chunk {i}")
            continue
        
        try:
            # 复用同一个模型实例，只重置视频会话
            chunk_video_path = _create_chunk_video(video_path, chunk_start, chunk_end, num_frames)
            sam2_module.initialize_video(chunk_video_path)
            
            # 在 chunk 的第一个帧（与前一个 chunk 重叠的边界）添加 prompt
            chunk_prompt_idx = boundary_frame - chunk_start
            sam2_module.add_prompt(frame_idx=chunk_prompt_idx, mask=boundary_mask)
            
            # 双向传播
            chunk_masks = sam2_module.propagate_masks()
            print(f"  ✓ Chunk {i} generated {len(chunk_masks)} masks")
            
            # 合并结果：在重叠区域使用加权平均
            for chunk_frame_idx, mask in chunk_masks.items():
                global_frame_idx = chunk_start + chunk_frame_idx
                
                # 确保 mask 是 2D 数组
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                if global_frame_idx in all_masks:
                    # 重叠区域：加权平均
                    overlap_end = prev_chunk_end
                    if global_frame_idx >= overlap_end:
                        # 非重叠区域，直接使用
                        all_masks[global_frame_idx] = mask
                    else:
                        # 重叠区域：加权平均
                        weight = (overlap_end - global_frame_idx) / overlap_size
                        weight = max(0.0, min(1.0, weight))  # 限制在 [0, 1]
                        existing_mask = all_masks[global_frame_idx]
                        if existing_mask.ndim > 2:
                            existing_mask = existing_mask.squeeze()
                        existing_mask = existing_mask.astype(np.float32)
                        new_mask = mask.astype(np.float32)
                        merged_mask = (1 - weight) * existing_mask + weight * new_mask
                        all_masks[global_frame_idx] = (merged_mask > 0.5).astype(np.uint8)
                else:
                    # 非重叠区域，直接使用
                    all_masks[global_frame_idx] = mask
            
            # 重置状态（清理视频会话，但保留模型）
            sam2_module.reset()
            if hasattr(sam2_module, 'video_frames') and hasattr(sam2_module.video_frames, 'clear_cache'):
                sam2_module.video_frames.clear_cache()
            
            if chunk_video_path != video_path and os.path.exists(chunk_video_path):
                if os.path.isdir(chunk_video_path):
                    shutil.rmtree(chunk_video_path)
                else:
                    os.remove(chunk_video_path)
                    
        except Exception as e:
            print(f"  ✗ Chunk {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 所有 chunks 处理完成后，清理模型
    if sam2_module is not None:
        try:
            sam2_module.cleanup()
            del sam2_module
            clear_gpu_memory()
        except Exception as e:
            print(f"  ⚠️  Error during final cleanup: {e}")
    
    print(f"\n  ✓ [Chunking] Completed: {len(all_masks)} masks generated")
    return all_masks


def load_stage1_stage2_data(
    base_dir: str,
    vid: str,
    sentence_type: str,
    stage2_subdir: str = "stage2",
) -> Optional[Tuple[Dict, Dict]]:
    """
    Load stage1 and stage2 data for a given video.
    
    Args:
        base_dir: Resolved input_root directory (already contains prefix, e.g., model/output/savg)
        vid: Video ID (e.g., "2400171624_1" or "all-terrain_vehicle_8_3")
        sentence_type: "declarative", "interrogative", or "savg"
        
    Returns:
        Tuple of (stage1_meta, stage2_meta) or None if not found
    """
    # base_dir is treated as the resolved input_root (…/vidstg_declarative, …/savg, etc.)
    # No need to add prefix again
    
    # Parse vid to get base_vid and idx
    # Format: "2400171624_1" -> base_vid="2400171624", idx="1"
    if "_" in vid:
        base_vid, idx = vid.rsplit("_", 1)
    else:
        base_vid = vid
        idx = None
    
    # Load stage1 metadata
    if idx:
        stage1_path = os.path.join(base_dir, "stage1", base_vid, str(idx), "metadata.json")
    else:
        stage1_path = os.path.join(base_dir, "stage1", base_vid, "metadata.json")
    
    if not os.path.exists(stage1_path):
        print(f"✗ Stage 1 metadata not found: {stage1_path}")
        return None
    
    with open(stage1_path, 'r') as f:
        stage1_meta = json.load(f)
    
    # Load stage2 metadata (for interrogative/savg, to get bbox)
    s2 = str(stage2_subdir or "stage2").strip() or "stage2"
    if idx:
        stage2_path = os.path.join(base_dir, s2, base_vid, str(idx), "metadata.json")
    else:
        stage2_path = os.path.join(base_dir, s2, base_vid, "metadata.json")
    
    stage2_meta = None
    if os.path.exists(stage2_path):
        with open(stage2_path, 'r') as f:
            stage2_meta = json.load(f)
    else:
        # Stage 2 不存在，但不返回 None，允许 Case 3/4 处理
        print(f"  ⚠️  Stage 2 metadata not found: {stage2_path} (will use Case 3/4 logic)")
    
    return stage1_meta, stage2_meta


def load_stage4_bbox(
    base_dir: str,
    vid: str,
    sentence_type: str
) -> Optional[Dict]:
    """
    Load stage4 bbox for a given video.
    
    Args:
        base_dir: Resolved input_root directory (already contains prefix, e.g., model/output/savg)
        vid: Video ID (e.g., "2400171624_1" or "all-terrain_vehicle_8_3")
        sentence_type: "declarative", "interrogative", or "savg"
        
    Returns:
        Bbox dictionary with keys 'xmin', 'ymin', 'xmax', 'ymax' or None if not found/failed
    
    Subdirectory under base_dir is ``stage4`` by default, or ``STAGE3_STAGE4_SUBDIR`` env /
    ``--stage4-subdir`` (set in main via environment for worker processes).
    """
    # Parse vid to get base_vid and idx
    if "_" in vid:
        base_vid, idx = vid.rsplit("_", 1)
    else:
        base_vid = vid
        idx = None

    stage4_rel = (os.environ.get("STAGE3_STAGE4_SUBDIR") or "stage4").strip().strip("/\\")
    
    # Load stage4 metadata
    if idx:
        stage4_path = os.path.join(base_dir, stage4_rel, base_vid, str(idx), "metadata.json")
    else:
        stage4_path = os.path.join(base_dir, stage4_rel, base_vid, "metadata.json")
    
    if not os.path.exists(stage4_path):
        print(f"  ⚠️  Stage4 bbox metadata not found (expected): {stage4_path}")
        return None
    
    try:
        with open(stage4_path, 'r') as f:
            stage4_meta = json.load(f)
        
        # Check if stage4 was successful
        success = stage4_meta.get('success', False)
        if not success:
            return None
        
        # Get bbox
        bbox = stage4_meta.get('bbox')
        if bbox is None:
            return None
        
        return bbox
    except Exception as e:
        print(f"  ⚠️  Error loading stage4 bbox: {e}")
        return None


def select_tube_with_stage4_bbox(
    all_tubes: Dict,
    stage4_bbox: Dict,
    key_frame_idx: int,
    frame_height: int,
    frame_width: int,
    selection_mode: str = 'relative',
    threshold: float = 0.3
) -> Tuple[Optional[int], Optional[float], Optional[Dict]]:
    """
    Select best tube based on stage4 bbox IoU (similar to Case 1 selection).
    
    Args:
        all_tubes: Dictionary of tubes {tube_id: {frame_idx: mask, ...}}
        stage4_bbox: Bbox dictionary with 'xmin', 'ymin', 'xmax', 'ymax'
        key_frame_idx: Key frame index
        frame_height: Frame height
        frame_width: Frame width
        selection_mode: 'relative' (intersection/mask_size) or 'absolute' (intersection/bbox_size)
        threshold: Minimum score threshold for selection
        
    Returns:
        Tuple of (selected_tube_id, selected_score, masks_dict) or (None, None, None) if no valid selection
    """
    if not all_tubes or not stage4_bbox:
        return None, None, None
    
    print(f"  Selecting best tube based on stage4 bbox overlap (mode: {selection_mode})...")
    bbox_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    bbox_mask[stage4_bbox['ymin']:stage4_bbox['ymax'], stage4_bbox['xmin']:stage4_bbox['xmax']] = 1
    bbox_total = bbox_mask.sum()
    
    tube_scores = {}
    for tube_id, frames in all_tubes.items():
        if key_frame_idx in frames:
            tube_mask = frames[key_frame_idx]
            intersection = np.logical_and(tube_mask, bbox_mask).sum()
            
            if selection_mode == 'absolute':
                # Absolute: intersection / bbox_size
                ratio = intersection / bbox_total if bbox_total > 0 else 0
            else:  # relative
                # Relative: intersection / mask_size
                mask_total = tube_mask.sum()
                ratio = intersection / mask_total if mask_total > 0 else 0
            
            tube_scores[tube_id] = ratio
            print(f"    Tube {tube_id}: {ratio:.4f} ({selection_mode})")
    
    if tube_scores:
        best_tube_id = max(tube_scores, key=tube_scores.get)
        best_score = tube_scores[best_tube_id]
        
        if best_score >= threshold:
            selected_tube_id = best_tube_id
            selected_score = best_score
            print(f"  ✓ Selected Tube {selected_tube_id}, score: {best_score:.4f}")
            
            # Use selected tube as masks
            masks_dict = all_tubes[selected_tube_id]
            return selected_tube_id, selected_score, masks_dict
        else:
            print(f"  ✗ Best tube score {best_score:.4f} < threshold {threshold}")
            return None, None, None
    else:
        return None, None, None


def process_stage3(
    base_dir: str,
    vid: str,
    sentence_type: str,
    dataset: str,
    output_dir: str,
    config: Dict,
    selection_mode: str = 'relative',
    each_tube_visualization: bool = False,
    viz_manager: Optional[object] = None,
    tube_bidirectional_expand: bool = False,
    retry_failed: bool = False,
    num_gpus: int = 1,
    skip_visualization: bool = False,
    stage2_subdir: str = "stage2",
    stage3_subdir: str = "stage3",
    sam3_prompt_override: Optional[str] = None,
) -> Optional[Dict]:
    """
    Process Stage 3: Video Segmentation using SAM3
    
    Args:
        base_dir: Resolved input_root directory (already contains prefix, e.g., model/output/savg)
        vid: Video ID (e.g., "2400171624_1" or "all-terrain_vehicle_8_3")
        sentence_type: Internal processing mode derived from dataset
        dataset: Dataset key (vidstg_declarative/hcstvg_v1/hcstvg_v2/savg)
        output_dir: Resolved output_root directory (already contains prefix, e.g., model/output/savg)
        config: Configuration dictionary
        selection_mode: Tube selection mode - 'relative' (intersection/mask_size) or 
                      'absolute' (intersection/bbox_size). Default: 'relative'
        each_tube_visualization: If True, generate individual visualization video for each tube ID
        viz_manager: VisualizationManager instance for async visualization (optional)
        
    Returns:
        Metadata dictionary or None if failed
    """
    print(f"\n{'='*80}")
    print(f"Stage 3: {vid} - Video Segmentation (SAM3) - {sentence_type}")
    print(f"{'='*80}\n")
#############################################################################################################################################################################
    # Check if already completed
    # Use output_dir (output_root) for checking completion, not base_dir (input_root)
    is_completed, stage3_output = check_stage3_completed(
        output_dir, vid, sentence_type, retry_failed=retry_failed, stage3_subdir=stage3_subdir
    )
    if is_completed:
        print(f"✓ Stage 3 already completed for {vid}")
        print(f"  Output: {stage3_output}")
        # Load and return existing metadata
        metadata_path = os.path.join(stage3_output, "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            print(f"  ⚠️  Error loading existing metadata: {e}")
            # Continue to reprocess
 #############################################################################################################################################################################   
    try:
        # Parse vid to get base_vid and idx
        # Format: "10001787725_3" -> base_vid="10001787725", idx="3"
        if "_" in vid:
            base_vid, idx = vid.rsplit("_", 1)
        else:
            base_vid = vid
            idx = None
        
        # Load stage1 and stage2 data
        data = load_stage1_stage2_data(
            base_dir, vid, sentence_type, stage2_subdir=stage2_subdir
        )
        if data is None:
            return None
        
        stage1_meta, stage2_meta = data
        
        video_path = _resolve_video_path_from_stage1(stage1_meta['video_path'])
        # Use original_frame_idx if available (original video frame index),
        # otherwise fall back to key_frame_idx (sampled frame index)
        # This is the frame index where we should add prompts for SAM3
        original_frame_idx = stage1_meta.get('original_frame_idx', stage1_meta['key_frame_idx'])
        key_frame_idx = original_frame_idx  # Keep for backward compatibility in variable names
        text_query = stage1_meta['text_query']
        
        # Get prompt for SAM3
        if sentence_type == "declarative" or sentence_type == "savg":
            # Both declarative and savg use target_object from stage1
            prompt = stage1_meta.get('target_object', text_query)
        else:  # interrogative
            prompt = stage2_meta.get('answer', text_query) if stage2_meta else text_query
        
        if sam3_prompt_override and str(sam3_prompt_override).strip():
            prompt = str(sam3_prompt_override).strip()
            print(f"  (SAM3 prompt override → '{prompt}')")
        
        # Get qwen detection result (handle stage2_meta being None for Case 3/4)
        if stage2_meta is not None:
            qwen_success = stage2_meta.get('success', False)
            bbox = stage2_meta.get('bbox')
        else:
            # Stage 2 failed or doesn't exist - qwen detection failed
            qwen_success = False
            bbox = None
            print(f"  ⚠️  Stage 2 metadata not available, treating as qwen_success=False")
        
        print(f"Video: {video_path}")
        print(f"Key frame (original_frame_idx): {key_frame_idx}")
        if 'original_frame_idx' in stage1_meta:
            print(f"  (sampled key_frame_idx was {stage1_meta['key_frame_idx']}, using original_frame_idx {original_frame_idx})")
        print(f"Text query: {text_query}")
        print(f"SAM3 prompt: {prompt}")
        print(f"Qwen success: {qwen_success}")
        print(f"BBox: {bbox}")
        
        qwen_key_frame_path: Optional[str] = None
        if idx and base_vid:
            _kfp = os.path.join(
                base_dir, stage2_subdir, base_vid, str(idx), "key_frame.png"
            )
            if os.path.isfile(_kfp):
                qwen_key_frame_path = _kfp
                print(f"  (SAM3 will prefer Qwen key frame image: {qwen_key_frame_path})")
        
        if not os.path.exists(video_path):
            print(f"✗ Video file not found")
            return None
        
        # Get video info - handle both video files and image directories (for SAVG)
        if os.path.isdir(video_path):
            # Image directory (e.g., savg dataset)
            image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            frame_files = [
                f for f in os.listdir(video_path)
                if any(f.lower().endswith(ext) for ext in image_extensions)
            ]
            # Sort by frame number (assuming format like 000001.jpg)
            try:
                frame_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except ValueError:
                # Fallback to lexicographic sort if frame numbers can't be parsed
                frame_files.sort()
            total_frames = len(frame_files)
            
            # Get frame dimensions from first image
            if frame_files:
                first_image_path = os.path.join(video_path, frame_files[0])
                first_image = Image.open(first_image_path)
                frame_width, frame_height = first_image.size
            else:
                print(f"  ⚠️  No image files found in directory: {video_path}")
                frame_width, frame_height = 0, 0
            
            print(f"Video info: {total_frames} frames, {frame_width}x{frame_height} (from image directory)")
        else:
            # Video file
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            print(f"Video info: {total_frames} frames, {frame_width}x{frame_height} (from video file)")
        
        # Validate video info
        if total_frames == 0:
            print(f"  ⚠️  Warning: Video has 0 frames, this may cause issues")
        if frame_width == 0 or frame_height == 0:
            print(f"  ⚠️  Warning: Video dimensions are 0x0, this may cause issues")
        
        # base_dir is already input_root (contains prefix), output_dir is already output_root (contains prefix)
        # No need to add prefix again
        
        # Check if sam3_seg exists and can be reused
        sam3_seg_dir = os.path.join(output_dir, stage3_subdir, base_vid, "sam3_seg")
        all_tubes = None
        sam3_success = False
        
        # Generate object_name from prompt for checking
        object_name = prompt.lower().strip()
        object_name = re.sub(r'[^\w\-]', '_', object_name)
        object_name = re.sub(r'_+', '_', object_name)
        object_name = object_name.strip('_')
        
        # PRIORITY 1: Check stage3_tubes directory first (from stage3_collect_tube.py)
        print(f"\nChecking for existing SAM3 tubes in stage3_tubes directory...")
        # For stage3_tubes, output_dir is like model/output/savg
        # load_tubes_from_stage3_tubes expects base_dir to be parent of prefix directory
        # So if output_dir is model/output/savg, base_dir should be model/output, prefix should be savg
        prefix = dataset
        # Extract base_dir: if output_dir ends with prefix, use its parent; otherwise use output_dir
        if os.path.basename(output_dir) == prefix:
            stage3_tubes_base_dir = os.path.dirname(output_dir)
        else:
            # output_dir might already be the full path, try to find prefix in it
            stage3_tubes_base_dir = output_dir
        all_tubes, has_zero_tubes = load_tubes_from_stage3_tubes(
            base_dir=stage3_tubes_base_dir,
            prefix=prefix,
            base_vid=base_vid,
            prompt=prompt,
            video_path=video_path
        )
        
        if has_zero_tubes is True:
            # Metadata exists with num_tubes=0, skip SAM3 text prompt segmentation
            sam3_success = False
            all_tubes = {}
            print(f"  ⊘ Skipping SAM3 text prompt segmentation (num_tubes=0 confirmed)")
        elif all_tubes and len(all_tubes) > 0:
            sam3_success = True
            print(f"  ✓ Found {len(all_tubes)} tubes in stage3_tubes directory")
        # else:
        #     print(f"  ✗ No tubes found in stage3_tubes directory, checking sam3_seg...")
        #     has_zero_tubes = False  # Initialize for later check
            
        #     # PRIORITY 2: Fallback to sam3_seg directory (backward compatibility)
        #     # Check for existing tubes for this specific object
        #     print(f"\nChecking for existing SAM3 segmentation in sam3_seg for '{object_name}'...")
            
        #     if os.path.exists(sam3_seg_dir):
        #         tubes_dir = os.path.join(sam3_seg_dir, f"{object_name}_tubes")
        #         metadata_path = os.path.join(tubes_dir, "metadata.json")
                
        #         if os.path.exists(metadata_path):
        #             try:
        #                 with open(metadata_path, 'r') as f:
        #                     sam3_meta = json.load(f)
                        
        #                 # Check if it's a failure record
        #                 if sam3_meta.get('status') == 'failed':
        #                     if (sam3_meta.get('prompt') == prompt and 
        #                         sam3_meta.get('video_path') == video_path):
        #                         print(f"  ⊘ SAM3 text prompt segmentation previously failed for '{object_name}'")
        #                         print(f"    Reason: {sam3_meta.get('failure_reason', 'unknown')}")
        #                         print(f"    Error: {sam3_meta.get('error_message', 'unknown')[:100]}...")
        #                         print(f"    Skipping to avoid redundant processing...")
        #                         sam3_success = False
        #                         all_tubes = {}
        #                         # Continue to fallback methods (Case 2, 3, 4)
        #                     else:
        #                         print(f"  ⚠️  Failure record exists but parameters don't match")
        #                         print(f"    Existing: prompt='{sam3_meta.get('prompt')}', video_path='{sam3_meta.get('video_path')}'")
        #                         print(f"    Current: prompt='{prompt}', video_path='{video_path}'")
        #                 # Check if we can reuse (same prompt and video_path)
        #                 elif (sam3_meta.get('prompt') == prompt and 
        #                       sam3_meta.get('video_path') == video_path):
        #                     print(f"  ✓ Found reusable SAM3 segmentation for '{object_name}'")
        #                     all_tubes = load_sam3_tubes(sam3_seg_dir, object_name=object_name)
        #                     if all_tubes and len(all_tubes) > 0:
        #                         sam3_success = True
        #                         print(f"  ✓ Reusing {len(all_tubes)} tubes for '{object_name}'")
                                
        #                         # Expand all tubes if enabled
        #                         if tube_bidirectional_expand:
        #                             print(f"  Expanding all tubes using bidirectional propagation...")
        #                             all_tubes = expand_all_tubes_bidirectionally(
        #                                 video_path=video_path,
        #                                 all_tubes=all_tubes,
        #                                 config=config
        #                             )
        #                     else:
        #                         print(f"  ✗ Failed to load tubes, will regenerate")
        #                 else:
        #                     print(f"  ✗ SAM3 segmentation exists but parameters don't match")
        #                     print(f"    Existing: prompt='{sam3_meta.get('prompt')}', video_path='{sam3_meta.get('video_path')}'")
        #                     print(f"    Current: prompt='{prompt}', video_path='{video_path}'")
        #             except Exception as e:
        #                 print(f"  ✗ Error reading metadata: {e}")
        #         else:
        #             # Also check old format (backward compatibility)
        #             old_metadata_path = os.path.join(sam3_seg_dir, "metadata.json")
        #             if os.path.exists(old_metadata_path):
        #                 try:
        #                     with open(old_metadata_path, 'r') as f:
        #                         sam3_meta = json.load(f)
                            
        #                     if (sam3_meta.get('prompt') == prompt and 
        #                         sam3_meta.get('video_path') == video_path):
        #                         print(f"  ✓ Found reusable SAM3 segmentation (old format)")
        #                         all_tubes = load_sam3_tubes(sam3_seg_dir)
        #                         if all_tubes and len(all_tubes) > 0:
        #                             sam3_success = True
        #                             print(f"  ✓ Reusing {len(all_tubes)} tubes")
        #                         else:
        #                             print(f"  ✗ Failed to load tubes, will regenerate")
        #                     else:
        #                         print(f"  ✗ SAM3 segmentation exists but parameters don't match (old format)")
        #                         print(f"    Existing: prompt='{sam3_meta.get('prompt')}', video_path='{sam3_meta.get('video_path')}'")
        #                         print(f"    Current: prompt='{prompt}', video_path='{video_path}'")
        #                 except Exception as e:
        #                     print(f"  ✗ Error reading old format metadata: {e}")
        #             else:
        #                 print(f"  ✗ No existing SAM3 segmentation found for '{object_name}'")
        #                 print(f"    (Checked: {metadata_path})")
        #     else:
        #         print(f"  ✗ SAM3 segmentation directory does not exist yet: {sam3_seg_dir}")
        
        clear_gpu_memory()
        
        # Try SAM3 for full video segmentation if not loaded
        threshold = config.get('stage3_threshold', 0.5)
        
        # Initialize predictor and session_id before the if block
        # so they can be safely referenced in the finally block
        predictor = None
        session_id = None
        
        # Qwen variant: only use tubes from stage3_tubes, do not run SAM3
        print(f"\n  (Qwen variant: using tubes from stage3_tubes only, no SAM3 run)")
        
        # Handle two cases: Case1 tubes+bbox (tube select, else bbox prop), Case2 no tubes+bbox (bbox prop)
        print(f"\nProcessing results (Qwen bbox only)...")
        print(f"  Qwen (bbox) success: {qwen_success}")
        print(f"  Tubes available: {sam3_success}")
        
        method = None
        masks_dict = None
        selected_tube_id = None
        selected_score = None
        tube_scores = {}
        
        if qwen_success and sam3_success:
            # Case 1: tubes + bbox → tube select; if fail → bbox propagation
            print(f"  Case 1: Tubes + Qwen bbox → tube select (else bbox propagation)")
            method = 'qwen_bbox_sam3_tube_selection'
            
            if bbox and all_tubes:
                print(f"  Selecting best tube based on bbox overlap (mode: {selection_mode})...")
                bbox_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                bbox_mask[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = 1
                bbox_total = bbox_mask.sum()
                
                for tube_id, frames in all_tubes.items():
                    if key_frame_idx in frames:
                        tube_mask = frames[key_frame_idx]
                        intersection = np.logical_and(tube_mask, bbox_mask).sum()
                        if selection_mode == 'absolute':
                            ratio = intersection / bbox_total if bbox_total > 0 else 0
                        else:
                            mask_total = tube_mask.sum()
                            ratio = intersection / mask_total if mask_total > 0 else 0
                        tube_scores[tube_id] = ratio
                        print(f"    Tube {tube_id}: {ratio:.4f} ({selection_mode})")
                
                if tube_scores:
                    best_tube_id = max(tube_scores, key=tube_scores.get)
                    best_score = tube_scores[best_tube_id]
                    if best_score >= threshold:
                        selected_tube_id = best_tube_id
                        selected_score = best_score
                        print(f"  ✓ Selected Tube {selected_tube_id}, score: {best_score:.4f}")
                        masks_dict = all_tubes[selected_tube_id]
            
            # Tube select failed or no selection → Case 2: bbox propagation
            if masks_dict is None and bbox:
                print(f"  Tube select failed or no selection → bbox bidirectional propagation")
                method = _get_bbox_propagation_method_name()
                initial_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                try:
                    masks_dict = propagate_mask_with_sam2(
                        video_path=video_path,
                        key_frame_idx=key_frame_idx,
                        initial_mask=initial_mask,
                        config=config,
                        bbox=bbox,
                        num_gpus=num_gpus,
                        stage1_meta=stage1_meta,
                        qwen_key_frame_path=qwen_key_frame_path,
                    )
                    if masks_dict is None or len(masks_dict) == 0:
                        masks_dict = None
                    else:
                        print(f"  ✓ Bbox propagation successful, generated {len(masks_dict)} masks")
                except Exception as e:
                    error_msg = str(e)
                    if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower() or "killed" in error_msg.lower():
                        print(f"  ✗ OOM during bbox propagation: {e}")
                        masks_dict = None
                    else:
                        raise
        
        elif qwen_success and not sam3_success:
            # Case 2: no tubes + bbox → bbox bidirectional propagation
            print(f"  Case 2: No tubes + Qwen bbox → bbox bidirectional propagation")
            method = _get_bbox_propagation_method_name()
            initial_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            try:
                masks_dict = propagate_mask_with_sam2(
                    video_path=video_path,
                    key_frame_idx=key_frame_idx,
                    initial_mask=initial_mask,
                    config=config,
                    bbox=bbox,
                    num_gpus=num_gpus,
                    stage1_meta=stage1_meta,
                    qwen_key_frame_path=qwen_key_frame_path,
                )
                if masks_dict is None or len(masks_dict) == 0:
                    masks_dict = None
                else:
                    print(f"  ✓ Bbox propagation successful, generated {len(masks_dict)} masks")
            except Exception as e:
                error_msg = str(e)
                if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower() or "killed" in error_msg.lower():
                    print(f"  ✗ OOM during bbox propagation: {e}")
                    masks_dict = None
                else:
                    raise
        
        else:
            # No bbox or other: failure
            print(f"  Failure: no Qwen bbox (or no tubes and no bbox)")
            method = 'no_output'
            masks_dict = None
        
        # Determine output directory based on sentence type
        # Structure: {base_vid}/{idx} instead of {vid}
        # output_dir is already output_root (contains prefix)
        
        if idx:
            stage3_output = os.path.join(output_dir, stage3_subdir, base_vid, str(idx))
        else:
            stage3_output = os.path.join(output_dir, stage3_subdir, base_vid)
        
        # Save results if we have masks
        # Check for empty masks or only 1 mask (propagation failed)
        num_masks = len(masks_dict) if masks_dict is not None else 0
        if masks_dict is None or num_masks == 0:
            print(f"\n✗ No masks generated")
            failure_reason = 'no_masks_generated'
        elif num_masks == 1:
            print(f"\n✗ Only 1 mask generated (propagation failed, expected multiple frames)")
            # Treat as failure - set masks_dict to None to trigger failure handling
            masks_dict = None
            failure_reason = 'insufficient_masks_propagation_failed'
        else:
            failure_reason = None  # Success case
        
        if masks_dict is None or (masks_dict is not None and len(masks_dict) == 0):
            
            # Record failure in metadata to prevent redundant processing
            os.makedirs(stage3_output, exist_ok=True)
            
            # Determine failure reason based on method and context (if not already set)
            if failure_reason is None:
                failure_reason = 'no_masks_generated'
            if method and ('tracker' in method.lower() or 'propagation' in method.lower()):
                # If we were trying to use tracker propagation and got empty result, likely OOM
                if failure_reason == 'no_masks_generated':
                    failure_reason = 'oom_or_propagation_failed'
            elif not sam3_success and qwen_success:
                # Case 2: SAM3 failed but qwen succeeded, likely OOM during tracker
                if failure_reason == 'no_masks_generated':
                    failure_reason = 'oom_or_propagation_failed'
            
            failure_metadata = {
                'status': 'failed',
                'vid': vid,
                'base_vid': base_vid,
                'idx': idx,
                'sentence_type': sentence_type,
                'failure_reason': failure_reason,
                'key_frame_idx': int(key_frame_idx),
                'text_query': text_query,
                'prompt': prompt,
                'qwen_success': qwen_success,
                'sam3_success': sam3_success,
                'has_bbox': bbox is not None,
                'video_path': video_path,
                'method': method if 'method' in locals() and method is not None else 'unknown',
                'total_frames': int(total_frames),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            failure_metadata_path = os.path.join(stage3_output, "metadata.json")
            with open(failure_metadata_path, 'w') as f:
                json.dump(failure_metadata, f, indent=2)
            print(f"  ✓ Recorded failure to {failure_metadata_path} (to prevent redundant processing)")
            
            return None
        
        masks_dir = os.path.join(stage3_output, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        for frame_idx, mask in masks_dict.items():
            if mask.ndim > 2:
                mask = mask.squeeze()
            mask_file = os.path.join(masks_dir, f"mask_{frame_idx:05d}.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_file)
        
        # Generate visualization video (skip if --no-visualization)
        output_video = None
        if not skip_visualization:
            if idx:
                output_video = os.path.join(stage3_output, f"{base_vid}_{idx}_stage3_visualization.mp4")
            else:
                output_video = os.path.join(stage3_output, f"{base_vid}_stage3_visualization.mp4")
            
            if viz_manager is not None:
                print(f"\n→ Submitting stage3 visualization to background...")
                viz_manager.submit(
                    generate_mask_video,
                    video_path, masks_dict, output_video,
                    mask_color=(128, 0, 128), mask_alpha=0.5
                )
            else:
                print(f"\nGenerating visualization video...")
                generate_mask_video(video_path, masks_dict, output_video,
                                  mask_color=(128, 0, 128), mask_alpha=0.5)
        else:
            print(f"\n⊘ Skipping visualization (--no-visualization)")
        
        # Save metadata
        # Convert all integers to Python native int to avoid JSON serialization issues
        metadata = {
            'status': 'success',  # Mark as successful
            'vid': vid,
            'base_vid': base_vid,
            'idx': idx,
            'sentence_type': sentence_type,
            'key_frame_idx': int(key_frame_idx),
            'text_query': text_query,
            'prompt': prompt,
            'num_frames': int(total_frames),
            'num_masks': int(len(masks_dict)),
            'method': method,
            'threshold': float(threshold),
            'qwen_success': bool(qwen_success),
            'sam3_success': bool(sam3_success),
            'selected_tube_id': int(selected_tube_id) if selected_tube_id is not None else None,
            'selected_score': float(selected_score) if selected_score is not None else None,
            'selection_mode': selection_mode,
            'output_video': output_video,
            'sam3_seg_dir': sam3_seg_dir,
            'mask_color': 'purple',
            'mask_alpha': 0.5,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(os.path.join(stage3_output, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Stage 3 completed successfully!")
        print(f"  Output: {stage3_output}")
        print(f"  Method: {method}")
        if selected_tube_id is not None:
            if selected_score is not None:
                print(f"  Selected Tube: {selected_tube_id}, score: {selected_score:.4f}")
            else:
                print(f"  Selected Tube: {selected_tube_id} (no score)")
        
        clear_gpu_memory()
        
        return metadata
        
    except RuntimeError as e:
        error_msg = str(e)
        
        # Determine output directory for failure recording
        # output_dir is already output_root (contains prefix)
        
        # Parse vid to get base_vid and idx (need to get from outer scope or re-parse)
        try:
            if "_" in vid:
                base_vid, idx = vid.rsplit("_", 1)
            else:
                base_vid = vid
                idx = None
            
            if idx:
                stage3_output = os.path.join(output_dir, stage3_subdir, base_vid, str(idx))
            else:
                stage3_output = os.path.join(output_dir, stage3_subdir, base_vid)
            
            os.makedirs(stage3_output, exist_ok=True)
            
            # Record failure
            is_oom = "out of memory" in error_msg.lower() or "cuda" in error_msg.lower()
            if is_oom:
                print(f"\n✗ Stage 3 CUDA OOM error: {e}")
                print("  Cleaning up GPU resources...")
                failure_reason = 'cuda_oom'
            else:
                print(f"\n✗ Stage 3 failed (RuntimeError): {e}")
                failure_reason = 'runtime_error'
            
            failure_metadata = {
                'status': 'failed',
                'vid': vid,
                'base_vid': base_vid,
                'idx': idx,
                'sentence_type': sentence_type,
                'failure_reason': failure_reason,
                'error_type': 'RuntimeError',
                'error_message': str(e),
                'is_oom': is_oom,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save failure metadata
            failure_metadata_path = os.path.join(stage3_output, "metadata.json")
            with open(failure_metadata_path, 'w') as f:
                json.dump(failure_metadata, f, indent=2)
            print(f"  ✓ Recorded failure to {failure_metadata_path}")
        except Exception as record_error:
            print(f"  ⚠️  Warning: Failed to record failure metadata: {record_error}")
        
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return None
        
    except Exception as e:
        error_msg = str(e)
        
        # Determine output directory for failure recording
        # output_dir is already output_root (contains prefix)
        
        # Parse vid to get base_vid and idx (need to get from outer scope or re-parse)
        try:
            if "_" in vid:
                base_vid, idx = vid.rsplit("_", 1)
            else:
                base_vid = vid
                idx = None
            
            if idx:
                stage3_output = os.path.join(output_dir, stage3_subdir, base_vid, str(idx))
            else:
                stage3_output = os.path.join(output_dir, stage3_subdir, base_vid)
            
            os.makedirs(stage3_output, exist_ok=True)
            
            # Record failure
            is_oom = "out of memory" in error_msg.lower() or "cuda" in error_msg.lower()
            if is_oom:
                print(f"\n✗ Stage 3 CUDA OOM error: {e}")
                print("  Cleaning up GPU resources...")
                failure_reason = 'cuda_oom'
            else:
                print(f"\n✗ Stage 3 failed: {e}")
                failure_reason = 'general_exception'
            
            failure_metadata = {
                'status': 'failed',
                'vid': vid,
                'base_vid': base_vid,
                'idx': idx,
                'sentence_type': sentence_type,
                'failure_reason': failure_reason,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'is_oom': is_oom,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save failure metadata
            failure_metadata_path = os.path.join(stage3_output, "metadata.json")
            with open(failure_metadata_path, 'w') as f:
                json.dump(failure_metadata, f, indent=2)
            print(f"  ✓ Recorded failure to {failure_metadata_path}")
        except Exception as record_error:
            print(f"  ⚠️  Warning: Failed to record failure metadata: {record_error}")
        
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        return None


def _apply_gpu_arg_early():
    """If unset, set CUDA_VISIBLE_DEVICES from --gpu / --gpu=N (before first CUDA use)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--gpu" and i + 1 < len(argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(argv[i + 1]).strip()
            return
        if a.startswith("--gpu="):
            os.environ["CUDA_VISIBLE_DEVICES"] = a.split("=", 1)[1].strip()
            return
        i += 1


def main():
    _apply_gpu_arg_early()
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done at the very beginning, before any multiprocessing operations
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set (e.g., in a subprocess), ignore
        pass
    
    parser = argparse.ArgumentParser(
        description='Stage 3: Video Segmentation (SAM3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific video (declarative)
  python model/stage3_sam3_savg.py --video-id 2400171624_1 --sentence-type declarative
  
  # Process specific video (interrogative)
  python model/stage3_sam3_savg.py --video-id 2400171624_1 --sentence-type interrogative
  
  # Process specific video (savg)
  python model/stage3_sam3_savg.py --video-id all-terrain_vehicle_8_3 --sentence-type savg
  
  # Process first 10 declarative videos
  python model/stage3_sam3_savg.py --sentence-type declarative --num 10
  
  # Process all interrogative videos
  python model/stage3_sam3_savg.py --sentence-type interrogative
  
  # Process all savg videos
  python model/stage3_sam3_savg.py --sentence-type savg
   python model/stage3_sam3_qwen.py --sentence-type savg --retry-failed --no-visualization --num-gpus 4

        """
    )
    
    parser.add_argument('--video-id', type=str, help='Video ID (e.g., 2400171624_1)')
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=list(SUPPORTED_DATASETS),
        help='Dataset key (recommended): vidstg_declarative / hcstvg_v1 / hcstvg_v2 / savg',
    )
    parser.add_argument(
        '--sentence-type',
        type=str,
        default=None,
        choices=['declarative', 'interrogative', 'savg'],
        help='Deprecated legacy switch; use --dataset instead.',
    )
    parser.add_argument('--num', type=int, default=None, help='Number of videos (default: all)')
    parser.add_argument('--base-dir', type=str, default='model', 
                        help='Base directory containing vidstg_declarative/interrogative/savg')
    parser.add_argument('--output-dir', type=str, default='model', 
                        help='Output directory (default: model)')
    parser.add_argument('--config', type=str, default='model/config.yaml', help='Config file')
    parser.add_argument('--selection-mode', type=str, default='absolute',
                        choices=['relative', 'absolute'],
                        help='Tube selection mode: relative (intersection/mask_size) or '
                             'absolute (intersection/bbox_size). Default: relative')
    parser.add_argument('--each-tube-visualization', action='store_true', default=False,
                        help='Generate individual visualization video for each tube ID')
    parser.add_argument('--num-gpus', type=int, default=1,
                        help='Number of GPUs to use for parallel processing (default: 1)')
    parser.add_argument(
        '--gpu',
        type=int,
        default=None,
        metavar='ID',
        help='Physical GPU id (sets CUDA_VISIBLE_DEVICES if not already set). Process sees it as cuda:0.',
    )
    parser.add_argument('--gpu-delay', type=float, default=2.0,
                        help='Delay between GPU model loading to avoid channel congestion (seconds, default: 2.0)')
    parser.add_argument('--tube-bidirectional-expand', action='store_true', default=False,
                        help='If enabled, extend existing tubes using bidirectional propagation instead of using bbox tracker. Original text prompt segmentation masks are preserved. (default: False)')
    parser.add_argument('--retry-failed', action='store_true', default=False,
                        help='Retry failed tasks (especially no_masks_generated). Default: False')
    parser.add_argument('--no-visualization', action='store_true', default=False,
                        help='Disable all video visualization generation to save memory and time. Default: False')
    parser.add_argument(
        '--stage2-subdir',
        '--stage2-dir-name',
        dest='stage2_subdir',
        type=str,
        default='stage2',
        help='Subfolder under each clip for stage2 artifacts (default: stage2; e.g. stage2_qwen2.5_7b)',
    )
    parser.add_argument(
        '--stage3-subdir',
        '--stage3-dir-name',
        dest='stage3_subdir',
        type=str,
        default='stage3',
        help='Subfolder under output_root for stage3 writes (default: stage3; e.g. stage3_qwen2.5_7b_sam3_propagation)',
    )
    parser.add_argument(
        '--sam3-prompt',
        type=str,
        default=None,
        help='Override SAM3/metadata prompt string (e.g. car). Box-prompt path still uses GT bbox.',
    )
    parser.add_argument(
        '--input-root',
        type=str,
        default=None,
        help='Absolute path to dataset root containing stage1/ (overrides config/base_dir for reads)',
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default=None,
        help='Absolute path to write stage3 under (overrides config for writes)',
    )
    parser.add_argument(
        '--stage4-subdir',
        type=str,
        default='stage4',
        help=(
            'Subfolder under input_root for Stage4 bbox metadata '
            '(default: stage4; ZeroOV e.g. stage4_zeroov_qwen2.5_7b). '
            'Also exported to env STAGE3_STAGE4_SUBDIR for worker processes.'
        ),
    )
    
    args = parser.parse_args()
    os.environ["STAGE3_STAGE4_SUBDIR"] = str(args.stage4_subdir or "stage4").strip()
    args.dataset, args.sentence_type = _resolve_dataset_and_sentence_type(
        args.dataset,
        args.sentence_type,
    )
    print(f"Dataset: {args.dataset} (sentence_type={args.sentence_type})")
    if args.gpu is not None and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))

    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    
    # Ensure config is a plain dict for multiprocessing serialization
    if not isinstance(config, dict):
        config = dict(config) if hasattr(config, '__dict__') else config
    
    # Create visualization manager for async CPU tasks (skip if --no-visualization)
    if args.no_visualization:
        viz_manager = None
        print("Note: Visualization disabled (--no-visualization)")
    else:
        viz_manager = VisualizationManager(max_workers=2)
    
    try:
        if args.video_id:
            # Single video mode
            # Resolve input and output directories (same as batch mode)
            prefix, input_root, output_root = resolve_io_dirs(
                dataset=args.dataset,
                base_dir=args.base_dir,
                output_dir=args.output_dir,
                config=config,
                input_root_override=args.input_root,
                output_root_override=args.output_root,
            )
            
            result = process_stage3(
                base_dir=input_root,  # Use resolved input_root instead of args.base_dir
                vid=args.video_id,
                sentence_type=args.sentence_type,
                dataset=args.dataset,
                output_dir=output_root,  # Use resolved output_root instead of args.output_dir
                config=config,
                selection_mode=args.selection_mode,
                each_tube_visualization=args.each_tube_visualization,
                viz_manager=viz_manager,
                tube_bidirectional_expand=args.tube_bidirectional_expand,
                retry_failed=args.retry_failed,
                num_gpus=args.num_gpus,
                skip_visualization=args.no_visualization,
                stage2_subdir=args.stage2_subdir,
                stage3_subdir=args.stage3_subdir,
                sam3_prompt_override=args.sam3_prompt,
            )
        
            if result:
                print(f"\n{'='*80}")
                print("Stage 3 Completed Successfully!")
                print(f"{'='*80}")
            else:
                print(f"\n{'='*80}")
                print("Stage 3 Failed!")
                print(f"{'='*80}")
            
            # Wait for visualization tasks to complete (skip if --no-visualization)
            if viz_manager is not None:
                print(f"\nWaiting for visualization tasks to complete...")
                results = viz_manager.wait_all()
                success_count = sum(1 for success, _, _ in results if success)
                fail_count = len(results) - success_count
                if len(results) > 0:
                    print(f"Visualization: {success_count} successful, {fail_count} failed")
                    for success, task_name, result in results:
                        if not success:
                            print(f"  ✗ {task_name} failed: {result}")
            else:
                print(f"\n⊘ Skipping visualization (--no-visualization)")
        
        else:
            # Batch mode
            # Resolve input and output directories
            prefix, input_root, output_root = resolve_io_dirs(
                dataset=args.dataset,
                base_dir=args.base_dir,
                output_dir=args.output_dir,
                config=config,
                input_root_override=args.input_root,
                output_root_override=args.output_root,
            )
            
            # Check stage1 directory (scan from stage1 to include videos without stage2)
            # This allows Case 3/4 to handle videos where stage2 failed
            stage1_base = os.path.join(input_root, "stage1")
            if not os.path.exists(stage1_base):
                # Try output_root as fallback
                stage1_base = os.path.join(output_root, "stage1")
                if not os.path.exists(stage1_base):
                    print(f"✗ Stage 1 directory not found in either:")
                    print(f"    {os.path.join(input_root, 'stage1')}")
                    print(f"    {os.path.join(output_root, 'stage1')}")
                    return
                else:
                    print(f"✓ Found Stage 1 directory in output_root: {stage1_base}")
            else:
                print(f"✓ Found Stage 1 directory in input_root: {stage1_base}")
            
            print(f"Scanning for Stage 1 outputs in {stage1_base}...")
            
            video_ids = []
            # Scan stage1 directory structure: stage1/{base_vid}/{idx}/
            for base_vid in os.listdir(stage1_base):
                base_vid_path = os.path.join(stage1_base, base_vid)
                if not os.path.isdir(base_vid_path):
                    continue
                
                # Check for subdirectories (idx)
                subitems = [s for s in os.listdir(base_vid_path) if s.isdigit()]
                if subitems:
                    for idx in sorted(subitems, key=int):
                        meta_path = os.path.join(base_vid_path, idx, "metadata.json")
                        if os.path.isfile(meta_path):
                            video_ids.append(f"{base_vid}_{idx}")
                else:
                    # Root folder case
                    meta_path = os.path.join(base_vid_path, "metadata.json")
                    if os.path.isfile(meta_path):
                        video_ids.append(base_vid)
            
            if not video_ids:
                print(f"✗ No Stage 1 outputs found")
                return
            
            # Filter out already completed videos
            print(f"Checking for already completed videos...")
            if args.retry_failed:
                print(f"  ↻ Retry mode enabled: will retry failed tasks")
            remaining_video_ids = []
            skipped_count = 0
            for vid in video_ids:
                # Use output_root instead of args.base_dir for checking completion
                is_completed, _ = check_stage3_completed(
                    output_root,
                    vid,
                    args.sentence_type,
                    retry_failed=args.retry_failed,
                    stage3_subdir=args.stage3_subdir,
                )
                if is_completed:
                    skipped_count += 1
                    # Don't print "already completed" for retried tasks
                    if not args.retry_failed:
                        print(f"  ⊘ Skipping {vid} (already completed)")
                else:
                    remaining_video_ids.append(vid)
            
            if skipped_count > 0:
                print(f"  ✓ Skipped {skipped_count} already completed videos")
            
            if not remaining_video_ids:
                print(f"✓ All videos already completed!")
                return
            
            # Limit if specified
            remaining_video_ids = sorted(remaining_video_ids)
            if args.num:
                remaining_video_ids = remaining_video_ids[:args.num]
            
            print(f"✓ Processing {len(remaining_video_ids)} videos" + (" (all)" if args.num is None else ""))
            
            # Multi-GPU parallel processing
            if args.num_gpus > 1 and len(remaining_video_ids) > 1:
                print(f"\nUsing {args.num_gpus} GPUs for parallel processing...")
                print(f"GPU loading delay: {args.gpu_delay}s (to avoid channel congestion)")
                
                mp_config = copy.deepcopy(config) if isinstance(config, dict) else config
                success_count, fail_count = process_videos_multi_gpu(
                    video_ids=remaining_video_ids,
                    base_dir=input_root,  # Use resolved input_root instead of args.base_dir
                    sentence_type=args.sentence_type,
                    dataset=args.dataset,
                    output_dir=output_root,  # Use resolved output_root instead of args.output_dir
                    config=mp_config,
                    selection_mode=args.selection_mode,
                    each_tube_visualization=args.each_tube_visualization,
                    num_gpus=args.num_gpus,
                    gpu_delay=args.gpu_delay,
                    tube_bidirectional_expand=args.tube_bidirectional_expand,
                    retry_failed=args.retry_failed,
                    skip_visualization=args.no_visualization,
                    stage2_subdir=args.stage2_subdir,
                    stage3_subdir=args.stage3_subdir,
                    sam3_prompt_override=args.sam3_prompt,
                )
            else:
                # Single GPU sequential processing
                success_count = 0
                fail_count = 0
                
                for idx, vid in enumerate(remaining_video_ids, 1):
                    print(f"\n{'='*80}")
                    print(f"Processing {idx}/{len(remaining_video_ids)}: {vid}")
                    print(f"{'='*80}")
                    
                    # Use input_root as base_dir for process_stage3 (it will handle paths correctly)
                    result = process_stage3(
                        base_dir=input_root,  # Use resolved input_root instead of args.base_dir
                        vid=vid,
                        sentence_type=args.sentence_type,
                        dataset=args.dataset,
                        output_dir=output_root,  # Use resolved output_root instead of args.output_dir
                        config=config,
                        selection_mode=args.selection_mode,
                        each_tube_visualization=args.each_tube_visualization,
                        viz_manager=viz_manager,
                        tube_bidirectional_expand=args.tube_bidirectional_expand,
                        retry_failed=args.retry_failed,
                        num_gpus=args.num_gpus,
                        skip_visualization=args.no_visualization,
                        stage2_subdir=args.stage2_subdir,
                        stage3_subdir=args.stage3_subdir,
                        sam3_prompt_override=args.sam3_prompt,
                    )
                    
                    if result:
                        success_count += 1
                    else:
                        fail_count += 1
            
            print(f"\n{'='*80}")
            print("Batch Processing Complete")
            print(f"{'='*80}")
            print(f"Total: {len(remaining_video_ids)}")
            print(f"Success: {success_count}")
            print(f"Failed: {fail_count}")
            if len(remaining_video_ids) > 0:
                print(f"Success rate: {success_count/len(remaining_video_ids)*100:.2f}%")
            print(f"{'='*80}")
            
            # Wait for all visualization tasks to complete (skip if --no-visualization)
            if viz_manager is not None:
                print(f"\nWaiting for all visualization tasks to complete...")
                results = viz_manager.wait_all()
                viz_success_count = sum(1 for success, _, _ in results if success)
                viz_fail_count = len(results) - viz_success_count
                if len(results) > 0:
                    print(f"Visualization tasks: {viz_success_count} successful, {viz_fail_count} failed")
                    for success, task_name, result in results:
                        if not success:
                            print(f"  ✗ {task_name} failed: {result}")
            else:
                print(f"\n⊘ Skipping visualization (--no-visualization)")
    
    finally:
        # Shutdown visualization manager (skip if --no-visualization)
        if viz_manager is not None:
            viz_manager.shutdown(wait=True)


if __name__ == '__main__':
    main()

