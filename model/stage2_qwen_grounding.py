#!/usr/bin/env python3
"""
Stage 2: Direct Visual Grounding using Qwen VL models (vLLM Accelerated)

This stage uses Qwen3-VL to directly predict bounding boxes from text queries.
Refactored to use vLLM for high-performance batch inference and tensor parallelism.

Usage:
    python model/stage2_qwen_grounding.py --output-dir model/output/savg --num-gpus 4 --batch-size 100
"""

import os
import sys
import json
import argparse
import re
import base64
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from PIL import Image
import cv2
import torch
import torch._dynamo as torch_dynamo
import gc

# vLLM Imports
from vllm import LLM, SamplingParams

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


def normalize_dataset_key(dataset: str) -> str:
    key = DATASET_ALIASES.get((dataset or "").strip())
    if key is None:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            f"Expected one of: {', '.join(SUPPORTED_DATASETS)}"
        )
    return key


def is_interrogative(text: str) -> bool:
    """Check if the text query is an interrogative sentence."""
    text = text.strip()
    if text.endswith('?'):
        return True
    question_words = ['what', 'where', 'who', 'which', 'when', 'how', 'why', 'whose', 'whom']
    text_lower = text.lower()
    for qw in question_words:
        if text_lower.startswith(qw + ' ') or text_lower.startswith(qw + '?'):
            return True
    return False


def parse_qwen_answer_and_bbox(
    text: str,
    image_width: int,
    image_height: int,
    is_question: bool = False,
    coord_mode: str = "auto",
) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]]]:
    """Parse Qwen3-VL output to extract answer (if any) and bbox."""
    answer = None
    bbox = None
    
    # Extract answer (only for questions)
    if is_question:
        match_ans = re.search(r'[Aa]nswer[:\s]+(.*?)(?:\n|Reasoning|Target|$)', text, re.IGNORECASE | re.DOTALL)
        if match_ans:
            raw_answer = match_ans.group(1).strip()
            stopwords = {
                'the', 'a', 'an', 'this', 'that', 'these', 'those',
                'another', 'other', 'others', 'one', 'two', 'three', 'four', 'five',
                'six', 'seven', 'eight', 'nine', 'ten',
                'yes', 'no', 'none', 'target', 'object', 'subject',
                'some', 'many', 'few', 'several', 'all', 'both',
                'first', 'second', 'third', 'last', 'next', 'previous'
            }
            words = re.sub(r'[^\w\s]', '', raw_answer.lower()).split()
            while words and words[0] in stopwords:
                words.pop(0)
            if words:
                answer = words[-1] if words[-1] not in stopwords else (words[-2] if len(words) > 1 else None)
    else:
        match_ans = re.search(r'[Aa]nswer[:\s]+([A-Za-z]+)', text)
        if match_ans:
            answer = match_ans.group(1).strip()
    
    bbox = parse_qwen_bbox_output(text, image_width, image_height, coord_mode=coord_mode)
    return answer, bbox


def parse_qwen_bbox_output(
    text: str,
    image_width: int,
    image_height: int,
    coord_mode: str = "auto",
) -> Optional[Tuple[int, int, int, int]]:
    """Universal parser for Qwen output coordinates."""
    text_lower = text.lower()
    if 'target: none' in text_lower or 'target:none' in text_lower:
        none_pattern = re.search(r'[Tt]arget\s*:\s*[Nn]one', text)
        if none_pattern:
            return None
    
    pattern_tokens = r'(?:<\|box_start\|\>|<tool_call>)\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*<\|box_end\|\>'
    pattern_plain = r'\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)'
    pattern_target = r'[Tt]arget\s*:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)'
    pattern_bracket = r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]'
    
    matches = list(re.finditer(pattern_tokens, text))
    if not matches: matches = list(re.finditer(pattern_plain, text))
    if not matches: matches = list(re.finditer(pattern_target, text))
    if not matches: matches = list(re.finditer(pattern_bracket, text))
    
    if matches:
        last_match = matches[-1]
        try:
            x1, y1, x2, y2 = map(int, last_match.groups())
            # coord_mode:
            # - "pixel": always treat as pixel coordinates (no scaling)
            # - "normalized": always treat as 0-1000 normalized coordinates
            # - "auto": keep backward-compatible behavior
            if coord_mode == "pixel":
                return (x1, y1, x2, y2)
            if coord_mode == "normalized":
                xmin = int(x1 / 1000.0 * image_width)
                ymin = int(y1 / 1000.0 * image_height)
                xmax = int(x2 / 1000.0 * image_width)
                ymax = int(y2 / 1000.0 * image_height)
                return (xmin, ymin, xmax, ymax)

            # auto mode (backward compatible)
            if max(x1, y1, x2, y2) <= 1000:
                xmin = int(x1 / 1000.0 * image_width)
                ymin = int(y1 / 1000.0 * image_height)
                xmax = int(x2 / 1000.0 * image_width)
                ymax = int(y2 / 1000.0 * image_height)
                return (xmin, ymin, xmax, ymax)
            else:
                return (x1, y1, x2, y2)
        except ValueError:
            return None
    return None


def get_image_base64_and_mime(image_path: str) -> Tuple[Optional[str], str]:
    """Convert image file to base64 string and get MIME type."""
    try:
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")
        
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_data, mime_type
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None, "image/jpeg"


def determine_sentence_type(output_dir: str) -> Optional[str]:
    """Determine sentence type from output_dir path."""
    output_dir = os.path.abspath(output_dir)
    if 'vidstg_declarative' in output_dir:
        return 'declarative'
    elif 'vidstg_interrogative' in output_dir:
        return 'interrogative'
    return None


def _sorted_image_filenames_in_dir(video_path: str) -> List[str]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = [f for f in os.listdir(video_path) if os.path.splitext(f)[1].lower() in exts]
    return sorted(files)


def resolve_key_frame_path(stage1_dir: str) -> Optional[str]:
    """Return key_frame.png path, materializing from stage1 metadata when missing."""
    key_frame_path = os.path.join(stage1_dir, "key_frame.png")
    if os.path.isfile(key_frame_path):
        return key_frame_path

    meta_path = os.path.join(stage1_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    video_path = meta.get("video_path")
    frame_idx = meta.get("original_frame_idx", meta.get("key_frame_idx"))
    if not video_path or frame_idx is None:
        return None

    try:
        frame_idx = int(frame_idx)
        if os.path.isdir(video_path):
            frame_files = _sorted_image_filenames_in_dir(video_path)
            if frame_idx < 0 or frame_idx >= len(frame_files):
                return None
            src = os.path.join(video_path, frame_files[frame_idx])
            img = Image.open(src)
        else:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return None
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(key_frame_path)
        return key_frame_path
    except Exception as e:
        print(f"  Warning: Failed to materialize key_frame for {stage1_dir}: {e}")
        return None


def build_vllm_messages(image_path: str, text_query: str) -> List[Dict]:
    """
    Constructs the prompt and message list for vLLM based on the CoT logic.
    """
    is_question = is_interrogative(text_query)
    
    if is_question:
        prompt = f"""Task: Answer the question and locate the target object: "{text_query}"

Steps:
1. **Answer**: Provide a SINGLE NOUN only (no articles like "the", "a", "an", no numbers like "two", "another").
   - Correct examples: "boy", "dog", "toy", "person"
   - Wrong examples: "the boy", "a dog", "another person", "two dogs"
2. **Analysis**: Briefly describe the visual features of the target object (color, clothing, location) to distinguish it from others.
3. **Locate**: Output the bounding box.

STRICT FORMAT (Start directly with "Answer:"):
Answer: [single noun only, no articles or numbers]
Reasoning: [analysis]
Target: <|box_start|>(x1,y1),(x2,y2)<|box_end|>

CRITICAL: Output only the noun word after "Answer:", do NOT include "the", "a", "an", "another", "other", "two", etc.

Start your response immediately with "Answer:".
"""
    else:
        prompt = f"""Task: Locate the object described as: "{text_query}"

CRITICAL INSTRUCTION:

1. **Core Subject Check (Strict)**: First, identify the main noun (the Subject, e.g., "boy", "car"). 

   - If the image does **NOT** contain this Subject at all, output "Target: None".

2. **Attribute Matching (Loose)**: If the Subject exists, you **MUST** locate it, even if the description is wrong.

   - Ignore wrong colors (e.g., text "red shirt", image "blue shirt" -> Select the person).
   - Ignore wrong actions (e.g., text "running", image "standing" -> Select the person).
   - Ignore missing items (e.g., text "holding toy", image "empty handed" -> Select the person).

Steps:
1. Extract the Core Subject from "{text_query}".
2. Does this Subject exist in the image? 
   - NO -> Output "Target: None".
   - YES -> Find the instance that is closest to the description (ignoring errors) and output the box.

Format:
Reasoning: [Analyze subject existence first, then attributes]
Target: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
OR
Target: None
"""

    # Convert image to base64 for vLLM
    base64_image, mime_type = get_image_base64_and_mime(image_path)
    if base64_image is None:
        raise ValueError(f"Failed to convert image to base64: {image_path}")
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]
    return messages


def check_stage2_completed(output_dir: str, vid_id: str, stage2_dir_name: str = "stage2") -> bool:
    """
    Check if Stage 2 output already exists for this video/sentence.
    Returns True if metadata.json exists (completed), False otherwise.
    """
    if "_" in vid_id:
        base_vid, idx = vid_id.rsplit("_", 1)
    else:
        base_vid = vid_id
        idx = None
    stage2_base = os.path.join(output_dir, stage2_dir_name)
    if idx:
        meta_path = os.path.join(stage2_base, base_vid, idx, "metadata.json")
    else:
        meta_path = os.path.join(stage2_base, base_vid, "metadata.json")
    return os.path.isfile(meta_path)


def save_result(output_dir: str, vid_id: str, text_query: str, target_object: str, 
                bbox: Tuple, answer: str, raw_output: str, image_path: str,
                stage2_dir_name: str = "stage2", method_name: str = "qwen3_vl_cot_vllm"):
    """Saves the result metadata and visualization to disk."""
    
    # Parse vid to get base_vid and idx
    if "_" in vid_id:
        base_vid, idx = vid_id.rsplit("_", 1)
    else:
        base_vid = vid_id
        idx = None
    
    stage2_base = os.path.join(output_dir, stage2_dir_name)
    if idx:
        final_output_dir = os.path.join(stage2_base, base_vid, idx)
    else:
        final_output_dir = os.path.join(stage2_base, base_vid)
    
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Copy key frame
    try:
        from shutil import copyfile
        copyfile(image_path, os.path.join(final_output_dir, "key_frame.png"))
    except Exception as e:
        print(f"  Warning: Failed to copy image: {e}")

    metadata = {
        'vid': vid_id,
        'text_query': text_query,
        'target_object': target_object,
        'bbox': None,
        'method': method_name,
        'success': False,
        'failure_reason': None,
        'raw_output': raw_output if raw_output else None
    }

    if bbox is None:
        failure_reason = "bbox_parsing_failed"
        if raw_output:
            text_lower = raw_output.lower()
            if 'target: none' in text_lower or 'target:none' in text_lower:
                failure_reason = "subject_not_found"
            elif 'subject does not exist' in text_lower or 'no subject' in text_lower:
                failure_reason = "subject_not_found"
            elif 'cannot find' in text_lower or 'not found' in text_lower:
                failure_reason = "subject_not_found"
            else:
                failure_reason = "bbox_not_detected"
        
        metadata['failure_reason'] = failure_reason
        with open(os.path.join(final_output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    else:
        # Visualization
        xmin, ymin, xmax, ymax = bbox
        try:
            img = cv2.imread(image_path)
            if img is not None:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(final_output_dir, "bbox_visualization.png"), img)
        except Exception as e:
            print(f"  Warning: Failed to create visualization: {e}")

        metadata['bbox'] = {
            'xmin': int(xmin), 'ymin': int(ymin),
            'xmax': int(xmax), 'ymax': int(ymax)
        }
        metadata['success'] = True
        if answer:
            metadata['answer'] = answer
        
        with open(os.path.join(final_output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def process_batch_vllm(
    tasks: List[Tuple],
    output_dir: str,
    config: Dict,
    num_gpus: int = 1,
    batch_size: int = 1000,
    qwen_model_path: Optional[str] = None,
    stage2_dir_name: str = "stage2",
    method_name: str = "qwen3_vl_cot_vllm",
    model_key: Optional[str] = None,
):
    """
    Main processing loop using vLLM with chunked batch processing to avoid OOM.
    Args:
        tasks: List of (image_path, text_query, target_object, vid_id)
        batch_size: Number of prompts to process in each batch (default: 1000)
    """
    if not tasks:
        print("No tasks to process.")
        return

    print(f"\n{'='*80}")
    print(f"Initializing vLLM Engine on {num_gpus} GPUs (Tensor Parallelism)...")
    print(f"{'='*80}")

    # Resolve Qwen model path & max_model_len
    qwen_cfg = config.get('qwen_grounding', {})
    if qwen_model_path is None:
        qwen_model_path = qwen_cfg.get(
            'model_path',
            '/home/xdu/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct'
        )
    max_model_len = int(qwen_cfg.get('max_model_len', 8192))

    # 禁用 TorchDynamo 编译，避免 tracer_output 未定义等问题，强制使用 eager 模式
    try:
        torch_dynamo.config.suppress_errors = True
        torch_dynamo.disable()
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("VLLM_TORCH_COMPILE_METHOD", "none")
    except Exception:
        pass

    # 1. Initialize Engine
    # enforce_eager=True 关闭 CUDA graph 捕获，避免在 "Capturing CUDA graphs" 阶段 OOM（尤其 4 卡 TP 时）
    # gpu_memory_utilization 适当降低，给显存留余量（可于 config 中覆盖）
    gpu_mem_util = float(qwen_cfg.get("gpu_memory_utilization", 0.85))
    try:
        llm = LLM(
            model=qwen_model_path,
            trust_remote_code=True,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            enforce_eager=True,  # 禁用 CUDA graph，避免 capture 阶段 OOM
        )
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize vLLM: {e}")
        return

    # 2. Define Sampling Parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=512,
        repetition_penalty=1.1,
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    print(f"✓ Engine loaded. Preparing {len(tasks)} prompts...")

    # 3. Build All Valid Tasks and Inputs
    valid_tasks = []
    all_inputs = []
    
    for i, (img_path, query, target, vid_id) in enumerate(tasks):
        if not os.path.exists(img_path):
            print(f"Skipping missing image: {img_path}")
            continue
            
        messages = build_vllm_messages(img_path, query)
        all_inputs.append(messages)
        valid_tasks.append((img_path, query, target, vid_id))

    if not all_inputs:
        print("No valid inputs created.")
        return

    total_tasks = len(all_inputs)
    num_batches = (total_tasks + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"\n{'='*80}")
    print(f"Processing {total_tasks} tasks in {num_batches} batches (batch_size={batch_size})")
    print(f"{'='*80}\n")

    # 4. Process in Chunks
    total_success_count = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_tasks)
        batch_inputs = all_inputs[start_idx:end_idx]
        batch_tasks = valid_tasks[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx + 1}/{num_batches}: Processing samples {start_idx+1}-{end_idx} ({len(batch_inputs)} samples)")
        print(f"{'='*60}")
        
        try:
            # Run batch inference
            outputs = llm.chat(messages=batch_inputs, sampling_params=sampling_params)
            
            # Process results for this batch
            batch_success = 0
            for i, output in enumerate(outputs):
                try:
                    generated_text = output.outputs[0].text
                except (AttributeError, IndexError) as e:
                    print(f"  Warning: Unexpected output format for sample {start_idx + i + 1}: {e}")
                    generated_text = ""
                
                img_path, query, target, vid_id = batch_tasks[i]
                
                # Get image size for bbox normalization
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                except:
                    w, h = 1920, 1080 # Fallback
                    
                is_question = is_interrogative(query)
                # Choose coordinate mode based on model_key:
                # - For Qwen2.5-VL models, treat outputs as pixel coordinates.
                # - For Qwen3-VL-8B, keep auto (0-1000 normalized or pixel fallback).
                if model_key in ("qwen2.5-vl-3b", "qwen2.5-vl-7b"):
                    coord_mode = "pixel"
                else:
                    coord_mode = "auto"
                answer, bbox = parse_qwen_answer_and_bbox(
                    generated_text,
                    w,
                    h,
                    is_question=is_question,
                    coord_mode=coord_mode,
                )
                
                # Save
                save_result(
                    output_dir,
                    vid_id,
                    query,
                    target,
                    bbox,
                    answer,
                    generated_text,
                    img_path,
                    stage2_dir_name=stage2_dir_name,
                    method_name=method_name,
                )
                
                status = "✓" if bbox else "✗"
                if bbox: 
                    batch_success += 1
                    total_success_count += 1
                
                # Print progress every 50 samples or at the end of batch
                if (i + 1) % 50 == 0 or (i + 1) == len(outputs):
                    print(f"  [{start_idx + i + 1}/{total_tasks}] {vid_id}: {status} | Query: {query[:40]}...")
            
            print(f"\n  Batch {batch_idx + 1} Complete: {batch_success}/{len(batch_tasks)} succeeded ({batch_success/len(batch_tasks)*100:.2f}%)")
            
            # Clean up batch input data (Python objects only, won't affect vLLM engine)
            del outputs
            del batch_inputs  # This contains base64 image data which can be large
            gc.collect()  # Trigger Python garbage collection
            
        except Exception as e:
            print(f"\n  ✗ Batch {batch_idx + 1} Failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Continuing with next batch...")
            # Clean up even on error
            if 'outputs' in locals():
                del outputs
            if 'batch_inputs' in locals():
                del batch_inputs
            gc.collect()
            continue

    # 5. Final Summary
    print(f"\n{'='*80}")
    print(f"All Batches Complete")
    print(f"{'='*80}")
    print(f"Total Processed: {total_tasks}")
    print(f"Total Success: {total_success_count}")
    print(f"Total Failed: {total_tasks - total_success_count}")
    if total_tasks > 0:
        print(f"Overall Success Rate: {total_success_count}/{total_tasks} ({total_success_count/total_tasks*100:.2f}%)")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Qwen VL Visual Grounding (vLLM Accelerated)')
    parser.add_argument('--video-id', type=str, help='Video ID from test set')
    parser.add_argument('--image-path', type=str, help='Direct path to image file')
    parser.add_argument('--text-query', type=str, help='Text query')
    parser.add_argument('--num', type=int, default=None, help='Number of videos to process')
    parser.add_argument(
        '--dataset',
        type=str,
        default='vidstg_declarative',
        choices=['vidstg_declarative', 'hcstvg_v1', 'hcstvg_v2', 'savg', 'vidstg', 'hcstvg-v1', 'hcstvg-v2'],
        help='Dataset key (recommended): vidstg_declarative / hcstvg_v1 / hcstvg_v2 / savg',
    )
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: datasets.<dataset>.output_dir in config)')
    parser.add_argument('--config', type=str, default='model/config.yaml', help='Config file')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs for Tensor Parallelism')
    parser.add_argument('--batch-size', type=int, default=1000, help='Number of prompts to process in each batch (default: 1000)')
    parser.add_argument(
        '--qwen-model',
        type=str,
        default='qwen3-vl-8b',
        choices=['qwen2.5-vl-3b', 'qwen2.5-vl-7b', 'qwen3-vl-8b'],
        help='Which Qwen VL model to use (must match keys in config.yaml, e.g., qwen2.5-vl-3b).',
    )
    parser.add_argument('--force', action='store_true', help='Force reprocess all, do not skip already completed stage2 outputs')
    
    args = parser.parse_args()
    
    args.dataset = normalize_dataset_key(args.dataset)
    config_manager = ConfigManager(args.config)
    config = config_manager.config
    if args.output_dir is None:
        ds_cfg = config.get("datasets", {}).get(args.dataset, {})
        args.output_dir = ds_cfg.get("output_dir")
        if not args.output_dir:
            raise ValueError(
                f"Missing output_dir for dataset '{args.dataset}' in config {args.config}. "
                "Please set datasets.<dataset>.output_dir or pass --output-dir."
            )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Dataset: {args.dataset}")
    print(f"Output dir: {args.output_dir}")

    tasks = [] # List of (image_path, text_query, target_object, vid_id)

    # Resolve Qwen model path & stage2 output folder based on args.qwen_model
    model_key = args.qwen_model
    # Map model key -> (stage2 folder name, default method name)
    stage2_dir_map = {
        'qwen2.5-vl-3b': 'stage2_qwen2.5_3b',
        'qwen2.5-vl-7b': 'stage2_qwen2.5_7b',
        'qwen3-vl-8b': 'stage2_qwen3_8b',
    }
    method_name_map = {
        'qwen2.5-vl-3b': 'qwen2.5_vl_3b_cot_vllm',
        'qwen2.5-vl-7b': 'qwen2.5_vl_7b_cot_vllm',
        'qwen3-vl-8b': 'qwen3_vl_8b_cot_vllm',
    }
    stage2_dir_name = stage2_dir_map.get(model_key, 'stage2')
    method_name = method_name_map.get(model_key, 'qwen3_vl_cot_vllm')

    # Model path: first try top-level config key (qwen2.5-vl-3b etc.), fallback to qwen_grounding.model_path
    qwen_model_path = config.get(model_key)
    if not qwen_model_path:
        qwen_grounding_cfg = config.get('qwen_grounding', {})
        qwen_model_path = qwen_grounding_cfg.get(
            'model_path',
            '/home/xdu/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct'
        )

    # ==========================================
    # Mode 1: Single Image (CLI)
    # ==========================================
    if args.image_path:
        if not args.text_query:
            parser.error("--text-query is required with --image-path")
        vid_id = args.video_id if args.video_id else Path(args.image_path).stem
        tasks.append((args.image_path, args.text_query, args.text_query, vid_id))

    # ==========================================
    # Mode 2: Specific Video ID (Folder Scan)
    # ==========================================
    elif args.video_id:
        stage1_base = os.path.join(args.output_dir, "stage1")
        stage1_output = os.path.join(stage1_base, args.video_id)
        
        if os.path.exists(stage1_output):
            # Check for sub-folders (sentences)
            subdirs = []
            for subitem in os.listdir(stage1_output):
                subdir_path = os.path.join(stage1_output, subitem)
                if os.path.isdir(subdir_path) and subitem.isdigit():
                    if os.path.exists(os.path.join(subdir_path, "metadata.json")):
                        subdirs.append((int(subitem), subdir_path))
            
            if subdirs:
                subdirs.sort()
                for sentence_idx, subdir_path in subdirs:
                    img_p = resolve_key_frame_path(subdir_path)
                    if not img_p:
                        continue
                    with open(os.path.join(subdir_path, "metadata.json"), 'r') as f:
                        meta = json.load(f)
                    tasks.append((
                        img_p,
                        meta['text_query'],
                        meta.get('target_object'),
                        f"{args.video_id}_{sentence_idx}"
                    ))
            else:
                # Root folder check
                meta_path = os.path.join(stage1_output, "metadata.json")
                img_p = resolve_key_frame_path(stage1_output)
                if os.path.exists(meta_path) and img_p:
                    with open(meta_path, 'r') as f: meta = json.load(f)
                    tasks.append((
                        img_p,
                        meta['text_query'],
                        meta.get('target_object'),
                        args.video_id
                    ))
        else:
            print(f"✗ Stage 1 output not found for video {args.video_id}")

    # ==========================================
    # Mode 3: Batch Process (Folder Scan)
    # ==========================================
    else:
        print(f"Scanning for Stage 1 outputs in {args.output_dir}...")
        stage1_base_dir = os.path.join(args.output_dir, "stage1")
        
        if os.path.exists(stage1_base_dir):
            video_dirs = os.listdir(stage1_base_dir)
            for vid in video_dirs:
                stage1_dir = os.path.join(stage1_base_dir, vid)
                if not os.path.isdir(stage1_dir): continue
                
                # Check sub-folders
                subitems = [s for s in os.listdir(stage1_dir) if s.isdigit()]
                if subitems:
                    for s in sorted(subitems, key=int):
                        p = os.path.join(stage1_dir, s)
                        meta_p = os.path.join(p, "metadata.json")
                        img_p = resolve_key_frame_path(p)
                        if os.path.exists(meta_p) and img_p:
                            with open(meta_p, 'r') as f: meta = json.load(f)
                            tasks.append((img_p, meta['text_query'], meta.get('target_object'), f"{vid}_{s}"))
                else:
                    # Root folder
                    meta_p = os.path.join(stage1_dir, "metadata.json")
                    img_p = resolve_key_frame_path(stage1_dir)
                    if os.path.exists(meta_p) and img_p:
                        with open(meta_p, 'r') as f: meta = json.load(f)
                        tasks.append((img_p, meta['text_query'], meta.get('target_object'), vid))

        if args.num:
            tasks = tasks[:args.num]

    # Skip already completed (Stage 2 output exists), unless --force
    if tasks and not args.force:
        remaining_tasks = []
        skipped_count = 0
        for task in tasks:
            img_path, text_query, target_object, vid_id = task
            if check_stage2_completed(args.output_dir, vid_id, stage2_dir_name=stage2_dir_name):
                skipped_count += 1
            else:
                remaining_tasks.append(task)
        if skipped_count > 0:
            print(f"✓ Skipped {skipped_count} already completed (stage2 output exists)")
        if not remaining_tasks:
            print("✓ All tasks already completed!")
            return
        tasks = remaining_tasks

    # Execute
    if tasks:
        print(f"✓ Found {len(tasks)} tasks.")
        process_batch_vllm(
            tasks,
            args.output_dir,
            config,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
            qwen_model_path=qwen_model_path,
            stage2_dir_name=stage2_dir_name,
            method_name=method_name,
            model_key=model_key,
        )
    else:
        print("No tasks found.")

if __name__ == '__main__':
    main()