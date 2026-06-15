#!/usr/bin/env python3
"""
Compatibility shim for Stage 3 Qwen bbox propagation.

This file was restored to keep legacy entrypoints/imports working:
`stage3_qwen_bbox_sam2_propagation.py` and
`stage3_qwen_bbox_sam3_propagation.py`.

Core implementation is delegated to `stage3_sam3_qwen.py`.
Use `--dataset` in the delegated entrypoint to select
{vidstg_declarative,hcstvg_v1,hcstvg_v2,savg}.
"""

from __future__ import annotations

import sys

import stage3_sam3_qwen as _impl


# Re-export commonly imported symbols
propagate_mask_with_sam2 = _impl.propagate_mask_with_sam2
propagate_mask_with_sam3_tracker = _impl.propagate_mask_with_sam3_tracker
get_mask_from_bbox_with_sam3 = _impl.get_mask_from_bbox_with_sam3
process_stage3 = _impl.process_stage3
load_stage1_stage2_data = _impl.load_stage1_stage2_data
load_stage4_bbox = _impl.load_stage4_bbox


def _strip_compat_args(argv: list[str]) -> list[str]:
    """Pass argv through; stage2/stage3 dir flags are defined in stage3_sam3_qwen."""
    return list(argv)


def main():
    # Keep monkey-patching behavior compatible with wrappers:
    # e.g. stage3_qwen_bbox_sam3_propagation.py patches this module's
    # `propagate_mask_with_sam2`. Mirror it to implementation module.
    _impl.propagate_mask_with_sam2 = propagate_mask_with_sam2
    _impl.propagate_mask_with_sam3_tracker = propagate_mask_with_sam3_tracker

    old_argv = sys.argv[:]
    try:
        sys.argv = _strip_compat_args(sys.argv)
        return _impl.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

