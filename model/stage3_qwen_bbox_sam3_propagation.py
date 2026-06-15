#!/usr/bin/env python3
"""
Stage 3 (Qwen bbox + SAM3 propagation-only entrypoint).

This wrapper enforces the bbox bidirectional propagation path:
1) Replace SAM2 propagation with SAM3 tracker propagation.
2) Disable tube loading so it will not short-circuit into tube selection.

Stage2/stage3 output folder names are parsed by stage3_sam3_qwen (e.g. --stage2-subdir / --stage3-subdir;
--stage2-dir-name / --stage3-dir-name are aliases).

GPU: pass --gpu 3 to use physical GPU 3 (sets CUDA_VISIBLE_DEVICES before importing torch).
Dataset: pass --dataset {vidstg_declarative,hcstvg_v1,hcstvg_v2,savg}.
"""

from __future__ import annotations

import os
import sys


def _strip_compat_args(argv: list[str]) -> list[str]:
    """Remove --gpu so inner argparse does not override CUDA_VISIBLE_DEVICES set above."""
    drop_next_for = {"--gpu"}
    out: list[str] = []
    i = 0
    while i < len(argv):
        cur = argv[i]
        if cur in drop_next_for:
            i += 2
            continue
        if cur.startswith("--gpu="):
            i += 1
            continue
        out.append(cur)
        i += 1
    return out


def _apply_gpu_from_argv(argv: list[str]) -> None:
    """Set CUDA_VISIBLE_DEVICES before torch import (only if not already set)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return
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
    """
    Force bbox propagation branch with SAM3 tracker.
    """
    _apply_gpu_from_argv(sys.argv[1:])

    import stage3_sam3_qwen as _impl

    # Replace SAM2 propagation with SAM3 propagation.
    _impl.propagate_mask_with_sam2 = _impl.propagate_mask_with_sam3_tracker

    # Force "no tubes" so execution always goes to bbox bidirectional propagation
    # when qwen bbox exists.
    _impl.load_tubes_from_stage3_tubes = lambda *args, **kwargs: (None, False)

    old_argv = sys.argv[:]
    try:
        sys.argv = _strip_compat_args(sys.argv)
        _impl.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
