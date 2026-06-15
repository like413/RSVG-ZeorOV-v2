#!/usr/bin/env python3
"""Stage 3 propagation CLI (SAM3).

Delegates to the proven ``stage3_sam3_qwen.py`` core, forcing the SAM3 tracker
propagation path:

- replaces the bbox propagator with the SAM3 tracker, and
- with ``--skip-tube-select`` (default on) always uses bbox propagation
  (no tube selection).

All other flags (``--dataset``, ``--stage2-subdir``, ``--stage3-subdir``,
``--num-gpus``, ``--gpu``, ``--input-root``, ``--output-root``, ...) are forwarded
to ``stage3_sam3_qwen`` unchanged.

Example::

    python -m stvg.cli.stage3_propagation --dataset savg \
        --stage2-subdir grounding --stage3-subdir stage3_sam3 --num-gpus 4
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _extract_store_true(argv, name):
    present = name in argv
    out = [a for a in argv if a != name]
    return present, out


def _extract_no_flag(argv, name):
    present = name in argv
    out = [a for a in argv if a != name]
    return present, out


def _apply_gpu_from_argv(argv) -> None:
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


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Tube selection is skipped by default (bbox propagation only); allow opting back in.
    skip_tube, argv = _extract_store_true(argv, "--skip-tube-select")
    keep_tube, argv = _extract_no_flag(argv, "--with-tube-select")
    skip_tube = skip_tube or (not keep_tube)

    # GPU env must be set before torch import inside the legacy module.
    _apply_gpu_from_argv(argv)

    from stvg._legacy import load_stage3

    impl = load_stage3()

    # Force SAM3 tracker propagation.
    impl.propagate_mask_with_sam2 = impl.propagate_mask_with_sam3_tracker
    if skip_tube:
        impl.load_tubes_from_stage3_tubes = lambda *a, **k: (None, False)

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + argv
        impl.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
