#!/usr/bin/env python3
"""Stage 2 grounding CLI (RSVG-ZeroOV).

Stage 2 produces a key-frame mask/bbox with RSVG-ZeroOV and writes the unified
grounding schema into ``--grounding-subdir``.

Example::

    python -m stvg.cli.stage2_grounding --dataset savg \
        --rsvg-dir /path/to/RSVG-ZeorOV
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure `stvg` is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stvg.config import dataset_output_dir, load_config  # noqa: E402
from stvg.pipeline.stage2_runner import run_stage2  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 spatial grounding (RSVG-ZeroOV)")
    p.add_argument("--dataset", default="savg", help="Dataset key (savg)")
    p.add_argument("--output-dir", default=None, help="Dataset root (default: from config)")
    p.add_argument("--config", default="model/config.yaml", help="Config YAML path")
    p.add_argument("--grounding-subdir", default="grounding",
                   help="Subdir to write unified grounding metadata (default: grounding)")
    p.add_argument("--num", type=int, default=None, help="Limit number of clips")
    p.add_argument("--force", action="store_true", help="Reprocess clips even if output exists")
    # ZeroOV options
    p.add_argument("--rsvg-dir", default="/mnt/data/disk2/zyu/videoVG/RSVG-ZeorOV",
                   help="Path to the external RSVG-ZeroOV repo")
    p.add_argument("--zeroov-qwen-model", default=None,
                   help="Override Qwen path used in the RSVG-ZeroOV attention step")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    output_dir = args.output_dir or dataset_output_dir(config, args.dataset)
    if not output_dir:
        raise SystemExit(
            f"No output_dir for dataset {args.dataset!r}; pass --output-dir or set it in config."
        )

    run_stage2(
        output_dir=output_dir,
        config=config,
        method="zeroov",
        grounding_subdir=args.grounding_subdir,
        num=args.num,
        force=args.force,
        grounding_kwargs={
            "rsvg_dir": args.rsvg_dir,
            "qwen_model": args.zeroov_qwen_model,
        },
    )


if __name__ == "__main__":
    main()
