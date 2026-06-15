#!/usr/bin/env python3
"""Run Stage 2 (RSVG-ZeroOV grounding) then Stage 3 (SAM3 propagation) in one process.

Convenience wrapper for small/demo runs. For large-scale, multi-GPU production runs
prefer the dedicated entrypoints (or the shell scripts under ``scripts/``).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stvg.config import dataset_output_dir, load_config  # noqa: E402
from stvg.io.stage1 import collect_stage1_records  # noqa: E402
from stvg.pipeline.stage2_runner import run_stage2  # noqa: E402
from stvg.pipeline.stage3_runner import run_stage3_clip  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Run Stage 2 (ZeroOV) + Stage 3 (SAM3)")
    p.add_argument("--dataset", default="savg")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--config", default="model/config.yaml")
    p.add_argument("--grounding-subdir", default="grounding")
    p.add_argument("--stage3-subdir", default="stage3")
    p.add_argument("--num", type=int, default=None)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--rsvg-dir", default="/mnt/data/disk2/zyu/videoVG/RSVG-ZeorOV")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    config = load_config(args.config)
    output_dir = args.output_dir or dataset_output_dir(config, args.dataset)
    if not output_dir:
        raise SystemExit(f"No output_dir for dataset {args.dataset!r}; pass --output-dir.")

    run_stage2(
        output_dir=output_dir,
        config=config,
        method="zeroov",
        grounding_subdir=args.grounding_subdir,
        num=args.num,
        force=args.force,
        grounding_kwargs={"rsvg_dir": args.rsvg_dir},
    )

    for record in collect_stage1_records(output_dir, limit=args.num):
        run_stage3_clip(
            output_dir=output_dir,
            vid=record.vid,
            config=config,
            backend="sam3",
            grounding_subdir=args.grounding_subdir,
            stage3_subdir=args.stage3_subdir,
            num_gpus=args.num_gpus,
        )


if __name__ == "__main__":
    main()
