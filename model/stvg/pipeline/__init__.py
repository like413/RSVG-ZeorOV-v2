"""Thin orchestrators that glue the stages together."""

from stvg.pipeline.stage2_runner import run_stage2
from stvg.pipeline.stage3_runner import run_stage3_clip

__all__ = ["run_stage2", "run_stage3_clip"]
