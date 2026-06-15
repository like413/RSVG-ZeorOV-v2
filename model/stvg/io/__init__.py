"""IO helpers: scanning Stage 1 outputs and reading/writing grounding metadata."""

from stvg.io.filesystem import split_vid, vid_output_dir
from stvg.io.grounding import load_grounding_result, save_grounding_result
from stvg.io.stage1 import iter_stage1_records, load_stage1_record

__all__ = [
    "split_vid",
    "vid_output_dir",
    "iter_stage1_records",
    "load_stage1_record",
    "load_grounding_result",
    "save_grounding_result",
]
