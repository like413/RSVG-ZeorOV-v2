"""Stage 2 runner: produce unified grounding output for a whole dataset.

Scans Stage 1 outputs, runs the chosen grounding strategy (Qwen or ZeroOV) via
the factory, and writes one canonical ``metadata.json`` per clip into a single
``grounding_subdir`` -- identical schema regardless of method.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from stvg.grounding import create_grounding_model
from stvg.io.grounding import grounding_completed, save_grounding_result
from stvg.io.stage1 import collect_stage1_records
from stvg.schemas import GroundingResult, Stage1Record


def run_stage2(
    output_dir: str,
    config: Dict[str, Any],
    method: str = "zeroov",
    grounding_subdir: str = "grounding",
    num: Optional[int] = None,
    force: bool = False,
    grounding_kwargs: Optional[Dict[str, Any]] = None,
) -> List[GroundingResult]:
    """Run Stage 2 over every Stage 1 record under ``output_dir``.

    Args:
        output_dir: Dataset root that already contains ``stage1/``.
        config: Loaded pipeline config dict.
        method: Grounding method registered in the factory (``qwen`` / ``zeroov``).
        grounding_subdir: Subdir to write unified metadata into.
        num: Optional cap on number of clips.
        force: Reprocess clips even if grounding output already exists.
        grounding_kwargs: Extra kwargs forwarded to the strategy constructor.

    Returns:
        The list of :class:`GroundingResult` that were (re)computed this run.
    """
    grounding_kwargs = dict(grounding_kwargs or {})
    if method == "zeroov":
        grounding_kwargs.setdefault("work_dir", output_dir)
        grounding_kwargs.setdefault("stage_dir_name", grounding_subdir)

    records = collect_stage1_records(output_dir, limit=num)
    if not force:
        records = [
            r for r in records if not grounding_completed(output_dir, grounding_subdir, r.vid)
        ]
    if not records:
        print("Stage 2: nothing to do (all clips already grounded).")
        return []

    print(f"Stage 2: grounding {len(records)} clip(s) with method={method!r}")
    model = create_grounding_model(method, config, **grounding_kwargs)
    try:
        results = model.ground_batch(records)
    finally:
        model.close()

    by_vid = {r.vid: r for r in records}
    for result in results:
        record: Optional[Stage1Record] = by_vid.get(result.vid)
        key_frame = record.key_frame_path if record else None
        save_grounding_result(output_dir, grounding_subdir, result, key_frame_path=key_frame)

    n_ok = sum(1 for r in results if r.success)
    print(f"Stage 2: done. {n_ok}/{len(results)} succeeded -> {grounding_subdir}/")
    return results
