"""Stable data contracts shared across STVG stages.

These dataclasses are the single source of truth for how data flows between
Stage 1 (temporal grounding) -> Stage 2 (spatial grounding) -> Stage 3 (mask
propagation). Every Stage 2 grounding strategy emits a :class:`GroundingResult`
with an identical on-disk schema (see :meth:`GroundingResult.to_metadata`), so
Stage 3 can consume the output without caring which model produced it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


# Canonical key order for the Stage 2 grounding metadata.json written to disk.
# Keeping this constant guarantees every strategy produces the *same* schema.
GROUNDING_METADATA_KEYS = (
    "vid",
    "text_query",
    "target_object",
    "method",
    "success",
    "bbox",
    "confidence",
    "answer",
    "mask_path",
    "raw_output",
    "failure_reason",
    "extras",
)


@dataclass
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def __post_init__(self) -> None:
        self.xmin = int(self.xmin)
        self.ymin = int(self.ymin)
        self.xmax = int(self.xmax)
        self.ymax = int(self.ymax)
        if self.xmax < self.xmin or self.ymax < self.ymin:
            raise ValueError(
                f"Invalid bbox: expected xmax>=xmin and ymax>=ymin, got {self.to_dict()}"
            )

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> Dict[str, int]:
        return {"xmin": self.xmin, "ymin": self.ymin, "xmax": self.xmax, "ymax": self.ymax}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Optional[BBox]":
        if data is None:
            return None
        try:
            return cls(
                xmin=data["xmin"],
                ymin=data["ymin"],
                xmax=data["xmax"],
                ymax=data["ymax"],
            )
        except KeyError as exc:
            raise KeyError(
                f"bbox dict missing required key {exc}; expected xmin/ymin/xmax/ymax"
            ) from exc

    @classmethod
    def from_xyxy(cls, coords: Any) -> "Optional[BBox]":
        """Build from an ``(x1, y1, x2, y2)`` tuple/list, or ``None``."""
        if coords is None:
            return None
        x1, y1, x2, y2 = coords
        return cls(xmin=x1, ymin=y1, xmax=x2, ymax=y2)


@dataclass
class Stage1Record:
    """Normalized Stage 1 output that feeds a Stage 2 grounding strategy.

    Built from a Stage 1 ``metadata.json`` plus its (possibly materialized)
    ``key_frame.png``.
    """

    vid: str
    key_frame_path: str
    text_query: str
    target_object: Optional[str] = None
    video_path: Optional[str] = None
    key_frame_idx: Optional[int] = None
    original_frame_idx: Optional[int] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def read_frame_idx(self) -> Optional[int]:
        """Frame index to decode the key frame from the source video."""
        if self.original_frame_idx is not None:
            return self.original_frame_idx
        return self.key_frame_idx

    @classmethod
    def from_metadata(
        cls,
        vid: str,
        metadata: Dict[str, Any],
        key_frame_path: str,
    ) -> "Stage1Record":
        return cls(
            vid=vid,
            key_frame_path=key_frame_path,
            text_query=metadata["text_query"],
            target_object=metadata.get("target_object"),
            video_path=metadata.get("video_path"),
            key_frame_idx=metadata.get("key_frame_idx"),
            original_frame_idx=metadata.get("original_frame_idx"),
            raw_metadata=dict(metadata),
        )


@dataclass
class GroundingResult:
    """Unified Stage 2 output. Identical schema for every grounding strategy.

    ``mask`` (an in-memory numpy array) is intentionally *not* a field: masks are
    large and strategy-specific. When a strategy produces a mask it is persisted
    to disk and referenced via ``mask_path`` plus mask-derived scores in
    ``extras`` (e.g. ZeroOV's CLIP similarities and mask coverage).
    """

    vid: str
    text_query: str
    success: bool
    method: str
    target_object: Optional[str] = None
    bbox: Optional[BBox] = None
    confidence: Optional[float] = None
    answer: Optional[str] = None
    mask_path: Optional[str] = None
    raw_output: Optional[str] = None
    failure_reason: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        """Serialize to the canonical Stage 2 ``metadata.json`` dict.

        The returned dict always has exactly :data:`GROUNDING_METADATA_KEYS`,
        regardless of the producing strategy.
        """
        meta = {
            "vid": self.vid,
            "text_query": self.text_query,
            "target_object": self.target_object,
            "method": self.method,
            "success": bool(self.success),
            "bbox": self.bbox.to_dict() if self.bbox is not None else None,
            "confidence": None if self.confidence is None else float(self.confidence),
            "answer": self.answer,
            "mask_path": self.mask_path,
            "raw_output": self.raw_output,
            "failure_reason": self.failure_reason,
            "extras": dict(self.extras),
        }
        # Defensive: enforce the exact key set/order.
        return {k: meta[k] for k in GROUNDING_METADATA_KEYS}

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "GroundingResult":
        return cls(
            vid=meta["vid"],
            text_query=meta["text_query"],
            success=bool(meta.get("success", False)),
            method=meta.get("method", "unknown"),
            target_object=meta.get("target_object"),
            bbox=BBox.from_dict(meta.get("bbox")),
            confidence=meta.get("confidence"),
            answer=meta.get("answer"),
            mask_path=meta.get("mask_path"),
            raw_output=meta.get("raw_output"),
            failure_reason=meta.get("failure_reason"),
            extras=dict(meta.get("extras", {})),
        )

    def save(self, output_dir: str, filename: str = "metadata.json") -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(self.to_metadata(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str) -> "GroundingResult":
        with open(path, "r") as f:
            return cls.from_metadata(json.load(f))


# Fields that Stage 3 strictly requires to run mask propagation.
PROPAGATION_REQUIRED_FIELDS = ("vid", "video_path", "key_frame_idx", "bbox", "key_frame_path")


@dataclass
class PropagationInput:
    """Unified Stage 3 input, assembled from a Stage 1 record + Stage 2 result.

    This decouples Stage 3 from the grounding model: whether the bbox came from
    Qwen or ZeroOV, Stage 3 receives the same structure.
    """

    vid: str
    video_path: str
    key_frame_idx: int
    bbox: Optional[BBox]
    text_query: str
    key_frame_path: str
    target_object: Optional[str] = None
    answer: Optional[str] = None
    grounding_success: bool = False
    grounding_method: Optional[str] = None

    @property
    def prompt(self) -> str:
        """Text prompt for SAM3 / tube selection (answer wins for questions)."""
        return self.answer or self.target_object or self.text_query

    @classmethod
    def from_stages(
        cls,
        stage1: Stage1Record,
        grounding: GroundingResult,
    ) -> "PropagationInput":
        if stage1.video_path is None:
            raise KeyError(
                f"Stage1Record for vid={stage1.vid!r} is missing 'video_path'; "
                "Stage 3 cannot locate the source video."
            )
        if stage1.read_frame_idx is None:
            raise KeyError(
                f"Stage1Record for vid={stage1.vid!r} is missing both "
                "'original_frame_idx' and 'key_frame_idx'."
            )
        return cls(
            vid=grounding.vid,
            video_path=stage1.video_path,
            key_frame_idx=int(stage1.read_frame_idx),
            bbox=grounding.bbox,
            text_query=grounding.text_query,
            key_frame_path=stage1.key_frame_path,
            target_object=grounding.target_object,
            answer=grounding.answer,
            grounding_success=grounding.success,
            grounding_method=grounding.method,
        )

    def validate(self) -> None:
        """Raise ``KeyError`` if a field Stage 3 requires is missing/empty."""
        missing = []
        for name in PROPAGATION_REQUIRED_FIELDS:
            value = getattr(self, name)
            if value is None:
                missing.append(name)
            elif isinstance(value, str) and value.strip() == "":
                missing.append(name)
        if missing:
            raise KeyError(
                f"PropagationInput for vid={self.vid!r} missing required field(s): "
                f"{', '.join(missing)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_dict() if self.bbox is not None else None
        return data
