"""Abstract base class for Stage 2 grounding strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from stvg.schemas import GroundingResult, Stage1Record


class BaseGroundingModel(ABC):
    """Common interface for every spatial-grounding strategy.

    Concrete implementations (Qwen-VL, RSVG-ZeroOV, ...) must return a
    :class:`~stvg.schemas.GroundingResult` whose on-disk schema is identical
    across strategies. That contract is what makes Stage 2 model-agnostic from
    Stage 3's point of view.
    """

    #: Stable identifier used in metadata ``method`` and for the factory.
    method_name: str = "base"

    @abstractmethod
    def ground(self, record: Stage1Record) -> GroundingResult:
        """Ground a single Stage 1 record into a unified result."""
        raise NotImplementedError

    def ground_batch(self, records: Sequence[Stage1Record]) -> List[GroundingResult]:
        """Ground many records. Override for engines that batch efficiently."""
        return [self.ground(record) for record in records]

    def close(self) -> None:
        """Release any heavy resources (model weights, engines). Optional."""

    def __enter__(self) -> "BaseGroundingModel":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
