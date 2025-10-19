from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import chess


@dataclass(frozen=True)
class MoveSuggestion:
    """Inference result describing the engine's chosen action."""

    move: chess.Move
    probability: float
    evaluation: float | None = None
    rationale: Sequence[str] | None = None


class InferenceService(Protocol):
    """Contract for selecting moves from the active model."""

    @property
    def model_version(self) -> str | None:
        """Return semantic identifier for the active model, if available."""

    def select_move(
        self,
        board: chess.Board,
        *,
        difficulty: str = "deterministic",
    ) -> MoveSuggestion:
        """Pick a legal move for the supplied board position."""


__all__ = ["InferenceService", "MoveSuggestion"]
