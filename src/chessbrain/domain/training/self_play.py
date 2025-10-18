from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import chess

try:  # pragma: no cover - runtime dependency injection
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from src.chessbrain.domain.models.move_encoding import (
    decode_index,
    legal_moves_mask,
    policy_from_masked_logits,
)
from src.chessbrain.domain.training.features import board_to_tensor


TensorLike = Any if torch is None else torch.Tensor


@dataclass(frozen=True)
class TrainingSample:
    features: TensorLike
    policy_target: TensorLike
    value_target: float
    legal_mask: TensorLike


@dataclass(frozen=True)
class SelfPlayEpisode:
    samples: List[TrainingSample]
    win_rate: float


class SelfPlayCollector:
    """Generate self-play experience using the current policy/value network."""

    def __init__(
        self,
        *,
        device: TensorLike,
        temperature: float = 1.0,
        max_moves: int = 160,
        exploration_epsilon: float = 0.0,
    ) -> None:
        if torch is None:  # pragma: no cover - enforced by import guards
            raise RuntimeError("PyTorch is required for SelfPlayCollector")
        self._device = device
        self._temperature = temperature
        self._max_moves = max_moves
        self._exploration_epsilon = exploration_epsilon

    def generate_episode(self, model: torch.nn.Module) -> SelfPlayEpisode:  # type: ignore[misc]
        board = chess.Board()
        pending: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor, chess.Color]] = []

        with torch.no_grad():
            for _ply in range(self._max_moves):
                if board.is_game_over(claim_draw=True):
                    break

                features = board_to_tensor(board, device=self._device)
                legal_mask = legal_moves_mask(board, device=self._device)

                output = model.inference(features.unsqueeze(0), temperature=None)
                logits = output.flatten_policy().squeeze(0)

                policy = policy_from_masked_logits(logits, legal_mask, temperature=self._temperature)
                policy = self._apply_exploration(policy, legal_mask)

                move_index = torch.multinomial(policy, 1).item()
                move = decode_index(board, move_index)
                if move is None:
                    legal_moves = list(board.legal_moves)
                    if not legal_moves:
                        break
                    move = legal_moves[0]
                    fallback_index = decode_index(board, move)
                    policy = torch.zeros_like(policy)
                    if fallback_index is not None:
                        policy[fallback_index] = 1.0

                pending.append((features, policy, legal_mask, board.turn))
                board.push(move)

            outcome = board.outcome(claim_draw=True)

        samples: List[TrainingSample] = []
        if outcome is None or outcome.winner is None:
            result = 0.0
        else:
            result = 1.0 if outcome.winner == chess.WHITE else -1.0

        for features, policy, legal_mask, color in pending:
            value = result if color == chess.WHITE else -result
            samples.append(
                TrainingSample(
                    features=features.detach(),
                    policy_target=policy.detach(),
                    value_target=float(value),
                    legal_mask=legal_mask.detach(),
                )
            )

        if result > 0:
            win_rate = 1.0
        elif result < 0:
            win_rate = 0.0
        else:
            win_rate = 0.5

        return SelfPlayEpisode(samples=samples, win_rate=win_rate)

    def _apply_exploration(self, policy: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        if self._exploration_epsilon <= 0.0:
            return policy
        legal_indices = legal_mask.nonzero(as_tuple=False).squeeze(-1)
        if legal_indices.numel() == 0:
            return policy
        uniform = torch.zeros_like(policy)
        uniform[legal_indices] = 1.0 / legal_indices.numel()
        return (1 - self._exploration_epsilon) * policy + self._exploration_epsilon * uniform


__all__ = ["SelfPlayCollector", "SelfPlayEpisode", "TrainingSample"]
