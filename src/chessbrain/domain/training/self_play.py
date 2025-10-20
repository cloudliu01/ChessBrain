from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

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
from src.chessbrain.domain.training.mcts import AlphaZeroMCTS


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
    moves: List[str]
    san_moves: List[str]
    termination: Optional[str]
    winner: Optional[str]
    result: float
    final_fen: str


class SelfPlayCollector:
    """Generate self-play experience using the current policy/value network."""

    _DEFAULT_OPENINGS: Sequence[Sequence[str]] = (
        ("e2e4", "e7e5", "g1f3", "b8c6"),
        ("d2d4", "d7d5", "c2c4", "e7e6"),
        ("c2c4", "e7e5", "g1f3", "b8c6"),
        ("g1f3", "d7d5", "d2d4", "g8f6"),
        ("f2f4", "e7e5", "g2g3", "d7d5"),
    )

    def __init__(
        self,
        *,
        device: TensorLike,
        temperature: float = 1.0,
        max_moves: int = 160,
        exploration_epsilon: float = 0.0,
        mcts_simulations: int = 64,
        mcts_c_puct: float = 1.5,
        opening_sequences: Optional[Sequence[Sequence[str]]] = None,
        opening_shuffle_moves: int = 4,
    ) -> None:
        if torch is None:  # pragma: no cover - enforced by import guards
            raise RuntimeError("PyTorch is required for SelfPlayCollector")
        self._device = device
        self._temperature = temperature
        self._max_moves = max_moves
        self._exploration_epsilon = exploration_epsilon
        self._mcts_simulations = mcts_simulations
        self._mcts_c_puct = mcts_c_puct
        self._opening_sequences: List[List[str]] = [
            list(sequence) for sequence in (opening_sequences or self._DEFAULT_OPENINGS)
        ]
        self._opening_shuffle_moves = max(0, opening_shuffle_moves)

    def generate_episode(self, model: torch.nn.Module) -> SelfPlayEpisode:  # type: ignore[misc]
        board = chess.Board()
        pending: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor, chess.Color, str]] = []
        moves: List[str] = []
        san_moves: List[str] = []

        opening_prefix = self._apply_opening_sequence(board)
        if opening_prefix:
            for uci, san in opening_prefix:
                moves.append(uci)
                san_moves.append(san)

        mcts = (
            AlphaZeroMCTS(
                device=self._device,
                simulations=self._mcts_simulations,
                c_puct=self._mcts_c_puct,
            )
            if self._mcts_simulations > 0
            else None
        )

        with torch.no_grad():
            for _ply in range(max(0, self._max_moves - len(opening_prefix))):
                if board.is_game_over(claim_draw=True):
                    break

                features = board_to_tensor(board, device=self._device)
                legal_mask = legal_moves_mask(board, device=self._device)

                if mcts is not None:
                    policy = mcts.run(board, model).detach()
                    if self._temperature != 1.0:
                        policy = self._apply_temperature(policy, self._temperature)
                else:
                    output = model.inference(features.unsqueeze(0), temperature=None)
                    logits = output.flatten_policy().squeeze(0)
                    policy = policy_from_masked_logits(logits, legal_mask, temperature=self._temperature)

                policy = self._apply_exploration(policy, legal_mask)
                policy = self._normalize_policy(policy, legal_mask)

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
                    else:
                        legal_indices = legal_mask.nonzero(as_tuple=False).squeeze(-1)
                        if legal_indices.numel() > 0:
                            policy[legal_indices] = 1.0 / legal_indices.numel()
                        else:
                            policy.fill_(1.0 / policy.numel())
                    policy = self._normalize_policy(policy, legal_mask)

                san = board.san(move)
                moves.append(move.uci())
                san_moves.append(san)
                pending.append((features, policy, legal_mask, board.turn, san))
                board.push(move)

            outcome = board.outcome(claim_draw=True)

        samples: List[TrainingSample] = []
        if outcome is None or outcome.winner is None:
            result = 0.0
        else:
            result = 1.0 if outcome.winner == chess.WHITE else -1.0

        for features, policy, legal_mask, color, _san in pending:
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

        termination = outcome.termination.name if outcome and outcome.termination else None
        winner = None
        if outcome and outcome.winner is not None:
            winner = "white" if outcome.winner == chess.WHITE else "black"

        final_fen = board.fen()

        return SelfPlayEpisode(
            samples=samples,
            win_rate=win_rate,
            moves=moves,
            san_moves=san_moves,
            termination=termination,
            winner=winner,
            result=float(result),
            final_fen=final_fen,
        )

    def _apply_exploration(self, policy: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        if self._exploration_epsilon <= 0.0:
            return policy
        legal_indices = legal_mask.nonzero(as_tuple=False).squeeze(-1)
        if legal_indices.numel() == 0:
            return policy
        uniform = torch.zeros_like(policy)
        uniform[legal_indices] = 1.0 / legal_indices.numel()
        return (1 - self._exploration_epsilon) * policy + self._exploration_epsilon * uniform

    @staticmethod
    def _apply_temperature(policy: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if math.isclose(temperature, 1.0):
            return policy
        scaled = policy.pow(1.0 / temperature)
        total = scaled.sum()
        if total > 0:
            return scaled / total
        return policy

    @staticmethod
    def _normalize_policy(policy: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        masked = policy * legal_mask
        total = masked.sum()
        if torch.isfinite(total) and total > 0:
            return masked / total
        legal_indices = legal_mask.nonzero(as_tuple=False).squeeze(-1)
        if legal_indices.numel() > 0:
            normalized = torch.zeros_like(policy)
            normalized[legal_indices] = 1.0 / legal_indices.numel()
            return normalized
        return torch.full_like(policy, 1.0 / policy.numel())

    def _apply_opening_sequence(self, board: chess.Board) -> List[tuple[str, str]]:
        if not self._opening_sequences:
            return []

        sequence = random.choice(self._opening_sequences)
        if not sequence:
            return []

        max_prefix = len(sequence)
        if self._opening_shuffle_moves > 0:
            max_prefix = min(max_prefix, self._opening_shuffle_moves)

        if max_prefix <= 0:
            return []

        prefix_len = random.randint(1, max_prefix)
        applied: List[tuple[str, str]] = []
        for uci in sequence[:prefix_len]:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            san = board.san(move)
            applied.append((uci, san))
            board.push(move)
            if board.is_game_over(claim_draw=True):
                break
        return applied


__all__ = ["SelfPlayCollector", "SelfPlayEpisode", "TrainingSample"]
