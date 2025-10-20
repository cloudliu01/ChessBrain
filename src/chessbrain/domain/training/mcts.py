from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import torch

from src.chessbrain.domain.models.move_encoding import (
    ACTION_SPACE_SIZE,
    decode_index,
    encode_index,
    legal_moves_mask,
    policy_from_masked_logits,
)
from src.chessbrain.domain.training.features import board_to_tensor


@dataclass
class _ChildStats:
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class _Node:
    board: chess.Board
    parent: Optional["_Node"] = None
    children: Dict[int, "_Node"] = field(default_factory=dict)
    stats: Dict[int, _ChildStats] = field(default_factory=dict)
    total_visits: int = 0
    is_terminal: bool = False


class AlphaZeroMCTS:
    """Minimal MCTS implementation producing policy targets via visit counts."""

    def __init__(
        self,
        *,
        device: torch.device,
        simulations: int = 64,
        c_puct: float = 1.5,
        evaluation_batch_size: int = 8,
    ) -> None:
        self._device = device
        self._simulations = simulations
        self._c_puct = c_puct
        self._eval_batch_size = max(1, evaluation_batch_size)

    def run(self, board: chess.Board, model: torch.nn.Module) -> torch.Tensor:
        root_board = board.copy(stack=False)
        root = _Node(board=root_board, is_terminal=root_board.is_game_over(claim_draw=True))

        if root.is_terminal:
            policy = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32, device=self._device)
            mask = legal_moves_mask(root_board, device=self._device)
            legal_count = int(mask.sum().item())
            if legal_count > 0:
                policy[mask > 0] = 1.0 / legal_count
            return policy

        self._evaluate_batch([root], [root_board], model)

        remaining = self._simulations
        while remaining > 0:
            batch_size = min(self._eval_batch_size, remaining)
            nodes: List[_Node] = []
            boards: List[chess.Board] = []
            paths: List[List[Tuple[_Node, int]]] = []

            for _ in range(batch_size):
                node, board_state, path = self._traverse(root, root_board)
                nodes.append(node)
                boards.append(board_state)
                paths.append(path)

            values = self._evaluate_batch(nodes, boards, model)
            for path, value in zip(paths, values):
                self._backpropagate(path, value)
            remaining -= len(nodes)

        policy = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32, device=self._device)
        for move_index, stats in root.stats.items():
            policy[move_index] = float(stats.visit_count)
        total = policy.sum()
        if total > 0:
            policy /= total
        else:
            mask = legal_moves_mask(root_board, device=self._device)
            legal_count = int(mask.sum().item())
            if legal_count > 0:
                policy[mask > 0] = 1.0 / legal_count
        return policy

    def _select_move(self, node: _Node) -> int:
        best_score = float("-inf")
        best_move = -1
        sqrt_total = math.sqrt(node.total_visits + 1)
        for move_index, stats in node.stats.items():
            q_value = stats.mean_value
            u_value = self._c_puct * stats.prior * sqrt_total / (1 + stats.visit_count)
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_move = move_index
        return best_move

    def _backpropagate(self, path: List[Tuple[_Node, int]], value: float) -> None:
        current_value = value
        for node, move_index in reversed(path):
            stats = node.stats.get(move_index)
            if stats is None:
                continue
            stats.visit_count += 1
            stats.value_sum += current_value
            node.total_visits += 1
            current_value = -current_value

    def _traverse(
        self,
        root: _Node,
        root_board: chess.Board,
    ) -> Tuple[_Node, chess.Board, List[Tuple[_Node, int]]]:
        node = root
        board_copy = root_board.copy(stack=False)
        path: List[Tuple[_Node, int]] = []

        while not node.is_terminal and node.stats:
            move_index = self._select_move(node)
            move = decode_index(board_copy, move_index)
            if move is None:
                node.stats.pop(move_index, None)
                continue
            board_copy.push(move)
            path.append((node, move_index))
            child = node.children.get(move_index)
            if child is None:
                child = _Node(
                    board=board_copy.copy(stack=False),
                    parent=node,
                    is_terminal=board_copy.is_game_over(claim_draw=True),
                )
                node.children[move_index] = child
                node = child
                break
            node = child
        return node, board_copy, path

    def _evaluate_batch(
        self,
        nodes: Sequence[_Node],
        boards: Sequence[chess.Board],
        model: torch.nn.Module,
    ) -> List[float]:
        features: List[torch.Tensor] = []
        mapping: List[int] = []
        values: List[float] = [0.0 for _ in nodes]

        for idx, (node, board) in enumerate(zip(nodes, boards)):
            if board.is_game_over(claim_draw=True):
                node.is_terminal = True
                node.stats.clear()
                node.children.clear()
                values[idx] = self._terminal_value(board)
            else:
                node.is_terminal = False
                features.append(board_to_tensor(board, device=self._device))
                mapping.append(idx)

        if features:
            stacked = torch.stack(features, dim=0)
            output = model.inference(stacked)
            logits = output.flatten_policy()
            value_tensor = output.value.squeeze(1)

            for pos, idx in enumerate(mapping):
                node = nodes[idx]
                board = boards[idx]
                mask = legal_moves_mask(board, device=self._device)
                priors = policy_from_masked_logits(logits[pos], mask)

                node.stats.clear()
                for move in board.legal_moves:
                    move_index = encode_index(board, move)
                    if move_index is None:
                        continue
                    node.stats[move_index] = _ChildStats(prior=float(priors[move_index].item()))

                node.total_visits = 0
                values[idx] = float(value_tensor[pos].item())

        return values

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0


__all__ = ["AlphaZeroMCTS"]
