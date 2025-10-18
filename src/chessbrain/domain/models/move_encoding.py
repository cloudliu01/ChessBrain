from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import chess
import torch

from src.chessbrain.domain.models.policy_value_network import ACTION_SPACE_PLANES, ACTION_SPACE_SIZE

SLIDING_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),   # North
    (1, 1),   # North-East
    (1, 0),   # East
    (1, -1),  # South-East
    (0, -1),  # South
    (-1, -1), # South-West
    (-1, 0),  # West
    (-1, 1),  # North-West
)

KNIGHT_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)

PROMOTION_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1),   # forward
    (-1, 1),  # capture left
    (1, 1),   # capture right
)

PROMOTION_PIECES: Tuple[chess.PieceType, ...] = (
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK, 
)


@dataclass(frozen=True)
class MoveEncoding:
    plane: int
    from_square: int

    @property
    def flattened_index(self) -> int:
        return self.plane * 64 + self.from_square


def _normalize_vector(dx: int, dy: int, color: chess.Color) -> Tuple[int, int]:
    if color == chess.BLACK:
        return -dx, -dy
    return dx, dy


def _sliding_plane(dx: int, dy: int) -> Optional[int]:
    if dx == 0 and dy == 0:
        return None
    steps = max(abs(dx), abs(dy))
    if steps == 0 or steps > 7:
        return None
    dir_vector = (dx // steps if dx != 0 else 0, dy // steps if dy != 0 else 0)
    if dir_vector not in SLIDING_DIRECTIONS:
        return None
    dir_index = SLIDING_DIRECTIONS.index(dir_vector)
    return dir_index * 7 + (steps - 1)


def _knight_plane(dx: int, dy: int) -> Optional[int]:
    offset = (dx, dy)
    if offset in KNIGHT_OFFSETS:
        return 56 + KNIGHT_OFFSETS.index(offset)
    return None


def _promotion_plane(dx: int, dy: int, promotion: chess.PieceType) -> Optional[int]:
    direction = (dx, dy)
    if direction not in PROMOTION_DIRECTIONS:
        return None
    if promotion not in PROMOTION_PIECES:
        return None
    direction_index = PROMOTION_DIRECTIONS.index(direction)
    promotion_index = PROMOTION_PIECES.index(promotion)
    return 64 + direction_index * len(PROMOTION_PIECES) + promotion_index


def encode_move(board: chess.Board, move: chess.Move) -> Optional[MoveEncoding]:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return None

    dx = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    dy = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    norm_dx, norm_dy = _normalize_vector(dx, dy, piece.color)

    plane: Optional[int] = None
    if move.promotion:
        plane = _promotion_plane(norm_dx, norm_dy, move.promotion)
        if plane is None and move.promotion == chess.QUEEN:
            plane = _sliding_plane(norm_dx, norm_dy)
    if plane is None:
        plane = _knight_plane(norm_dx, norm_dy)
    if plane is None:
        plane = _sliding_plane(norm_dx, norm_dy)

    if plane is None or plane >= ACTION_SPACE_PLANES:
        return None

    return MoveEncoding(plane=plane, from_square=move.from_square)


def encode_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    encoding = encode_move(board, move)
    if encoding is None:
        return None
    return encoding.flattened_index


def legal_moves_mask(board: chess.Board, *, device: Optional[torch.device] = None) -> torch.Tensor:
    mask = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32, device=device)
    for move in board.legal_moves:
        index = encode_index(board, move)
        if index is not None:
            mask[index] = 1.0
    return mask


def decode_index(board: chess.Board, index: int) -> Optional[chess.Move]:
    if index < 0 or index >= ACTION_SPACE_SIZE:
        return None
    target_plane = index // 64
    from_square = index % 64
    for move in board.legal_moves:
        if move.from_square != from_square:
            continue
        encoding = encode_move(board, move)
        if encoding and encoding.plane == target_plane:
            return move
    return None


def policy_from_masked_logits(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    masked_logits = logits.clone()
    illegal = legal_mask < 0.5
    masked_logits[illegal] = -1e9
    if temperature != 1.0:
        masked_logits = masked_logits / max(temperature, 1e-5)
    probs = torch.softmax(masked_logits, dim=-1)
    if torch.isnan(probs).any():
        legal_count = legal_mask.sum()
        if legal_count > 0:
            probs = legal_mask / legal_count
        else:
            probs = torch.full_like(legal_mask, 1.0 / legal_mask.numel())
    return probs


__all__ = [
    "MoveEncoding",
    "decode_index",
    "encode_index",
    "encode_move",
    "legal_moves_mask",
    "policy_from_masked_logits",
]
