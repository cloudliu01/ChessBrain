from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess
import torch

BOARD_HEIGHT = 8
BOARD_WIDTH = 8
TOTAL_FEATURE_PLANES = 20

WHITE_PIECE_TYPES = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
BLACK_PIECE_TYPES = WHITE_PIECE_TYPES


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for tensor feature extraction."""

    halfmove_normalizer: float = 100.0
    fullmove_normalizer: float = 200.0


def _piece_plane_index(piece_type: chess.PieceType, color: chess.Color) -> int:
    offset = 0 if color == chess.WHITE else len(WHITE_PIECE_TYPES)
    return offset + (piece_type - 1)


def board_to_tensor(
    board: chess.Board,
    *,
    device: Optional[torch.device] = None,
    config: FeatureConfig | None = None,
) -> torch.Tensor:
    """Convert a chess.Board state into AlphaZero-style feature planes."""

    feature_config = config or FeatureConfig()
    tensor = torch.zeros(
        (TOTAL_FEATURE_PLANES, BOARD_HEIGHT, BOARD_WIDTH),
        dtype=torch.float32,
        device=device,
    )

    # Piece planes
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in WHITE_PIECE_TYPES:
            channel = _piece_plane_index(piece_type, color)
            for square in board.pieces(piece_type, color):
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                tensor[channel, rank, file] = 1.0

    # Side to move
    tensor[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)

    # Castling rights
    tensor[13].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    tensor[14].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    tensor[15].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    tensor[16].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)

    # Halfmove clock normalized
    tensor[17].fill_(min(board.halfmove_clock, feature_config.halfmove_normalizer) / feature_config.halfmove_normalizer)

    # Fullmove number normalized
    tensor[18].fill_(min(board.fullmove_number, feature_config.fullmove_normalizer) / feature_config.fullmove_normalizer)

    # En passant file indicator
    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor[19, rank, file] = 1.0

    return tensor


__all__ = ["BOARD_HEIGHT", "BOARD_WIDTH", "FeatureConfig", "TOTAL_FEATURE_PLANES", "board_to_tensor"]
