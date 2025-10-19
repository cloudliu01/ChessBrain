from __future__ import annotations

import chess
import pytest

torch = pytest.importorskip("torch")

from src.chessbrain.domain.models.move_encoding import (
    ACTION_SPACE_SIZE,
    decode_index,
    encode_index,
    legal_moves_mask,
    policy_from_masked_logits,
)


def test_encode_decode_opening_moves() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    index = encode_index(board, move)
    assert index is not None
    decoded = decode_index(board, index)
    assert decoded == move

    board.push(move)
    response = chess.Move.from_uci("e7e5")
    response_index = encode_index(board, response)
    assert response_index is not None
    assert decode_index(board, response_index) == response


def test_legal_moves_mask_matches_board() -> None:
    board = chess.Board()
    mask = legal_moves_mask(board, device=torch.device("cpu"))
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert int(mask.sum().item()) == board.legal_moves.count()


def test_underpromotion_encoding_round_trip() -> None:
    fen = "k7/7P/8/8/8/8/8/K7 w - - 0 1"
    board = chess.Board(fen)
    move = chess.Move.from_uci("h7h8r")
    index = encode_index(board, move)
    assert index is not None
    assert decode_index(board, index) == move


def test_policy_from_masked_logits_respects_illegal_moves() -> None:
    board = chess.Board()
    mask = legal_moves_mask(board, device=torch.device("cpu"))
    logits = torch.zeros(ACTION_SPACE_SIZE)
    probs = policy_from_masked_logits(logits, mask)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))
    illegal_positions = (mask < 0.5).nonzero(as_tuple=False).squeeze(-1)
    if illegal_positions.numel() > 0:
        assert torch.all(probs[illegal_positions] == 0)
