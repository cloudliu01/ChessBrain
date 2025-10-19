from __future__ import annotations

from pathlib import Path

import chess
import pytest


def _resolve_model_path() -> Path:
    primary = Path("models/checkpoints/c3ade8cf-step185-20251019171110.pt")
    if primary.exists():
        return primary.resolve()
    fallback = Path("models/checkpoints.archive/c3ade8cf-step185-20251019171110.pt")
    if fallback.exists():
        return fallback.resolve()
    pytest.skip("No checkpoint artifact available for gameplay integration test.")


@pytest.fixture()
def gameplay_client(app):
    path = _resolve_model_path()
    app.config["ACTIVE_MODEL_PATH"] = str(path)
    app.config["ACTIVE_MODEL_VERSION"] = path.stem
    return app.test_client()


def test_gameplay_flow(gameplay_client):
    create_response = gameplay_client.post(
        "/api/v1/sessions",
        json={"playerColor": "white", "difficulty": "deterministic"},
    )
    assert create_response.status_code == 201
    session = create_response.get_json()
    session_id = session["id"]

    board = chess.Board(session["currentFen"])
    assert board.turn == chess.WHITE

    for _ in range(2):
        human_move = next(iter(board.legal_moves))
        move_response = gameplay_client.post(
            f"/api/v1/sessions/{session_id}/moves",
            json={"uci": human_move.uci()},
        )
        assert move_response.status_code == 200
        payload = move_response.get_json()
        assert len(payload["moves"]) >= 2
        board = chess.Board(payload["currentFen"])
        assert payload["status"] == "in_progress"

    final_state = gameplay_client.get(f"/api/v1/sessions/{session_id}")
    assert final_state.status_code == 200
    state_payload = final_state.get_json()
    assert state_payload["id"] == session_id
    assert len(state_payload["moves"]) >= 4
