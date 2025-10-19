from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable
from uuid import UUID

import chess
import pytest


def _resolve_model_path() -> Path:
    """Return an existing checkpoint path for inference tests."""
    primary = Path("models/checkpoints/c3ade8cf-step185-20251019171110.pt")
    if primary.exists():
        return primary.resolve()

    fallback_candidates: Iterable[Path] = [
        Path("models/checkpoints.archive/c3ade8cf-step185-20251019171110.pt"),
        Path("models/checkpoints/8f8b5c49-step1-20251019185213.pt"),
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate.resolve()

    pytest.skip("No checkpoint artifact available for inference contract tests.")


@pytest.fixture()
def client(app):
    model_path = _resolve_model_path()
    app.config["ACTIVE_MODEL_PATH"] = str(model_path)
    app.config["ACTIVE_MODEL_VERSION"] = model_path.stem
    return app.test_client()


def test_create_session_as_white(client):
    response = client.post(
        "/api/v1/sessions",
        json={"playerColor": "white", "difficulty": "deterministic"},
    )
    assert response.status_code == 201
    payload = response.get_json()
    assert UUID(payload["id"])
    assert payload["status"] == "in_progress"
    assert payload["playerColor"] == "white"
    assert payload["currentFen"] == chess.Board().fen()
    assert payload["moves"] == []
    assert payload["activeModelVersion"]
    assert payload["undoCount"] == 0
    assert isinstance(datetime.fromisoformat(payload["startedAt"]), datetime)


def test_create_session_as_black_triggers_ai_opening(client):
    response = client.post(
        "/api/v1/sessions",
        json={"playerColor": "black", "difficulty": "deterministic"},
    )
    assert response.status_code == 201
    payload = response.get_json()
    assert payload["playerColor"] == "black"
    assert payload["moves"], "AI should move first when human is black"
    ai_move = payload["moves"][0]
    assert ai_move["actor"] == "ai"
    assert ai_move["san"]
    assert ai_move["uci"]
    chess.Board(payload["currentFen"])  # should not raise


def test_submit_illegal_move_returns_conflict(client):
    create = client.post(
        "/api/v1/sessions",
        json={"playerColor": "white", "difficulty": "deterministic"},
    )
    session_id = create.get_json()["id"]

    response = client.post(
        f"/api/v1/sessions/{session_id}/moves",
        json={"uci": "e7e5"},
    )
    assert response.status_code == 409
    payload = response.get_json()
    assert payload["code"] == "illegal_move"


def test_submit_legal_move_includes_ai_reply(client):
    create = client.post(
        "/api/v1/sessions",
        json={"playerColor": "white", "difficulty": "deterministic"},
    )
    session_id = create.get_json()["id"]

    move_response = client.post(
        f"/api/v1/sessions/{session_id}/moves",
        json={"uci": "e2e4"},
    )
    assert move_response.status_code == 200
    payload = move_response.get_json()
    assert len(payload["moves"]) >= 2
    human_move = payload["moves"][-2]
    ai_move = payload["moves"][-1]
    assert human_move["actor"] == "human"
    assert ai_move["actor"] == "ai"
    assert ai_move["uci"]
    chess.Board(payload["currentFen"])  # final position is valid


def test_get_session_returns_latest_state(client):
    create = client.post(
        "/api/v1/sessions",
        json={"playerColor": "white", "difficulty": "deterministic"},
    )
    session_id = create.get_json()["id"]
    client.post(f"/api/v1/sessions/{session_id}/moves", json={"uci": "d2d4"})

    response = client.get(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["id"] == session_id
    assert len(payload["moves"]) >= 2


def test_undo_rewinds_last_pair_of_moves(client):
    create = client.post(
        "/api/v1/sessions",
        json={"playerColor": "white", "difficulty": "deterministic"},
    )
    session_id = create.get_json()["id"]
    client.post(f"/api/v1/sessions/{session_id}/moves", json={"uci": "g1f3"})

    undo = client.post(f"/api/v1/sessions/{session_id}/undo")
    assert undo.status_code == 200
    payload = undo.get_json()
    assert payload["moves"] == []
    assert payload["undoCount"] == 1
    assert payload["currentFen"] == chess.Board().fen()


def test_resign_completes_game(client):
    create = client.post(
        "/api/v1/sessions",
        json={"playerColor": "white", "difficulty": "deterministic"},
    )
    session_id = create.get_json()["id"]

    resign = client.post(f"/api/v1/sessions/{session_id}/resign")
    assert resign.status_code == 200
    payload = resign.get_json()
    assert payload["status"] in {"black_won", "white_won"}
    assert payload["endedAt"] is not None
