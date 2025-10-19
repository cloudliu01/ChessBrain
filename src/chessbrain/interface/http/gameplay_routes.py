from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID, uuid4

from flask import Blueprint, current_app, jsonify, request

from src.chessbrain.domain.chess import (
    GameSession,
    IllegalMoveError,
    PlayerColor,
    SessionCompletedError,
    SessionDifficulty,
    SessionManager,
    SessionNotFoundError,
    SessionStatus,
    UndoNotAvailableError,
)
from src.chessbrain.infrastructure.config import AppConfig
from src.chessbrain.infrastructure.persistence.base import session_scope
from src.chessbrain.infrastructure.persistence.game_session_repository import (
    SqlAlchemyGameSessionRepository,
)
from src.chessbrain.infrastructure.rl.device import resolve_device
from src.chessbrain.infrastructure.rl.inference_adapter import TorchInferenceAdapter
from src.chessbrain.interface.telemetry.logging import bind_trace, get_logger

gameplay_bp = Blueprint("gameplay", __name__)
logger = get_logger("chessbrain.api.sessions")


def _app_config() -> AppConfig | None:
    cfg = current_app.config.get("APP_CONFIG")
    return cfg


def _inference_service():
    adapter = current_app.extensions.get("inference_adapter")
    if adapter is not None:
        return adapter

    model_path = current_app.config.get("ACTIVE_MODEL_PATH")
    if not model_path:
        raise RuntimeError("ACTIVE_MODEL_PATH is not configured for inference.")

    cfg: AppConfig | None = _app_config()
    device = resolve_device(cfg) if cfg else "cpu"
    adapter = TorchInferenceAdapter(Path(model_path), device=device)
    current_app.extensions["inference_adapter"] = adapter
    return adapter


@contextmanager
def _session_manager() -> Iterator[SessionManager]:
    factory = current_app.config.get("SESSION_FACTORY")
    if factory is None:
        cfg = _app_config()
        with session_scope(cfg) as db_session:
            repository = SqlAlchemyGameSessionRepository(db_session)
            manager = SessionManager(repository, _inference_service())
            yield manager
        return

    session = factory()
    try:
        repository = SqlAlchemyGameSessionRepository(session)
        manager = SessionManager(repository, _inference_service())
        yield manager
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _serialize_session(session: GameSession, trace_id: str | None = None) -> dict[str, Any]:
    return {
        "id": str(session.id),
        "status": session.status.value,
        "playerColor": session.player_color.value,
        "difficulty": session.difficulty.value,
        "currentFen": session.current_fen,
        "moves": [
            {
                "san": move.san,
                "uci": move.uci,
                "actor": move.actor.value,
                "timestamp": move.timestamp.isoformat(),
                "evaluation": move.evaluation,
                "rationale": move.rationale,
            }
            for move in session.moves
        ],
        "activeModelVersion": session.active_model_version,
        "undoCount": session.undo_count,
        "evaluation": session.evaluation,
        "startedAt": session.started_at.isoformat(),
        "updatedAt": session.updated_at.isoformat(),
        "endedAt": session.ended_at.isoformat() if session.ended_at else None,
        "traceId": trace_id,
    }


def _domain_error(code: str, message: str, status: int = 400, detail: Any | None = None):
    payload: dict[str, Any] = {"code": code, "message": message}
    if detail is not None:
        payload["detail"] = detail
    return jsonify(payload), status


@gameplay_bp.post("")
def create_session():
    payload = request.get_json(silent=True) or {}
    trace_id = request.headers.get("X-Trace-Id") or uuid4().hex
    log = bind_trace(logger, trace_id)

    try:
        player_color = PlayerColor(payload.get("playerColor", "white"))
    except ValueError:
        return _domain_error("invalid_color", "playerColor must be 'white' or 'black'.")

    try:
        difficulty = SessionDifficulty(payload.get("difficulty", "deterministic"))
    except ValueError:
        return _domain_error("invalid_difficulty", "difficulty must be 'deterministic' or 'stochastic'.")

    with _session_manager() as manager:
        session = manager.create_session(player_color=player_color, difficulty=difficulty)

    log.info(
        "session_created",
        session_id=str(session.id),
        player_color=player_color.value,
        difficulty=difficulty.value,
        model_version=session.active_model_version,
    )
    return jsonify(_serialize_session(session, trace_id=trace_id)), 201


@gameplay_bp.get("/<session_id>")
def get_session(session_id: str):
    trace_id = request.headers.get("X-Trace-Id") or uuid4().hex
    log = bind_trace(logger, trace_id, session_id=session_id)
    try:
        session_uuid = UUID(session_id)
    except ValueError:
        return _domain_error("invalid_session_id", "sessionId must be a valid UUID.", status=400)

    try:
        with _session_manager() as manager:
            session = manager.get_session(session_uuid)
    except SessionNotFoundError:
        log.warning("session_not_found", session_id=session_id)
        return _domain_error("session_not_found", "Session not found.", status=404)

    return jsonify(_serialize_session(session, trace_id=trace_id)), 200


@gameplay_bp.post("/<session_id>/moves")
def submit_move(session_id: str):
    payload = request.get_json(silent=True) or {}
    trace_id = request.headers.get("X-Trace-Id") or uuid4().hex
    log = bind_trace(logger, trace_id, session_id=session_id)

    try:
        session_uuid = UUID(session_id)
    except ValueError:
        return _domain_error("invalid_session_id", "sessionId must be a valid UUID.", status=400)

    uci = payload.get("uci")
    if not isinstance(uci, str):
        return _domain_error("invalid_move", "uci must be provided as a string.", status=400)

    try:
        with _session_manager() as manager:
            session = manager.submit_move(session_uuid, uci)
    except SessionNotFoundError:
        log.warning("session_not_found", session_id=session_id)
        return _domain_error("session_not_found", "Session not found.", status=404)
    except IllegalMoveError as exc:
        log.warning("illegal_move_rejected", uci=uci, detail=str(exc))
        return _domain_error("illegal_move", str(exc), status=409)
    except SessionCompletedError as exc:
        log.warning("move_after_completion", reason=str(exc))
        return _domain_error("session_completed", "Session already completed.", status=409)

    log.info("move_accepted", uci=uci, total_moves=len(session.moves))
    return jsonify(_serialize_session(session, trace_id=trace_id)), 200


@gameplay_bp.post("/<session_id>/undo")
def undo_move(session_id: str):
    trace_id = request.headers.get("X-Trace-Id") or uuid4().hex
    log = bind_trace(logger, trace_id, session_id=session_id)

    try:
        session_uuid = UUID(session_id)
    except ValueError:
        return _domain_error("invalid_session_id", "sessionId must be a valid UUID.", status=400)

    try:
        with _session_manager() as manager:
            session = manager.undo_last(session_uuid)
    except SessionNotFoundError:
        log.warning("session_not_found", session_id=session_id)
        return _domain_error("session_not_found", "Session not found.", status=404)
    except UndoNotAvailableError as exc:
        log.warning("undo_unavailable", reason=str(exc))
        return _domain_error("undo_unavailable", str(exc), status=409)

    log.info("undo_applied", remaining_moves=len(session.moves))
    return jsonify(_serialize_session(session, trace_id=trace_id)), 200


@gameplay_bp.post("/<session_id>/resign")
def resign_session(session_id: str):
    trace_id = request.headers.get("X-Trace-Id") or uuid4().hex
    log = bind_trace(logger, trace_id, session_id=session_id)

    try:
        session_uuid = UUID(session_id)
    except ValueError:
        return _domain_error("invalid_session_id", "sessionId must be a valid UUID.", status=400)

    try:
        with _session_manager() as manager:
            session = manager.resign(session_uuid)
    except SessionNotFoundError:
        log.warning("session_not_found", session_id=session_id)
        return _domain_error("session_not_found", "Session not found.", status=404)

    status = "in_progress" if session.status is SessionStatus.in_progress else session.status.value
    log.info("session_resigned", resulting_status=status)
    return jsonify(_serialize_session(session, trace_id=trace_id)), 200


__all__ = ["gameplay_bp"]
