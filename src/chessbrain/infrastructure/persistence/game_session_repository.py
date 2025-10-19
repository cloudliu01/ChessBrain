from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional
from uuid import UUID

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Session

from src.chessbrain.domain.chess import (
    GameSession,
    GameSessionRepository,
    MoveActor,
    MoveRecord,
    PlayerColor,
    SessionDifficulty,
    SessionStatus,
)
from src.chessbrain.infrastructure.persistence.base import Base


class GameSessionRecord(Base):  # type: ignore[misc]
    __tablename__ = "game_sessions"

    id = Column(String(36), primary_key=True)
    status = Column(String(32), nullable=False)
    player_color = Column(String(8), nullable=False)
    difficulty = Column(String(16), nullable=False)
    initial_fen = Column(Text, nullable=False)
    current_fen = Column(Text, nullable=False)
    moves = Column(JSON, nullable=False, default=list)
    active_model_version = Column(String(128), nullable=True)
    evaluation = Column(Float, nullable=True)
    undo_count = Column(Integer, nullable=False, default=0)
    metadata_payload = Column(JSON, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    ended_at = Column(DateTime(timezone=True), nullable=True)


def _serialize_moves(moves: Iterable[MoveRecord]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for move in moves:
        payload.append(
            {
                "san": move.san,
                "uci": move.uci,
                "actor": move.actor.value,
                "timestamp": move.timestamp.isoformat(),
                "evaluation": move.evaluation,
                "rationale": list(move.rationale),
            }
        )
    return payload


def _deserialize_moves(items: Optional[Iterable[dict[str, Any]]]) -> List[MoveRecord]:
    if not items:
        return []
    records: List[MoveRecord] = []
    for item in items:
        timestamp = datetime.fromisoformat(item["timestamp"])
        rationale = item.get("rationale") or []
        records.append(
            MoveRecord(
                san=item["san"],
                uci=item["uci"],
                actor=MoveActor(item["actor"]),
                timestamp=timestamp,
                evaluation=item.get("evaluation"),
                rationale=list(rationale),
            )
        )
    return records


class SqlAlchemyGameSessionRepository(GameSessionRepository):
    """SQLAlchemy-backed repository for chess sessions."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, session_entity: GameSession) -> GameSession:
        record = GameSessionRecord(
            id=str(session_entity.id),
            status=session_entity.status.value,
            player_color=session_entity.player_color.value,
            difficulty=session_entity.difficulty.value,
            initial_fen=session_entity.initial_fen,
            current_fen=session_entity.current_fen,
            moves=_serialize_moves(session_entity.moves),
            active_model_version=session_entity.active_model_version,
            evaluation=session_entity.evaluation,
            undo_count=session_entity.undo_count,
            metadata_payload=session_entity.metadata or {},
            started_at=session_entity.started_at,
            updated_at=session_entity.updated_at,
            ended_at=session_entity.ended_at,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return self._to_entity(record)

    def get(self, session_id: UUID) -> GameSession | None:
        record = self._session.get(GameSessionRecord, str(session_id))
        return self._to_entity(record) if record else None

    def save(self, session_entity: GameSession) -> GameSession:
        record = self._session.get(GameSessionRecord, str(session_entity.id))
        if record is None:
            raise ValueError(f"Session {session_entity.id} not found.")

        record.status = session_entity.status.value
        record.player_color = session_entity.player_color.value
        record.difficulty = session_entity.difficulty.value
        record.initial_fen = session_entity.initial_fen
        record.current_fen = session_entity.current_fen
        record.moves = _serialize_moves(session_entity.moves)
        record.active_model_version = session_entity.active_model_version
        record.evaluation = session_entity.evaluation
        record.undo_count = session_entity.undo_count
        record.metadata_payload = session_entity.metadata or {}
        record.started_at = session_entity.started_at
        record.updated_at = session_entity.updated_at
        record.ended_at = session_entity.ended_at

        self._session.commit()
        self._session.refresh(record)
        return self._to_entity(record)

    @staticmethod
    def _to_entity(record: GameSessionRecord) -> GameSession:
        return GameSession(
            id=UUID(record.id),
            status=SessionStatus(record.status),
            player_color=PlayerColor(record.player_color),
            difficulty=SessionDifficulty(record.difficulty),
            initial_fen=record.initial_fen,
            current_fen=record.current_fen,
            moves=_deserialize_moves(record.moves),
            active_model_version=record.active_model_version,
            evaluation=record.evaluation if record.evaluation is None else float(record.evaluation),
            undo_count=record.undo_count or 0,
            metadata=record.metadata_payload or {},
            started_at=record.started_at or datetime.now(timezone.utc),
            updated_at=record.updated_at or datetime.now(timezone.utc),
            ended_at=record.ended_at,
        )


__all__ = ["SqlAlchemyGameSessionRepository", "GameSessionRecord"]
