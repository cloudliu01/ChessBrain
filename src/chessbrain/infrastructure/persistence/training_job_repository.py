from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import JSON, Column, DateTime, Enum, Integer, String, Text, func
from sqlalchemy.orm import Session

from src.chessbrain.infrastructure.persistence.base import Base


class TrainingJobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    paused = "paused"
    completed = "completed"
    failed = "failed"


@dataclass(frozen=True, slots=True)
class TrainingJob:
    id: UUID
    status: TrainingJobStatus
    config: dict[str, Any]
    metrics_uri: str
    checkpoint_version: Optional[str]
    episodes_played: int
    last_heartbeat: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str]


class TrainingJobRecord(Base):  # type: ignore[misc]
    __tablename__ = "training_jobs"

    id = Column(String(36), primary_key=True)
    status = Column(Enum(TrainingJobStatus, native_enum=False), nullable=False)
    config = Column(JSON, nullable=False)
    metrics_uri = Column(String, nullable=False)
    checkpoint_version = Column(String, nullable=True)
    episodes_played = Column(Integer, nullable=False, default=0)
    last_heartbeat = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    error_message = Column(Text, nullable=True)


class TrainingJobRepository:
    """Repository for persisting training job lifecycle metadata."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, job_id: UUID, config: dict[str, Any], metrics_uri: str) -> TrainingJob:
        record = TrainingJobRecord(
            id=str(job_id),
            status=TrainingJobStatus.queued,
            config=config,
            metrics_uri=metrics_uri,
            episodes_played=0,
        )
        self._session.add(record)
        self._session.commit()
        self._session.refresh(record)
        return self._to_entity(record)

    def mark_running(self, job_id: UUID) -> TrainingJob:
        return self._update(
            job_id,
            status=TrainingJobStatus.running,
            last_heartbeat=datetime.now(timezone.utc),
        )

    def record_progress(
        self,
        job_id: UUID,
        episodes_played: int,
        last_heartbeat: Optional[datetime] = None,
    ) -> TrainingJob:
        return self._update(
            job_id,
            episodes_played=episodes_played,
            last_heartbeat=last_heartbeat or datetime.now(timezone.utc),
        )

    def mark_completed(
        self,
        job_id: UUID,
        checkpoint_version: str,
        episodes_played: int,
    ) -> TrainingJob:
        return self._update(
            job_id,
            status=TrainingJobStatus.completed,
            checkpoint_version=checkpoint_version,
            episodes_played=episodes_played,
            last_heartbeat=datetime.now(timezone.utc),
        )

    def set_failed(self, job_id: UUID, error_message: str) -> TrainingJob:
        return self._update(
            job_id,
            status=TrainingJobStatus.failed,
            error_message=error_message,
            last_heartbeat=datetime.now(timezone.utc),
        )

    def get(self, job_id: UUID) -> Optional[TrainingJob]:
        record = self._session.get(TrainingJobRecord, str(job_id))
        return self._to_entity(record) if record else None

    def _update(self, job_id: UUID, **values: Any) -> TrainingJob:
        record = self._session.get(TrainingJobRecord, str(job_id))
        if record is None:
            raise ValueError(f"Training job {job_id} not found")

        for key, value in values.items():
            setattr(record, key, value)

        self._session.commit()
        self._session.refresh(record)
        return self._to_entity(record)

    @staticmethod
    def _to_entity(record: TrainingJobRecord) -> TrainingJob:
        return TrainingJob(
            id=UUID(record.id),
            status=record.status,
            config=record.config,
            metrics_uri=record.metrics_uri,
            checkpoint_version=record.checkpoint_version,
            episodes_played=record.episodes_played,
            last_heartbeat=record.last_heartbeat,
            created_at=record.created_at,
            updated_at=record.updated_at,
            error_message=record.error_message,
        )


__all__ = ["TrainingJob", "TrainingJobRepository", "TrainingJobStatus"]
