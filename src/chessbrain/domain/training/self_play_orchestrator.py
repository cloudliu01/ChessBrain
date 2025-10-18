from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from uuid import UUID, uuid4

from src.chessbrain.infrastructure.persistence.training_job_repository import (
    TrainingJob,
    TrainingJobRepository,
)
from src.chessbrain.infrastructure.rl.checkpoint_publisher import (
    CheckpointArtifact,
    FileCheckpointPublisher,
)
from src.chessbrain.infrastructure.rl.training_loop import (
    EpisodeMetrics,
    TrainingConfig,
    TrainingLoop,
)


class SelfPlayOrchestrator:
    """Coordinate training jobs across repository, loop, and artifact publishing."""

    def __init__(
        self,
        repository: TrainingJobRepository,
        training_loop: TrainingLoop,
        checkpoint_publisher: FileCheckpointPublisher,
        tensorboard_root: Path,
    ) -> None:
        self._repository = repository
        self._training_loop = training_loop
        self._checkpoint_publisher = checkpoint_publisher
        self._tensorboard_root = tensorboard_root

    def start_job(self, config: TrainingConfig) -> TrainingJob:
        job_id = uuid4()
        metrics_dir = self._tensorboard_root / job_id.hex
        metrics_dir.mkdir(parents=True, exist_ok=True)

        job = self._repository.create(
            job_id=job_id,
            config=asdict(config),
            metrics_uri=str(metrics_dir),
        )
        job = self._repository.mark_running(job_id)

        try:
            result = self._training_loop.run(
                config=config,
                start_episode=job.episodes_played,
                max_episodes=config.total_episodes,
            )

            episodes_played = job.episodes_played + result.episodes_played
            job = self._repository.record_progress(job_id, episodes_played)
            self._write_metrics(metrics_dir, result.metrics)

            checkpoint_state = result.checkpoint_state
            if checkpoint_state is None:
                last_episode = result.metrics[-1].episode_index if result.metrics else episodes_played
                checkpoint_state = {
                    "global_step": last_episode,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

            artifact = self._checkpoint_publisher.publish(job_id, checkpoint_state)
            job = self._repository.mark_completed(
                job_id=job_id,
                checkpoint_version=artifact.version,
                episodes_played=episodes_played,
            )
            return job
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._repository.set_failed(job_id, str(exc))
            raise

    def _write_metrics(self, metrics_dir: Path, metrics: Iterable[EpisodeMetrics]) -> None:
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "metrics.jsonl"
        with metrics_path.open("a", encoding="utf-8") as handle:
            for metric in metrics:
                handle.write(
                    json.dumps(
                        {
                            "episode_index": metric.episode_index,
                            "policy_loss": metric.policy_loss,
                            "value_loss": metric.value_loss,
                            "win_rate": metric.win_rate,
                        }
                    )
                )
                handle.write("\n")


__all__ = ["SelfPlayOrchestrator"]
