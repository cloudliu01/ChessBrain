from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Iterable, Optional, Any
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

        start_time = time.time()

        def progress(metric, current_episode: int, total_episodes: int) -> None:
            percent = 0
            if total_episodes > 0:
                percent = int(current_episode * 100 / total_episodes)
            elapsed = time.time() - start_time
            print(
                f"[training] episode {current_episode}/{total_episodes}"
                f" ({percent}%) | policy_loss={metric.policy_loss:.4f}"
                f" value_loss={metric.value_loss:.4f} win_rate={metric.win_rate:.2f}"
                f" elapsed={elapsed:.1f}s",
                flush=True,
            )

        last_artifact: Optional[CheckpointArtifact] = None
        last_state: Optional[dict[str, Any]] = None

        def on_checkpoint(step: int, state: dict[str, Any]) -> None:
            nonlocal last_artifact, last_state
            last_artifact = self._checkpoint_publisher.publish(job_id, state)
            last_state = state

        try:
            checkpoint_every = max(1, config.total_episodes // 10)
            result = self._training_loop.run(
                config=config,
                start_episode=job.episodes_played,
                max_episodes=config.total_episodes,
                progress_callback=progress,
                checkpoint_interval=checkpoint_every,
                on_checkpoint=on_checkpoint,
            )

            episodes_played = job.episodes_played + result.episodes_played
            job = self._repository.record_progress(job_id, episodes_played)
            self._write_metrics(metrics_dir, result.metrics)

            checkpoint_state = result.checkpoint_state or last_state
            if checkpoint_state is None:
                last_episode = result.metrics[-1].episode_index if result.metrics else episodes_played
                checkpoint_state = {
                    "global_step": last_episode,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

            if last_artifact is None:
                last_artifact = self._checkpoint_publisher.publish(job_id, checkpoint_state)
            artifact = last_artifact
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
