from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Iterable, Optional, Any, Dict

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None
from uuid import UUID, uuid4

import chess
import chess.pgn

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

    def start_job(
        self,
        config: TrainingConfig,
        *,
        start_episode: int = 0,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> TrainingJob:
        job_id = uuid4()
        metrics_dir = self._tensorboard_root / job_id.hex
        metrics_dir.mkdir(parents=True, exist_ok=True)

        job = self._repository.create(
            job_id=job_id,
            config=asdict(config),
            metrics_uri=str(metrics_dir),
        )
        job = self._repository.mark_running(job_id)
        if start_episode > 0:
            job = self._repository.record_progress(job_id, start_episode)

        metrics_path = metrics_dir / "metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("a", encoding="utf-8")
        metrics_written = False
        writer = SummaryWriter(str(metrics_dir)) if SummaryWriter is not None else None

        start_time = time.time()

        def progress(metric, current_episode: int, total_episodes: int) -> None:
            nonlocal metrics_written
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
            record = {
                "episode_index": current_episode,
                "total_episodes": total_episodes,
                "policy_loss": metric.policy_loss,
                "value_loss": metric.value_loss,
                "win_rate": metric.win_rate,
                "elapsed_seconds": elapsed,
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()
            metrics_written = True
            if writer is not None:
                writer.add_scalar("loss/policy", metric.policy_loss, current_episode)
                writer.add_scalar("loss/value", metric.value_loss, current_episode)
                writer.add_scalar("performance/win_rate", metric.win_rate, current_episode)
                writer.flush()

        last_artifact: Optional[CheckpointArtifact] = None
        last_state: Optional[dict[str, Any]] = resume_state.copy() if resume_state else None

        def on_checkpoint(step: int, state: dict[str, Any]) -> None:
            nonlocal last_artifact, last_state
            last_artifact = self._checkpoint_publisher.publish(job_id, state)
            last_state = state

        games_dir = metrics_dir / "games"
        games_dir.mkdir(parents=True, exist_ok=True)

        best_loss = float("-inf")

        def episode_callback(step: int, episode, stats: Optional[dict[str, float]]) -> None:
            nonlocal best_loss
            if stats is not None:
                total_loss = stats.get("total_loss", 0.0)
                if total_loss > best_loss:
                    best_loss = total_loss
                    self._save_episode_game(
                        games_dir / f"worst_loss_step_{step}.pgn",
                        episode,
                        step,
                        stats,
                    )
            if episode.termination == "CHECKMATE":
                self._save_episode_game(
                    games_dir / f"checkmate_step_{step}.pgn",
                    episode,
                    step,
                    stats,
                )

        try:
            remaining = max(config.total_episodes - start_episode, 1)
            checkpoint_every = max(1, remaining // 10)
            result = self._training_loop.run(
                config=config,
                start_episode=start_episode,
                max_episodes=config.total_episodes,
                progress_callback=progress,
                checkpoint_interval=checkpoint_every,
                on_checkpoint=on_checkpoint,
                episode_callback=episode_callback,
                resume_state=resume_state,
            )

            final_episode = (
                result.metrics[-1].episode_index
                if result.metrics
                else start_episode
            )
            job = self._repository.record_progress(job_id, final_episode)
            if not metrics_written:
                self._write_metrics(metrics_dir, result.metrics)

            checkpoint_state = result.checkpoint_state or last_state
            if checkpoint_state is None:
                last_episode = final_episode
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
                episodes_played=final_episode,
            )
            return job
        except Exception as exc:  # pragma: no cover - defensive fallback
            self._repository.set_failed(job_id, str(exc))
            raise
        finally:
            metrics_file.close()
            if writer is not None:
                writer.flush()
                writer.close()

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

    def _save_episode_game(
        self,
        path: Path,
        episode,
        episode_index: int,
        stats: Optional[Dict[str, float]],
    ) -> None:
        game = chess.pgn.Game()
        game.headers["Event"] = "ChessBrain Self-Play"
        game.headers["Round"] = str(episode_index)
        game.headers["Result"] = self._episode_result_header(episode)
        if episode.termination:
            game.headers["Termination"] = episode.termination
        if episode.winner:
            game.headers["Winner"] = episode.winner
        game.headers["FinalFEN"] = episode.final_fen
        game.headers["WinRate"] = f"{episode.win_rate:.2f}"
        if stats:
            game.headers["PolicyLoss"] = f"{stats.get('policy_loss', 0.0):.6f}"
            game.headers["ValueLoss"] = f"{stats.get('value_loss', 0.0):.6f}"
            game.headers["TotalLoss"] = f"{stats.get('total_loss', 0.0):.6f}"

        node = game
        board = chess.Board()
        for move_uci in episode.moves:
            move = chess.Move.from_uci(move_uci)
            node = node.add_variation(move)
            board.push(move)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(str(game))
            handle.write("\n")

    @staticmethod
    def _episode_result_header(episode) -> str:
        if episode.result > 0:
            return "1-0"
        if episode.result < 0:
            return "0-1"
        return "1/2-1/2"


__all__ = ["SelfPlayOrchestrator"]
