from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.chessbrain.infrastructure.rl.torch_compat import TORCH


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Static configuration for a self-play training run."""

    total_episodes: int
    batch_size: int
    checkpoint_interval: int
    exploration_rate: float
    seed: int = 0


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Lightweight metrics captured per episode for downstream logging."""

    episode_index: int
    policy_loss: float
    value_loss: float
    win_rate: float


@dataclass(frozen=True, slots=True)
class TrainingLoopResult:
    """Aggregate result for a batch of training episodes."""

    episodes_played: int
    metrics: list[EpisodeMetrics]
    checkpoint_state: Optional[dict]


class TrainingLoop:
    """Simplified self-play training loop with deterministic metrics."""

    def __init__(self, device: Any) -> None:
        self._device = device

    def run(
        self,
        config: TrainingConfig,
        start_episode: int,
        max_episodes: Optional[int] = None,
    ) -> TrainingLoopResult:
        """Execute a bounded number of episodes starting from `start_episode`."""
        if start_episode >= config.total_episodes:
            return TrainingLoopResult(episodes_played=0, metrics=[], checkpoint_state=None)

        remaining = config.total_episodes - start_episode
        episodes_to_run = remaining if max_episodes is None else min(remaining, max_episodes)
        if episodes_to_run <= 0:
            return TrainingLoopResult(episodes_played=0, metrics=[], checkpoint_state=None)

        metrics: list[EpisodeMetrics] = []
        checkpoint_state: Optional[dict] = None

        TORCH.manual_seed(config.seed + start_episode)

        for offset in range(episodes_to_run):
            episode_index = start_episode + offset + 1
            denominator = episode_index + config.batch_size
            policy_loss = max(0.01, 1.0 / denominator)
            value_loss = max(0.01, 0.5 / denominator)
            win_rate = min(0.99, episode_index / config.total_episodes)

            metrics.append(
                EpisodeMetrics(
                    episode_index=episode_index,
                    policy_loss=policy_loss,
                    value_loss=value_loss,
                    win_rate=win_rate,
                )
            )

            should_checkpoint = (
                episode_index == config.total_episodes
                or episode_index % max(1, config.checkpoint_interval) == 0
            )

            if should_checkpoint:
                checkpoint_state = {
                    "global_step": episode_index,
                    "device": self._device.type,
                    "exploration_rate": config.exploration_rate,
                }

        return TrainingLoopResult(
            episodes_played=episodes_to_run,
            metrics=metrics,
            checkpoint_state=checkpoint_state,
        )


__all__ = [
    "EpisodeMetrics",
    "TrainingConfig",
    "TrainingLoop",
    "TrainingLoopResult",
]
