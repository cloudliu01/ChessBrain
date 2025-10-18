from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.chessbrain.domain.models.policy_value_network import (
    AlphaZeroResidualNetwork,
    BOARD_CHANNELS,
)
from src.chessbrain.infrastructure.rl.torch_compat import HAS_TORCH, TORCH


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

    def __init__(self, device: Any, model: Optional[Any] = None) -> None:
        self._device = device
        self._model = model
        if HAS_TORCH and self._model is None:
            self._model = AlphaZeroResidualNetwork().to(self._device)
        if HAS_TORCH and self._model is not None:
            self._model.to(self._device)

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
            if HAS_TORCH and self._model is not None:
                self._model.train()
                features = TORCH.randn(
                    (1, BOARD_CHANNELS, 8, 8),
                    device=self._device,
                )
                target_policy = TORCH.randint(
                    0, self._model.action_space_size, (1,), device=self._device
                )
                target_value = TORCH.rand((1, 1), device=self._device) * 2 - 1

                output = self._model(features)
                policy_logits = output.flatten_policy()
                policy_loss_tensor = TORCH.nn.functional.cross_entropy(
                    policy_logits, target_policy
                )
                value_loss_tensor = TORCH.nn.functional.mse_loss(output.value, target_value)

                policy_loss = float(policy_loss_tensor.item())
                value_loss = float(value_loss_tensor.item())
                win_rate = float(((target_value > 0).float().mean()).item())
            else:
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
                if HAS_TORCH and self._model is not None:
                    checkpoint_state["model_state_dict"] = {
                        key: value.detach().cpu()
                        for key, value in self._model.state_dict().items()
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
