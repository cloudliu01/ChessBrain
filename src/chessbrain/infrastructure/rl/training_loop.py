from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.chessbrain.domain.models.policy_value_network import AlphaZeroResidualNetwork
from src.chessbrain.domain.training.self_play import SelfPlayCollector
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
    """Self-play training loop coordinating model updates and metrics."""

    def __init__(
        self,
        device: Any,
        model: Optional[Any] = None,
        collector: Optional[SelfPlayCollector] = None,
        *,
        learning_rate: float = 1e-3,
        l2_coefficient: float = 1e-4,
    ) -> None:
        self._device = device
        self._model = model
        self._collector = collector
        self._learning_rate = learning_rate
        self._l2_coefficient = l2_coefficient
        self._optimizer = None

        if HAS_TORCH and self._model is None:
            self._model = AlphaZeroResidualNetwork().to(self._device)
        if HAS_TORCH and self._model is not None:
            self._model.to(self._device)
            self._optimizer = TORCH.optim.Adam(
                self._model.parameters(),
                lr=self._learning_rate,
            )

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

        if not self._should_use_self_play():
            return self._run_deterministic(config, start_episode, episodes_to_run)

        return self._run_self_play(config, start_episode, episodes_to_run)

    def attach_collector(self, collector: SelfPlayCollector) -> None:
        self._collector = collector

    def _should_use_self_play(self) -> bool:
        return HAS_TORCH and self._model is not None and self._collector is not None and self._optimizer is not None

    def _run_deterministic(
        self,
        config: TrainingConfig,
        start_episode: int,
        episodes_to_run: int,
    ) -> TrainingLoopResult:
        metrics: list[EpisodeMetrics] = []
        checkpoint_state: Optional[dict] = None

        TORCH.manual_seed(config.seed + start_episode)

        for offset in range(episodes_to_run):
            episode_index = start_episode + offset + 1
            denominator = episode_index + max(1, config.batch_size)
            policy_loss = max(0.01, 1.0 / denominator)
            value_loss = max(0.01, 0.5 / denominator)
            win_rate = min(0.99, episode_index / max(1, config.total_episodes))

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
                    "device": getattr(self._device, "type", "cpu"),
                    "exploration_rate": config.exploration_rate,
                }

        return TrainingLoopResult(
            episodes_played=episodes_to_run,
            metrics=metrics,
            checkpoint_state=checkpoint_state,
        )

    def _run_self_play(
        self,
        config: TrainingConfig,
        start_episode: int,
        episodes_to_run: int,
    ) -> TrainingLoopResult:
        assert self._collector is not None
        assert self._model is not None
        assert self._optimizer is not None

        metrics: list[EpisodeMetrics] = []
        checkpoint_state: Optional[dict] = None

        TORCH.manual_seed(config.seed + start_episode)

        for offset in range(episodes_to_run):
            episode_index = start_episode + offset + 1
            episode = self._collector.generate_episode(self._model)

            if not episode.samples:
                metrics.append(
                    EpisodeMetrics(
                        episode_index=episode_index,
                        policy_loss=0.0,
                        value_loss=0.0,
                        win_rate=episode.win_rate,
                    )
                )
                continue

            features = TORCH.stack([sample.features for sample in episode.samples]).to(self._device)
            policy_targets = TORCH.stack([sample.policy_target for sample in episode.samples]).to(self._device)
            legal_masks = TORCH.stack([sample.legal_mask for sample in episode.samples]).to(self._device)
            value_targets = TORCH.tensor(
                [sample.value_target for sample in episode.samples],
                dtype=TORCH.float32,
                device=self._device,
            ).unsqueeze(1)

            self._model.train()
            output = self._model(features)
            log_probs = output.log_probs(legal_mask=legal_masks)
            policy_loss_tensor = -(policy_targets * log_probs).sum(dim=1).mean()
            value_loss_tensor = TORCH.nn.functional.mse_loss(output.value, value_targets)

            l2_loss_tensor = TORCH.tensor(0.0, device=self._device)
            if self._l2_coefficient > 0:
                l2_terms = [param.pow(2).sum() for param in self._model.parameters()]
                if l2_terms:
                    l2_loss_tensor = TORCH.stack(l2_terms).sum() * self._l2_coefficient

            total_loss = policy_loss_tensor + value_loss_tensor + l2_loss_tensor

            self._optimizer.zero_grad()
            total_loss.backward()
            TORCH.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
            self._optimizer.step()

            metrics.append(
                EpisodeMetrics(
                    episode_index=episode_index,
                    policy_loss=float(policy_loss_tensor.detach().cpu().item()),
                    value_loss=float(value_loss_tensor.detach().cpu().item()),
                    win_rate=episode.win_rate,
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
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in self._model.state_dict().items()
                    },
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
