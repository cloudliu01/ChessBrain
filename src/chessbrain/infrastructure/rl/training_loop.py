from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.chessbrain.domain.models.policy_value_network import AlphaZeroResidualNetwork
from src.chessbrain.domain.training.replay_buffer import ReplayBuffer
from src.chessbrain.domain.training.self_play import SelfPlayCollector
from src.chessbrain.domain.training.episode_producer_pool import EpisodeProducerPool
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
        replay_buffer: Optional[ReplayBuffer] = None,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        collector_config: Optional[dict] = None,
        producer_workers: int = 0,
        producer_queue_size: int = 16,
        producer_device: str = "cpu",
    ) -> None:
        self._device = device
        self._model = model
        self._collector = collector
        self._learning_rate = learning_rate
        self._l2_coefficient = l2_coefficient
        self._replay_buffer = replay_buffer or ReplayBuffer()
        self._optimizer = None
        self._grad_accum_steps = max(1, grad_accum_steps)
        self._use_amp = False
        self._scaler = None
        self._collector_config = collector_config or {}
        self._producer_pool: Optional[EpisodeProducerPool] = None

        if HAS_TORCH and self._model is None:
            self._model = AlphaZeroResidualNetwork().to(self._device)
        if HAS_TORCH and self._model is not None:
            self._model.to(self._device)
            self._optimizer = TORCH.optim.Adam(
                self._model.parameters(),
                lr=self._learning_rate,
            )
            if use_amp and getattr(self._device, "type", "") == "cuda":
                self._use_amp = True
                self._scaler = TORCH.cuda.amp.GradScaler()

        if self._collector is None and producer_workers <= 0:
            raise ValueError("SelfPlayCollector is required when no producer workers are configured.")

        if producer_workers > 0 and self._model is not None:
            model_kwargs = self._extract_model_kwargs(self._model)
            collector_kwargs = dict(self._collector_config)
            collector_kwargs.setdefault("temperature", 1.0)
            collector_kwargs.setdefault("max_moves", 160)
            collector_kwargs.setdefault("exploration_epsilon", 0.0)
            collector_kwargs.setdefault("mcts_simulations", 64)
            collector_kwargs.setdefault("mcts_c_puct", 1.5)
            self._producer_pool = EpisodeProducerPool(
                workers=producer_workers,
                model_kwargs=model_kwargs,
                collector_kwargs=collector_kwargs,
                device=producer_device,
                queue_size=producer_queue_size,
            )
            self._producer_pool.start(self._model.state_dict())

    def run(
        self,
        config: TrainingConfig,
        start_episode: int,
        max_episodes: Optional[int] = None,
        progress_callback: Optional[Callable[[EpisodeMetrics, int, int], None]] = None,
    ) -> TrainingLoopResult:
        """Execute a bounded number of episodes starting from `start_episode`."""
        if start_episode >= config.total_episodes:
            return TrainingLoopResult(episodes_played=0, metrics=[], checkpoint_state=None)

        remaining = config.total_episodes - start_episode
        episodes_to_run = remaining if max_episodes is None else min(remaining, max_episodes)
        if episodes_to_run <= 0:
            return TrainingLoopResult(episodes_played=0, metrics=[], checkpoint_state=None)

        if not self._should_use_self_play():
            return self._run_deterministic(
                config,
                start_episode,
                episodes_to_run,
                progress_callback,
            )

        return self._run_self_play(
            config,
            start_episode,
            episodes_to_run,
            progress_callback,
        )

    def attach_collector(self, collector: SelfPlayCollector) -> None:
        self._collector = collector

    def _should_use_self_play(self) -> bool:
        has_generator = self._collector is not None or self._producer_pool is not None
        return HAS_TORCH and self._model is not None and has_generator and self._optimizer is not None

    def _run_deterministic(
        self,
        config: TrainingConfig,
        start_episode: int,
        episodes_to_run: int,
        progress_callback: Optional[Callable[[EpisodeMetrics, int, int], None]] = None,
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

            metric = EpisodeMetrics(
                episode_index=episode_index,
                policy_loss=policy_loss,
                value_loss=value_loss,
                win_rate=win_rate,
            )
            metrics.append(metric)

            if progress_callback is not None:
                progress_callback(metric, episode_index, config.total_episodes)

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
        progress_callback: Optional[Callable[[EpisodeMetrics, int, int], None]] = None,
    ) -> TrainingLoopResult:
        if self._collector is None and self._producer_pool is None:
            raise RuntimeError("No episode generator available; provide collector or producer pool")
        assert self._model is not None
        assert self._optimizer is not None

        metrics: list[EpisodeMetrics] = []
        checkpoint_state: Optional[dict] = None

        TORCH.manual_seed(config.seed + start_episode)

        self._optimizer.zero_grad(set_to_none=True)
        accum_counter = 0

        for offset in range(episodes_to_run):
            episode_index = start_episode + offset + 1
            episode = self._next_episode()
            while episode is None:
                episode = self._next_episode()
            self._replay_buffer.add_episode(episode)

            samples = self._replay_buffer.sample(config.batch_size)
            if not samples:
                metrics.append(
                    EpisodeMetrics(
                        episode_index=episode_index,
                        policy_loss=0.0,
                        value_loss=0.0,
                        win_rate=episode.win_rate,
                    )
                )
                continue

            batch = self._replay_buffer.as_batch(samples, device=self._device)

            self._model.train()
            if self._use_amp:
                assert self._scaler is not None
                with TORCH.cuda.amp.autocast():
                    output = self._model(batch.features)
                    log_probs = output.log_probs(legal_mask=batch.legal_masks)
                    policy_loss_tensor = -(batch.policy_targets * log_probs).sum(dim=1).mean()
                    value_loss_tensor = TORCH.nn.functional.mse_loss(
                        output.value, batch.value_targets
                    )
                    l2_loss_tensor = self._compute_l2_penalty()
                    total_loss = policy_loss_tensor + value_loss_tensor + l2_loss_tensor
                loss = total_loss / self._grad_accum_steps
                self._scaler.scale(loss).backward()
            else:
                output = self._model(batch.features)
                log_probs = output.log_probs(legal_mask=batch.legal_masks)
                policy_loss_tensor = -(batch.policy_targets * log_probs).sum(dim=1).mean()
                value_loss_tensor = TORCH.nn.functional.mse_loss(
                    output.value, batch.value_targets
                )
                l2_loss_tensor = self._compute_l2_penalty()
                total_loss = policy_loss_tensor + value_loss_tensor + l2_loss_tensor
                loss = total_loss / self._grad_accum_steps
                loss.backward()

            accum_counter += 1
            should_step = (
                accum_counter % self._grad_accum_steps == 0
                or offset == episodes_to_run - 1
            )

            if should_step:
                if self._use_amp:
                    assert self._scaler is not None
                    self._scaler.unscale_(self._optimizer)
                TORCH.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
                if self._use_amp:
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)

                if self._producer_pool is not None:
                    self._producer_pool.update_from_model(self._model)

            metric = EpisodeMetrics(
                episode_index=episode_index,
                policy_loss=float(policy_loss_tensor.detach().cpu().item()),
                value_loss=float(value_loss_tensor.detach().cpu().item()),
                win_rate=episode.win_rate,
            )
            metrics.append(metric)

            if progress_callback is not None:
                progress_callback(metric, episode_index, config.total_episodes)

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

    def _next_episode(self) -> Optional[Any]:
        if self._producer_pool is not None:
            return self._producer_pool.get_episode(timeout=10)
        if self._collector is None:
            return None
        return self._collector.generate_episode(self._model)

    def _extract_model_kwargs(self, model: AlphaZeroResidualNetwork) -> dict:
        residual_blocks = len(getattr(model, "residual_blocks", []))
        channels = getattr(model.residual_blocks[0].conv1, "out_channels", 192) if residual_blocks else 192
        return {
            "residual_blocks": residual_blocks or 16,
            "channels": channels,
            "input_channels": getattr(model, "input_channels", 20),
            "action_space_size": getattr(model, "action_space_size", 4672),
        }

    def shutdown(self) -> None:
        if self._producer_pool is not None:
            self._producer_pool.shutdown()
            self._producer_pool = None

    def _compute_l2_penalty(self) -> TORCH.Tensor:
        if self._l2_coefficient <= 0:
            return TORCH.tensor(0.0, device=self._device)
        l2_terms = [param.pow(2).sum() for param in self._model.parameters()]
        if not l2_terms:
            return TORCH.tensor(0.0, device=self._device)
        return TORCH.stack(l2_terms).sum() * self._l2_coefficient


__all__ = [
    "EpisodeMetrics",
    "TrainingConfig",
    "TrainingLoop",
    "TrainingLoopResult",
]
