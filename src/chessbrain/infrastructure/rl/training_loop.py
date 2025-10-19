from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Dict

from src.chessbrain.domain.models.policy_value_network import AlphaZeroResidualNetwork
from src.chessbrain.domain.training.replay_buffer import ReplayBuffer
from src.chessbrain.domain.training.self_play import SelfPlayCollector, SelfPlayEpisode
from src.chessbrain.infrastructure.rl.torch_compat import HAS_TORCH, TORCH

if HAS_TORCH:
    import torch  # type: ignore
else:  # pragma: no cover - thin environments
    torch = TORCH  # type: ignore


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
        micro_batch_size: Optional[int] = None,
    ) -> None:
        self._device = device
        self._model = model
        self._collector = collector
        self._learning_rate = learning_rate
        self._l2_coefficient = l2_coefficient
        self._replay_buffer = replay_buffer
        self._optimizer = None
        self._grad_accum_steps = max(1, grad_accum_steps)
        self._use_amp = False
        self._scaler = None
        self._micro_batch_size = micro_batch_size

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

    def run(
        self,
        config: TrainingConfig,
        start_episode: int,
        max_episodes: Optional[int] = None,
        progress_callback: Optional[Callable[[EpisodeMetrics, int, int], None]] = None,
        checkpoint_interval: Optional[int] = None,
        on_checkpoint: Optional[Callable[[int, dict[str, Any]], None]] = None,
        episode_callback: Optional[Callable[[int, SelfPlayEpisode, Optional[Dict[str, float]]], None]] = None,
    ) -> TrainingLoopResult:
        """Execute a bounded number of episodes starting from `start_episode`."""
        if start_episode >= config.total_episodes:
            return TrainingLoopResult(episodes_played=0, metrics=[], checkpoint_state=None)

        remaining = config.total_episodes - start_episode
        episodes_to_run = remaining if max_episodes is None else min(remaining, max_episodes)
        if episodes_to_run <= 0:
            return TrainingLoopResult(episodes_played=0, metrics=[], checkpoint_state=None)

        if self._replay_buffer is None:
            capacity = max(config.batch_size * 4, config.batch_size)
            self._replay_buffer = ReplayBuffer(capacity=capacity)

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
            checkpoint_interval,
            on_checkpoint,
            episode_callback,
        )

    def attach_collector(self, collector: SelfPlayCollector) -> None:
        self._collector = collector

    def _should_use_self_play(self) -> bool:
        return HAS_TORCH and self._model is not None and self._collector is not None and self._optimizer is not None

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
        checkpoint_interval: Optional[int] = None,
        on_checkpoint: Optional[Callable[[int, dict[str, Any]], None]] = None,
        episode_callback: Optional[Callable[[int, SelfPlayEpisode, Optional[Dict[str, float]]], None]] = None,
    ) -> TrainingLoopResult:
        assert self._collector is not None
        assert self._model is not None
        assert self._optimizer is not None

        metrics: list[EpisodeMetrics] = []
        checkpoint_state: Optional[dict] = None

        TORCH.manual_seed(config.seed + start_episode)

        self._optimizer.zero_grad(set_to_none=True)
        accum_counter = 0
        last_checkpoint_state: Optional[dict[str, Any]] = None
        effective_interval = checkpoint_interval or config.checkpoint_interval

        for offset in range(episodes_to_run):
            episode_index = start_episode + offset + 1
            episode = self._collector.generate_episode(self._model)
            self._replay_buffer.add_episode(episode)

            episode_stats: Optional[dict[str, float]] = None
            if episode.samples:
                with TORCH.no_grad():
                    features = TORCH.stack([sample.features for sample in episode.samples]).to(self._device)
                    policy_targets = TORCH.stack([sample.policy_target for sample in episode.samples]).to(self._device)
                    legal_masks = TORCH.stack([sample.legal_mask for sample in episode.samples]).to(self._device)
                    safe_targets = self._normalize_targets(policy_targets, legal_masks)
                    value_targets = TORCH.tensor(
                        [sample.value_target for sample in episode.samples],
                        dtype=TORCH.float32,
                        device=self._device,
                    ).unsqueeze(1)
                    outputs = self._model(features)
                    episode_log_probs = outputs.log_probs(legal_mask=legal_masks)
                    ep_policy_loss = -(safe_targets * episode_log_probs).sum(dim=1).mean().item()
                    ep_value_loss = TORCH.nn.functional.mse_loss(outputs.value, value_targets).item()
                    episode_stats = {
                        "policy_loss": float(ep_policy_loss),
                        "value_loss": float(ep_value_loss),
                        "total_loss": float(ep_policy_loss + ep_value_loss),
                    }

            if episode_callback is not None:
                episode_callback(episode_index, episode, episode_stats)

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

            safe_policy_targets = self._normalize_targets(batch.policy_targets, batch.legal_masks)

            self._model.train()
            total_examples = batch.features.size(0)
            micro_size = self._micro_batch_size or total_examples
            micro_size = max(1, min(micro_size, total_examples))

            policy_loss_accum = 0.0
            value_loss_accum = 0.0

            for start in range(0, total_examples, micro_size):
                end = min(start + micro_size, total_examples)
                fraction = (end - start) / total_examples
                features_chunk = batch.features[start:end]
                targets_chunk = safe_policy_targets[start:end]
                masks_chunk = batch.legal_masks[start:end]
                values_chunk = batch.value_targets[start:end]

                if self._use_amp:
                    assert self._scaler is not None
                    with TORCH.cuda.amp.autocast():
                        output = self._model(features_chunk)
                        log_probs = output.log_probs(legal_mask=masks_chunk)
                        chunk_policy_loss = -(targets_chunk * log_probs).sum(dim=1).mean()
                        chunk_value_loss = TORCH.nn.functional.mse_loss(
                            output.value, values_chunk
                        )
                        chunk_loss = chunk_policy_loss + chunk_value_loss
                    self._scaler.scale(chunk_loss * fraction / self._grad_accum_steps).backward()
                else:
                    output = self._model(features_chunk)
                    log_probs = output.log_probs(legal_mask=masks_chunk)
                    chunk_policy_loss = -(targets_chunk * log_probs).sum(dim=1).mean()
                    chunk_value_loss = TORCH.nn.functional.mse_loss(
                        output.value, values_chunk
                    )
                    chunk_loss = chunk_policy_loss + chunk_value_loss
                    (chunk_loss * fraction / self._grad_accum_steps).backward()

                policy_loss_accum += chunk_policy_loss.detach().item() * (end - start)
                value_loss_accum += chunk_value_loss.detach().item() * (end - start)

            l2_loss_tensor = self._compute_l2_penalty()
            if self._use_amp:
                assert self._scaler is not None
                self._scaler.scale(l2_loss_tensor / self._grad_accum_steps).backward()
            else:
                (l2_loss_tensor / self._grad_accum_steps).backward()

            policy_loss_tensor = TORCH.tensor(
                policy_loss_accum / total_examples, device=self._device
            )
            value_loss_tensor = TORCH.tensor(
                value_loss_accum / total_examples, device=self._device
            )
            total_loss = policy_loss_tensor + value_loss_tensor + l2_loss_tensor

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

            metric = EpisodeMetrics(
                episode_index=episode_index,
                policy_loss=float(policy_loss_tensor.detach().cpu().item()),
                value_loss=float(value_loss_tensor.detach().cpu().item()),
                win_rate=episode.win_rate,
            )
            metrics.append(metric)

            if progress_callback is not None:
                progress_callback(metric, episode_index, config.total_episodes)

            emit_checkpoint = False
            if effective_interval:
                if episode_index % effective_interval == 0:
                    emit_checkpoint = True
            if episode_index == config.total_episodes:
                emit_checkpoint = True

            if emit_checkpoint:
                checkpoint_state = {
                    "global_step": episode_index,
                    "device": self._device.type,
                    "exploration_rate": config.exploration_rate,
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in self._model.state_dict().items()
                    },
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                last_checkpoint_state = checkpoint_state
                if on_checkpoint is not None:
                    on_checkpoint(episode_index, dict(checkpoint_state))

        return TrainingLoopResult(
            episodes_played=episodes_to_run,
            metrics=metrics,
            checkpoint_state=last_checkpoint_state,
        )

    @staticmethod
    def _normalize_targets(targets: TORCH.Tensor, masks: TORCH.Tensor) -> TORCH.Tensor:
        masked = targets * masks
        sums = masked.sum(dim=1, keepdim=True)
        legal_counts = masks.sum(dim=1, keepdim=True)
        legal_counts = TORCH.clamp(legal_counts, min=1.0)
        uniform = TORCH.where(
            masks > 0,
            masks / legal_counts,
            TORCH.zeros_like(masks),
        )
        normalized = TORCH.where(
            sums > 0,
            masked / TORCH.clamp(sums, min=1e-8),
            uniform,
        )
        return normalized

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
