from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Iterable, List, Sequence

import random

import torch

from .self_play import SelfPlayEpisode, TrainingSample


@dataclass(frozen=True)
class ReplayBatch:
    features: torch.Tensor
    policy_targets: torch.Tensor
    legal_masks: torch.Tensor
    value_targets: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity buffer storing self-play training samples."""

    def __init__(self, *, capacity: int = 32768) -> None:
        self._capacity = capacity
        self._storage: Deque[TrainingSample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def add_episode(self, episode: SelfPlayEpisode) -> None:
        for sample in episode.samples:
            self._storage.append(
                TrainingSample(
                    features=sample.features.detach().cpu(),
                    policy_target=sample.policy_target.detach().cpu(),
                    value_target=sample.value_target,
                    legal_mask=sample.legal_mask.detach().cpu(),
                )
            )

    def sample(self, batch_size: int) -> Sequence[TrainingSample]:
        if not self._storage:
            return ()
        if batch_size <= 0:
            return ()

        window = max(batch_size * 4, batch_size * 2)
        candidates = list(self._storage)[-window:]
        if batch_size >= len(candidates):
            return tuple(candidates)
        return tuple(random.sample(candidates, batch_size))

    def as_batch(self, samples: Iterable[TrainingSample], *, device: torch.device) -> ReplayBatch:
        samples_list = list(samples)
        if not samples_list:
            raise ValueError("ReplayBatch requires at least one sample")

        features = torch.stack([sample.features for sample in samples_list]).to(device)
        policy_targets = torch.stack([sample.policy_target for sample in samples_list]).to(device)
        legal_masks = torch.stack([sample.legal_mask for sample in samples_list]).to(device)
        value_targets = torch.tensor(
            [sample.value_target for sample in samples_list],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        return ReplayBatch(
            features=features,
            policy_targets=policy_targets,
            legal_masks=legal_masks,
            value_targets=value_targets,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize buffer contents for checkpointing."""
        samples_payload: list[dict[str, torch.Tensor | float]] = []

        for sample in self._storage:
            samples_payload.append(
                {
                    "features": sample.features.detach().clone().cpu(),
                    "policy_target": sample.policy_target.detach().clone().cpu(),
                    "legal_mask": sample.legal_mask.detach().clone().cpu(),
                    "value_target": float(sample.value_target),
                }
            )

        return {
            "capacity": self._capacity,
            "samples": samples_payload,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore buffer contents from a previously serialized snapshot."""
        capacity = int(state.get("capacity", self._capacity))
        samples_payload = state.get("samples", [])

        self._capacity = capacity
        self._storage = deque(maxlen=capacity)

        for payload in samples_payload:
            features = payload["features"]
            policy_target = payload["policy_target"]
            legal_mask = payload["legal_mask"]
            value_target = float(payload["value_target"])
            if isinstance(features, torch.Tensor):
                features = features.detach().clone().cpu()
            if isinstance(policy_target, torch.Tensor):
                policy_target = policy_target.detach().clone().cpu()
            if isinstance(legal_mask, torch.Tensor):
                legal_mask = legal_mask.detach().clone().cpu()
            self._storage.append(
                TrainingSample(
                    features=features,
                    policy_target=policy_target,
                    value_target=value_target,
                    legal_mask=legal_mask,
                )
            )


__all__ = ["ReplayBatch", "ReplayBuffer"]
