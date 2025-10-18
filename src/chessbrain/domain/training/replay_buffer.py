from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Sequence

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

    def __init__(self, *, capacity: int = 8192) -> None:
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
        if batch_size <= 0 or batch_size >= len(self._storage):
            return tuple(self._storage)
        return tuple(random.sample(list(self._storage), batch_size))

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


__all__ = ["ReplayBatch", "ReplayBuffer"]
