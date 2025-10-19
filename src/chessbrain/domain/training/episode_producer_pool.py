from __future__ import annotations

import multiprocessing as mp
import queue
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch

from src.chessbrain.domain.models.policy_value_network import AlphaZeroResidualNetwork
from src.chessbrain.domain.training.self_play import (
    SelfPlayCollector,
    SelfPlayEpisode,
    TrainingSample,
)


def _to_cpu_sample(sample: TrainingSample) -> TrainingSample:
    return TrainingSample(
        features=sample.features.cpu(),
        policy_target=sample.policy_target.cpu(),
        value_target=sample.value_target,
        legal_mask=sample.legal_mask.cpu(),
    )


def _to_cpu_episode(episode: SelfPlayEpisode) -> SelfPlayEpisode:
    cpu_samples = [_to_cpu_sample(sample) for sample in episode.samples]
    return SelfPlayEpisode(samples=cpu_samples, win_rate=episode.win_rate)


def _producer_worker(
    state_queue: mp.Queue,
    episode_queue: mp.Queue,
    stop_event: mp.Event,
    model_kwargs: Dict,
    collector_kwargs: Dict,
    device_str: str,
) -> None:
    torch.set_num_threads(1)
    device = torch.device(device_str)
    model = AlphaZeroResidualNetwork(**model_kwargs).to(device)
    collector = SelfPlayCollector(device=device, **collector_kwargs)

    # Load initial state (blocking)
    initial_state = state_queue.get()
    if initial_state is None:
        return
    model.load_state_dict(initial_state)

    while not stop_event.is_set():
        try:
            # Drain any queued updates
            while True:
                state_message = state_queue.get_nowait()
                if state_message is None:
                    stop_event.set()
                    return
                model.load_state_dict(state_message)
        except queue.Empty:
            pass

        episode = collector.generate_episode(model)
        cpu_episode = _to_cpu_episode(episode)

        try:
            episode_queue.put(cpu_episode, timeout=1)
        except queue.Full:
            continue


@dataclass
class EpisodeProducerPool:
    workers: int
    model_kwargs: Dict
    collector_kwargs: Dict
    device: str = "cpu"
    queue_size: int = 16

    def __post_init__(self) -> None:
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._stop_event = ctx.Event()
        self._episode_queue: mp.Queue = ctx.Queue(maxsize=self.queue_size)
        self._state_queues: list[mp.Queue] = []
        self._processes: list[mp.Process] = []
        self._started = False

    def start(self, initial_state: Dict[str, torch.Tensor]) -> None:
        if self._started or self.workers <= 0:
            return
        for _ in range(self.workers):
            state_queue: mp.Queue = self._ctx.Queue()
            process = self._ctx.Process(
                target=_producer_worker,
                args=(
                    state_queue,
                    self._episode_queue,
                    self._stop_event,
                    self.model_kwargs,
                    self.collector_kwargs,
                    self.device,
                ),
            )
            process.daemon = True
            process.start()
            self._state_queues.append(state_queue)
            self._processes.append(process)

        self.broadcast_state(initial_state)
        self._started = True

    def broadcast_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        if not self._started:
            return
        cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
        for state_queue in self._state_queues:
            state_queue.put(cpu_state)

    def update_from_model(self, model: AlphaZeroResidualNetwork) -> None:
        if not self._started:
            return
        self.broadcast_state(model.state_dict())

    def get_episode(self, timeout: float = 5.0) -> Optional[SelfPlayEpisode]:
        if not self._started:
            return None
        try:
            episode = self._episode_queue.get(timeout=timeout)
            return episode
        except queue.Empty:
            return None

    def shutdown(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        for state_queue in self._state_queues:
            state_queue.put(None)
        for process in self._processes:
            process.join(timeout=5)
        while not self._episode_queue.empty():
            try:
                self._episode_queue.get_nowait()
            except queue.Empty:
                break
        self._started = False


__all__ = ["EpisodeProducerPool"]
