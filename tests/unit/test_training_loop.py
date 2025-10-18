from __future__ import annotations

import math
from pathlib import Path

from src.chessbrain.infrastructure.rl.training_loop import (
    TrainingConfig,
    TrainingLoop,
)
from src.chessbrain.infrastructure.rl.torch_compat import TORCH


def test_training_loop_emits_metrics_and_checkpoint(tmp_path: Path) -> None:
    config = TrainingConfig(
        total_episodes=4,
        batch_size=8,
        checkpoint_interval=2,
        exploration_rate=0.15,
        seed=42,
    )
    loop = TrainingLoop(device=TORCH.device("cpu"))

    first_batch = loop.run(config=config, start_episode=0, max_episodes=1)
    assert first_batch.episodes_played == 1
    assert first_batch.metrics[0].episode_index == 1
    assert math.isclose(first_batch.metrics[0].win_rate, 0.25, rel_tol=1e-6)
    assert first_batch.checkpoint_state is None

    second_batch = loop.run(config=config, start_episode=1, max_episodes=1)
    assert second_batch.episodes_played == 1
    assert second_batch.metrics[0].episode_index == 2
    assert second_batch.metrics[0].policy_loss < first_batch.metrics[0].policy_loss
    assert second_batch.checkpoint_state is not None
    assert second_batch.checkpoint_state["global_step"] == 2

    final_batch = loop.run(config=config, start_episode=2, max_episodes=2)
    assert final_batch.episodes_played == 2
    assert final_batch.metrics[-1].episode_index == 4
    assert final_batch.checkpoint_state is not None
    assert final_batch.checkpoint_state["global_step"] == 4
