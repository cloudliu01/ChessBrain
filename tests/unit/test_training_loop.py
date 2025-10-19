from __future__ import annotations

from pathlib import Path

from src.chessbrain.domain.models.policy_value_network import AlphaZeroResidualNetwork
from src.chessbrain.infrastructure.rl.training_loop import TrainingConfig, TrainingLoop
from src.chessbrain.infrastructure.rl.torch_compat import HAS_TORCH, TORCH


def test_training_loop_emits_metrics_and_checkpoint(tmp_path: Path) -> None:
    config = TrainingConfig(
        total_episodes=4,
        batch_size=8,
        checkpoint_interval=2,
        exploration_rate=0.15,
        seed=42,
    )
    model = None
    collector = None
    if HAS_TORCH:
        from src.chessbrain.domain.training.self_play import SelfPlayCollector

        model = AlphaZeroResidualNetwork(residual_blocks=1, channels=64)
        collector = SelfPlayCollector(
            device=TORCH.device("cpu"),
            max_moves=8,
            exploration_epsilon=0.1,
            mcts_simulations=8,
        )
    loop = TrainingLoop(device=TORCH.device("cpu"), model=model, collector=collector)

    first_batch = loop.run(config=config, start_episode=0, max_episodes=1)
    assert first_batch.episodes_played == 1
    assert first_batch.metrics[0].episode_index == 1
    assert 0.0 <= first_batch.metrics[0].win_rate <= 1.0
    assert first_batch.checkpoint_state is None

    second_batch = loop.run(config=config, start_episode=1, max_episodes=1)
    assert second_batch.episodes_played == 1
    assert second_batch.metrics[0].episode_index == 2
    assert second_batch.checkpoint_state is not None
    assert second_batch.checkpoint_state["global_step"] == 2
    if HAS_TORCH and "model_state_dict" in second_batch.checkpoint_state:
        # Ensure serialized tensors moved to CPU
        values = list(second_batch.checkpoint_state["model_state_dict"].values())
        assert all(getattr(value, "device", None) is None or value.device.type == "cpu" for value in values)

    final_batch = loop.run(config=config, start_episode=2, max_episodes=2)
    assert final_batch.episodes_played == 2
    assert final_batch.metrics[-1].episode_index == 4
    assert final_batch.checkpoint_state is not None
    assert final_batch.checkpoint_state["global_step"] == 4
