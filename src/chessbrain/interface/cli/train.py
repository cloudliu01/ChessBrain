from __future__ import annotations

import click

from src.chessbrain.domain.models.policy_value_network import AlphaZeroResidualNetwork
from src.chessbrain.domain.training.self_play import SelfPlayCollector
from src.chessbrain.domain.training.self_play_orchestrator import SelfPlayOrchestrator
from src.chessbrain.infrastructure.config import load_config
from src.chessbrain.infrastructure.persistence.base import (
    Base,
    create_engine_from_config,
    session_scope,
)
from src.chessbrain.infrastructure.persistence.training_job_repository import (
    TrainingJobRepository,
)
from src.chessbrain.infrastructure.rl.checkpoint_publisher import FileCheckpointPublisher
from src.chessbrain.infrastructure.rl.device import resolve_device
from src.chessbrain.infrastructure.rl.training_loop import TrainingConfig, TrainingLoop


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--episodes", type=int, required=True, help="Total self-play episodes to run.")
@click.option("--batch-size", type=int, default=256, show_default=True)
@click.option("--checkpoint-interval", type=int, default=50, show_default=True)
@click.option("--exploration-rate", type=float, default=0.1, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--grad-accum-steps", type=int, default=1, show_default=True)
@click.option("--no-amp", is_flag=True, help="Disable automatic mixed precision (CUDA only).")
@click.option("--mcts-simulations", type=int, default=64, show_default=True, help="Number of MCTS rollouts per move.")
@click.option("--mcts-cpuct", type=float, default=1.5, show_default=True, help="Exploration constant for MCTS.")
@click.option("--producer-workers", type=int, default=0, show_default=True, help="Spawn N parallel episode producers (CPU).")
@click.option("--producer-queue-size", type=int, default=16, show_default=True)
@click.option("--producer-device", default="cpu", show_default=True)
def main(
    episodes: int,
    batch_size: int,
    checkpoint_interval: int,
    exploration_rate: float,
    seed: int,
    grad_accum_steps: int,
    no_amp: bool,
    mcts_simulations: int,
    mcts_cpuct: float,
    producer_workers: int,
    producer_queue_size: int,
    producer_device: str,
) -> None:
    """Run a self-play training cycle and persist resulting artifacts."""
    config = load_config()
    engine = create_engine_from_config(config)
    Base.metadata.create_all(bind=engine)

    device = resolve_device(config)
    model = AlphaZeroResidualNetwork()
    use_amp = (getattr(device, "type", "") == "cuda") and not no_amp

    collector_config = {
        "temperature": 1.0,
        "exploration_epsilon": exploration_rate,
        "max_moves": 128,
        "mcts_simulations": mcts_simulations,
        "mcts_c_puct": mcts_cpuct,
    }

    collector = None
    if producer_workers == 0:
        collector = SelfPlayCollector(device=device, **collector_config)
    training_loop = TrainingLoop(
        device=device,
        model=model,
        collector=collector,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        collector_config=collector_config,
        producer_workers=producer_workers,
        producer_queue_size=producer_queue_size,
        producer_device=producer_device,
    )
    checkpoint_publisher = FileCheckpointPublisher(config.model_checkpoint_dir)

    run_config = TrainingConfig(
        total_episodes=episodes,
        batch_size=batch_size,
        checkpoint_interval=checkpoint_interval,
        exploration_rate=exploration_rate,
        seed=seed,
    )

    with session_scope(config) as session:
        repository = TrainingJobRepository(session)
        orchestrator = SelfPlayOrchestrator(
            repository=repository,
            training_loop=training_loop,
            checkpoint_publisher=checkpoint_publisher,
            tensorboard_root=config.tensorboard_log_dir,
        )
        job = orchestrator.start_job(run_config)

    click.secho(
        f"Training job {job.id} completed; checkpoint version {job.checkpoint_version}",
        fg="green",
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main"]
