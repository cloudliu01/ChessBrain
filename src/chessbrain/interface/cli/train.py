from __future__ import annotations

import click

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
def main(
    episodes: int,
    batch_size: int,
    checkpoint_interval: int,
    exploration_rate: float,
    seed: int,
) -> None:
    """Run a self-play training cycle and persist resulting artifacts."""
    config = load_config()
    engine = create_engine_from_config(config)
    Base.metadata.create_all(bind=engine)

    device = resolve_device(config)
    training_loop = TrainingLoop(device=device)
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
