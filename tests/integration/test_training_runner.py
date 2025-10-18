from __future__ import annotations

from pathlib import Path

from src.chessbrain.domain.training.self_play_orchestrator import SelfPlayOrchestrator
from src.chessbrain.infrastructure.persistence.training_job_repository import (
    TrainingJobRepository,
    TrainingJobStatus,
)
from src.chessbrain.infrastructure.rl.checkpoint_publisher import FileCheckpointPublisher
from src.chessbrain.infrastructure.rl.training_loop import TrainingConfig, TrainingLoop
from src.chessbrain.infrastructure.rl.torch_compat import TORCH


def test_training_job_lifecycle(app_config, db_session) -> None:  # type: ignore[no-untyped-def]
    checkpoint_root = app_config.model_checkpoint_dir
    tensorboard_root = app_config.tensorboard_log_dir
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tensorboard_root.mkdir(parents=True, exist_ok=True)

    repository = TrainingJobRepository(db_session)
    training_loop = TrainingLoop(device=TORCH.device("cpu"))
    publisher = FileCheckpointPublisher(root=checkpoint_root)

    orchestrator = SelfPlayOrchestrator(
        repository=repository,
        training_loop=training_loop,
        checkpoint_publisher=publisher,
        tensorboard_root=tensorboard_root,
    )

    config = TrainingConfig(
        total_episodes=4,
        batch_size=16,
        checkpoint_interval=2,
        exploration_rate=0.1,
        seed=99,
    )

    job = orchestrator.start_job(config)

    assert job.status is TrainingJobStatus.completed
    assert job.episodes_played == 4
    checkpoint_path = Path(checkpoint_root) / f"{job.checkpoint_version}.pt"
    assert checkpoint_path.exists()

    persisted = repository.get(job.id)
    assert persisted is not None
    assert persisted.status is TrainingJobStatus.completed
    assert persisted.metrics_uri is not None
    assert Path(persisted.metrics_uri).exists()
