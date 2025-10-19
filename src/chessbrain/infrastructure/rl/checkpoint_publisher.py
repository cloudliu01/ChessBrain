from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from src.chessbrain.infrastructure.rl.torch_compat import TORCH


@dataclass(frozen=True, slots=True)
class CheckpointArtifact:
    """Metadata describing a published checkpoint artifact."""

    version: str
    path: Path


class FileCheckpointPublisher:
    """Persist checkpoints as `.pt` files within the configured root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def publish(self, job_id: UUID, state: dict[str, Any]) -> CheckpointArtifact:
        self._root.mkdir(parents=True, exist_ok=True)

        global_step = state.get("global_step", 0)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        version = f"{job_id.hex[:8]}-step{global_step}-{timestamp}"
        path = self._root / f"{version}.pt"

        TORCH.save(state, path)
        return CheckpointArtifact(version=version, path=path)


__all__ = ["CheckpointArtifact", "FileCheckpointPublisher"]
