from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Sequence

import chess

from src.chessbrain.domain.models.inference_service import InferenceService, MoveSuggestion
from src.chessbrain.domain.models.policy_value_network import AlphaZeroResidualNetwork
from src.chessbrain.domain.training.features import board_to_tensor
from src.chessbrain.domain.models.move_encoding import (
    decode_index,
    legal_moves_mask,
    policy_from_masked_logits,
)
from src.chessbrain.infrastructure.rl.torch_compat import HAS_TORCH, TORCH

if HAS_TORCH:  # pragma: no cover - import guarded for thin envs
    import torch
else:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class TorchInferenceAdapter(InferenceService):
    """PyTorch-backed inference service loading AlphaZero checkpoints."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for inference.")

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Checkpoint {self._model_path} not found.")

        self._device = torch.device(device)
        self._model: AlphaZeroResidualNetwork | None = None
        self._model_version = self._model_path.stem
        self._lock = Lock()

    @property
    def model_version(self) -> str | None:
        return self._model_version

    def select_move(
        self,
        board: chess.Board,
        *,
        difficulty: str = "deterministic",
    ) -> MoveSuggestion:
        model = self._ensure_model()

        with torch.no_grad():
            features = board_to_tensor(board, device=self._device).unsqueeze(0)
            legal_mask = legal_moves_mask(board, device=self._device)
            output = model.inference(features)
            logits = output.flatten_policy().squeeze(0)
            probabilities = policy_from_masked_logits(logits, legal_mask)

            if probabilities.sum().item() <= 0:
                probabilities = legal_mask / max(1, int(legal_mask.sum().item()))

            if difficulty == "stochastic":
                sampled_index = torch.multinomial(probabilities, num_samples=1).item()
            else:
                sampled_index = torch.argmax(probabilities).item()

            move = decode_index(board, int(sampled_index))
            if move is None:
                move = self._fallback_move(board)

            move_probability = float(probabilities[int(sampled_index)].detach().cpu().item())
            evaluation = float(output.value.squeeze().detach().cpu().item())
            rationale = self._top_rationale(board, probabilities)

        return MoveSuggestion(
            move=move,
            probability=move_probability,
            evaluation=evaluation,
            rationale=rationale,
        )

    def _ensure_model(self) -> AlphaZeroResidualNetwork:
        with self._lock:
            if self._model is not None:
                return self._model

            checkpoint = TORCH.load(self._model_path, map_location=self._device)
            state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
            model = AlphaZeroResidualNetwork()
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()

            version = checkpoint.get("version") if isinstance(checkpoint, dict) else None
            if isinstance(version, str):
                self._model_version = version

            self._model = model
            return model

    def _fallback_move(self, board: chess.Board) -> chess.Move:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise RuntimeError("No legal moves available for fallback inference.")
        return legal_moves[0]

    def _top_rationale(self, board: chess.Board, probs: torch.Tensor, k: int = 3) -> Sequence[str]:
        if probs.numel() == 0:
            return []
        topk = torch.topk(probs, k=min(k, probs.numel()))
        rationale: list[str] = []
        for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
            if prob <= 0:
                continue
            move = decode_index(board, int(idx))
            if move is None:
                continue
            rationale.append(f"{board.san(move)} ({prob:.3f})")
        return rationale


__all__ = ["TorchInferenceAdapter"]
