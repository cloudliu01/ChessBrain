from __future__ import annotations

from typing import Any, Optional

from src.chessbrain.infrastructure.config import AppConfig, load_config
from src.chessbrain.infrastructure.rl.torch_compat import HAS_TORCH, TORCH


def resolve_device(config: Optional[AppConfig] = None) -> Any:
    """Determine the preferred torch.device based on configuration."""
    cfg = config or load_config()
    preference = cfg.pytorch_device_preference.lower()

    if preference == "cuda" and TORCH.cuda.is_available():
        return TORCH.device("cuda")
    mps_backend = getattr(TORCH.backends, "mps", None)

    if preference == "mps" and mps_backend and mps_backend.is_available():
        return TORCH.device("mps")
    if preference == "cpu":
        return TORCH.device("cpu")

    if TORCH.cuda.is_available():
        return TORCH.device("cuda")
    if mps_backend and mps_backend.is_available():
        return TORCH.device("mps")
    return TORCH.device("cpu")


def device_summary(config: Optional[AppConfig] = None) -> dict[str, str]:
    """Provide metadata about the selected device for telemetry."""
    device = resolve_device(config)
    summary = {"type": device.type}
    if device.type == "cuda" and HAS_TORCH:
        summary["name"] = TORCH.cuda.get_device_name(device)
    return summary


__all__ = ["resolve_device", "device_summary"]
