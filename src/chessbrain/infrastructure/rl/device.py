from __future__ import annotations

from typing import Optional
import torch

from src.chessbrain.infrastructure.config import AppConfig, load_config


def resolve_device(config: Optional[AppConfig] = None) -> torch.device:
    """Determine the preferred torch.device based on configuration."""
    cfg = config or load_config()
    preference = cfg.pytorch_device_preference.lower()

    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)

    if preference == "mps" and mps_backend and mps_backend.is_available():
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_backend and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_summary(config: Optional[AppConfig] = None) -> dict[str, str]:
    """Provide metadata about the selected device for telemetry."""
    device = resolve_device(config)
    summary = {"type": device.type}
    if device.type == "cuda":
        summary["name"] = torch.cuda.get_device_name(device)
    return summary


__all__ = ["resolve_device", "device_summary"]
