from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

try:  # pragma: no cover - runtime detection
    import torch as TORCH  # type: ignore[assignment]
    HAS_TORCH = True
except ModuleNotFoundError:  # pragma: no cover - thin environments
    HAS_TORCH = False

    class _StubDevice:
        def __init__(self, device_type: str) -> None:
            self.type = device_type

        def __repr__(self) -> str:
            return f"device(type='{self.type}')"

    class _CudaNamespace:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def get_device_name(device: _StubDevice) -> str:
            return "cpu"

    class _MpsNamespace:
        @staticmethod
        def is_available() -> bool:
            return False

    def _device_factory(device_type: str) -> _StubDevice:
        return _StubDevice(device_type)

    def _manual_seed(_: int) -> None:
        return None

    def _save(obj: object, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(obj, handle)

    TORCH = SimpleNamespace(
        device=_device_factory,
        manual_seed=_manual_seed,
        save=_save,
        cuda=_CudaNamespace(),
        backends=SimpleNamespace(mps=_MpsNamespace()),
    )


__all__ = ["HAS_TORCH", "TORCH"]
