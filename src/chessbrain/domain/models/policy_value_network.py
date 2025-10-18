from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, TYPE_CHECKING

import math

try:  # pragma: no cover - executed at import
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - when torch missing
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]
    TORCH_AVAILABLE = False


if TYPE_CHECKING:  # pragma: no cover
    import torch as torch_typing
    TensorLike = torch_typing.Tensor
else:
    TensorLike = Any


# === Chess-specific shapes ===
ACTION_SPACE_PLANES = 73  # AlphaZero-like directional planes
ACTION_SPACE_SIZE = ACTION_SPACE_PLANES * 8 * 8
BOARD_CHANNELS = 20       # 12 piece planes + stm + castling + halfmove + extras


class PolicyValueOutput(NamedTuple):
    """
    policy_logits: (B, PLANES, 8, 8)
    value: (B, 1)
    """
    policy_logits: TensorLike
    value: TensorLike

    def flatten_policy(self) -> TensorLike:
        """Flatten policy logits into (B, ACTION_SPACE_SIZE)."""
        return self.policy_logits.reshape(self.policy_logits.size(0), -1)

    def log_probs(self, legal_mask: Optional[TensorLike] = None) -> TensorLike:
        """
        Return log probabilities with optional legal mask.
        legal_mask should be a bool tensor with shape (B, ACTION_SPACE_SIZE).
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to compute log probabilities.")
        logits = self.flatten_policy()
        if legal_mask is not None:
            # Ensure boolean mask for stable masking; illegal -> -inf
            if legal_mask.dtype != torch.bool:
                legal_mask = legal_mask > 0
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        return F.log_softmax(logits, dim=-1)


@dataclass(frozen=True)
class ResidualBlockConfig:
    channels: int = 128
    kernel_size: int = 3


if TORCH_AVAILABLE:

    class ResidualBlock(nn.Module):
        """Single AlphaZero-style residual block with zero-init on the last BN gamma."""

        def __init__(self, config: ResidualBlockConfig) -> None:
            super().__init__()
            padding = config.kernel_size // 2
            C = config.channels
            self.conv1 = nn.Conv2d(C, C, config.kernel_size, padding=padding, bias=False)
            self.bn1 = nn.BatchNorm2d(C)
            self.conv2 = nn.Conv2d(C, C, config.kernel_size, padding=padding, bias=False)
            self.bn2 = nn.BatchNorm2d(C)

            # Zero init bn2.weight (gamma) so the block starts as near-identity.
            nn.init.zeros_(self.bn2.weight)
            nn.init.zeros_(self.bn2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x = x + residual
            return F.relu(x)


    class PolicyHead(nn.Module):
        """Head producing logits for directional planes over the board."""

        def __init__(self, channels: int, action_planes: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, 64, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, action_planes, kernel_size=1, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = F.relu(self.bn1(self.conv1(x)))
            return self.conv2(x)  # (B, PLANES, 8, 8)


    class ValueHead(nn.Module):
        """Head approximating scalar value in range [-1, 1] with GAP."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(channels, 64, kernel_size=1, bias=False)
            self.bn = nn.BatchNorm2d(64)
            self.fc1 = nn.Linear(64, 256)  # after GAP -> 64 dims
            self.fc2 = nn.Linear(256, 1)

            # Optional: small init on final layer
            bound = 1 / math.sqrt(self.fc2.out_features)
            nn.init.uniform_(self.fc2.weight, -bound, bound)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = F.relu(self.bn(self.conv(x)))
            x = x.mean(dim=(-2, -1))        # Global Average Pooling -> (B, 64)
            x = F.relu(self.fc1(x))
            return torch.tanh(self.fc2(x))  # (B, 1)


    class AlphaZeroResidualNetwork(nn.Module):
        """Residual tower with policy and value heads inspired by AlphaZero."""

        def __init__(
            self,
            residual_blocks: int = 16,
            channels: int = 192,
            input_channels: int = BOARD_CHANNELS,
            action_space_size: int = ACTION_SPACE_SIZE,
        ) -> None:
            super().__init__()
            self.input_channels = input_channels
            self.action_space_size = action_space_size
            self.action_space_planes = max(1, action_space_size // (8 * 8))

            # Sanity check: planes * 64 must match the flat action size
            assert self.action_space_planes * 64 == self.action_space_size, (
                "Action space shape mismatch: action_space_planes * 64 must equal action_space_size. "
                f"Got planes={self.action_space_planes}, size={self.action_space_size}"
            )

            self.initial_conv = nn.Conv2d(
                input_channels, channels, kernel_size=3, padding=1, bias=False
            )
            self.initial_bn = nn.BatchNorm2d(channels)

            block_config = ResidualBlockConfig(channels=channels)
            self.residual_blocks = nn.ModuleList(
                ResidualBlock(block_config) for _ in range(residual_blocks)
            )

            self.policy_head = PolicyHead(channels, self.action_space_planes)
            self.value_head = ValueHead(channels)

            self._initialize_parameters()

        def _initialize_parameters(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    # Do not overwrite bn2 in ResidualBlock (already zero-inited)
                    if m.weight is not None and torch.count_nonzero(m.weight).item() != 0:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None and torch.count_nonzero(m.bias).item() != 0:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    bound = 1 / math.sqrt(m.out_features)
                    nn.init.uniform_(m.weight, -bound, bound)
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> PolicyValueOutput:  # type: ignore[override]
            if x.ndim != 4:
                raise ValueError("Expected input tensor of shape (batch, channels, height, width)")
            if x.size(1) != self.input_channels:
                raise ValueError(f"Expected {self.input_channels} channels, received {x.size(1)}")

            x = F.relu(self.initial_bn(self.initial_conv(x)))
            for block in self.residual_blocks:
                x = block(x)

            policy_logits = self.policy_head(x)  # (B, PLANES, 8, 8)
            value = self.value_head(x)           # (B, 1)
            return PolicyValueOutput(policy_logits=policy_logits, value=value)

        @torch.no_grad()
        def inference(self, features: torch.Tensor, temperature: Optional[float] = None) -> PolicyValueOutput:
            """
            Run forward pass plus optional temperature scaling for inference.
            Note: Apply your legal mask and softmax outside when sampling actions.
            """
            self.eval()
            output = self.forward(features)
            if temperature is None or temperature == 1:
                return output
            scaled_logits = output.policy_logits / max(temperature, 1e-5)
            return PolicyValueOutput(policy_logits=scaled_logits, value=output.value)


else:

    class AlphaZeroResidualNetwork:  # type: ignore[no-redef]
        """Placeholder when torch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required to instantiate AlphaZeroResidualNetwork.")

        def inference(self, *args: Any, **kwargs: Any) -> PolicyValueOutput:
            raise RuntimeError("PyTorch is required to run inference.")


__all__ = [
    "ACTION_SPACE_SIZE",
    "AlphaZeroResidualNetwork",
    "BOARD_CHANNELS",
    "PolicyValueOutput",
]