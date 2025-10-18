#!/usr/bin/env python
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from src.chessbrain.domain.models.policy_value_network import (
    ACTION_SPACE_PLANES,
    AlphaZeroResidualNetwork,
    BOARD_CHANNELS,
)


def benchmark(
    *,
    device: str,
    residual_blocks: int,
    channels: int,
    batch_size: int,
    warmup_steps: int,
    iterations: int,
    lr: float,
) -> tuple[float, float]:
    torch.manual_seed(42)
    model = AlphaZeroResidualNetwork(residual_blocks=residual_blocks, channels=channels).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    x = torch.randn(batch_size, BOARD_CHANNELS, 8, 8, device=device)
    mask = torch.ones(batch_size, ACTION_SPACE_PLANES * 64, dtype=torch.bool, device=device)
    pi_targets = torch.softmax(
        torch.randn(batch_size, ACTION_SPACE_PLANES * 64, device=device),
        dim=-1,
    )
    z_targets = torch.rand(batch_size, device=device) * 2 - 1

    for _ in range(warmup_steps):
        loss = _forward_backward(model, optim, x, mask, pi_targets, z_targets)

    torch.cuda.empty_cache() if device.startswith("cuda") else None

    t0 = time.time()
    for _ in range(iterations):
        loss = _forward_backward(model, optim, x, mask, pi_targets, z_targets)
    t1 = time.time()

    elapsed = max(t1 - t0, 1e-9)
    steps_per_s = iterations / elapsed
    samples_per_s = steps_per_s * batch_size
    return steps_per_s, samples_per_s


def _forward_backward(model, optim, x, mask, pi_targets, z_targets):
    out = model(x)
    logits = out.flatten_policy().masked_fill(~mask, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1)
    policy_loss = -(pi_targets * logp).sum(dim=-1).mean()
    value_loss = F.mse_loss(out.value.squeeze(-1), z_targets)
    loss = policy_loss + value_loss
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AlphaZeroResidualNetwork training throughput.")
    parser.add_argument("--device", default="mps", help="Device string (cpu, cuda, mps, etc.).")
    parser.add_argument("--residual-blocks", type=int, default=8)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    steps_per_s, samples_per_s = benchmark(
        device=device,
        residual_blocks=args.residual_blocks,
        channels=args.channels,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        iterations=args.iterations,
        lr=args.lr,
    )
    print(f"steps/s={steps_per_s:.2f}, samples/s={samples_per_s:.0f}")


if __name__ == "__main__":
    main()
