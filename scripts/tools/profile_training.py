#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

from src.chessbrain.domain.models.policy_value_network import (
    ACTION_SPACE_PLANES,
    AlphaZeroResidualNetwork,
    BOARD_CHANNELS,
)


def build_activities(device: torch.device) -> list[ProfilerActivity]:
    activities: list[ProfilerActivity] = [ProfilerActivity.CPU]
    if device.type == "cuda" and hasattr(ProfilerActivity, "CUDA"):
        activities.append(ProfilerActivity.CUDA)  # type: ignore[arg-type]
    elif device.type == "mps" and hasattr(ProfilerActivity, "PrivateUse1"):
        activities.append(getattr(ProfilerActivity, "PrivateUse1"))
    return activities


def make_synthetic_batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(1234)
    inputs = torch.randn(batch_size, BOARD_CHANNELS, 8, 8, device=device)
    legal_mask = torch.ones(batch_size, ACTION_SPACE_PLANES * 64, dtype=torch.bool, device=device)
    pi_targets = torch.softmax(
        torch.randn(batch_size, ACTION_SPACE_PLANES * 64, device=device),
        dim=-1,
    )
    z_targets = torch.rand(batch_size, device=device) * 2 - 1
    return inputs, legal_mask, pi_targets, z_targets


def train_step(
    *,
    model: AlphaZeroResidualNetwork,
    optim: torch.optim.Optimizer,
    inputs: torch.Tensor,
    legal_mask: torch.Tensor,
    pi_targets: torch.Tensor,
    z_targets: torch.Tensor,
) -> torch.Tensor:
    optim.zero_grad(set_to_none=True)
    output = model(inputs)
    logits = output.flatten_policy().masked_fill(~legal_mask, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1)
    policy_loss = -(pi_targets * logp).sum(dim=-1).mean()
    value_loss = F.mse_loss(output.value.squeeze(-1), z_targets)
    loss = policy_loss + value_loss
    loss.backward()
    optim.step()
    return loss.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile AlphaZeroResidualNetwork training step performance.")
    parser.add_argument("--device", default=None, help="Torch device string (cpu, cuda, mps). Defaults to auto-detect.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=8)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--wait", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--active", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=12, help="Total train steps to execute during profiling.")
    parser.add_argument("--logdir", default="./tb_logs", help="TensorBoard trace output directory.")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = AlphaZeroResidualNetwork(
        residual_blocks=args.residual_blocks,
        channels=args.channels,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    inputs, legal_mask, pi_targets, z_targets = make_synthetic_batch(args.batch_size, device)

    activities = build_activities(device)
    trace_dir = Path(args.logdir).expanduser().resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)

    prof_schedule = schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=args.repeat)

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for step in range(args.iterations):
            loss = train_step(
                model=model,
                optim=optim,
                inputs=inputs,
                legal_mask=legal_mask,
                pi_targets=pi_targets,
                z_targets=z_targets,
            )
            prof.step()

    print(f"Trace written to: {trace_dir}")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))


if __name__ == "__main__":
    main()
