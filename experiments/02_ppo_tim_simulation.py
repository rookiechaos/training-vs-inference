#!/usr/bin/env python3
"""Exp 02: PPO collapse simulation under TIM (recompute vs bypass)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tim.compare import inject_synthetic_drift
from tim.ppo_ratios import PpoRatioMode, compute_ppo_ratios, ppo_clipped_loss


def simulate_training_steps(
    steps: int = 100,
    batch: int = 4,
    seq_len: int = 64,
    tim_noise: float = 5e-4,
    seed: int = 0,
) -> dict:
    gen = torch.Generator().manual_seed(seed)
    log_policy = torch.randn(batch, seq_len, generator=gen) * 0.2 - 1.0
    mask = torch.ones(batch, seq_len)

    history = {"recompute": [], "bypass": [], "decoupled": []}

    for step in range(steps):
        # Rollout from "inference engine"
        log_rollout = log_policy.detach()
        # Training engine recomputes with TIM noise
        log_train = inject_synthetic_drift(log_rollout, mask, noise_std=tim_noise, seed=seed + step)
        # Policy update drift
        log_policy = log_policy + torch.randn(batch, seq_len, generator=gen) * 0.02
        advantages = torch.randn(batch, seq_len, generator=gen)

        for mode in PpoRatioMode:
            ratios = compute_ppo_ratios(log_policy, log_train, log_rollout, mask, mode=mode)
            loss = ppo_clipped_loss(ratios["ratio"], advantages, mask)
            # Track ratio explosion as collapse proxy
            r = ratios["ratio"][mask.bool()]
            history[mode.value].append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "ratio_max": float(r.max().item()),
                    "ratio_mean": float(r.mean().item()),
                }
            )

    return history


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    history = simulate_training_steps()
    # Summarize collapse: steps where ratio_max > 10
    summary = {}
    for mode, records in history.items():
        explosions = sum(1 for r in records if r["ratio_max"] > 10)
        summary[mode] = {
            "ratio_explosion_steps": explosions,
            "final_ratio_max": records[-1]["ratio_max"],
            "mean_loss": sum(r["loss"] for r in records) / len(records),
        }

    report = {"experiment": "02_ppo_tim_simulation", "summary": summary, "history": history}
    path = out_dir / "02_ppo_tim_simulation.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary": summary}, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
