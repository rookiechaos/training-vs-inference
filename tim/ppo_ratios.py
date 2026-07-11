"""PPO / GRPO importance sampling ratio implementations under TIM."""

from __future__ import annotations

from enum import Enum

import torch


class PpoRatioMode(str, Enum):
    """How to compute the behavior-policy denominator in PPO."""

    RECOMPUTE = "recompute"  # π_old from training engine (Megatron) — incorrect under TIM
    BYPASS = "bypass"  # π_old from rollout engine (SGLang) — correct behavior policy
    DECOUPLED = "decoupled"  # separate proximal (train) and behavior (rollout) policies


def compute_ppo_ratios(
    log_probs_current: torch.Tensor,
    log_probs_old_train: torch.Tensor,
    log_probs_old_rollout: torch.Tensor,
    mask: torch.Tensor,
    mode: PpoRatioMode = PpoRatioMode.BYPASS,
) -> dict[str, torch.Tensor]:
    """
    Compute PPO policy ratios under different TIM handling strategies.

    Returns dict with:
        - ratio: primary ratio used for PPO clip objective
        - ratio_proximal: π_θ / π_old_train (for decoupled clip anchor)
        - ratio_behavior: π_θ / π_old_rollout (for IS correction)
        - log_ratio: log of primary ratio
    """
    log_current = log_probs_current * mask
    log_old_train = log_probs_old_train * mask
    log_old_rollout = log_probs_old_rollout * mask

    ratio_proximal = torch.exp((log_current - log_old_train).clamp(min=-20.0, max=20.0))
    ratio_behavior = torch.exp((log_current - log_old_rollout).clamp(min=-20.0, max=20.0))
    ratio_recompute = ratio_proximal  # alias: classic (broken under TIM) implementation

    if mode == PpoRatioMode.RECOMPUTE:
        ratio = ratio_recompute
        log_ratio = log_current - log_old_train
    elif mode == PpoRatioMode.BYPASS:
        ratio = ratio_behavior
        log_ratio = log_current - log_old_rollout
    elif mode == PpoRatioMode.DECOUPLED:
        # Decoupled PPO uses behavior ratio for IS, proximal for clipping
        ratio = ratio_behavior
        log_ratio = log_current - log_old_rollout
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "ratio": ratio * mask + (1 - mask),
        "ratio_proximal": ratio_proximal * mask + (1 - mask),
        "ratio_behavior": ratio_behavior * mask + (1 - mask),
        "ratio_recompute": ratio_recompute * mask + (1 - mask),
        "log_ratio": log_ratio * mask,
    }


def ppo_clipped_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Standard PPO clipped surrogate (token-level)."""
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(unclipped, clipped)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def decoupled_ppo_loss(
    ratio_proximal: torch.Tensor,
    ratio_behavior: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    is_threshold: float = 2.0,
) -> torch.Tensor:
    """
    Decoupled PPO: clip on proximal ratio, IS-correct with behavior ratio.

    L = -E[ min(r_prox * A, clip(r_prox) * A) * truncate(r_beh) ]
    """
    unclipped = ratio_proximal * advantages
    clipped = torch.clamp(ratio_proximal, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    surrogate = torch.min(unclipped, clipped)
    is_weight = torch.clamp(ratio_behavior, max=is_threshold)
    loss = -(surrogate * is_weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1)
