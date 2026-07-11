"""TIM diagnostic metrics for rollout vs training logprob mismatch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


def _masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask.bool()]


def compute_log_ratio(
    log_probs_train: torch.Tensor,
    log_probs_rollout: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """log(π_train / π_rollout) per token."""
    log_ratio = log_probs_train - log_probs_rollout
    if mask is not None:
        log_ratio = log_ratio * mask
    return log_ratio


def compute_importance_weights(
    log_ratio: torch.Tensor,
    mask: torch.Tensor,
    *,
    level: str = "token",
    threshold: float | None = None,
    batch_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute IS weights w = exp(log_ratio) with optional truncation.

    level: "token" | "sequence" | "geometric"
    """
    masked_ratio = _masked(log_ratio, mask)
    if masked_ratio.numel() == 0:
        return torch.zeros_like(log_ratio), {"is_mean": 1.0, "is_max": 1.0, "ess": 0.0}

    if level == "token":
        weights = torch.exp(log_ratio.clamp(min=-20.0, max=20.0))
        weights = weights * mask
        active = _masked(weights, mask)
    elif level == "sequence":
        seq_log_ratio = (log_ratio * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
        weights = torch.exp(seq_log_ratio.clamp(min=-20.0, max=20.0)).expand_as(log_ratio) * mask
        active = weights[mask.bool()]
    elif level == "geometric":
        log_geom = (log_ratio * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
        weights = torch.exp(log_geom.clamp(min=-20.0, max=20.0)).expand_as(log_ratio) * mask
        active = weights[mask.bool()]
    else:
        raise ValueError(f"Unknown IS level: {level}")

    if threshold is not None:
        weights = torch.clamp(weights, max=threshold) * mask
        active = _masked(weights, mask)

    if batch_normalize and active.numel() > 0:
        mean_w = active.mean()
        if mean_w > 0:
            weights = (weights / mean_w) * mask
            active = _masked(weights, mask)

    ess = float((active.sum().item() ** 2) / (active.pow(2).sum().item() + 1e-12)) if active.numel() else 0.0
    stats = {
        "is_mean": float(active.mean().item()) if active.numel() else 1.0,
        "is_max": float(active.max().item()) if active.numel() else 1.0,
        "is_min": float(active.min().item()) if active.numel() else 1.0,
        "ess": ess,
    }
    return weights, stats


def compute_perplexity(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence perplexity from token log probs."""
    token_counts = mask.sum(dim=-1).clamp(min=1)
    nll = -(log_probs * mask).sum(dim=-1) / token_counts
    return torch.exp(nll)


def compute_kl_divergence(
    log_probs_p: torch.Tensor,
    log_probs_q: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Mean KL(P||Q) ≈ E[log P - log Q] over masked tokens."""
    diff = _masked(log_probs_p - log_probs_q, mask)
    if diff.numel() == 0:
        return 0.0
    return float(diff.mean().item())


@dataclass
class TimDiagnostics:
    """Aggregated TIM metrics for a batch of sequences."""

    rollout_ppl: float
    training_ppl: float
    log_ppl_diff: float
    ppl_ratio: float
    kl_train_vs_rollout: float
    is_mean: float
    is_max: float
    is_min: float
    ess: float
    token_log_ratio_mean: float
    token_log_ratio_std: float
    token_log_ratio_max: float
    outlier_fraction: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollout_ppl": self.rollout_ppl,
            "training_ppl": self.training_ppl,
            "log_ppl_diff": self.log_ppl_diff,
            "ppl_ratio": self.ppl_ratio,
            "kl_train_vs_rollout": self.kl_train_vs_rollout,
            "is_mean": self.is_mean,
            "is_max": self.is_max,
            "is_min": self.is_min,
            "ess": self.ess,
            "token_log_ratio_mean": self.token_log_ratio_mean,
            "token_log_ratio_std": self.token_log_ratio_std,
            "token_log_ratio_max": self.token_log_ratio_max,
            "outlier_fraction": self.outlier_fraction,
            **self.extra,
        }

    def severity(self) -> str:
        """Heuristic TIM severity for quick diagnosis."""
        if self.is_mean > 1.5 or self.is_mean < 0.67 or self.kl_train_vs_rollout > 0.01:
            return "critical"
        if abs(self.log_ppl_diff) > 0.05 or self.is_max > 2.0:
            return "warning"
        return "ok"


def compute_tim_metrics(
    log_probs_train: torch.Tensor,
    log_probs_rollout: torch.Tensor,
    mask: torch.Tensor,
    *,
    is_threshold: float = 2.0,
    outlier_log_ratio: float = 0.1,
) -> TimDiagnostics:
    """
    Full TIM diagnostic suite.

    Args:
        log_probs_train: log π_train(a|s), shape [B, T]
        log_probs_rollout: log π_rollout(a|s), shape [B, T]
        mask: 1 for valid response tokens, 0 for padding
        is_threshold: truncate IS weights at this value
        outlier_log_ratio: |log ratio| above this counts as outlier token
    """
    log_ratio = compute_log_ratio(log_probs_train, log_probs_rollout, mask)
    _, is_stats = compute_importance_weights(
        log_ratio, mask, level="token", threshold=is_threshold
    )

    rollout_ppl = compute_perplexity(log_probs_rollout, mask)
    training_ppl = compute_perplexity(log_probs_train, mask)
    log_ppl_diff = torch.log(training_ppl) - torch.log(rollout_ppl)

    masked_ratio = _masked(log_ratio, mask)
    outlier_frac = float((masked_ratio.abs() > outlier_log_ratio).float().mean().item()) if masked_ratio.numel() else 0.0

    return TimDiagnostics(
        rollout_ppl=float(rollout_ppl.mean().item()),
        training_ppl=float(training_ppl.mean().item()),
        log_ppl_diff=float(log_ppl_diff.mean().item()),
        ppl_ratio=float((training_ppl / rollout_ppl.clamp(min=1e-12)).mean().item()),
        kl_train_vs_rollout=compute_kl_divergence(log_probs_train, log_probs_rollout, mask),
        is_mean=is_stats["is_mean"],
        is_max=is_stats["is_max"],
        is_min=is_stats["is_min"],
        ess=is_stats["ess"],
        token_log_ratio_mean=float(masked_ratio.mean().item()) if masked_ratio.numel() else 0.0,
        token_log_ratio_std=float(masked_ratio.std(unbiased=False).item()) if masked_ratio.numel() > 1 else 0.0,
        token_log_ratio_max=float(masked_ratio.abs().max().item()) if masked_ratio.numel() else 0.0,
        outlier_fraction=outlier_frac,
    )


def rejection_mask(
    log_ratio: torch.Tensor,
    mask: torch.Tensor,
    *,
    level: str = "token",
    threshold: float = 0.5,
) -> torch.Tensor:
    """Binary mask: 1 = keep sample, 0 = reject due to TIM outlier."""
    if level == "token":
        keep = (log_ratio.abs() <= threshold).float()
        return keep * mask
    seq_max = (log_ratio.abs() * mask).max(dim=-1, keepdim=True).values
    keep = (seq_max <= threshold).float().expand_as(log_ratio)
    return keep * mask
