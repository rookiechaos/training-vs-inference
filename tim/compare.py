"""Compare logprobs across engines and aggregate drift statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from tim.metrics import TimDiagnostics, compute_tim_metrics


@dataclass
class LogprobRecord:
    """Single-engine logprob snapshot for one sequence."""

    engine: str
    prompt: str
    token_ids: list[int]
    log_probs: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class DriftReport:
    """Cross-engine logprob comparison for identical token sequences."""

    engine_a: str
    engine_b: str
    num_tokens: int
    max_abs_logprob_diff: float
    mean_abs_logprob_diff: float
    max_ratio: float
    diagnostics: TimDiagnostics | None = None
    per_token_diffs: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        base = {
            "engine_a": self.engine_a,
            "engine_b": self.engine_b,
            "num_tokens": self.num_tokens,
            "max_abs_logprob_diff": self.max_abs_logprob_diff,
            "mean_abs_logprob_diff": self.mean_abs_logprob_diff,
            "max_ratio": self.max_ratio,
            "per_token_diffs": self.per_token_diffs,
        }
        if self.diagnostics:
            base["diagnostics"] = self.diagnostics.to_dict()
        return base


def compare_logprob_records(
    record_a: LogprobRecord,
    record_b: LogprobRecord,
    *,
    assume_a_is_train: bool = True,
) -> DriftReport:
    """Compare two logprob records on the same token sequence."""
    if record_a.token_ids != record_b.token_ids:
        raise ValueError(
            f"Token mismatch: {len(record_a.token_ids)} vs {len(record_b.token_ids)}"
        )

    la = torch.tensor(record_a.log_probs, dtype=torch.float64)
    lb = torch.tensor(record_b.log_probs, dtype=torch.float64)
    diffs = (la - lb).abs()
    ratios = torch.exp(la - lb)

    log_train = la if assume_a_is_train else lb
    log_rollout = lb if assume_a_is_train else la
    mask = torch.ones(1, len(la))
    diag = compute_tim_metrics(
        log_train.unsqueeze(0).float(),
        log_rollout.unsqueeze(0).float(),
        mask,
    )

    return DriftReport(
        engine_a=record_a.engine,
        engine_b=record_b.engine,
        num_tokens=len(la),
        max_abs_logprob_diff=float(diffs.max().item()),
        mean_abs_logprob_diff=float(diffs.mean().item()),
        max_ratio=float(ratios.max().item()),
        diagnostics=diag,
        per_token_diffs=diffs.tolist(),
    )


def aggregate_drift_reports(reports: Sequence[DriftReport]) -> dict[str, float]:
    """Summarize multiple drift reports (e.g., across batch sizes)."""
    if not reports:
        return {}
    return {
        "count": len(reports),
        "max_abs_diff_mean": sum(r.max_abs_logprob_diff for r in reports) / len(reports),
        "max_abs_diff_worst": max(r.max_abs_logprob_diff for r in reports),
        "mean_abs_diff_mean": sum(r.mean_abs_logprob_diff for r in reports) / len(reports),
        "max_ratio_worst": max(r.max_ratio for r in reports),
        "critical_count": sum(1 for r in reports if r.diagnostics and r.diagnostics.severity() == "critical"),
    }


def inject_synthetic_drift(
    log_probs: torch.Tensor,
    mask: torch.Tensor,
    *,
    noise_std: float = 1e-5,
    seed: int = 0,
) -> torch.Tensor:
    """Add Gaussian noise to log probs to simulate TIM (for unit tests / ablations)."""
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(log_probs.shape, generator=gen, dtype=log_probs.dtype) * noise_std
    return log_probs + noise * mask
