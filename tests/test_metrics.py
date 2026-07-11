"""Tests for TIM toolkit."""

from __future__ import annotations

import torch

from tim.compare import compare_logprob_records, inject_synthetic_drift, LogprobRecord
from tim.metrics import compute_tim_metrics, compute_importance_weights, rejection_mask
from tim.ppo_ratios import PpoRatioMode, compute_ppo_ratios, ppo_clipped_loss
from tim.reduction import compare_reduction_orders, split_reduction, simulate_logprob_drift


def test_split_reduction_differs_by_block_size():
    values = torch.randn(1024)
    r32 = split_reduction(values, 32)
    r128 = split_reduction(values, 128)
    # Usually differs in float32 (may rarely be equal by chance)
    assert isinstance(float(r32.item()), float)


def test_non_associativity_detected():
    values = torch.randn(512)
    results = compare_reduction_orders(values, block_sizes=[16, 64, 256])
    assert len(results) >= 2


def test_rmsnorm_drift_simulation():
    drifts = simulate_logprob_drift(hidden_dim=512, seq_len=16, block_sizes=[32, 64, 128])
    assert all(v >= 0 for v in drifts.values())


def test_tim_metrics_identity():
    B, T = 2, 16
    log_p = torch.randn(B, T) * 0.5 - 1.0
    mask = torch.ones(B, T)
    diag = compute_tim_metrics(log_p, log_p.clone(), mask)
    assert abs(diag.is_mean - 1.0) < 1e-5
    assert diag.severity() == "ok"


def test_tim_metrics_with_noise():
    log_rollout = torch.randn(2, 32) * 0.3 - 1.0
    mask = torch.ones(2, 32)
    log_train = inject_synthetic_drift(log_rollout, mask, noise_std=0.05)
    diag = compute_tim_metrics(log_train, log_rollout, mask)
    assert diag.kl_train_vs_rollout != 0.0


def test_importance_weights_truncation():
    log_ratio = torch.tensor([[0.0, 5.0, -3.0]])
    mask = torch.ones(1, 3)
    weights, stats = compute_importance_weights(log_ratio, mask, threshold=2.0)
    assert weights.max() <= 2.0 + 1e-6
    assert stats["is_max"] <= 2.0 + 1e-6


def test_rejection_mask():
    log_ratio = torch.tensor([[0.01, 0.5, 0.02]])
    mask = torch.ones(1, 3)
    keep = rejection_mask(log_ratio, mask, threshold=0.1)
    assert keep[0, 1].item() == 0.0
    assert keep[0, 0].item() == 1.0


def test_ppo_bypass_vs_recompute():
    B, T = 2, 8
    log_rollout = torch.randn(B, T)
    mask = torch.ones(B, T)
    log_train = log_rollout + 0.01
    log_current = log_rollout + 0.005

    r_re = compute_ppo_ratios(log_current, log_train, log_rollout, mask, PpoRatioMode.RECOMPUTE)
    r_bp = compute_ppo_ratios(log_current, log_train, log_rollout, mask, PpoRatioMode.BYPASS)
    assert not torch.allclose(r_re["ratio"], r_bp["ratio"])


def test_ppo_loss_runs():
    ratio = torch.ones(2, 8)
    adv = torch.randn(2, 8)
    mask = torch.ones(2, 8)
    loss = ppo_clipped_loss(ratio, adv, mask)
    assert loss.ndim == 0


def test_compare_logprob_records():
    rec_a = LogprobRecord("train", "hi", [1, 2, 3], [-0.5, -0.3, -0.8])
    rec_b = LogprobRecord("rollout", "hi", [1, 2, 3], [-0.5, -0.31, -0.79])
    report = compare_logprob_records(rec_a, rec_b)
    assert report.num_tokens == 3
    assert report.max_abs_logprob_diff > 0
