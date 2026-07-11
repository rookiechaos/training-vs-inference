"""Floating-point reduction order demos — root cause of batch-dependent drift."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ReductionResult:
    left_assoc: float | torch.Tensor
    right_assoc: float | torch.Tensor
    split_blocks: float | torch.Tensor
    max_diff: float


def split_reduction(values: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Simulate GPU split-reduction: sum values in fixed-size blocks, then sum block sums.

    Different block_size mimics different batch shapes changing kernel tiling.
    """
    flat = values.flatten()
    n = flat.numel()
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    blocks = []
    for start in range(0, n, block_size):
        blocks.append(flat[start : start + block_size].sum())
    return torch.stack(blocks).sum()


def compare_reduction_orders(
    values: torch.Tensor,
    block_sizes: list[int] | None = None,
) -> dict[int, ReductionResult]:
    """
    Show (a+b)+c ≠ a+(b+c) and batch-dependent block splitting effects.

    Returns per-block-size comparison of reduction strategies.
    """
    flat = values.flatten().float()
    n = flat.numel()
    if block_sizes is None:
        block_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    block_sizes = [b for b in block_sizes if b <= n]

    left = flat.sum()
    # Right-assoc via sequential pairwise fold
    acc = flat[0]
    for i in range(1, n):
        acc = acc + flat[i]
    right = acc

    results: dict[int, ReductionResult] = {}
    for bs in block_sizes:
        split = split_reduction(flat, bs)
        max_diff = max(abs(left - split).item(), abs(right - split).item(), abs(left - right).item())
        results[bs] = ReductionResult(
            left_assoc=float(left.item()),
            right_assoc=float(right.item()),
            split_blocks=float(split.item()),
            max_diff=max_diff,
        )
    return results


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm in high precision."""
    x64 = x.double()
    rms = (x64.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    return (x64 / rms * weight.double()).float()


def rmsnorm_split_reduce(
    x: torch.Tensor,
    weight: torch.Tensor,
    block_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm with split-reduction on the hidden-dim variance sum."""
    x32 = x.float()
    sq = x32.pow(2)
    b, t, h = sq.shape
    out = torch.empty_like(x32)
    for bi in range(b):
        for ti in range(t):
            row = sq[bi, ti, :]
            var_sum = split_reduction(row, min(block_size, h))
            mean_sq = var_sum / h
            rms = (mean_sq + eps).sqrt()
            out[bi, ti, :] = x32[bi, ti, :] / rms * weight
    return out


def attention_softmax_split(
    scores: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Softmax with split-reduction on exp sum.

    scores: [batch, heads, seq, seq]
    """
    max_score = scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores - max_score)
    # Split reduce along last dim (key dimension)
    b, h, q, k = exp_scores.shape
    flat = exp_scores.reshape(b * h * q, k)
    denom_list = []
    for row in flat:
        denom_list.append(split_reduction(row, min(block_size, k)))
    denom = torch.stack(denom_list).reshape(b, h, q, 1)
    return exp_scores / denom


def simulate_logprob_drift(
    hidden_dim: int = 4096,
    seq_len: int = 128,
    block_sizes: list[int] | None = None,
    seed: int = 42,
) -> dict[int, float]:
    """
    Simulate activation drift through RMSNorm under different reduction block sizes.

    Returns max abs diff vs reference for each block size.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, seq_len, hidden_dim, generator=gen)
    weight = torch.ones(hidden_dim)

    ref = rmsnorm_reference(x, weight)
    if block_sizes is None:
        block_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]

    drifts: dict[int, float] = {}
    for bs in block_sizes:
        if bs > hidden_dim:
            continue
        out = rmsnorm_split_reduce(x, weight, bs)
        drifts[bs] = float((out - ref).abs().max().item())
    return drifts


def matmul_split_reduction(
    a: torch.Tensor,
    b: torch.Tensor,
    k_block: int,
) -> torch.Tensor:
    """
    Matrix multiply C = A @ B with split-reduction along K dimension.

    Mimics GEMM tiling where reduction order depends on tile size.
    """
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    c = torch.zeros(m, n, dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            acc = torch.zeros((), dtype=a.dtype)
            for kb in range(0, k, k_block):
                acc = acc + (a[i, kb : kb + k_block] * b[kb : kb + k_block, j]).sum()
            c[i, j] = acc
    return c
