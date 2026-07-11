#!/usr/bin/env python3
"""Exp 01: Floating-point non-associativity and split-reduction."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tim.reduction import compare_reduction_orders, simulate_logprob_drift


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    gen = torch.Generator().manual_seed(42)
    values = torch.randn(8192, generator=gen, dtype=torch.float32)
    reduction = {str(k): v.max_diff for k, v in compare_reduction_orders(values).items()}
    rmsnorm = {str(k): v for k, v in simulate_logprob_drift().items()}

    report = {
        "experiment": "01_fp_non_associativity",
        "reduction_max_diff_by_block_size": reduction,
        "rmsnorm_max_drift_by_block_size": rmsnorm,
    }
    path = out_dir / "01_fp_non_associativity.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
