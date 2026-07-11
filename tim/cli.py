"""CLI for TIM diagnostics and experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.table import Table

from tim.compare import compare_logprob_records, inject_synthetic_drift
from tim.metrics import compute_tim_metrics
from tim.ppo_ratios import PpoRatioMode, compute_ppo_ratios, ppo_clipped_loss
from tim.reduction import compare_reduction_orders, simulate_logprob_drift

app = typer.Typer(no_args_is_help=True, help="Training-Inference Mismatch (TIM) toolkit")
console = Console()


@app.command("demo-reduction")
def demo_reduction(
    size: int = typer.Option(4096, help="Number of values to reduce"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Demonstrate floating-point non-associativity and split-reduction drift."""
    gen = torch.Generator().manual_seed(seed)
    values = torch.randn(size, generator=gen, dtype=torch.float32)
    results = compare_reduction_orders(values)

    table = Table(title="Split-Reduction Order Effects (float32)")
    table.add_column("Block Size")
    table.add_column("Left Assoc")
    table.add_column("Split Reduce")
    table.add_column("Max Diff")

    for bs, r in sorted(results.items()):
        table.add_row(str(bs), f"{r.left_assoc:.8f}", f"{r.split_blocks:.8f}", f"{r.max_diff:.2e}")

    console.print(table)
    console.print("\n[bold]Takeaway:[/bold] Different block sizes → different sums → downstream logprob drift.")


@app.command("demo-rmsnorm-drift")
def demo_rmsnorm_drift(
    hidden_dim: int = typer.Option(4096, help="Hidden dimension"),
    seq_len: int = typer.Option(128, help="Sequence length"),
) -> None:
    """Simulate RMSNorm activation drift under varying reduction block sizes."""
    drifts = simulate_logprob_drift(hidden_dim=hidden_dim, seq_len=seq_len)

    table = Table(title="RMSNorm Max Activation Drift vs Reference (float64)")
    table.add_column("Block Size")
    table.add_column("Max |Δ|")

    for bs, d in sorted(drifts.items()):
        table.add_row(str(bs), f"{d:.2e}")

    console.print(table)


@app.command("demo-metrics")
def demo_metrics(
    noise_std: float = typer.Option(1e-4, help="Synthetic logprob noise std"),
    seq_len: int = typer.Option(64, help="Sequence length"),
) -> None:
    """Run TIM metrics on synthetic rollout/train logprob mismatch."""
    gen = torch.Generator().manual_seed(0)
    log_rollout = torch.randn(4, seq_len, generator=gen) * 0.5 - 2.0
    mask = torch.ones(4, seq_len)
    log_train = inject_synthetic_drift(log_rollout, mask, noise_std=noise_std)

    diag = compute_tim_metrics(log_train, log_rollout, mask)
    console.print_json(json.dumps(diag.to_dict(), indent=2))
    console.print(f"\n[bold]Severity:[/bold] {diag.severity()}")


@app.command("demo-ppo")
def demo_ppo(
    noise_std: float = typer.Option(1e-3, help="TIM noise between train and rollout"),
) -> None:
    """Compare PPO loss under recompute vs bypass ratio modes."""
    B, T = 2, 32
    gen = torch.Generator().manual_seed(1)
    log_old_rollout = torch.randn(B, T, generator=gen) * 0.3 - 1.0
    mask = torch.ones(B, T)
    log_old_train = inject_synthetic_drift(log_old_rollout, mask, noise_std=noise_std, seed=2)
    log_current = log_old_rollout + torch.randn(B, T, generator=gen) * 0.01
    advantages = torch.randn(B, T, generator=gen)

    losses = {}
    for mode in PpoRatioMode:
        ratios = compute_ppo_ratios(log_current, log_old_train, log_old_rollout, mask, mode=mode)
        losses[mode.value] = float(ppo_clipped_loss(ratios["ratio"], advantages, mask).item())

    table = Table(title="PPO Loss by Ratio Mode (lower = more stable under this synthetic TIM)")
    table.add_column("Mode")
    table.add_column("Loss")
    for k, v in losses.items():
        table.add_row(k, f"{v:.6f}")
    console.print(table)


@app.command("compare-engines")
def compare_engines(
    sglang_url: str = typer.Option("http://127.0.0.1:30000", help="SGLang server URL"),
    model: str = typer.Option(..., help="HF model name for PyTorch reference"),
    prompt: str = typer.Option("The capital of France is", help="Test prompt"),
    output: Optional[Path] = typer.Option(None, help="Save JSON report"),
) -> None:
    """Compare SGLang rollout logprobs vs PyTorch training reference."""
    from tim.backends.sglang_client import SGLangClient, SGLangConfig, wait_for_server
    from tim.backends.torch_reference import TorchReferenceModel

    client = SGLangClient(SGLangConfig(base_url=sglang_url))
    if not wait_for_server(client, retries=3, interval=1.0):
        console.print("[red]SGLang server not reachable.[/red] Start with:")
        console.print("  python -m sglang.launch_server --model-path <model> --port 30000")
        raise typer.Exit(1)

    sglang_rec = client.logprob_record(prompt, label="sglang")
    if not sglang_rec.token_ids:
        console.print("[red]No logprobs returned. Ensure return_logprob=True is supported.[/red]")
        raise typer.Exit(1)

    ref = TorchReferenceModel(model)
    tid = sglang_rec.token_ids[0]
    _, lp = ref.next_token_logprob(prompt, token_id=tid)

    from tim.compare import LogprobRecord

    train_rec = LogprobRecord(
        engine="torch_train",
        prompt=prompt,
        token_ids=[tid],
        log_probs=[lp],
    )
    report = compare_logprob_records(train_rec, sglang_rec, assume_a_is_train=True)
    console.print_json(json.dumps(report.to_dict(), indent=2))

    if output:
        output.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"Saved to {output}")


@app.command("batch-drift")
def batch_drift(
    config: Path = typer.Option(..., help="YAML config for batch drift experiment"),
    output: Path = typer.Option(Path("results/batch_drift.json"), help="Output JSON"),
) -> None:
    """Run batch composition logprob drift experiment (requires SGLang)."""
    import yaml

    from experiments.batch_drift import run_batch_drift_experiment

    with config.open() as f:
        cfg = yaml.safe_load(f)
    results = run_batch_drift_experiment(cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    console.print_json(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
