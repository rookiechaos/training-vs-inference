# Training vs Inference

Toolkit to **diagnose, reproduce, and mitigate Training–Inference Mismatch (TIM)** in modern RL / generative stacks.

Same weights, different engines → different logprobs (LLMs) or different videos (diffusion). That breaks on-policy assumptions in PPO/GRPO and shows up as “alignment tax” when a LoRA trained on one path is evaluated on another.

| Track | What you get |
|-------|----------------|
| **LLM TIM** | FP non-associativity demos, PPL/KL/IS/ESS metrics, PPO ratio modes, SGLang/vLLM clients |
| **Video TIM (Wan2.1)** | M3 Mac practice: infer-path drift, tiny LoRA train (Apple Container), cross-path eval |

---

## Install

```bash
git clone https://github.com/<you>/training-vs-inference.git
cd training-vs-inference
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Torch is a dependency; use a machine with MPS (Mac) or CUDA for Wan experiments.

**Not in git (download locally):**

| Artifact | How |
|----------|-----|
| `Wan2.1-T2V-1.3B/` (~16–20 GB) | `hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir Wan2.1-T2V-1.3B` |
| `wan-infer/` | `./generate.sh --setup` (clones HighDoping/Wan2.1-Mac + applies patches) |
| `wan/vendor/musubi-tuner/` | cloned by train scripts |
| `wan/data/clips/*.mp4` | `python wan/scripts/download_mixkit_clips.py --count 30 --theme cat` |

---

## Quick start — LLM TIM

```bash
tim demo-reduction
tim demo-ppo
python experiments/02_ppo_tim_simulation.py
pytest -q
```

```python
from tim.metrics import compute_tim_metrics
from tim.ppo_ratios import PpoRatioMode, compute_ppo_ratios

diag = compute_tim_metrics(log_probs_train, log_probs_rollout, mask)
print(diag.severity())  # ok | warning | critical
```

---

## Quick start — Wan2.1 on Apple M3 (18 GB)

**Host Mac = MPS inference. Apple Container = LoRA train (deps isolated).**

```bash
# Infer setup (once)
./generate.sh --setup
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir Wan2.1-T2V-1.3B

# Phase 0 — same checkpoint, two infer paths (proxy TIM)
./generate.sh --mode optimized
python wan/experiments/run_tim_pair.py

# Phase 1 — real TIM: train LoRA in container, eval on host
./wan/scripts/container_build.sh
./wan/scripts/container_up.sh
python wan/scripts/download_mixkit_clips.py --count 30 --theme cat --replace
./wan/scripts/train_lora_container.sh
python wan/experiments/run_lora_tim_pair.py --skip-train \
  --lora wan/output/lora_tiny/wan_lora_tiny.safetensors
```

Docs: [`wan/README.md`](wan/README.md) · [`docs/wan_tim_code_structure.md`](docs/wan_tim_code_structure.md) · [`wan/container/README.md`](wan/container/README.md) · [`docs/wan21_m3_practice.md`](docs/wan21_m3_practice.md)

---

## Why TIM matters

```
Rollout engine (SGLang / optimized infer)  ──samples──▶  Training engine
         π_rollout(a|s)                                      π_train(a|s)
                │                                                  │
                └──────── same θ, different forward ───────────────┘
```

PPO/GRPO importance ratios assume the behavior policy matches the training forward. Kernel-level drift (batch shape, tiling, reduction order) makes that false. Token-level error can look tiny (~1e-5) and still compound across sequence length and steps.

**Hardware root:** floating-point add is not associative; GPU/MPS split-reductions change with batch layout. See `tim/reduction.py` and `experiments/01_fp_non_associativity.py`.

---

## Repo layout

```
tim/                      # Core library (LLM + wan/ video helpers)
experiments/              # Synthetic LLM TIM demos
configs/                  # Example YAML
tests/                    # pytest
generate.sh               # Wan host infer entry (MPS)
wan/
  experiments/            # run_tim_pair, run_lora_tim_pair, …
  scripts/                # container_*, train_lora_*, download_mixkit_*
  tools/                  # apply_lora, patch_mps, patch_lora_support
  container/              # Apple Container image for musubi train
  configs/                # dataset_tiny.toml
  data/clips/             # local only (.gitkeep tracked)
docs/                     # Deep dives
```

---

## License

Apache-2.0 — see [`LICENSE`](LICENSE).

Third-party: [Wan2.1](https://github.com/Wan-Video/Wan2.1), [HighDoping/Wan2.1-Mac](https://github.com/HighDoping/Wan2.1-Mac), [musubi-tuner](https://github.com/kohya-ss/musubi-tuner), [Mixkit](https://mixkit.co/license/) stock clips.

## References

- [SGLang Deterministic Inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference.html)
- [Towards Deterministic Inference in SGLang (LMSYS)](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [Thinking Machines Lab — batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops)
- [Diagnosing TIM in LLM RL (arXiv:2605.14220)](https://arxiv.org/html/2605.14220v1)
- [On the Rollout–Training Mismatch (OpenReview)](https://openreview.net/pdf?id=8MHqvb4lK9)
- [Miles / slime TIM notes](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/mismatch/blog-en.md)
