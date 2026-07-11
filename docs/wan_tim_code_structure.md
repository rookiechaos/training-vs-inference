# Wan2.1: Training vs Inference Code Structure

How this repo maps **LLM TIM** concepts to your working **Wan2.1** setup.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        training-vs-inference                     │
├─────────────────────────────────────────────────────────────────┤
│  tim/                    Core TIM library (modality-agnostic)      │
│    metrics.py            PPL, KL, IS, ESS  (LLM logprob TIM)     │
│    ppo_ratios.py         recompute / bypass / decoupled PPO      │
│    reduction.py          FP split-reduction demos                  │
│    wan/                  Wan video TIM layer                       │
│      profiles.py         train_like vs rollout (+ optional LoRA) │
│      metrics.py          frame MSE drift reports                 │
├─────────────────────────────────────────────────────────────────┤
│  generate.sh             Infer entry (--lora / --lora-scale)     │
│  wan-infer/              Mac infer stack (LoRA merge after load)  │
│  Wan2.1-T2V-1.3B/        Shared base weights                     │
│  wan/vendor/musubi-tuner Training stack (LoRA)                   │
├─────────────────────────────────────────────────────────────────┤
│  wan/experiments/        Runnable TIM studies                    │
│    run_tim_pair.py       Phase 0: infer-only A/B (proxy)         │
│    run_lora_tim_pair.py  Phase 1: train LoRA → cross-path eval   │
│  results/                JSON reports + mp4 outputs                │
└─────────────────────────────────────────────────────────────────┘
```

## Concept mapping

| LLM RL stack | Wan2.1 on your M3 | This repo |
|--------------|-------------------|-----------|
| Rollout engine (SGLang) | `optimized` generate path | `tim/wan/profiles.py` → `rollout` |
| Training engine (Megatron) | musubi LoRA train + `canonical` infer | `train_lora_18g.sh` + `train_like` |
| Token logprob π(a\|s) | Per-step latent / per-frame pixels | `compare_videos`, `compare_latents` |
| Importance ratio r(θ) | Frame drift growth over denoise steps | `late_frame_mse_mean` |
| Alignment tax | LoRA effect weakens on optimized path | Phase 1 report severity |
| PPO bypass / IS | (Phase 3) Flow-DPO / GRPO on cloud | `tim/ppo_ratios.py` (LLM); cloud later |

## Phase 0 — infer-only proxy

Same **base** checkpoint, two infer configs. Proves paths diverge; **no training**.

```
prompt + seed
     │
     ├─► generate.sh --mode canonical  ──► train_like.mp4
     │
     └─► generate.sh --mode optimized  ──► rollout.mp4
                    │
                    ▼
         compare_videos → results/wan_tim/
```

```bash
python wan/experiments/run_tim_pair.py
python wan/experiments/run_tim_pair.py --skip-generate
```

## Phase 1 — real TIM (LoRA train → cross-path eval)

```
clips ──► musubi-tuner LoRA train ──► lora.safetensors
                                            │
              ┌─────────────────────────────┼─────────────────────────────┐
              ▼                             ▼                             │
   generate train_like + LoRA    generate rollout + LoRA                  │
              │                             │                             │
              └──────────► compare_videos ──┘                             │
                           results/wan_tim_lora/                          │
```

```bash
# Bootstrap clips from existing Phase 0 mp4s
./wan/scripts/prepare_tiny_clips.sh

# Or download 30 Mixkit cat clips (clearer LoRA signal; free stock license)
python wan/scripts/download_mixkit_clips.py --count 30 --theme cat --replace

# Full pipeline (train may take hours / OOM on 18 GB)
python wan/experiments/run_lora_tim_pair.py

# Or reuse an existing LoRA
python wan/experiments/run_lora_tim_pair.py \
  --skip-train --lora wan/output/lora_tiny/wan_lora_tiny.safetensors

# Compare only
python wan/experiments/run_lora_tim_pair.py --skip-train --skip-generate \
  --lora wan/output/lora_tiny/wan_lora_tiny.safetensors
```

Single-path with LoRA:

```bash
./generate.sh --mode canonical --lora wan/output/lora_tiny/wan_lora_tiny.safetensors
./generate.sh --mode optimized --lora wan/output/lora_tiny/wan_lora_tiny.safetensors
```

## Commands (other)

```bash
./generate.sh --mode optimized
./generate.sh --mode canonical --output results/generate/canonical_seed42.mp4
tim demo-reduction
tim demo-ppo
```

## Phases

| Phase | Goal | Code |
|-------|------|------|
| **0** | Prove infer paths diverge (proxy) | `run_tim_pair.py`, `generate.sh` |
| **1** | Train LoRA, eval on train_like vs rollout | `prepare_tiny_clips.sh`, `train_lora_18g.sh`, `run_lora_tim_pair.py`, `apply_lora.py` |
| **1b** | Per-step latent drift | `apply_latent_hook.py`, `compare_latents.py` |
| **3** | Real RL (GRPO/DPO) on cloud | DanceGRPO / epipolar-dpo |

## File conventions

```
results/
  generate/              single runs from generate.sh
  wan_tim/               Phase 0 pair
  wan_tim_lora/          Phase 1 pair (+ LoRA)
    train_like_lora_seed42.mp4
    rollout_lora_seed42.mp4
    tim_report_lora_seed42.json
    manifest_lora_seed42.json
wan/
  data/clips/            tiny train set (.mp4 + .txt)
  output/lora_tiny/      trained LoRA safetensors
  tools/apply_lora.py    merge helper used by wan-infer
```

## Extending

```python
INFER_PROFILES["aggressive"] = InferProfile(
    name="aggressive",
    role="rollout",
    description="More tiling, fewer steps",
    mode="optimized",
    frame_num=9,
    t5_quant=True,
)
```

```bash
python wan/experiments/run_lora_tim_pair.py --profile-a train_like --profile-b aggressive \
  --skip-train --lora wan/output/lora_tiny/wan_lora_tiny.safetensors
```

## What NOT to put in Apple Container

- `generate.sh` / MPS inference → **host Mac only**
- LoRA **train** → **Apple Container** (`./wan/scripts/train_lora_container.sh`) — isolates musubi from Anaconda
- Container data prep / `hf` → OK (`tim-wan-lora`, repo mounted at `/workspace`)

See [wan/container/README.md](../wan/container/README.md).
