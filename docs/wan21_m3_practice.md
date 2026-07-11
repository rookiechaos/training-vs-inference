# Practicing TIM with Wan2.1 on Apple M3

A realistic playbook for studying training–inference mismatch on video diffusion models when you only have an M3 Mac.

## Bottom Line

| Goal | Feasible on M3? |
|------|-----------------|
| Run Wan2.1 **inference** (1.3B) | Yes, with Mac forks and memory tricks |
| **LoRA fine-tuning** (tiny clips, low res) | Painful but possible on 24 GB+ unified memory |
| Full **Flow-GRPO / DPO RL** (DanceGRPO-style) | No — needs multi-GPU cloud (8× H800 class) |
| **Measure denoise-step drift** between train vs infer paths | Yes — this is the right TIM exercise for M3 |

Do not aim for production RL on M3. Aim to **reproduce the mechanism**: same weights, different forward paths, drift that amplifies across denoising steps.

---

## Hardware Expectations

| M3 config | Wan2.1-T2V-1.3B infer | Wan2.1-T2V-1.3B LoRA | Wan2.1-T2V-14B |
|-----------|------------------------|----------------------|----------------|
| 16 GB RAM | Tight — few frames, heavy offload | Very hard | Not recommended |
| **18 GB RAM** | **512×320, 9 frames — use `wan/scripts/`** | **Marginal** | Not recommended |
| 24 GB RAM | Practical at 480p, ~17–25 frames | Possible with tiny dataset | Disk offload only, very slow |
| 36 GB+ RAM | Comfortable for 1.3B | More headroom | 14B infer with offload |

Unified memory on Apple Silicon is shared between CPU and GPU. Techniques like `blocks_to_swap` (CUDA VRAM savers) are **counterproductive** on M3 — they add CPU↔MPS copy overhead. Prefer model offload and VAE tiling instead.

### 18 GB in-repo quick path

All commands run inside this monorepo under `wan/`:

```bash
cd wan && ./scripts/setup.sh
source .venv/bin/activate && source scripts/env.sh
./scripts/run_phase0.sh
```

Defaults: `512*320`, 9 frames, 15 denoise steps, seed 42. Outputs: `wan/results/phase0/`.

---

## Recommended Stack

```
Training path (static, gradient)     Inference path (optimized, dynamic)
─────────────────────────────        ───────────────────────────────────
musubi-tuner                         HighDoping/Wan2.1-Mac
LoRA on DiT                          generate.py with tiling / offload
fixed resolution & frame count       varying tile_size, frame_num, offload
bf16/fp16 train forward              MPS + CPU fallback
```

| Component | Repo | Role |
|-----------|------|------|
| Base model | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | Start here, not 14B |
| Mac inference | [HighDoping/Wan2.1-Mac](https://github.com/HighDoping/Wan2.1-Mac) | MPS, VAE tiling, T5 quant |
| LoRA training | [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) | Wan2.1 LoRA docs in `docs/wan.md` |
| RL reference (cloud only) | [DanceGRPO](https://github.com/XueZeyue/DanceGRPO) | Wan-2.1 GRPO scripts — run on GPU cloud later |

---

## Phase 0: Inference-Only TIM (Start Here, 1–2 Days)

No training required. You only compare **two inference configurations** on the **same checkpoint** and **same seed**.

### Setup

```bash
git clone https://github.com/HighDoping/Wan2.1-Mac.git
cd Wan2.1-Mac

pip install -r requirements.txt
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### Run A — "canonical" (closest to training forward)

```bash
python generate.py \
  --task t2v-1.3B \
  --size "832*480" \
  --frame_num 17 \
  --sample_steps 25 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --offload_model False \
  --tile_size 0 \
  --t5_cpu \
  --device mps \
  --base_seed 42 \
  --prompt "A cat running on a soccer field." \
  --save_file out_canonical.mp4
```

### Run B — "production infer" (tiling + offload)

```bash
python generate.py \
  --task t2v-1.3B \
  --size "832*480" \
  --frame_num 17 \
  --sample_steps 25 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --offload_model True \
  --tile_size 256 \
  --t5_quant \
  --device mps \
  --base_seed 42 \
  --prompt "A cat running on a soccer field." \
  --save_file out_optimized.mp4
```

### What to observe (Part IV — chaotic amplification)

1. **Early steps (1–5)**: outputs may look nearly identical.
2. **Late steps (20–25)**: optimized path may show limb drift, texture melt, or scene jumps.
3. **Change `frame_num`** (9 vs 17 vs 25) while holding seed fixed — spatiotemporal "batch shape" changes, similar to dynamic batching in LLM serving.

This is the video analogue of logprob drift: small numerical differences in early denoising steps amplify through the DiT chain.

### Optional: save per-step latents

Patch `generate.py` to `torch.save()` the latent tensor at steps 1, 5, 10, 20, 25 for both runs. Then:

```python
import torch
a = torch.load("latent_step20_canonical.pt")
b = torch.load("latent_step20_optimized.pt")
print((a - b).abs().max())  # should grow vs earlier steps
```

---

## Phase 1: Tiny LoRA Train + Cross-Path Eval (1–2 Weeks)

Train a minimal LoRA, then sample the **same LoRA** on train_like vs rollout infer paths.

**In-repo one-liner:**

```bash
./wan/scripts/prepare_tiny_clips.sh
python wan/experiments/run_lora_tim_pair.py
# or, if train OOMs:
python wan/experiments/run_lora_tim_pair.py --skip-train --lora wan/output/lora_tiny/wan_lora_tiny.safetensors
```

See [docs/wan_tim_code_structure.md](wan_tim_code_structure.md).

### Dataset (keep it tiny)

```
wan/data/clips/
  clip01.mp4   # 2–3 seconds
  clip01.txt   # caption
  clip02.mp4
  clip02.txt
  ...          # 2–10 clips total
```

`prepare_tiny_clips.sh` bootstraps from Phase 0 mp4s. For a clearer LoRA signal, download **30 Mixkit cat clips** (free stock):

```bash
python wan/scripts/download_mixkit_clips.py --count 30 --theme cat --replace
```

Recommended resolution for training: **320×512**, **9 frames**. Do not start at 720p on M3.

### musubi-tuner on Mac

```bash
git clone https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
pip install -e .

accelerate config
# NO to CPU-only
# mixed_precision: fp16  (NOT bf16 on MPS)
```

Mac-specific flags (from community reports):

```bash
accelerate launch --mixed_precision fp16 src/musubi_tuner/wan_train_network.py \
  --task t2v-1.3B \
  --dit ./Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
  --dataset_config ./dataset.toml \
  --sdpa \
  --mixed_precision fp16 \
  --optimizer_type adamw \
  --learning_rate 2e-4 \
  --gradient_checkpointing \
  --split_attn \
  --blocks_to_swap 8 \
  --network_module networks.lora \
  --network_dim 16 \
  --max_train_epochs 4 \
  --save_every_n_epochs 2 \
  --output_dir ./output/wan_lora_tiny
```

> `adamw8bit` / bitsandbytes often falls back to CPU on MPS. Use plain `adamw`.
> Expect hours per epoch on M3, not minutes.

### TIM experiment after LoRA

```bash
python wan/experiments/run_lora_tim_pair.py --skip-train \
  --lora wan/output/lora_tiny/wan_lora_tiny.safetensors
```

Compare:
- Visual quality / LoRA effect strength on train_like vs rollout
- Frame MSE report under `results/wan_tim_lora/` (style may "disappear" on optimized path if drift dominates)

If you later add RL (cloud), this is exactly the failure mode: reward computed on rollout videos does not match what the training forward would produce.

---

## Phase 2: Simulated "RL" Without Full GRPO (M3-Friendly)

Full GRPO needs log-probabilities under flow-matching SDE — heavy engineering. On M3, simulate the **credit assignment failure** instead:

### Proxy reward loop

```
for epoch in range(N):
    1. Sample video with optimized infer (path B)
    2. Score with cheap proxy: CLIP text-video score, aesthetic predictor, or your eyes
    3. Train LoRA one step with musubi (path A forward)
    4. Log: reward vs. visual degradation over epochs
```

You are not running correct Flow-GRPO. You are reproducing the **symptom**: optimizer chases rewards from a policy (infer path B) that the trainer (path A) does not implement.

Watch for:

- Reward flat or noisy while videos get worse
- Late-epoch "melting" especially in last 30% of frames
- LoRA overfitting to caption, underfitting motion

---

## Phase 3: Real RL on Cloud (When M3 Proves the Problem)

Once Phase 0–2 show drift matters, move RL to GPU cloud:

| Framework | Model | Notes |
|-----------|-------|-------|
| [DanceGRPO](https://github.com/XueZeyue/DanceGRPO) | Wan-2.1 T2V 1.3B | `finetune_wan_2_1_grpo.sh`, 8× H800 |
| [epipolar-dpo](https://github.com/JackZhouSz/epipolar-dpo) | Wan + Flow-DPO | Preference pairs, geometry reward |
| OpenReview flowDPO+DM | Wan 2.1 T2V 1.3B | Unified distillation + alignment |

Bring back to M3 only for **evaluation** — and expect eval drift unless you match training inference settings.

---

## Mapping to This Repo

| TIM concept | Wan2.1 M3 exercise |
|-------------|------------------|
| Split-reduction / batch shape | Change `frame_num`, `tile_size`, `sample_steps` |
| Train vs rollout logprob mismatch | Latent L2 at step *t* between canonical vs optimized infer |
| PPO ratio explosion | N/A directly — use latent drift growth rate instead |
| Alignment tax | LoRA improves on train path, vanishes on optimized infer |
| Chaotic amplification | Compare step 5 vs step 25 latent delta |
| L1 fix (deterministic) | Same flags on both paths: no tiling, no offload, fixed frames |
| L3 fix (noise injection) | Add tiling to musubi training forward (advanced, not default) |

Core library demos still help build intuition:

```bash
cd training-vs-inference
tim demo-reduction          # why float order matters
tim demo-rmsnorm-drift      # activation-level drift
python experiments/02_ppo_tim_simulation.py  # IS ratio collapse (LLM analogue)
```

---

## Minimal `dataset.toml` Example

```toml
[general]
resolution = [384, 640]
batch_size = 1
enable_bucket = false
caption_extension = ".txt"

[[datasets]]
video_directory = "./data/wan_tiny"
cache_directory = "./cache/wan_tiny"
target_frames = [9]
frame_extraction = "head"
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| MPS OOM / hang | `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, reduce `frame_num`, enable `--offload_model True` |
| bf16 error in musubi | Use `fp16`, not `bf16` |
| bitsandbytes crash | `--optimizer_type adamw` instead of `adamw8bit` |
| NaN loss | Add `--split_attn` |
| 14B won't load | Use 1.3B, or `--disk_offload --mps_ram 10GB` (very slow) |
| Trained LoRA has no effect on Mac infer | Check LoRA loader path; check TIM drift overwhelming LoRA signal |

---

## Suggested 7-Day Schedule

| Day | Task |
|-----|------|
| 1 | Phase 0: canonical vs optimized infer, same seed |
| 2 | Save per-step latents, plot max abs diff vs step |
| 3 | Sweep `frame_num` and `tile_size`, record drift |
| 4 | Prepare 5–10 clip dataset |
| 5–6 | Train tiny LoRA with musubi-tuner |
| 7 | Cross-eval LoRA on both infer paths; write up drift vs quality |

---

## References

- [Wan2.1 Official](https://github.com/Wan-Video/Wan2.1)
- [Wan2.1-Mac](https://github.com/HighDoping/Wan2.1-Mac)
- [musubi-tuner Wan docs](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/wan.md)
- [DanceGRPO Wan-2.1](https://github.com/XueZeyue/DanceGRPO)
- [M1 Max Wan2.1 benchmarks](https://zenn.dev/kemmm/articles/wan21-mac-studio-benchmark?locale=en)
