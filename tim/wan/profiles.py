"""Inference path profiles for Wan2.1 TIM experiments.

Maps the LLM TIM concept to video diffusion:

    training-like path  ≈  canonical inference (no VAE tiling, full T5)
    rollout path        ≈  optimized inference (tiling, disk offload, …)

In full RL stacks these would be separate engines (trainer vs vLLM/SGLang).
On M3 we approximate them with two generate.sh modes.

Phase 1: attach the same LoRA to both profiles via ``lora`` / ``lora_scale``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class InferProfile:
    """One reproducible inference configuration (= behavior policy π_rollout)."""

    name: str
    role: str  # "train_like" | "rollout"
    description: str
    # generate.sh / wan-infer flags
    mode: str  # canonical | optimized
    size: str = "480*832"
    frame_num: int = 5
    sample_steps: int = 10
    disk_offload: bool = True
    mps_ram: str = "6GB"
    t5_quant: bool = False
    tile_size: int | None = None  # None → derived from mode
    lora: str | None = None
    lora_scale: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.tile_size is None:
            d["tile_size"] = 0 if self.mode == "canonical" else 128
        return d

    def with_lora(self, lora: str, lora_scale: float = 1.0) -> InferProfile:
        """Return a copy that merges the given LoRA at generate time."""
        return replace(self, lora=lora, lora_scale=lora_scale)

    def generate_argv(self) -> list[str]:
        """CLI args for ./generate.sh."""
        args = [
            "--mode",
            self.mode,
            "--size",
            self.size,
            "--frames",
            str(self.frame_num),
            "--steps",
            str(self.sample_steps),
            "--mps-ram",
            self.mps_ram,
        ]
        if not self.disk_offload:
            args.append("--no-disk-offload")
        if self.t5_quant:
            args.append("--t5-quant")
        if self.lora:
            args.extend(["--lora", self.lora, "--lora-scale", str(self.lora_scale)])
        return args


# Default pair for 18 GB M3 TIM study
INFER_PROFILES: dict[str, InferProfile] = {
    "train_like": InferProfile(
        name="train_like",
        role="train_like",
        description="Closest to static training forward: no VAE tiling, full T5 on CPU.",
        mode="canonical",
    ),
    "rollout": InferProfile(
        name="rollout",
        role="rollout",
        description="Production Mac inference: VAE tiling + disk offload (behavior policy).",
        mode="optimized",
    ),
}


def get_profile(name: str) -> InferProfile:
    if name not in INFER_PROFILES:
        raise KeyError(f"Unknown profile {name!r}. Choose from: {list(INFER_PROFILES)}")
    return INFER_PROFILES[name]
