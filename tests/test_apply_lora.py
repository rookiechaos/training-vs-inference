"""Smoke test for musubi/kohya LoRA merge helper."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from wan.tools.apply_lora import apply_lora


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.add_module("q", nn.Linear(8, 8, bias=False))


class _TinyWanLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block()])


def test_apply_lora_merges_delta(tmp_path: Path) -> None:
    model = _TinyWanLike()
    with torch.no_grad():
        model.blocks[0].self_attn.q.weight.fill_(0.0)

    rank = 2
    down = torch.randn(rank, 8)
    up = torch.randn(8, rank)
    alpha = float(rank)

    from safetensors.torch import save_file

    lora_path = tmp_path / "tiny.safetensors"
    save_file(
        {
            "lora_unet_blocks_0_self_attn_q.lora_down.weight": down,
            "lora_unet_blocks_0_self_attn_q.lora_up.weight": up,
            "lora_unet_blocks_0_self_attn_q.alpha": torch.tensor(alpha),
        },
        str(lora_path),
    )

    n = apply_lora(model, lora_path, scale=1.0)
    assert n == 1
    expected = (up @ down) * (alpha / rank)
    got = model.blocks[0].self_attn.q.weight.detach().cpu()
    assert torch.allclose(got, expected, atol=1e-5)
