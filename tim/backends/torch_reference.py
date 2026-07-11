"""PyTorch reference forward for TIM comparison (training-side simulation)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TorchReferenceConfig:
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


class TorchReferenceModel:
    """
    Minimal HF-compatible wrapper for extracting next-token logprobs.

    Used to simulate the training engine (Megatron/FSDP) side of TIM.
    """

    def __init__(self, model_name: str, config: TorchReferenceConfig | None = None):
        self.config = config or TorchReferenceConfig()
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.config.dtype,
            device_map=self.config.device,
        )
        self.model.eval()

    @torch.no_grad()
    def next_token_logprob(self, prompt: str, token_id: int | None = None) -> tuple[int, float]:
        """Return (token_id, log_prob) for the first generated token (greedy if token_id None)."""
        if self.model is None:
            self.load()
        assert self.tokenizer is not None and self.model is not None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**inputs).logits[:, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)

        if token_id is None:
            token_id = int(log_probs.argmax(dim=-1).item())
        else:
            token_id = int(token_id)

        return token_id, float(log_probs[0, token_id].item())

    @torch.no_grad()
    def sequence_logprobs(self, prompt: str, token_ids: list[int]) -> list[float]:
        """Teacher-forced logprobs for a token sequence appended to prompt."""
        if self.model is None:
            self.load()
        assert self.tokenizer is not None and self.model is not None

        full_ids = self.tokenizer.encode(prompt, add_special_tokens=True) + token_ids
        input_ids = torch.tensor([full_ids], device=self.model.device)
        logits = self.model(input_ids).logits

        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=True))
        log_probs_out: list[float] = []
        for i, tid in enumerate(token_ids):
            pos = prompt_len + i - 1
            if pos < 0:
                continue
            lp = F.log_softmax(logits[0, pos, :], dim=-1)
            log_probs_out.append(float(lp[tid].item()))
        return log_probs_out
