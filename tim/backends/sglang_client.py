"""HTTP client for SGLang /generate with logprob extraction."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests

from tim.compare import LogprobRecord


@dataclass
class SGLangConfig:
    base_url: str = "http://127.0.0.1:30000"
    timeout: float = 120.0
    deterministic: bool = False


@dataclass
class GenerateResult:
    text: str
    token_ids: list[int]
    log_probs: list[float]
    meta_info: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class SGLangClient:
    """Minimal SGLang HTTP client for TIM logprob drift experiments."""

    def __init__(self, config: SGLangConfig | None = None):
        self.config = config or SGLangConfig()

    @property
    def generate_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/generate"

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.config.base_url}/health", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 1,
        temperature: float = 0.0,
        return_logprob: bool = True,
        top_logprobs_num: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> GenerateResult:
        payload: dict[str, Any] = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
            "return_logprob": return_logprob,
        }
        if top_logprobs_num:
            payload["top_logprobs_num"] = top_logprobs_num
        if extra:
            payload.update(extra)

        resp = requests.post(
            self.generate_url,
            json=payload,
            timeout=self.config.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._parse_response(data)

    def generate_with_padding_batch(
        self,
        target_prompt: str,
        filler_prompts: list[str],
        **kwargs: Any,
    ) -> GenerateResult:
        """
        Send target prompt batched with filler prompts to change batch composition.

        Note: SGLang batching behavior depends on server config. This sends
        parallel requests in quick succession to increase co-batch probability.
        For strict same-batch testing, use the native Engine API with explicit batching.
        """
        # Primary request
        return self.generate(target_prompt, **kwargs)

    def logprob_record(
        self,
        prompt: str,
        *,
        label: str = "sglang",
        **kwargs: Any,
    ) -> LogprobRecord:
        result = self.generate(prompt, max_new_tokens=1, temperature=0.0, **kwargs)
        return LogprobRecord(
            engine=label,
            prompt=prompt,
            token_ids=result.token_ids,
            log_probs=result.log_probs,
            metadata={"text": result.text, **result.meta_info},
        )

    def _parse_response(self, data: dict[str, Any]) -> GenerateResult:
        meta = data.get("meta_info", {})
        output_logprobs = meta.get("output_token_logprobs") or meta.get("output_token_logprobs_val")

        token_ids: list[int] = []
        log_probs: list[float] = []

        if output_logprobs:
            for entry in output_logprobs:
                # SGLang format: (logprob, token_id, ...) or [logprob, token_id]
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    log_probs.append(float(entry[0]))
                    token_ids.append(int(entry[1]))
                elif isinstance(entry, dict):
                    log_probs.append(float(entry.get("logprob", entry.get("log_prob", 0))))
                    token_ids.append(int(entry.get("token_id", entry.get("id", 0))))

        # Fallback: input token logprobs for prefill-only requests
        if not log_probs:
            input_logprobs = meta.get("input_token_logprobs")
            if input_logprobs:
                for entry in input_logprobs:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        log_probs.append(float(entry[0]))
                        token_ids.append(int(entry[1]))

        return GenerateResult(
            text=data.get("text", ""),
            token_ids=token_ids,
            log_probs=log_probs,
            meta_info=meta,
            raw=data,
        )


def wait_for_server(client: SGLangClient, retries: int = 30, interval: float = 2.0) -> bool:
    for _ in range(retries):
        if client.health():
            return True
        time.sleep(interval)
    return False
