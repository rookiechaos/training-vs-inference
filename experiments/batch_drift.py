"""Batch composition logprob drift experiment against live SGLang server."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any

from tim.backends.sglang_client import SGLangClient, SGLangConfig, wait_for_server
from tim.compare import LogprobRecord, aggregate_drift_reports, compare_logprob_records


def _burst_requests(
    client: SGLangClient,
    prompts: list[str],
    **kwargs: Any,
) -> list[LogprobRecord]:
    """Fire concurrent requests to increase dynamic batch co-location."""
    records: list[LogprobRecord] = []

    def _one(p: str, idx: int) -> LogprobRecord:
        return client.logprob_record(p, label=f"burst_{idx}", **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futs = [pool.submit(_one, p, i) for i, p in enumerate(prompts)]
        for f in concurrent.futures.as_completed(futs):
            records.append(f.result())
    return records


def run_batch_drift_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """
    Compare target prompt logprobs under different batch compositions.

    Config schema:
        sglang_url: str
        target_prompt: str
        filler_prompts: list[str]  # prompts to co-batch with
        batch_sizes: list[int]     # number of concurrent fillers (1, 4, 16, ...)
        repeats: int
    """
    url = config.get("sglang_url", "http://127.0.0.1:30000")
    client = SGLangClient(SGLangConfig(base_url=url))
    if not wait_for_server(client):
        raise RuntimeError(f"SGLang not reachable at {url}")

    target = config["target_prompt"]
    fillers: list[str] = config.get("filler_prompts", [])
    batch_sizes: list[int] = config.get("batch_sizes", [1, 4, 8])
    repeats: int = config.get("repeats", 3)

    # Baseline: isolated request
    baseline = client.logprob_record(target, label="baseline")
    if not baseline.token_ids:
        raise RuntimeError("No logprobs from baseline request")

    reports_by_batch: dict[str, list[dict]] = {}
    drift_by_batch: dict[str, list] = {}

    baseline_rec = LogprobRecord(
        engine="baseline",
        prompt=target,
        token_ids=baseline.token_ids,
        log_probs=baseline.log_probs,
    )

    for bs in batch_sizes:
        batch_reports: list[dict] = []
        drift_objs = []
        for _rep in range(repeats):
            selected = fillers[: max(bs - 1, 0)]
            burst = [target] + selected
            records = _burst_requests(client, burst, max_new_tokens=1, temperature=0.0)
            target_rec = next(r for r in records if r.prompt == target)

            report = compare_logprob_records(
                baseline_rec,
                target_rec,
                assume_a_is_train=False,
            )
            batch_reports.append(report.to_dict())
            drift_objs.append(report)
            time.sleep(0.1)

        reports_by_batch[str(bs)] = batch_reports
        drift_by_batch[str(bs)] = drift_objs

    return {
        "experiment": "batch_drift",
        "baseline": {
            "token_ids": baseline.token_ids,
            "log_probs": baseline.log_probs,
        },
        "reports_by_batch_size": reports_by_batch,
        "aggregate": {bs: aggregate_drift_reports(objs) for bs, objs in drift_by_batch.items()},
    }
