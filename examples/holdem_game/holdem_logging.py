"""Rollout logging for holdem agentic RL training.

Plugged in via --custom-rollout-log-function-path generate_with_holdem.log_rollout_data.
Produces console summaries and an append-only JSONL file alongside checkpoints.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

from holdem_reward import extract_tool_calls

logger = logging.getLogger(__name__)

_RESPONSE_PREVIEW_CHARS = 500
_FULL_RESPONSE_SAMPLE_K = 3


# ---------------------------------------------------------------------------
# JSONL sink (pattern from playground-ray/src/rl_logging.py)
# ---------------------------------------------------------------------------

class _JsonlSink:
    """Append-only JSONL event sink, flushed after every write."""

    def __init__(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = out.open("a", encoding="utf-8")

    def emit(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False))
        self._handle.write("\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


# Module-level sink; created lazily on first use, keyed by path.
_sinks: dict[str, _JsonlSink] = {}


def _get_sink(path: str) -> _JsonlSink:
    if path not in _sinks:
        _sinks[path] = _JsonlSink(path)
    return _sinks[path]


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def _compute_grpo_advantages(samples) -> dict[int, float]:
    """Compute per-sample whitened GRPO advantage.

    advantage_i = (score_i - group_mean) / (group_std + 1e-8)

    Returns a dict mapping list-index → advantage.
    """
    # group scores by group_index
    groups: dict[Any, list[tuple[int, float]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        reward = getattr(sample, "reward", None)
        if not isinstance(reward, dict):
            continue
        score = reward.get("score")
        if score is None:
            continue
        groups[sample.group_index].append((idx, float(score)))

    advantages: dict[int, float] = {}
    for members in groups.values():
        scores = [s for _, s in members]
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        for idx, score in members:
            advantages[idx] = (score - mean) / (std + 1e-8)

    return advantages


# ---------------------------------------------------------------------------
# Per-sample info extraction
# ---------------------------------------------------------------------------

def _collect_sample_info(
    sample,
    advantage: float | None,
    full_response: bool,
) -> dict[str, Any]:
    reward = sample.reward if isinstance(sample.reward, dict) else {}
    details = reward.get("details", {})
    protocol = details.get("protocol", {})
    action = details.get("action", {})
    efficiency = details.get("efficiency", {})

    # tool call sequence from response text
    try:
        tool_calls_raw = extract_tool_calls(sample.response or "")
        tool_sequence = [
            {"name": name, "arguments": args}
            for name, args in tool_calls_raw
        ]
    except Exception:
        tool_sequence = []

    response_text = sample.response or ""
    if full_response:
        response_field = response_text
    else:
        response_field = response_text[:_RESPONSE_PREVIEW_CHARS]

    protocol_violations = {
        k: v for k, v in protocol.items() if k != "score"
    }

    return {
        "score": reward.get("score"),
        "grpo_advantage": advantage,
        "pred": reward.get("pred"),
        "label": sample.label,
        "protocol_score": protocol.get("score"),
        "protocol_violations": protocol_violations if protocol_violations else None,
        "action_score": action.get("score"),
        "efficiency_penalty": efficiency.get("penalty"),
        "tool_turns": efficiency.get("turns"),
        "response_length": sample.response_length,
        "status": sample.status.value,
        "group_index": sample.group_index,
        "tool_sequence": tool_sequence,
        "response": response_field,
        "full_response": full_response,
    }


def _empty_sample_info(
    *,
    status: str,
    group_index: int | None = None,
    label: Any = None,
    response_length: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    info = {
        "score": None,
        "grpo_advantage": None,
        "pred": None,
        "label": label,
        "protocol_score": None,
        "protocol_violations": None,
        "action_score": None,
        "efficiency_penalty": None,
        "tool_turns": None,
        "response_length": response_length,
        "status": status,
        "group_index": group_index,
        "tool_sequence": [],
        "response": "",
        "full_response": False,
    }
    if error is not None:
        info["error"] = error
    return info


# ---------------------------------------------------------------------------
# Full-response sampling: pick K samples with highest |advantage|
# ---------------------------------------------------------------------------

def _select_full_response_indices(
    advantages: dict[int, float],
    k: int,
) -> set[int]:
    """Return k indices uniformly sampled from samples sorted by grpo_advantage."""
    if k <= 0:
        return set()
    ranked = sorted(advantages.keys(), key=lambda i: advantages[i])
    n = len(ranked)
    if n <= k:
        return set(ranked)
    # pick k evenly-spaced positions across the sorted list
    step = (n - 1) / (k - 1) if k > 1 else 0
    return {ranked[round(i * step)] for i in range(k)}


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _action_distribution(sample_infos: list[dict[str, Any]]) -> dict[str, int]:
    dist: dict[str, int] = defaultdict(int)
    for info in sample_infos:
        pred = info.get("pred") or "unknown"
        # normalize: keep only action type (strip amount)
        action_type = pred.split()[0] if pred else "unknown"
        dist[action_type] += 1
    return dict(dist)


def _console_summary(
    rollout_id: int,
    sample_infos: list[dict[str, Any]],
    rollout_time: float,
) -> None:
    scores = [i.get("score") for i in sample_infos if i.get("score") is not None]
    advantages = [i.get("grpo_advantage") for i in sample_infos if i.get("grpo_advantage") is not None]
    proto_scores = [i.get("protocol_score") for i in sample_infos if i.get("protocol_score") is not None]
    action_scores = [i.get("action_score") for i in sample_infos if i.get("action_score") is not None]
    turns = [i.get("tool_turns") for i in sample_infos if i.get("tool_turns") is not None]

    n = len(sample_infos)
    truncated = sum(1 for i in sample_infos if i.get("status") == "truncated")
    failed = sum(1 for i in sample_infos if i.get("status") == "failed")
    collection_error = sum(1 for i in sample_infos if i.get("status") == "collection_error")

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def _std(xs):
        if len(xs) < 2:
            return 0.0
        m = _mean(xs)
        return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

    action_dist = _action_distribution(sample_infos)
    action_dist_str = " ".join(f"{k}={v}" for k, v in sorted(action_dist.items()))

    summary = (
        f"[holdem][rollout {rollout_id}] "
        f"n={n} time={rollout_time:.1f}s "
        f"reward={_mean(scores):.3f}\u00b1{_std(scores):.3f} "
        f"advantage={_mean(advantages):.2f}\u00b1{_std(advantages):.2f} "
        f"proto={_mean(proto_scores):.3f} "
        f"action={_mean(action_scores):.3f} "
        f"turns={_mean(turns):.1f} "
        f"trunc={truncated}/{n}"
    )
    if failed:
        summary += f" failed={failed}/{n}"
    if collection_error:
        summary += f" collect_err={collection_error}/{n}"

    logger.info(summary)
    logger.info(f"  [holdem][rollout {rollout_id}] action_dist: {action_dist_str}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def log_rollout_data(
    rollout_id: int,
    args,
    samples,
    rollout_extra_metrics,
    rollout_time: float,
) -> bool:
    """Custom rollout log hook for holdem training.

    Called by slime after every rollout via --custom-rollout-log-function-path.
    Returns False so slime's default wandb/tensorboard logging still runs.
    """
    ts = time.time()

    # determine JSONL output path
    save_dir = getattr(args, "save", None)
    sink: _JsonlSink | None = None
    if save_dir:
        jsonl_path = os.path.join(save_dir, "holdem_rollout_log.jsonl")
        try:
            sink = _get_sink(jsonl_path)
        except Exception as exc:
            logger.warning(f"[holdem] failed to open JSONL sink at {jsonl_path}: {exc}")

    # keep full responses for top-K absolute-advantage samples.
    full_response_k = _FULL_RESPONSE_SAMPLE_K

    # compute per-sample GRPO advantages
    advantages = _compute_grpo_advantages(samples)

    # select which samples get full response logged
    full_response_indices = _select_full_response_indices(advantages, full_response_k)

    # collect per-sample info
    sample_infos: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        adv = advantages.get(idx)
        full = idx in full_response_indices
        try:
            info = _collect_sample_info(sample, adv, full_response=full)
        except Exception as exc:
            logger.warning(f"[holdem] failed to collect info for sample {idx}: {exc}")
            info = _empty_sample_info(
                status="collection_error",
                group_index=getattr(sample, "group_index", None),
                label=getattr(sample, "label", None),
                response_length=getattr(sample, "response_length", None),
                error=str(exc),
            )
        sample_infos.append(info)

        if sink is not None:
            sink.emit({
                "type": "sample",
                "rollout_id": rollout_id,
                "ts": ts,
                **info,
            })

    # console summary
    try:
        _console_summary(rollout_id, sample_infos, rollout_time)
    except Exception as exc:
        logger.warning(f"[holdem] failed to print rollout summary for rollout {rollout_id}: {exc}")

    # rollout-level JSONL summary
    if sink is not None:
        scores = [i["score"] for i in sample_infos if i.get("score") is not None]
        advs = [i["grpo_advantage"] for i in sample_infos if i.get("grpo_advantage") is not None]
        proto = [i["protocol_score"] for i in sample_infos if i.get("protocol_score") is not None]
        action = [i["action_score"] for i in sample_infos if i.get("action_score") is not None]
        turns = [i["tool_turns"] for i in sample_infos if i.get("tool_turns") is not None]

        def _safe_mean(xs):
            return sum(xs) / len(xs) if xs else None

        def _safe_std(xs):
            if len(xs) < 2:
                return None
            m = _safe_mean(xs)
            return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5

        action_dist = _action_distribution(sample_infos)
        status_dist = defaultdict(int)
        for info in sample_infos:
            status_dist[info.get("status", "unknown")] += 1

        sink.emit({
            "type": "rollout",
            "rollout_id": rollout_id,
            "ts": ts,
            "rollout_time": rollout_time,
            "n_samples": len(sample_infos),
            "reward_mean": _safe_mean(scores),
            "reward_std": _safe_std(scores),
            "advantage_mean": _safe_mean(advs),
            "advantage_std": _safe_std(advs),
            "protocol_score_mean": _safe_mean(proto),
            "action_score_mean": _safe_mean(action),
            "tool_turns_mean": _safe_mean(turns),
            "action_distribution": action_dist,
            "status_distribution": dict(status_dist),
        })

    return False


__all__ = ["log_rollout_data"]
