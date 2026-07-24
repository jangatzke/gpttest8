"""Data models for llama-log-analyzer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TaskMetrics:
    """Represents the collected metrics for a single task."""

    task_id: int
    prompt_tokens: Optional[int] = None
    prompt_time_ms: Optional[float] = None
    prompt_tokens_per_second: Optional[float] = None
    eval_tokens: Optional[int] = None
    eval_time_ms: Optional[float] = None
    eval_tokens_per_second: Optional[float] = None
    total_time_ms: Optional[float] = None
    total_tokens: Optional[int] = None
    graphs_reused: Optional[int] = None
    draft_acceptance: Optional[float] = None
    draft_accepted: Optional[int] = None
    draft_generated: Optional[int] = None
    mean_len: Optional[float] = None
    n_tokens: Optional[int] = None
    truncated: Optional[int] = None

    # ---- helpers ----

    @property
    def is_complete(self) -> bool:
        """Return True if the task has enough fields to be considered complete.

        A task is considered complete when it has at least:
          - prompt_tokens, prompt_tokens_per_second
          - eval_tokens, eval_tokens_per_second
          - total_time_ms
        """
        return (
            self.prompt_tokens is not None
            and self.prompt_tokens_per_second is not None
            and self.eval_tokens is not None
            and self.eval_tokens_per_second is not None
            and self.total_time_ms is not None
        )

    def to_dict(self) -> dict:
        """Convert to a plain dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def __repr__(self) -> str:
        parts = [f"Task(id={self.task_id})"]
        if self.prompt_tokens is not None:
            parts.append(f"prompt={self.prompt_tokens}")
        if self.eval_tokens is not None:
            parts.append(f"output={self.eval_tokens}")
        return f"<{', '.join(parts)}>"


class Summary:
    """Aggregate statistics across all completed tasks."""

    def __init__(self, tasks: list[TaskMetrics]) -> None:
        self.tasks = tasks

    # ---- public properties ----

    @property
    def complete_tasks(self) -> int:
        return sum(1 for t in self.tasks if t.is_complete)

    # Weighted average: sum(tokens) / sum(time_in_seconds)
    @property
    def weighted_avg_prompt_tps(self) -> Optional[float]:
        total_tokens = sum(t.prompt_tokens for t in self.tasks if t.prompt_tokens is not None)
        total_time_s = sum(
            (t.prompt_time_ms or 0) / 1000.0 for t in self.tasks if t.prompt_time_ms is not None
        )
        if total_time_s == 0:
            return None
        return round(total_tokens / total_time_s, 2)

    @property
    def median_prompt_tps(self) -> Optional[float]:
        values = [t.prompt_tokens_per_second for t in self.tasks if t.prompt_tokens_per_second is not None]
        return _median(values) if values else None

    @property
    def weighted_avg_output_tps(self) -> Optional[float]:
        total_tokens = sum(t.eval_tokens for t in self.tasks if t.eval_tokens is not None)
        total_time_s = sum(
            (t.eval_time_ms or 0) / 1000.0 for t in self.tasks if t.eval_time_ms is not None
        )
        if total_time_s == 0:
            return None
        return round(total_tokens / total_time_s, 2)

    @property
    def median_output_tps(self) -> Optional[float]:
        values = [t.eval_tokens_per_second for t in self.tasks if t.eval_tokens_per_second is not None]
        return _median(values) if values else None

    @property
    def total_prompt_tokens(self) -> int:
        return sum(t.prompt_tokens for t in self.tasks if t.prompt_tokens is not None)

    @property
    def total_output_tokens(self) -> int:
        return sum(t.eval_tokens for t in self.tasks if t.eval_tokens is not None)

    @property
    def total_time_ms(self) -> Optional[float]:
        times = [t.total_time_ms for t in self.tasks if t.total_time_ms is not None]
        return sum(times) if times else None

    @property
    def fastest_task(self) -> Optional[tuple[int, float]]:
        """Return (task_id, eval_tokens_per_second) for the fastest task."""
        best: Optional[tuple[int, float]] = None
        for t in self.tasks:
            if t.eval_tokens_per_second is not None:
                if best is None or t.eval_tokens_per_second > best[1]:
                    best = (t.task_id, t.eval_tokens_per_second)
        return best

    @property
    def slowest_task(self) -> Optional[tuple[int, float]]:
        """Return (task_id, eval_tokens_per_second) for the slowest task."""
        worst: Optional[tuple[int, float]] = None
        for t in self.tasks:
            if t.eval_tokens_per_second is not None:
                if worst is None or t.eval_tokens_per_second < worst[1]:
                    worst = (t.task_id, t.eval_tokens_per_second)
        return worst

    def to_dict(self) -> dict[str, object]:
        return {
            "complete_tasks": self.complete_tasks,
            "weighted_avg_prompt_tps": self.weighted_avg_prompt_tps,
            "median_prompt_tps": self.median_prompt_tps,
            "weighted_avg_output_tps": self.weighted_avg_output_tps,
            "median_output_tps": self.median_output_tps,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_time_ms": self.total_time_ms,
            "fastest_task": _task_tuple_to_dict(self.fastest_task),
            "slowest_task": _task_tuple_to_dict(self.slowest_task),
        }


def _median(values: list[float]) -> float:
    """Return the median of a list of floats."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _task_tuple_to_dict(t: Optional[tuple]) -> Optional[dict]:
    """Convert a (task_id, tps) tuple to a dict, or None."""
    if t is None:
        return None
    return {"task_id": t[0], "tokens_per_second": t[1]}
