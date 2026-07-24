"""Statistics computation for llama-log-analyzer."""

from __future__ import annotations

from .models import TaskMetrics, Summary


def compute_summary(tasks: list[TaskMetrics]) -> Summary:
    """Compute aggregate statistics from a list of TaskMetrics."""
    return Summary(tasks)
