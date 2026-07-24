"""Parser for llama.cpp / LocalAI log lines."""

from __future__ import annotations

import re
from typing import Optional

from .models import TaskMetrics


# ---------- compiled patterns ----------

# I slot print_timing: id 0 | task 2687 | prompt eval time = 28183.05 ms / 14055 tokens (2.01 ms per token, 498.70 tokens per second)
_RE_PROMPT = re.compile(
    r"task\s+(\d+)"
    r".*?prompt\s+eval\s+time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    r"(?:\s*\([\d.]+\s*ms\s+per\s+token,\s*([\d.]+)\s*tokens\s+per\s+second\))?"
)

# I slot print_timing: id 0 | task 2687 | eval time = 18479.88 ms / 423 tokens (43.69 ms per token, 22.89 tokens per second)
_RE_EVAL = re.compile(
    r"task\s+(\d+)"
    r".*?eval\s+time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
    r"(?:\s*\([\d.]+\s*ms\s+per\s+token,\s*([\d.]+)\s*tokens\s+per\s+second\))?"
)

# I slot print_timing: id 0 | task 2687 | total time = 46662.93 ms / 14478 tokens
_RE_TOTAL = re.compile(
    r"task\s+(\d+)"
    r".*?total\s+time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
)

# I slot print_timing: id 0 | task 2687 | graphs reused = 2539
_RE_GRAPHS = re.compile(
    r"task\s+(\d+)"
    r".*?graphs\s+reused\s*=\s*(\d+)"
)

# I slot print_timing: id 0 | task 2687 | draft acceptance = 0.28296 (191 accepted / 675 generated), mean len = 13.73
_RE_DRAFT = re.compile(
    r"task\s+(\d+)"
    r".*?draft\s+acceptance\s*=\s*([\d.]+)"
    r"\s*\((\d+)\s+accepted\s*/\s*(\d+)\s+generated\)"
    r"(?:,\s*mean\s+len\s*=\s*([\d.]+))?"
)

# I slot release: id 0 | task 2687 | stop processing: n_tokens = 36752, truncated = 0
_RE_STOP = re.compile(
    r"task\s+(\d+)"
    r".*?stop\s+processing.*?n_tokens\s*=\s*(\d+)"
    r"(?:,\s*truncated\s*=\s*(\d+))?"
)


class LogParser:
    """Incrementally parse llama.cpp log lines into TaskMetrics."""

    def __init__(self) -> None:
        # task_id -> TaskMetrics accumulator
        self._tasks: dict[int, TaskMetrics] = {}

    # ---- public API ----

    def parse_line(self, line: str) -> None:
        """Parse a single log line and update the corresponding TaskMetrics."""
        line = line.strip()
        if not line:
            return

        # Extract task id from the line if possible
        task_id = self._extract_task_id(line)
        if task_id is None:
            return

        metrics = self._ensure_task(task_id)

        if _RE_PROMPT.search(line):
            self._parse_prompt(metrics, line)
        elif _RE_EVAL.search(line):
            self._parse_eval(metrics, line)
        elif _RE_TOTAL.search(line):
            self._parse_total(metrics, line)
        elif _RE_GRAPHS.search(line):
            self._parse_graphs(metrics, line)
        elif _RE_DRAFT.search(line):
            self._parse_draft(metrics, line)
        elif _RE_STOP.search(line):
            self._parse_stop(metrics, line)

    def parse_file(self, filepath: str) -> list[TaskMetrics]:
        """Parse an entire log file and return a list of TaskMetrics (in order of first appearance)."""
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                self.parse_line(line)
        return list(self._tasks.values())

    def get_tasks(self) -> list[TaskMetrics]:
        """Return all accumulated TaskMetrics in insertion order."""
        return list(self._tasks.values())

    # ---- internals ----

    @staticmethod
    def _extract_task_id(line: str) -> Optional[int]:
        m = re.search(r"task\s+(\d+)", line)
        return int(m.group(1)) if m else None

    def _ensure_task(self, task_id: int) -> TaskMetrics:
        if task_id not in self._tasks:
            self._tasks[task_id] = TaskMetrics(task_id=task_id)
        return self._tasks[task_id]

    @staticmethod
    def _parse_prompt(metrics: TaskMetrics, line: str) -> None:
        m = _RE_PROMPT.search(line)
        if m:
            metrics.prompt_time_ms = float(m.group(2))
            metrics.prompt_tokens = int(m.group(3))
            if m.group(4):
                metrics.prompt_tokens_per_second = float(m.group(4))

    @staticmethod
    def _parse_eval(metrics: TaskMetrics, line: str) -> None:
        m = _RE_EVAL.search(line)
        if m:
            metrics.eval_time_ms = float(m.group(2))
            metrics.eval_tokens = int(m.group(3))
            if m.group(4):
                metrics.eval_tokens_per_second = float(m.group(4))

    @staticmethod
    def _parse_total(metrics: TaskMetrics, line: str) -> None:
        m = _RE_TOTAL.search(line)
        if m:
            metrics.total_time_ms = float(m.group(2))
            metrics.total_tokens = int(m.group(3))

    @staticmethod
    def _parse_graphs(metrics: TaskMetrics, line: str) -> None:
        m = _RE_GRAPHS.search(line)
        if m:
            metrics.graphs_reused = int(m.group(2))

    @staticmethod
    def _parse_draft(metrics: TaskMetrics, line: str) -> None:
        m = _RE_DRAFT.search(line)
        if m:
            metrics.draft_acceptance = float(m.group(2))
            metrics.draft_accepted = int(m.group(3))
            metrics.draft_generated = int(m.group(4))
            if m.group(5):
                metrics.mean_len = float(m.group(5))

    @staticmethod
    def _parse_stop(metrics: TaskMetrics, line: str) -> None:
        m = _RE_STOP.search(line)
        if m:
            metrics.n_tokens = int(m.group(2))
            if m.group(3):
                metrics.truncated = int(m.group(3))
