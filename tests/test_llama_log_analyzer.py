"""Comprehensive tests for llama-log-analyzer."""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import List

import pytest

# Ensure the src directory is on the path
src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from llama_log_analyzer.models import TaskMetrics, Summary, _median, _task_tuple_to_dict
from llama_log_analyzer.parser import LogParser
from llama_log_analyzer.statistics import compute_summary
from llama_log_analyzer.cli import main, build_parser


# ==================== Fixtures ====================

SAMPLE_LOG_LINE_PROMPT = (
    "I slot print_timing: id 0 | task 2687 | prompt eval time = 28183.05 ms / 14055 tokens (2.01 ms per token, 498.70 tokens per second)"
)

SAMPLE_LOG_LINE_EVAL = (
    "I slot print_timing: id 0 | task 2687 | eval time = 18479.88 ms / 423 tokens (43.69 ms per token, 22.89 tokens per second)"
)

SAMPLE_LOG_LINE_TOTAL = (
    "I slot print_timing: id 0 | task 2687 | total time = 46662.93 ms / 14478 tokens"
)

SAMPLE_LOG_LINE_GRAPHS = (
    "I slot print_timing: id 0 | task 2687 | graphs reused = 2539"
)

SAMPLE_LOG_LINE_DRAFT = (
    "I slot print_timing: id 0 | task 2687 | draft acceptance = 0.28296 (191 accepted / 675 generated), mean len = 13.73"
)

SAMPLE_LOG_LINE_STOP = (
    "I slot release: id 0 | task 2687 | stop processing: n_tokens = 36752, truncated = 0"
)


def _full_task_log() -> str:
    """Return a log string with all fields for a single complete task."""
    return "\n".join([
        SAMPLE_LOG_LINE_PROMPT,
        SAMPLE_LOG_LINE_EVAL,
        SAMPLE_LOG_LINE_TOTAL,
        SAMPLE_LOG_LINE_GRAPHS,
        SAMPLE_LOG_LINE_DRAFT,
        SAMPLE_LOG_LINE_STOP,
    ]) + "\n"


def _two_tasks_interleaved_log() -> str:
    """Return a log string with two interleaved tasks."""
    lines = [
        "I slot print_timing: id 0 | task 100 | prompt eval time = 1000.00 ms / 100 tokens (10.00 ms per token, 100.00 tokens per second)",
        "I slot print_timing: id 0 | task 200 | prompt eval time = 2000.00 ms / 200 tokens (10.00 ms per token, 100.00 tokens per second)",
        "I slot print_timing: id 0 | task 100 | eval time = 500.00 ms / 50 tokens (10.00 ms per token, 100.00 tokens per second)",
        "I slot print_timing: id 0 | task 200 | eval time = 1000.00 ms / 100 tokens (10.00 ms per token, 100.00 tokens per second)",
        "I slot print_timing: id 0 | task 100 | total time = 1500.00 ms / 150 tokens",
        "I slot print_timing: id 0 | task 200 | total time = 3000.00 ms / 300 tokens",
        "I slot print_timing: id 0 | task 100 | graphs reused = 100",
        "I slot print_timing: id 0 | task 200 | graphs reused = 200",
        "I slot print_timing: id 0 | task 100 | draft acceptance = 0.50000 (100 accepted / 200 generated), mean len = 5.00",
        "I slot print_timing: id 0 | task 200 | draft acceptance = 0.30000 (50 accepted / 166 generated), mean len = 3.33",
        "I slot release: id 0 | task 100 | stop processing: n_tokens = 500, truncated = 0",
        "I slot release: id 0 | task 200 | stop processing: n_tokens = 1000, truncated = 1",
    ]
    return "\n".join(lines) + "\n"


def _no_draft_log() -> str:
    """Return a log string without speculative decoding fields."""
    lines = [
        "I slot print_timing: id 0 | task 300 | prompt eval time = 500.00 ms / 50 tokens (10.00 ms per token, 100.00 tokens per second)",
        "I slot print_timing: id 0 | task 300 | eval time = 250.00 ms / 25 tokens (10.00 ms per token, 100.00 tokens per second)",
        "I slot print_timing: id 0 | task 300 | total time = 750.00 ms / 75 tokens",
    ]
    return "\n".join(lines) + "\n"


def _incomplete_task_log() -> str:
    """Return a log string with only partial info for a task."""
    return "I slot print_timing: id 0 | task 400 | prompt eval time = 100.00 ms / 10 tokens (10.00 ms per token, 100.00 tokens per second)\n"


def _unknown_lines_log() -> str:
    """Return a log string with unknown and empty lines."""
    lines = [
        "",
        "I some random unknown log line",
        "I another unrelated line",
        "I slot print_timing: id 0 | task 500 | prompt eval time = 200.00 ms / 20 tokens (10.00 ms per token, 100.00 tokens per second)",
        "garbage data 12345",
        "I slot print_timing: id 0 | task 500 | eval time = 100.00 ms / 10 tokens (10.00 ms per token, 100.00 tokens per second)",
        "I slot print_timing: id 0 | task 500 | total time = 300.00 ms / 30 tokens",
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture
def tmp_log_file(tmp_path: Path) -> Path:
    """Create a temporary log file with a full task."""
    p = tmp_path / "test.log"
    p.write_text(_full_task_log())
    return p


@pytest.fixture
def tmp_interleaved_file(tmp_path: Path) -> Path:
    """Create a temporary log file with two interleaved tasks."""
    p = tmp_path / "interleaved.log"
    p.write_text(_two_tasks_interleaved_log())
    return p


@pytest.fixture
def tmp_no_draft_file(tmp_path: Path) -> Path:
    """Create a temporary log file without speculative decoding."""
    p = tmp_path / "no_draft.log"
    p.write_text(_no_draft_log())
    return p


@pytest.fixture
def tmp_incomplete_file(tmp_path: Path) -> Path:
    """Create a temporary log file with an incomplete task."""
    p = tmp_path / "incomplete.log"
    p.write_text(_incomplete_task_log())
    return p


@pytest.fixture
def tmp_unknown_file(tmp_path: Path) -> Path:
    """Create a temporary log file with unknown lines."""
    p = tmp_path / "unknown.log"
    p.write_text(_unknown_lines_log())
    return p


# ==================== Tests: Data Model ====================


class TestTaskMetrics:
    """Tests for TaskMetrics model."""

    def test_complete_task(self):
        t = TaskMetrics(
            task_id=1,
            prompt_tokens=100,
            prompt_time_ms=1000.0,
            prompt_tokens_per_second=100.0,
            eval_tokens=50,
            eval_time_ms=500.0,
            eval_tokens_per_second=100.0,
            total_time_ms=1500.0,
            total_tokens=150,
        )
        assert t.is_complete is True

    def test_incomplete_task(self):
        t = TaskMetrics(task_id=1, prompt_tokens=100)
        assert t.is_complete is False

    def test_to_dict_excludes_none(self):
        t = TaskMetrics(task_id=1, prompt_tokens=100)
        d = t.to_dict()
        assert d == {"task_id": 1, "prompt_tokens": 100}
        assert "eval_tokens" not in d

    def test_repr(self):
        t = TaskMetrics(task_id=42, prompt_tokens=10, eval_tokens=5)
        assert "Task(id=42)" in repr(t)


class TestSummary:
    """Tests for Summary statistics."""

    def test_complete_tasks_count(self):
        tasks = [
            TaskMetrics(1, prompt_tokens=100, prompt_time_ms=1000, prompt_tokens_per_second=100,
                        eval_tokens=50, eval_time_ms=500, eval_tokens_per_second=100, total_time_ms=1500),
            TaskMetrics(2, prompt_tokens=100, prompt_time_ms=1000, prompt_tokens_per_second=100,
                        eval_tokens=50, eval_time_ms=500, eval_tokens_per_second=100, total_time_ms=1500),
            TaskMetrics(3, prompt_tokens=100),  # incomplete
        ]
        s = Summary(tasks)
        assert s.complete_tasks == 2

    def test_weighted_avg_prompt_tps(self):
        """Weighted avg = total_prompt_tokens / total_time_s."""
        # Task 1: 100 tokens in 1s => 100 t/s
        # Task 2: 200 tokens in 2s => 100 t/s
        # Weighted: 300 tokens / 3s = 100 t/s
        tasks = [
            TaskMetrics(1, prompt_tokens=100, prompt_time_ms=1000, prompt_tokens_per_second=100,
                        eval_tokens=50, eval_time_ms=500, eval_tokens_per_second=100, total_time_ms=1500),
            TaskMetrics(2, prompt_tokens=200, prompt_time_ms=2000, prompt_tokens_per_second=100,
                        eval_tokens=100, eval_time_ms=1000, eval_tokens_per_second=100, total_time_ms=3000),
        ]
        s = Summary(tasks)
        assert s.weighted_avg_prompt_tps == 100.0

    def test_weighted_avg_output_tps(self):
        tasks = [
            TaskMetrics(1, prompt_tokens=100, prompt_time_ms=1000, prompt_tokens_per_second=100,
                        eval_tokens=50, eval_time_ms=500, eval_tokens_per_second=100, total_time_ms=1500),
            TaskMetrics(2, prompt_tokens=200, prompt_time_ms=2000, prompt_tokens_per_second=100,
                        eval_tokens=100, eval_time_ms=1000, eval_tokens_per_second=100, total_time_ms=3000),
        ]
        s = Summary(tasks)
        assert s.weighted_avg_output_tps == 100.0

    def test_median_output_tps(self):
        tasks = [
            TaskMetrics(1, prompt_tokens=100, prompt_time_ms=1000, prompt_tokens_per_second=100,
                        eval_tokens=50, eval_time_ms=500, eval_tokens_per_second=50, total_time_ms=1500),
            TaskMetrics(2, prompt_tokens=200, prompt_time_ms=2000, prompt_tokens_per_second=100,
                        eval_tokens=100, eval_time_ms=1000, eval_tokens_per_second=100, total_time_ms=3000),
            TaskMetrics(3, prompt_tokens=300, prompt_time_ms=3000, prompt_tokens_per_second=100,
                        eval_tokens=150, eval_time_ms=1500, eval_tokens_per_second=150, total_time_ms=4500),
        ]
        s = Summary(tasks)
        # median of [50, 100, 150] = 100
        assert s.median_output_tps == 100.0

    def test_median_output_tps_even(self):
        tasks = [
            TaskMetrics(1, prompt_tokens=100, prompt_time_ms=1000, prompt_tokens_per_second=100,
                        eval_tokens=50, eval_time_ms=500, eval_tokens_per_second=50, total_time_ms=1500),
            TaskMetrics(2, prompt_tokens=200, prompt_time_ms=2000, prompt_tokens_per_second=100,
                        eval_tokens=100, eval_time_ms=1000, eval_tokens_per_second=150, total_time_ms=3000),
        ]
        s = Summary(tasks)
        # median of [50, 150] = 100
        assert s.median_output_tps == 100.0

    def test_total_tokens(self):
        tasks = [
            TaskMetrics(1, prompt_tokens=100, eval_tokens=50),
            TaskMetrics(2, prompt_tokens=200, eval_tokens=100),
        ]
        s = Summary(tasks)
        assert s.total_prompt_tokens == 300
        assert s.total_output_tokens == 150

    def test_total_time_ms(self):
        tasks = [
            TaskMetrics(1, total_time_ms=1000.0),
            TaskMetrics(2, total_time_ms=2000.0),
        ]
        s = Summary(tasks)
        assert s.total_time_ms == 3000.0

    def test_fastest_slowest_task(self):
        tasks = [
            TaskMetrics(1, eval_tokens_per_second=50.0),
            TaskMetrics(2, eval_tokens_per_second=200.0),
            TaskMetrics(3, eval_tokens_per_second=100.0),
        ]
        s = Summary(tasks)
        assert s.fastest_task == (2, 200.0)
        assert s.slowest_task == (1, 50.0)

    def test_to_dict(self):
        tasks = [
            TaskMetrics(1, prompt_tokens=100, prompt_time_ms=1000, prompt_tokens_per_second=100,
                        eval_tokens=50, eval_time_ms=500, eval_tokens_per_second=100, total_time_ms=1500),
        ]
        s = Summary(tasks)
        d = s.to_dict()
        assert d["complete_tasks"] == 1
        assert d["weighted_avg_prompt_tps"] == 100.0
        assert d["fastest_task"] == {"task_id": 1, "tokens_per_second": 100.0}

    def test_empty_summary(self):
        s = Summary([])
        assert s.complete_tasks == 0
        assert s.weighted_avg_prompt_tps is None
        assert s.median_prompt_tps is None
        assert s.total_prompt_tokens == 0


class TestHelpers:
    """Tests for helper functions."""

    def test_median_odd(self):
        assert _median([3, 1, 2]) == 2

    def test_median_even(self):
        assert _median([4, 1, 3, 2]) == 2.5

    def test_median_single(self):
        assert _median([42]) == 42

    def test_median_empty(self):
        assert _median([]) == 0.0

    def test_task_tuple_to_dict(self):
        assert _task_tuple_to_dict((1, 100.0)) == {"task_id": 1, "tokens_per_second": 100.0}
        assert _task_tuple_to_dict(None) is None


# ==================== Tests: Parser ====================


class TestLogParser:
    """Tests for LogParser."""

    def test_parse_single_complete_task(self, tmp_log_file: Path):
        parser = LogParser()
        tasks = parser.parse_file(str(tmp_log_file))
        assert len(tasks) == 1
        t = tasks[0]
        assert t.task_id == 2687
        assert t.prompt_tokens == 14055
        assert abs(t.prompt_time_ms - 28183.05) < 0.01
        assert abs(t.prompt_tokens_per_second - 498.70) < 0.01
        assert t.eval_tokens == 423
        assert abs(t.eval_time_ms - 18479.88) < 0.01
        assert abs(t.eval_tokens_per_second - 22.89) < 0.01
        assert abs(t.total_time_ms - 46662.93) < 0.01
        assert t.total_tokens == 14478
        assert t.graphs_reused == 2539
        assert abs(t.draft_acceptance - 0.28296) < 0.0001
        assert t.draft_accepted == 191
        assert t.draft_generated == 675
        assert abs(t.mean_len - 13.73) < 0.01
        assert t.n_tokens == 36752
        assert t.truncated == 0
        assert t.is_complete is True

    def test_parse_interleaved_two_tasks(self, tmp_interleaved_file: Path):
        parser = LogParser()
        tasks = parser.parse_file(str(tmp_interleaved_file))
        assert len(tasks) == 2
        by_id = {t.task_id: t for t in tasks}
        assert 100 in by_id
        assert 200 in by_id
        assert by_id[100].prompt_tokens == 100
        assert by_id[100].eval_tokens == 50
        assert by_id[200].prompt_tokens == 200
        assert by_id[200].eval_tokens == 100
        assert by_id[100].is_complete is True
        assert by_id[200].is_complete is True

    def test_parse_no_speculative_decoding(self, tmp_no_draft_file: Path):
        parser = LogParser()
        tasks = parser.parse_file(str(tmp_no_draft_file))
        assert len(tasks) == 1
        t = tasks[0]
        assert t.task_id == 300
        assert t.draft_acceptance is None
        assert t.mean_len is None
        assert t.is_complete is True

    def test_parse_incomplete_task(self, tmp_incomplete_file: Path):
        parser = LogParser()
        tasks = parser.parse_file(str(tmp_incomplete_file))
        assert len(tasks) == 1
        assert tasks[0].is_complete is False
        assert tasks[0].prompt_tokens == 10

    def test_parse_unknown_lines(self, tmp_unknown_file: Path):
        parser = LogParser()
        tasks = parser.parse_file(str(tmp_unknown_file))
        assert len(tasks) == 1
        assert tasks[0].task_id == 500
        assert tasks[0].prompt_tokens == 20
        assert tasks[0].eval_tokens == 10

    def test_parse_empty_file(self, tmp_path: Path):
        p = tmp_path / "empty.log"
        p.write_text("")
        parser = LogParser()
        tasks = parser.parse_file(str(p))
        assert len(tasks) == 0

    def test_parse_line_directly(self):
        parser = LogParser()
        parser.parse_line(SAMPLE_LOG_LINE_PROMPT)
        parser.parse_line(SAMPLE_LOG_LINE_EVAL)
        parser.parse_line(SAMPLE_LOG_LINE_TOTAL)
        parser.parse_line(SAMPLE_LOG_LINE_GRAPHS)
        parser.parse_line(SAMPLE_LOG_LINE_DRAFT)
        parser.parse_line(SAMPLE_LOG_LINE_STOP)
        tasks = parser.get_tasks()
        assert len(tasks) == 1
        assert tasks[0].is_complete is True

    def test_parse_empty_line(self):
        parser = LogParser()
        parser.parse_line("")
        assert len(parser.get_tasks()) == 0

    def test_parse_line_no_task_id(self):
        parser = LogParser()
        parser.parse_line("I some random line without task id")
        assert len(parser.get_tasks()) == 0


# ==================== Tests: CLI ====================


class TestCLI:
    """Tests for CLI interface."""

    def test_table_output(self, tmp_log_file: Path):
        result = main([str(tmp_log_file)])
        assert result == 0

    def test_table_output_to_stdout(self, tmp_log_file: Path, capsys):
        result = main([str(tmp_log_file)])
        assert result == 0
        captured = capsys.readouterr()
        assert "Task" in captured.out
        assert "2687" in captured.out
        assert "--- Summary ---" in captured.out
        assert "Complete tasks" in captured.out

    def test_json_output(self, tmp_log_file: Path, capsys):
        result = main([str(tmp_log_file), "--format", "json"])
        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["task_id"] == 2687
        assert "summary" in data
        assert data["summary"]["complete_tasks"] == 1

    def test_csv_output(self, tmp_log_file: Path, capsys):
        result = main([str(tmp_log_file), "--format", "csv"])
        assert result == 0
        captured = capsys.readouterr()
        reader = csv.reader(io.StringIO(captured.out))
        rows = [r for r in reader if r]  # skip empty rows
        assert rows[0] == ["Task", "Prompt Tokens", "Prompt t/s", "Output Tokens", "Output t/s", "Total Time", "Draft Acceptance", "Mean Draft Len", "Truncated"]
        # Find the data row for task 2687
        data_rows = [r for r in rows if r[0] == "2687"]
        assert len(data_rows) == 1

    def test_csv_output_summary(self, tmp_log_file: Path, capsys):
        result = main([str(tmp_log_file), "--format", "csv"])
        assert result == 0
        captured = capsys.readouterr()
        assert "--- Summary ---" in captured.out
        assert "Complete tasks" in captured.out

    def test_output_to_file(self, tmp_log_file: Path, tmp_path: Path):
        out_file = tmp_path / "output.txt"
        result = main([str(tmp_log_file), "--output", str(out_file)])
        assert result == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert "2687" in content

    def test_filter_by_task(self, tmp_interleaved_file: Path, capsys):
        result = main([str(tmp_interleaved_file), "--task", "100"])
        assert result == 0
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "200" not in captured.out

    def test_filter_by_task_no_match(self, tmp_log_file: Path, capsys):
        result = main([str(tmp_log_file), "--task", "9999"])
        assert result == 0
        captured = capsys.readouterr()
        assert "9999" not in captured.out

    def test_file_not_found(self):
        result = main(["nonexistent_file.log"])
        assert result != 0

    def test_file_not_found_error_message(self, capfd):
        result = main(["nonexistent_file.log"])
        assert result != 0
        captured = capfd.readouterr()
        assert "not found" in captured.err.lower() or "Not found" in captured.err

    def test_interleaved_tasks_table(self, tmp_interleaved_file: Path, capsys):
        result = main([str(tmp_interleaved_file)])
        assert result == 0
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "200" in captured.out
        assert "--- Summary ---" in captured.out

    def test_incomplete_task_no_crash(self, tmp_incomplete_file: Path, capsys):
        result = main([str(tmp_incomplete_file)])
        assert result == 0

    def test_unknown_lines_no_crash(self, tmp_unknown_file: Path, capsys):
        result = main([str(tmp_unknown_file)])
        assert result == 0

    def test_no_draft_output(self, tmp_no_draft_file: Path, capsys):
        result = main([str(tmp_no_draft_file)])
        assert result == 0
        captured = capsys.readouterr()
        assert "300" in captured.out


# ==================== Tests: Weighted Average Calculation ====================


class TestWeightedAverageCalculation:
    """Tests that verify token-weighted average computation."""

    def test_weighted_avg_vs_simple_avg(self):
        """Weighted average should differ from simple average when tasks have different sizes."""
        # Task 1: 1000 prompt tokens in 10s => 100 t/s
        # Task 2: 10 prompt tokens in 0.1s => 100 t/s
        # Weighted: 1010 / 10.1 = 100 t/s (same in this case)
        # Let's make it different:
        # Task 1: 1000 tokens in 10s => 100 t/s
        # Task 2: 10 tokens in 1s => 10 t/s
        # Weighted: 1010 / 11 = 91.82 t/s
        # Simple avg of [100, 10] = 55 t/s
        tasks = [
            TaskMetrics(1, prompt_tokens=1000, prompt_time_ms=10000, prompt_tokens_per_second=100,
                        eval_tokens=100, eval_time_ms=1000, eval_tokens_per_second=100, total_time_ms=11000),
            TaskMetrics(2, prompt_tokens=10, prompt_time_ms=1000, prompt_tokens_per_second=10,
                        eval_tokens=10, eval_time_ms=100, eval_tokens_per_second=10, total_time_ms=1100),
        ]
        s = Summary(tasks)
        # Weighted: 1010 tokens / 11s = 91.82 t/s
        assert abs(s.weighted_avg_prompt_tps - 91.82) < 0.1
        # Simple avg would be (100 + 10) / 2 = 55
        assert s.weighted_avg_prompt_tps > 55  # weighted should be higher here

    def test_weighted_output_avg(self):
        tasks = [
            TaskMetrics(1, eval_tokens=1000, eval_time_ms=10000, eval_tokens_per_second=100, total_time_ms=11000),
            TaskMetrics(2, eval_tokens=10, eval_time_ms=100, eval_tokens_per_second=100, total_time_ms=1100),
        ]
        s = Summary(tasks)
        # Weighted: 1010 / 10.1 = 100 t/s
        assert abs(s.weighted_avg_output_tps - 100.0) < 0.1


# ==================== Tests: Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_completely_empty_log(self, tmp_path: Path):
        p = tmp_path / "empty.log"
        p.write_text("")
        result = main([str(p)])
        assert result == 0
        # Should produce empty output without crashing

    def test_log_with_only_unknown_lines(self, tmp_path: Path):
        p = tmp_path / "unknown.log"
        p.write_text("random line 1\nrandom line 2\n")
        result = main([str(p)])
        assert result == 0

    def test_decimal_points_in_log(self, tmp_path: Path):
        """Test that decimal numbers with dots are parsed correctly."""
        lines = [
            "I slot print_timing: id 0 | task 600 | prompt eval time = 1234.56 ms / 100 tokens (12.34 ms per token, 81.02 tokens per second)",
            "I slot print_timing: id 0 | task 600 | eval time = 5678.90 ms / 50 tokens (113.58 ms per token, 8.80 tokens per second)",
            "I slot print_timing: id 0 | task 600 | total time = 6913.46 ms / 150 tokens",
        ]
        p = tmp_path / "decimal.log"
        p.write_text("\n".join(lines) + "\n")
        parser = LogParser()
        tasks = parser.parse_file(str(p))
        assert len(tasks) == 1
        assert abs(tasks[0].prompt_time_ms - 1234.56) < 0.01
        assert abs(tasks[0].eval_time_ms - 5678.90) < 0.01

    def test_multiple_tasks_different_ids(self, tmp_path: Path):
        lines = [
            "I slot print_timing: id 0 | task 1 | prompt eval time = 100.00 ms / 10 tokens (10.00 ms per token, 100.00 tokens per second)",
            "I slot print_timing: id 0 | task 2 | prompt eval time = 200.00 ms / 20 tokens (10.00 ms per token, 100.00 tokens per second)",
            "I slot print_timing: id 0 | task 3 | prompt eval time = 300.00 ms / 30 tokens (10.00 ms per token, 100.00 tokens per second)",
        ]
        p = tmp_path / "multi.log"
        p.write_text("\n".join(lines) + "\n")
        parser = LogParser()
        tasks = parser.parse_file(str(p))
        assert len(tasks) == 3
        ids = [t.task_id for t in tasks]
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids


# ==================== Tests: Argument Parser ====================


class TestArgumentParser:
    """Tests for CLI argument parsing."""

    def test_default_format(self):
        p = build_parser()
        args = p.parse_args(["test.log"])
        assert args.output_format == "table"

    def test_json_format(self):
        p = build_parser()
        args = p.parse_args(["test.log", "--format", "json"])
        assert args.output_format == "json"

    def test_csv_format(self):
        p = build_parser()
        args = p.parse_args(["test.log", "--format", "csv"])
        assert args.output_format == "csv"

    def test_output_option(self):
        p = build_parser()
        args = p.parse_args(["test.log", "--output", "out.txt"])
        assert args.output == "out.txt"

    def test_task_option(self):
        p = build_parser()
        args = p.parse_args(["test.log", "--task", "42"])
        assert args.task == 42


# ==================== Tests: Real-World Fixtures ====================


class TestRealWorldFixtures:
    """Tests using real-world log data collected from the internet."""

    def test_parse_speculative_decoding_log(self, tmp_path: Path):
        """Parse a server log with speculative decoding (NGRAM/MTP)."""
        from fixtures import llama_cpp_server_speculative_decoding_log
        p = tmp_path / "spec.log"
        p.write_text(llama_cpp_server_speculative_decoding_log())
        parser = LogParser()
        tasks = parser.parse_file(str(p))
        # Two complete tasks
        assert len(tasks) == 2
        by_id = {t.task_id: t for t in tasks}
        # Task 100
        assert 100 in by_id
        t100 = by_id[100]
        assert t100.is_complete is True
        assert t100.prompt_tokens == 550
        assert abs(t100.prompt_tokens_per_second - 500.0) < 0.1
        assert t100.eval_tokens == 275
        assert abs(t100.eval_tokens_per_second - 50.0) < 0.1
        assert t100.graphs_reused == 25
        assert abs(t100.draft_acceptance - 0.74177) < 0.001
        assert t100.draft_accepted == 1126
        assert t100.draft_generated == 1518
        assert abs(t100.mean_len - 5.85) < 0.1
        assert t100.n_tokens == 825
        assert t100.truncated == 0
        # Task 200
        assert 200 in by_id
        t200 = by_id[200]
        assert t200.is_complete is True
        assert t200.prompt_tokens == 600
        assert t200.eval_tokens == 300
        assert abs(t200.draft_acceptance - 0.65) < 0.001
        assert t200.draft_accepted == 195
        assert t200.draft_generated == 300
        assert abs(t200.mean_len - 3.90) < 0.1

    def test_parse_prefix_match_log(self, tmp_path: Path):
        """Parse server log with KV-cache prefix-match hits (zero-token prompts)."""
        from fixtures import llama_cpp_prefix_match_log
        p = tmp_path / "prefix.log"
        p.write_text(llama_cpp_prefix_match_log())
        parser = LogParser()
        tasks = parser.parse_file(str(p))
        # Three tasks: task 0 (full), task 1 (prefix match), task 2 (prefix match)
        assert len(tasks) == 3
        by_id = {t.task_id: t for t in tasks}
        # Task 0: full prompt
        assert by_id[0].prompt_tokens == 1000
        assert by_id[0].eval_tokens == 75
        # Task 1: prefix match (0 tokens prompt)
        assert by_id[1].prompt_tokens == 0
        assert by_id[1].eval_tokens == 77
        # Task 2: prefix match (0 tokens prompt)
        assert by_id[2].prompt_tokens == 0
        assert by_id[2].eval_tokens == 70

    def test_parse_parallel_requests_log(self, tmp_path: Path):
        """Parse server log with parallel requests from multiple slots."""
        from fixtures import llama_cpp_parallel_requests_log
        p = tmp_path / "parallel.log"
        p.write_text(llama_cpp_parallel_requests_log())
        parser = LogParser()
        tasks = parser.parse_file(str(p))
        assert len(tasks) == 3
        by_id = {t.task_id: t for t in tasks}
        # Task 0 (slot 0)
        assert by_id[0].prompt_tokens == 750
        assert by_id[0].eval_tokens == 150
        assert by_id[0].truncated == 0
        # Task 1 (slot 1) - truncated
        assert by_id[1].prompt_tokens == 800
        assert by_id[1].eval_tokens == 160
        assert by_id[1].truncated == 1
        # Task 2 (slot 0 again)
        assert by_id[2].prompt_tokens == 700
        assert by_id[2].eval_tokens == 125

    def test_parse_gpu_offload_log(self, tmp_path: Path):
        """Parse server log from GPU offloaded RTX 6000 with speculative decoding."""
        from fixtures import llama_cpp_gpu_offload_log
        p = tmp_path / "gpu.log"
        p.write_text(llama_cpp_gpu_offload_log())
        parser = LogParser()
        tasks = parser.parse_file(str(p))
        assert len(tasks) == 1
        t = tasks[0]
        assert t.task_id == 1768
        assert t.prompt_tokens == 18
        assert abs(t.prompt_tokens_per_second - 8.67) < 0.1
        assert t.eval_tokens == 128
        assert abs(t.eval_tokens_per_second - 7.11) < 0.1
        assert abs(t.draft_acceptance - 0.94156) < 0.001
        assert t.draft_accepted == 7556
        assert t.draft_generated == 8026
        assert abs(t.mean_len - 9.42) < 0.1

    def test_parse_dflash_log(self, tmp_path: Path):
        """Parse server log with DFlash speculative decoding (low acceptance rate)."""
        from fixtures import llama_cpp_dflash_log
        p = tmp_path / "dflash.log"
        p.write_text(llama_cpp_dflash_log())
        parser = LogParser()
        tasks = parser.parse_file(str(p))
        assert len(tasks) == 1
        t = tasks[0]
        assert t.task_id == 56
        assert t.prompt_tokens == 17
        assert abs(t.prompt_tokens_per_second - 79.49) < 0.1
        assert t.eval_tokens == 886
        assert abs(t.eval_tokens_per_second - 5.82) < 0.1
        assert abs(t.draft_acceptance - 0.09722) < 0.001
        assert t.draft_accepted == 86
        assert t.draft_generated == 885
        assert abs(t.mean_len - 1.05) < 0.1

    def test_parse_system_info_log(self, tmp_path: Path):
        """Parse server log with system info lines that should be ignored."""
        from fixtures import llama_cpp_system_info_log
        p = tmp_path / "system.log"
        p.write_text(llama_cpp_system_info_log())
        parser = LogParser()
        tasks = parser.parse_file(str(p))
        assert len(tasks) == 1
        t = tasks[0]
        assert t.task_id == 0
        assert t.prompt_tokens == 500
        assert t.eval_tokens == 100
        assert t.is_complete is True

    def test_parse_truncated_log(self, tmp_path: Path):
        """Parse log with truncated output (max tokens reached)."""
        from fixtures import llama_cpp_truncated_log
        p = tmp_path / "truncated.log"
        p.write_text(llama_cpp_truncated_log())
        parser = LogParser()
        tasks = parser.parse_file(str(p))
        assert len(tasks) == 1
        t = tasks[0]
        assert t.truncated == 1
        assert t.n_tokens == 1500

    def test_cli_with_real_world_log(self, tmp_path: Path, capsys):
        """Test CLI output with a real-world log fixture."""
        from fixtures import llama_cpp_server_speculative_decoding_log
        p = tmp_path / "real.log"
        p.write_text(llama_cpp_server_speculative_decoding_log())
        result = main([str(p)])
        assert result == 0
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "200" in captured.out
        assert "--- Summary ---" in captured.out

    def test_json_with_real_world_log(self, tmp_path: Path, capsys):
        """Test JSON output with a real-world log fixture."""
        from fixtures import llama_cpp_server_speculative_decoding_log
        p = tmp_path / "real.log"
        p.write_text(llama_cpp_server_speculative_decoding_log())
        result = main([str(p), "--format", "json"])
        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data["tasks"]) == 2
        assert data["tasks"][0]["task_id"] == 100
        assert data["tasks"][1]["task_id"] == 200
        assert data["summary"]["complete_tasks"] == 2

    def test_csv_with_real_world_log(self, tmp_path: Path, capsys):
        """Test CSV output with a real-world log fixture."""
        from fixtures import llama_cpp_server_speculative_decoding_log
        p = tmp_path / "real.log"
        p.write_text(llama_cpp_server_speculative_decoding_log())
        result = main([str(p), "--format", "csv"])
        assert result == 0
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "200" in captured.out

    def test_filter_task_with_real_world_log(self, tmp_path: Path, capsys):
        """Test --task filter with a real-world log fixture."""
        from fixtures import llama_cpp_server_speculative_decoding_log
        p = tmp_path / "real.log"
        p.write_text(llama_cpp_server_speculative_decoding_log())
        result = main([str(p), "--task", "100"])
        assert result == 0
        captured = capsys.readouterr()
        assert "100" in captured.out
        assert "200" not in captured.out

    def test_weighted_stats_with_real_world_log(self, tmp_path: Path, capsys):
        """Test weighted statistics with a real-world log fixture."""
        from fixtures import llama_cpp_parallel_requests_log
        p = tmp_path / "parallel.log"
        p.write_text(llama_cpp_parallel_requests_log())
        result = main([str(p), "--format", "json"])
        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        summary = data["summary"]
        # Should have 3 complete tasks
        assert summary["complete_tasks"] == 3
        # Weighted avg should be computed correctly
        assert summary["weighted_avg_prompt_tps"] is not None
        assert summary["weighted_avg_output_tps"] is not None
        # Median should be computed from individual task values
        assert summary["median_prompt_tps"] is not None
        assert summary["median_output_tps"] is not None
