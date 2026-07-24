"""CLI entry point for llama-log-analyzer."""

from __future__ import annotations

import argparse
import csv
import io
import sys
from typing import Optional


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="llama-log-analyzer",
        description="Analyze llama.cpp / LocalAI log files for performance metrics.",
    )
    parser.add_argument(
        "logfile",
        help="Path to the log file to analyze.",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        dest="output_format",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DATEI",
        help="Write output to file instead of stdout.",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=None,
        metavar="TASK_ID",
        help="Filter output to a specific task ID.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---- parse log file ----
    from .parser import LogParser
    from .statistics import compute_summary

    try:
        log_parser = LogParser()
        log_parser.parse_file(args.logfile)
    except FileNotFoundError:
        print(f"Error: Log file not found: '{args.logfile}'", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: Failed to parse log file: {exc}", file=sys.stderr)
        return 1

    tasks = log_parser.get_tasks()

    # Filter by task if requested
    if args.task is not None:
        tasks = [t for t in tasks if t.task_id == args.task]

    summary = compute_summary(tasks)

    # ---- format output ----
    if args.output_format == "json":
        output = _format_json(tasks, summary)
    elif args.output_format == "csv":
        output = _format_csv(tasks, summary)
    else:
        output = _format_table(tasks, summary)

    # ---- write output ----
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
    else:
        print(output, end="")

    return 0


# ==================== formatters ====================


def _format_table(tasks: list[TaskMetrics], summary: Summary) -> str:
    lines: list[str] = []

    header = (
        f"{'Task':<8}"
        f"{'Prompt Tokens':>14} "
        f"{'Prompt t/s':>12} "
        f"{'Output Tokens':>14} "
        f"{'Output t/s':>12} "
        f"{'Total Time':>14} "
        f"{'Draft Acceptance':>17} "
        f"{'Mean Draft Len':>15} "
        f"{'Truncated':>10}"
    )
    lines.append(header)

    for t in tasks:
        pt = _fmt(t.prompt_tokens)
        ptps = _fmt(t.prompt_tokens_per_second)
        ot = _fmt(t.eval_tokens)
        otps = _fmt(t.eval_tokens_per_second)
        tt = _fmt_time(t.total_time_ms)
        da = _fmt(t.draft_acceptance)
        ml = _fmt(t.mean_len)
        tr = _fmt(t.truncated)

        line = (
            f"{t.task_id:<8}"
            f"{pt:>14} "
            f"{ptps:>12} "
            f"{ot:>14} "
            f"{otps:>12} "
            f"{tt:>14} "
            f"{da:>17} "
            f"{ml:>15} "
            f"{tr:>10}"
        )
        lines.append(line)

    # ---- summary ----
    lines.append("")
    lines.append("--- Summary ---")
    lines.append(f"Complete tasks: {summary.complete_tasks}")

    wapt = summary.weighted_avg_prompt_tps
    lines.append(f"Weighted avg prompt t/s: {_fmt(wapt)}")
    lines.append(f"Median prompt t/s: {_fmt(summary.median_prompt_tps)}")

    waot = summary.weighted_avg_output_tps
    lines.append(f"Weighted avg output t/s: {_fmt(waot)}")
    lines.append(f"Median output t/s: {_fmt(summary.median_output_tps)}")

    lines.append(f"Total prompt tokens: {summary.total_prompt_tokens}")
    lines.append(f"Total output tokens: {summary.total_output_tokens}")

    tt = _fmt_time(summary.total_time_ms)
    lines.append(f"Total measured time: {tt}")

    ft = summary.fastest_task
    st = summary.slowest_task
    if ft:
        lines.append(f"Fastest task: {ft[0]} ({ft[1]} t/s)")
    if st:
        lines.append(f"Slowest task: {st[0]} ({st[1]} t/s)")

    return "\n".join(lines)


def _format_json(tasks: list[TaskMetrics], summary: Summary) -> str:
    import json as _json

    data = {
        "tasks": [t.to_dict() for t in tasks],
        "summary": {
            "complete_tasks": summary.complete_tasks,
            "weighted_avg_prompt_tps": summary.weighted_avg_prompt_tps,
            "median_prompt_tps": summary.median_prompt_tps,
            "weighted_avg_output_tps": summary.weighted_avg_output_tps,
            "median_output_tps": summary.median_output_tps,
            "total_prompt_tokens": summary.total_prompt_tokens,
            "total_output_tokens": summary.total_output_tokens,
            "total_time_ms": summary.total_time_ms,
            "fastest_task": _task_tuple_to_dict(summary.fastest_task),
            "slowest_task": _task_tuple_to_dict(summary.slowest_task),
        },
    }
    return _json.dumps(data, indent=2, ensure_ascii=False)


def _format_csv(tasks: list[TaskMetrics], summary: Summary) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)

    header = [
        "Task",
        "Prompt Tokens",
        "Prompt t/s",
        "Output Tokens",
        "Output t/s",
        "Total Time",
        "Draft Acceptance",
        "Mean Draft Len",
        "Truncated",
    ]
    writer.writerow(header)

    for t in tasks:
        writer.writerow([
            t.task_id,
            t.prompt_tokens,
            t.prompt_tokens_per_second,
            t.eval_tokens,
            t.eval_tokens_per_second,
            t.total_time_ms,
            t.draft_acceptance,
            t.mean_len,
            t.truncated,
        ])

    # Summary rows
    writer.writerow([])
    writer.writerow(["--- Summary ---"])
    writer.writerow(["Complete tasks", summary.complete_tasks])
    writer.writerow(["Weighted avg prompt t/s", summary.weighted_avg_prompt_tps])
    writer.writerow(["Median prompt t/s", summary.median_prompt_tps])
    writer.writerow(["Weighted avg output t/s", summary.weighted_avg_output_tps])
    writer.writerow(["Median output t/s", summary.median_output_tps])
    writer.writerow(["Total prompt tokens", summary.total_prompt_tokens])
    writer.writerow(["Total output tokens", summary.total_output_tokens])
    writer.writerow(["Total time ms", summary.total_time_ms])

    ft = summary.fastest_task
    if ft:
        writer.writerow(["Fastest task", ft[0], f"{ft[1]} t/s"])
    st = summary.slowest_task
    if st:
        writer.writerow(["Slowest task", st[0], f"{st[1]} t/s"])

    return buf.getvalue()


# ==================== helpers ====================


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"


def _fmt_time(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f} ms"


def _task_tuple_to_dict(t: Optional[tuple]) -> Optional[dict]:
    if t is None:
        return None
    return {"task_id": t[0], "tokens_per_second": t[1]}
