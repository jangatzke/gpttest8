# Llama Log Analyzer

A command-line tool for analyzing llama.cpp / LocalAI log files and extracting performance metrics per task plus aggregate statistics.

## Features

- Parses llama.cpp / LocalAI timing log lines
- Extracts per-task metrics: prompt tokens, output tokens, tokens/second, draft acceptance, etc.
- Outputs results in **table**, **JSON**, or **CSV** format
- Computes token-weighted average throughput and median values
- Filters by specific task ID
- Handles incomplete, corrupted, or interleaved log entries gracefully

## Installation

```bash
git clone <repository>
cd llama-log-analyzer
pip install -e .
```

Or install directly from PyPI (when published):

```bash
pip install llama-log-analyzer
```

For development / testing:

```bash
pip install -e ".[dev]"
```

## Usage

```bash
python -m llama_log_analyzer logfile.txt
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--format {table,json,csv}` | Output format | `table` |
| `--output DATEI` | Write output to file instead of stdout | stdout |
| `--task TASK_ID` | Filter output to a specific task ID | all tasks |

### Examples

**Table output (default):**

```bash
python -m llama_log_analyzer logs/localai.log
```

**JSON output:**

```bash
python -m llama_log_analyzer logs/localai.log --format json
```

**CSV output to file:**

```bash
python -m llama_log_analyzer logs/localai.log --format csv --output results.csv
```

**Filter by task:**

```bash
python -m llama_log_analyzer logs/localai.log --task 2687
```

## Supported Log Lines

The analyzer recognizes the following llama.cpp log patterns:

| Pattern | Description |
|---------|-------------|
| `prompt eval time = ... ms / ... tokens (...)` | Prompt evaluation timing |
| `eval time = ... ms / ... tokens (...)` | Token generation timing |
| `total time = ... ms / ... tokens` | Total task duration |
| `graphs reused = ...` | KV-cache graph reuse count |
| `draft acceptance = ... (...)` | Speculative decoding metrics |
| `stop processing: n_tokens = ..., truncated = ...` | Task completion info |

Unknown or malformed lines are silently ignored.

## Output

### Table Format

```
Task      Prompt Tokens  Prompt t/s   Output Tokens  Output t/s   Total Time     Draft Acceptance  Mean Draft Length  Truncated
2687      14055          498.70       423            22.89        46662.93 ms    0.28296           13.73              0

--- Summary ---
Complete tasks: 1
Weighted avg prompt t/s: 498.70
Median prompt t/s: 498.70
Weighted avg output t/s: 22.89
Median output t/s: 22.89
Total prompt tokens: 14055
Total output tokens: 423
Total measured time: 46662.93 ms
Fastest task: 2687 (22.89 t/s)
Slowest task: 2687 (22.89 t/s)
```

### JSON Format

```json
{
  "tasks": [
    {
      "task_id": 2687,
      "prompt_tokens": 14055,
      "prompt_tokens_per_second": 498.70,
      ...
    }
  ],
  "summary": {
    "complete_tasks": 1,
    ...
  }
}
```

## Project Structure

```
llama-log-analyzer/
├── src/llama_log_analyzer/
│   ├── __init__.py          # Package marker
│   ├── __main__.py          # Entry point for `python -m`
│   ├── models.py            # Data models (TaskMetrics)
│   ├── parser.py            # Log line parser
│   ├── statistics.py        # Statistics computation
│   └── cli.py               # CLI interface & output formatting
├── tests/
│   └── test_llama_log_analyzer.py
├── pyproject.toml
└── README.md
```

## Running Tests

```bash
pip install -e ".[test]"
pytest
```

## License

MIT
