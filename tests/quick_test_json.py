"""Quick test for JSON log parsing fix."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from llama_log_analyzer.parser import LogParser
import json
import tempfile

# Test 1: Create a sample JSON log file
print("Test 1: JSON format log file")
json_log_entries = [
    {
        "timestamp": "2026-07-24T14:09:47.381592175Z",
        "stream": "stderr",
        "text": "0.12.178.956 I slot print_timing: id  0 | task 0 | prompt eval time =    1301.85 ms /   265 tokens        (    4.91 ms per token, 53.91 tokens per second)"
    },
    {
        "timestamp": "2026-07-24T14:09:47.481592175Z",
        "stream": "stderr",
        "text": "0.12.178.956 I slot print_timing: id  0 | task 0 | eval time =     802.50 ms /    10 tokens       (   80.25 ms per token,    12.46 tokens per second)"
    },
    {
        "timestamp": "2026-07-24T14:09:47.581592175Z",
        "stream": "stderr",
        "text": "0.12.178.956 I slot print_timing: id  0 | task 0 | total time =    2104.35 ms /   275 tokens"
    },
    {
        "timestamp": "2026-07-24T14:09:48.381592175Z",
        "stream": "stderr",
        "text": "0.12.178.956 I slot print_timing: id  0 | task 1 | prompt eval time =    1500.00 ms /   300 tokens        (    5.00 ms per token, 60.00 tokens per second)"
    },
    {
        "timestamp": "2026-07-24T14:09:48.481592175Z",
        "stream": "stderr",
        "text": "0.12.178.956 I slot print_timing: id  0 | task 1 | eval time =     900.00 ms /    15 tokens       (   60.00 ms per token,    16.67 tokens per second)"
    },
    {
        "timestamp": "2026-07-24T14:09:48.581592175Z",
        "stream": "stderr",
        "text": "0.12.178.956 I slot print_timing: id  0 | task 1 | total time =    2400.00 ms /   315 tokens"
    }
]

# Write to temp file
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    for entry in json_log_entries:
        f.write(json.dumps(entry) + "\n")
    temp_json_path = f.name

# Test parsing JSON log file
parser = LogParser()
tasks = parser.parse_file(temp_json_path)
print(f"  Found {len(tasks)} tasks")
for task in tasks:
    print(f"  Task {task.task_id}: prompt_tokens={task.prompt_tokens}, eval_tokens={task.eval_tokens}, total_time_ms={task.total_time_ms}")
    assert task.task_id in (0, 1), f"Unexpected task_id: {task.task_id}"
print("  Test 1 PASSED!")

# Test 2: Plain text log file (backward compatibility)
print("\nTest 2: Plain text format log file")
plain_log_lines = [
    "0.12.178.956 I slot print_timing: id  0 | task 10 | prompt eval time =    1301.85 ms /   265 tokens        (    4.91 ms per token, 53.91 tokens per second)",
    "0.12.178.956 I slot print_timing: id  0 | task 10 | eval time =     802.50 ms /    10 tokens       (   80.25 ms per token,    12.46 tokens per second)",
    "0.12.178.956 I slot print_timing: id  0 | task 10 | total time =    2104.35 ms /   275 tokens",
]

with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
    for line in plain_log_lines:
        f.write(line + "\n")
    temp_plain_path = f.name

parser2 = LogParser()
tasks2 = parser2.parse_file(temp_plain_path)
print(f"  Found {len(tasks2)} tasks")
for task in tasks2:
    print(f"  Task {task.task_id}: prompt_tokens={task.prompt_tokens}, eval_tokens={task.eval_tokens}, total_time_ms={task.total_time_ms}")
    assert task.task_id == 10, f"Expected task_id 10, got {task.task_id}"
print("  Test 2 PASSED!")

# Cleanup
os.unlink(temp_json_path)
os.unlink(temp_plain_path)

print("\n=== All tests passed! ===")
