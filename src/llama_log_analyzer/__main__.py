"""Allow running as python -m llama_log_analyzer."""

from .cli import main

import sys

sys.exit(main())
