#!/usr/bin/env bash
# Exact model override: --model MODEL_ID; preview: --dry-run; synthetic test: --probe.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../.venv/bin/python" "$SCRIPT_DIR/_eval_so101.py" gemini "$@"
