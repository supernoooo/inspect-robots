from __future__ import annotations

import sys
from pathlib import Path

# The service is an independent installable package, but offline tests run from
# the source checkout without installing torch or downloading weights.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
