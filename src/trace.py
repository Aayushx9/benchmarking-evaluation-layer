"""JSON trace logging for later evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def save_trace(trace: dict, trace_dir: str = "traces") -> str:
    """Write trace dict as JSON, return the file path."""
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = Path(trace_dir) / f"trace_{ts}.json"
    # Make sure everything is JSON-serializable.
    safe = json.loads(json.dumps(trace, default=str))
    path.write_text(json.dumps(safe, indent=2), encoding="utf-8")
    return str(path)
