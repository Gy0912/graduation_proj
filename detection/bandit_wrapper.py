from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def run_bandit(file_path: str | Path) -> dict[str, Any]:
    """
    对单个 Python 文件执行 Bandit，并返回统一结构：
    {
      "is_vulnerable": bool,
      "issues": list[dict]
    }
    """
    file_path = str(file_path)
    cmd = ["bandit", file_path, "-f", "json"]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = _safe_parse_bandit_json(proc.stdout)
    issues = payload.get("results", []) if isinstance(payload, dict) else []
    if not isinstance(issues, list):
        issues = []
    return {
        "is_vulnerable": len(issues) > 0,
        "issues": issues,
    }


def _safe_parse_bandit_json(stdout: str) -> dict[str, Any]:
    text = (stdout or "").strip()
    if not text:
        return {"results": []}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"results": []}
    if not isinstance(parsed, dict):
        return {"results": []}
    return parsed
