"""兼容入口：转发到 evaluation/evaluate.py --model baseline"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    config = "configs/default.yaml"
    if len(sys.argv) >= 3 and sys.argv[1] == "--config":
        config = sys.argv[2]
    cmd = [sys.executable, str(ROOT / "evaluation" / "evaluate.py"), "--config", config, "--model", "baseline"]
    raise SystemExit(subprocess.run(cmd, cwd=str(ROOT), check=False).returncode)


if __name__ == "__main__":
    main()
