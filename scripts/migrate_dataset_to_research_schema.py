"""
将旧版扁平 JSON（train_expanded / eval_expanded 等）迁移为研究 schema，并写入：

  data/combined/train.json, data/combined/eval.json
  data/generation/, data/fix/ 下 train/eval 拆分

不删除源文件。若缺字段则从 attack_type 等推断。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.research_schema import to_research_record, write_research_splits


def _normalize_train_row(row: dict) -> dict:
    out = dict(row)
    if "vulnerability_type" not in out and "attack_type" in out:
        out["vulnerability_type"] = out["attack_type"]
    if "task_type" not in out:
        out["task_type"] = "generation"
    if "expected_vulnerable" not in out:
        out["expected_vulnerable"] = False
    if "difficulty" not in out:
        out["difficulty"] = "medium"
    if "input" in out and "input_code" not in out:
        out["input_code"] = out.get("input")
    return out


def _normalize_eval_row(row: dict) -> dict:
    out = _normalize_train_row(row)
    if "prompt" in out and "instruction" not in out:
        pass
    return out


def load_train_like(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    return [_normalize_train_row(r) for r in data if isinstance(r, dict)]


def load_eval_like(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array")
    out: list[dict] = []
    for r in data:
        if not isinstance(r, dict):
            continue
        er = _normalize_eval_row(r)
        if "instruction" not in er and er.get("prompt"):
            p = str(er["prompt"])
            if "### Instruction:" in p and "### Input:" in p:
                try:
                    rest = p.split("### Instruction:", 1)[1]
                    instr, inp = rest.split("### Input:", 1)
                    inp, _ = inp.split("### Response:", 1)
                    er["instruction"] = instr.strip()
                    er["input"] = inp.strip()
                except ValueError:
                    er.setdefault("instruction", "")
                    er.setdefault("input", "")
            else:
                er.setdefault("instruction", p)
                er.setdefault("input", "")
        out.append(er)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Migrate legacy dataset JSON to research schema paths")
    p.add_argument("--train", type=Path, default=ROOT / "data" / "train_expanded.json")
    p.add_argument("--eval", type=Path, default=ROOT / "data" / "eval_expanded.json")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将写入的样本数，不写文件",
    )
    args = p.parse_args()

    if not args.train.exists():
        raise FileNotFoundError(args.train)
    if not args.eval.exists():
        raise FileNotFoundError(args.eval)

    train_rows = load_train_like(args.train)
    eval_rows = load_eval_like(args.eval)

    for i, r in enumerate(train_rows):
        if not r.get("instruction") and not r.get("output"):
            print(f"[warn] train row {i} may be incomplete: keys={list(r.keys())}", file=sys.stderr)

    if args.dry_run:
        print(f"[dry-run] train={len(train_rows)} eval={len(eval_rows)}")
        return

    write_research_splits(train_rows, eval_rows, ROOT)
    print(f"[OK] wrote data/combined, data/generation, data/fix from {args.train} + {args.eval}")


if __name__ == "__main__":
    main()
