"""
研究用数据集样本 schema 转换与稳定 id。

规范字段：id, task_type, instruction, input_code, expected_vulnerable,
vulnerability_type, difficulty；训练 split 另含 output。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def stable_sample_id(row: dict[str, Any]) -> str:
    """由 instruction + input + task_type + vulnerability_type 生成稳定短 id。"""
    instr = str(row.get("instruction", "")).strip()
    inp = str(row.get("input", row.get("input_code", "")) or "").strip()
    task = str(row.get("task_type", "")).strip()
    vuln_t = str(row.get("attack_type", row.get("vulnerability_type", ""))).strip()
    payload = f"{task}\n{vuln_t}\n{instr}\n{inp}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"sqlsec-{h}"


def to_research_record(
    row: dict[str, Any],
    *,
    include_output: bool = True,
) -> dict[str, Any]:
    raw_input = row.get("input", row.get("input_code"))
    if raw_input is None:
        inp_code = None
    else:
        s = str(raw_input).strip()
        inp_code = s if s else None

    attack = str(row.get("attack_type", row.get("vulnerability_type", "unknown")))

    rec: dict[str, Any] = {
        "id": row.get("id") or stable_sample_id(
            {
                **row,
                "attack_type": attack,
                "input": raw_input or "",
            }
        ),
        "task_type": str(row.get("task_type", "generation")),
        "instruction": str(row.get("instruction", "")),
        "input_code": inp_code,
        "expected_vulnerable": bool(row.get("expected_vulnerable", False)),
        "vulnerability_type": attack,
        "difficulty": str(row.get("difficulty", "medium")),
    }
    if include_output and "output" in row:
        rec["output"] = str(row.get("output", ""))
    return rec


def split_by_task(rows: Iterable[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    gen: list[dict] = []
    fix: list[dict] = []
    for r in rows:
        if str(r.get("task_type")) == "fix":
            fix.append(r)
        else:
            gen.append(r)
    return gen, fix


def write_research_splits(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    root: Path,
) -> None:
    """写入 data/combined、data/generation、data/fix。"""
    combined = root / "data" / "combined"
    gen_dir = root / "data" / "generation"
    fix_dir = root / "data" / "fix"
    for d in (combined, gen_dir, fix_dir):
        d.mkdir(parents=True, exist_ok=True)

    train_r = [to_research_record(r, include_output=True) for r in train_rows]
    eval_r = [to_research_record(r, include_output=False) for r in eval_rows]

    with open(combined / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_r, f, ensure_ascii=False, indent=2)
    with open(combined / "eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_r, f, ensure_ascii=False, indent=2)

    tg, tf = split_by_task(train_r)
    eg, ef = split_by_task(eval_r)
    with open(gen_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(tg, f, ensure_ascii=False, indent=2)
    with open(gen_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(eg, f, ensure_ascii=False, indent=2)
    with open(fix_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(tf, f, ensure_ascii=False, indent=2)
    with open(fix_dir / "eval.json", "w", encoding="utf-8") as f:
        json.dump(ef, f, ensure_ascii=False, indent=2)
