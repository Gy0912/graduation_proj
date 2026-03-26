"""从评测文件加载评测样本（支持 JSONL 或 JSON 数组）。"""
from __future__ import annotations

import json
from pathlib import Path


def load_eval_prompts(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在: {path}")
    suf = path.suffix.lower()
    if suf == ".jsonl":
        samples: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                samples.append(_normalize_sample(row))
        return samples
    if suf == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} 应为 JSON 数组")
        samples: list[dict] = []
        for row in data:
            if isinstance(row, str):
                samples.append(_normalize_sample({"prompt": row}))
            elif isinstance(row, dict):
                samples.append(_normalize_sample(row))
            else:
                raise ValueError(f"无法解析评测样本: {row!r}")
        return samples
    raise ValueError(f"不支持的评测文件类型: {path}")


def _normalize_sample(row: dict) -> dict:
    prompt = row.get("prompt")
    if not prompt:
        instruction = row.get("instruction", "")
        user_input = row.get("input", "")
        if instruction or user_input:
            prompt = f"{instruction}\n{user_input}".strip()
    if not prompt:
        raise ValueError(f"无法解析 prompt/instruction/input 字段: {row!r}")

    return {
        "prompt": prompt,
        "instruction": row.get("instruction"),
        "input": row.get("input"),
        "output": row.get("output"),
        "attack_type": row.get("attack_type", "unknown"),
        "difficulty": row.get("difficulty", "unknown"),
        "task_type": row.get("task_type", "unknown"),
    }
