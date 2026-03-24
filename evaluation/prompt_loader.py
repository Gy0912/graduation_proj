"""从评测文件加载 prompt 列表（支持 JSONL 或 JSON 数组）。"""
from __future__ import annotations

import json
from pathlib import Path


def load_eval_prompts(path: Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在: {path}")
    suf = path.suffix.lower()
    if suf == ".jsonl":
        prompts: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompts.append(row["prompt"])
        return prompts
    if suf == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} 应为 JSON 数组")
        prompts = []
        for row in data:
            if isinstance(row, str):
                prompts.append(row)
            elif isinstance(row, dict) and "prompt" in row:
                prompts.append(row["prompt"])
            else:
                raise ValueError(f"无法解析 prompt 字段: {row!r}")
        return prompts
    raise ValueError(f"不支持的评测文件类型: {path}")
