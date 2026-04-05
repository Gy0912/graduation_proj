"""从评测文件加载评测样本（自动识别 JSON 数组或 JSONL）。"""
from __future__ import annotations

import json
from pathlib import Path


def _instruction_input_prompt(instruction: str, user_input: str) -> str:
    """与训练集 `template_prompt` 一致，保证评测分布与 SFT 对齐。"""
    return (
        "### Instruction:\n"
        + (instruction or "").strip()
        + "\n\n### Input:\n"
        + (user_input or "").strip()
        + "\n\n### Response:\n"
    )


def load_eval_prompts(path: Path) -> list[dict]:
    """
    支持两种格式（由首个非空白字符判断）：
    - 以 ``[`` 开头：标准 JSON 数组
    - 否则：JSONL（每行一个 JSON 对象）
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"评测文件不存在: {path}")

    # utf-8-sig：兼容带 BOM 的 UTF-8 文件
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []

    stripped = text.lstrip()
    first_char = stripped[0]

    if first_char == "[":
        print("Detected JSON format")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"评测 JSON 解析失败 ({path}): {e.msg} (line {e.lineno}, column {e.colno})"
            ) from e
        if not isinstance(data, list):
            raise ValueError(f"{path} 应为 JSON 数组（顶层必须是列表）")
        samples: list[dict] = []
        for idx, row in enumerate(data):
            if isinstance(row, str):
                samples.append(_normalize_sample({"prompt": row}))
            elif isinstance(row, dict):
                samples.append(_normalize_sample(row))
            else:
                raise ValueError(
                    f"{path} 数组第 {idx} 个元素类型无效，期望 str 或 dict，得到 {type(row).__name__}"
                )
        return samples

    print("Detected JSONL format")
    samples: list[dict] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        try:
            row = json.loads(line_stripped)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"评测 JSONL 第 {line_no} 行解析失败 ({path}): {e.msg} (column {e.colno})"
            ) from e
        if not isinstance(row, dict):
            raise ValueError(
                f"评测 JSONL 第 {line_no} 行应为 JSON 对象，得到 {type(row).__name__}"
            )
        try:
            samples.append(_normalize_sample(row))
        except ValueError as e:
            raise ValueError(f"评测 JSONL 第 {line_no} 行字段无效: {e}") from e
    return samples


def _normalize_sample(row: dict) -> dict:
    prompt = row.get("prompt")
    if not prompt:
        instruction = row.get("instruction", "")
        user_input = row.get("input")
        if user_input is None:
            user_input = row.get("input_code", "")
        user_input = user_input or ""
        if instruction or str(user_input or "").strip():
            prompt = _instruction_input_prompt(instruction, str(user_input or ""))
    if not prompt:
        raise ValueError(f"无法解析 prompt/instruction/input 字段: {row!r}")

    vuln_type = row.get("vulnerability_type") or row.get("attack_type", "unknown")

    return {
        "id": row.get("id"),
        "prompt": prompt,
        "instruction": row.get("instruction"),
        "input": row.get("input") if row.get("input") is not None else row.get("input_code"),
        "input_code": row.get("input_code"),
        "output": row.get("output"),
        "attack_type": str(vuln_type),
        "vulnerability_type": str(vuln_type),
        "difficulty": row.get("difficulty", "unknown"),
        "task_type": row.get("task_type", "unknown"),
        "expected_vulnerable": bool(row.get("expected_vulnerable", False)),
    }
