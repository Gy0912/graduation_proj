"""轻量 SQL 注入 fallback 检测与 Python 代码抽取。"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionResult:
    is_vulnerable: bool
    violations: list[str] = field(default_factory=list)
    matched_patterns: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class SQLInjectionDetector:
    """可选的轻量 fallback：只做简单模式检查。"""

    def __init__(self) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = [
            (
                "fstring_sql",
                re.compile(
                    r'(?:execute|executemany)\s*\(\s*f["\']',
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
            (
                "concat_plus_sql",
                re.compile(
                    r'["\']\s*(?:SELECT|INSERT|UPDATE|DELETE)\b[^"\']*["\']\s*\+',
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
            (
                "format_sql",
                re.compile(
                    r'(?:execute|executemany)\s*\(\s*["\'][^"\']*\{[^}]+\}[^"\']*["\']\s*\.format',
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
            (
                "percent_format_sql",
                re.compile(
                    r'(?:execute|executemany)\s*\(\s*["\'][^"\']*%[sd][^"\']*["\']\s*%',
                    re.IGNORECASE | re.MULTILINE,
                ),
            ),
        ]

    def analyze(self, code: str) -> DetectionResult:
        text = code or ""
        violations: list[str] = []
        matched: list[str] = []

        for name, pat in self._patterns:
            if pat.search(text):
                violations.append(name)
                matched.append(name)

        # 启发式：execute 参数中包含拼接 / f-string 痕迹
        if self._unsafe_execute_heuristic(text):
            if "unsafe_execute_heuristic" not in violations:
                violations.append("unsafe_execute_heuristic")
                matched.append("unsafe_execute_heuristic")

        is_vulnerable = len(violations) > 0

        return DetectionResult(
            is_vulnerable=is_vulnerable,
            violations=violations,
            matched_patterns=matched,
            details={},
        )

    def _unsafe_execute_heuristic(self, text: str) -> bool:
        # execute("... " + var) 类
        if re.search(r'execute\s*\(\s*[^)]*\+', text, re.IGNORECASE):
            return True
        if re.search(r'execute\s*\(\s*f["\']', text, re.IGNORECASE):
            return True
        return False


def detect_sql_injection(code: str) -> DetectionResult:
    return SQLInjectionDetector().analyze(code)


def extract_python_code(model_output: str) -> str | None:
    """
    从原始模型输出中提取“可解析的 Python 代码”。
    - 去除 instruction/response 标签、JSON 块和非代码描述文本
    - 返回可通过 ast.parse 的代码；失败返回 None
    """
    text = (model_output or "").strip()
    if not text:
        return None

    # 优先提取 markdown python 代码块。
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [c.strip() for c in fenced if c.strip()]
    if not candidates:
        candidates.append(text)

    for cand in candidates:
        clean = _strip_non_code_text(cand)
        valid = _best_valid_python(clean)
        if valid is not None:
            return valid
    return None


def _strip_non_code_text(text: str) -> str:
    # 移除 markdown json 代码块，避免把结构化说明当作 Python 输入。
    text = re.sub(r"```json\s*\n.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)

    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        low = line.strip().lower()
        if not low:
            lines.append("")
            continue
        if low.startswith("### instruction") or low.startswith("### response"):
            continue
        if low.startswith("instruction:") or low.startswith("response:"):
            continue
        lines.append(line)

    merged = "\n".join(lines).strip()

    # 若整体是 JSON，尝试抽取其中的 code/output 字段。
    if merged.startswith("{") and merged.endswith("}"):
        try:
            obj = json.loads(merged)
            if isinstance(obj, dict):
                for k in ("code", "python", "output", "response"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
        except json.JSONDecodeError:
            pass
    return merged


def _best_valid_python(text: str) -> str | None:
    stripped = (text or "").strip()
    if not stripped:
        return None

    if _is_valid_python(stripped):
        return stripped

    # 回退：仅保留“看起来像代码”的行，再次验证。
    code_lines: list[str] = []
    for ln in stripped.splitlines():
        s = ln.strip()
        if not s:
            code_lines.append("")
            continue
        if _looks_like_python_line(s):
            code_lines.append(ln)
    candidate = "\n".join(code_lines).strip()
    if candidate and _is_valid_python(candidate):
        return candidate
    return None


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def _looks_like_python_line(line: str) -> bool:
    keywords = (
        "def ",
        "class ",
        "import ",
        "from ",
        "if ",
        "elif ",
        "else:",
        "for ",
        "while ",
        "try:",
        "except",
        "finally:",
        "with ",
        "return ",
        "raise ",
        "sql",
        "cursor",
        "execute(",
        "=",
    )
    if line.startswith("#"):
        return True
    return any(k in line for k in keywords)
