"""从汇总对比 JSON 绘制各模型 SQL 注入率、FPR、FNR、安全代码生成率柱状图。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHOD_ORDER: tuple[str, ...] = (
    "baseline",
    "lora_only",
    "lora_sft",
    "lora_dpo",
    "qlora_only",
    "qlora_sft",
    "qlora_dpo",
)


def _legacy_flat_metrics(data: dict) -> dict[str, dict]:
    """兼容无 per_model 字段的旧版 comparison_summary。"""
    out: dict[str, dict] = {}
    for m in METHOD_ORDER:
        inj_k = f"{m}_sql_injection_rate"
        safe_k = f"{m}_safe_code_generation_rate"
        if inj_k not in data or safe_k not in data:
            continue
        out[m] = {
            "sql_injection_rate": float(data[inj_k]),
            "safe_code_generation_rate": float(data[safe_k]),
            "fpr": 0.0,
            "fnr": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }
    return out


def load_per_model(data: dict) -> dict[str, dict]:
    pm = data.get("per_model")
    if isinstance(pm, dict) and pm:
        return {k: v for k, v in pm.items() if isinstance(v, dict)}
    return _legacy_flat_metrics(data)


def _bar_chart(labels: list[str], values: list[float], ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("model")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="汇总对比 JSON（含 per_model）")
    p.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="图像输出目录（默认 outputs/plots）",
    )
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_model = load_per_model(data)
    if not per_model:
        raise ValueError("未找到任何模型指标（per_model 或扁平字段）")

    out_dir = Path(args.output_dir)
    labels: list[str] = []
    inj: list[float] = []
    fpr: list[float] = []
    fnr: list[float] = []
    safe: list[float] = []
    for key in METHOD_ORDER:
        if key not in per_model:
            continue
        row = per_model[key]
        labels.append(key)
        inj.append(float(row["sql_injection_rate"]))
        fpr.append(float(row["fpr"]))
        fnr.append(float(row["fnr"]))
        safe.append(float(row["safe_code_generation_rate"]))

    _bar_chart(labels, inj, "sql_injection_rate", out_dir / "injection_rate.png")
    _bar_chart(labels, fpr, "fpr", out_dir / "fpr.png")
    _bar_chart(labels, fnr, "fnr", out_dir / "fnr.png")
    _bar_chart(labels, safe, "safe_code_generation_rate", out_dir / "safe_rate.png")
    print(f"[OK] wrote {out_dir / 'injection_rate.png'} etc.")


if __name__ == "__main__":
    main()
