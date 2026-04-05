"""从各模型评测 JSON 生成柱状图：SQL 注入率；FPR / FNR（matplotlib 默认配色）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METHODS: tuple[str, ...] = (
    "baseline",
    "lora_only",
    "lora_sft",
    "lora_dpo",
    "qlora_only",
    "qlora_sft",
    "qlora_dpo",
)


def _load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["summary"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="输出目录（将写入 sql_injection_rate.png, fpr_fnr.png）",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="缺失的结果文件跳过该柱，不全则仍出图",
    )
    args = p.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    outs = cfg["outputs"]
    method_to_output = {
        "baseline": outs["baseline_results"],
        "lora_only": outs["lora_only_results"],
        "lora_sft": outs["lora_sft_results"],
        "lora_dpo": outs["lora_dpo_results"],
        "qlora_only": outs["qlora_only_results"],
        "qlora_sft": outs["qlora_sft_results"],
        "qlora_dpo": outs["qlora_dpo_results"],
    }

    labels: list[str] = []
    inj_rates: list[float] = []
    fprs: list[float] = []
    fnrs: list[float] = []

    for m in METHODS:
        rel = method_to_output[m]
        path = ROOT / rel
        if not path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"missing results: {path}")
        summ = _load_summary(path)
        cls_ = summ.get("classification_vs_expected", {})
        labels.append(m)
        inj_rates.append(float(summ.get("sql_injection_rate", 0.0)))
        fprs.append(float(cls_.get("false_positive_rate", 0.0)))
        fnrs.append(float(cls_.get("false_negative_rate", 0.0)))

    if not labels:
        raise SystemExit("no result files found")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(labels, inj_rates)
    ax1.set_ylabel("rate")
    ax1.set_title("SQL injection rate (merged detector)")
    ax1.tick_params(axis="x", rotation=45)
    fig1.tight_layout()
    fig1.savefig(out_dir / "sql_injection_rate.png")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    x = range(len(labels))
    w = 0.38
    ax2.bar([i - w / 2 for i in x], fprs, width=w, label="FPR")
    ax2.bar([i + w / 2 for i in x], fnrs, width=w, label="FNR")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=45, ha="right")
    ax2.set_ylabel("rate")
    ax2.set_title("False positive / false negative rate vs expected_vulnerable")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "fpr_fnr.png")
    plt.close(fig2)

    print(f"[OK] wrote {out_dir / 'sql_injection_rate.png'}")
    print(f"[OK] wrote {out_dir / 'fpr_fnr.png'}")


if __name__ == "__main__":
    main()
