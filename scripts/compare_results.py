"""对比 7 组实验结果并计算相对 baseline 的 SQL 注入率下降百分比。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["summary"]


def pct_drop(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (before - after) / before * 100.0


METHODS: tuple[str, ...] = (
    "baseline",
    "lora_only",
    "lora_sft",
    "lora_dpo",
    "qlora_only",
    "qlora_sft",
    "qlora_dpo",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
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

    missing = [m for m, p in method_to_output.items() if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少实验结果文件（请先完整重跑评测，禁止复用旧汇总）: {missing}"
        )

    base = load_summary(ROOT / method_to_output["baseline"])
    summary = {
        "eval_dataset": cfg["files"]["eval_prompts"],
        "baseline_sql_injection_rate": base["sql_injection_rate"],
        "baseline_safe_code_generation_rate": base["safe_code_generation_rate"],
    }
    for method in METHODS:
        s = load_summary(ROOT / method_to_output[method])
        inj = float(s["sql_injection_rate"])
        safe = float(s["safe_code_generation_rate"])
        summary[f"{method}_sql_injection_rate"] = inj
        summary[f"{method}_safe_code_generation_rate"] = safe
        summary[f"{method}_sql_injection_reduction_vs_baseline_pct"] = pct_drop(
            float(base["sql_injection_rate"]), inj
        )

    out_file = ROOT / outs.get("comparison_summary", "outputs/comparison_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {out_file}")


if __name__ == "__main__":
    main()
