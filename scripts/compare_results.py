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


def metrics_block_from_eval_json(path: Path) -> dict:
    """从单模型评测 JSON 的 summary 读取指标（不重算检测）。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    s = data["summary"]
    cve = s.get("classification_vs_expected") or {}
    cm = cve.get("confusion_matrix") or {}
    return {
        "sql_injection_rate": float(s["sql_injection_rate"]),
        "safe_code_generation_rate": float(s["safe_code_generation_rate"]),
        "fpr": float(cve.get("false_positive_rate", 0.0)),
        "fnr": float(cve.get("false_negative_rate", 0.0)),
        "tp": int(cm.get("TP", 0)),
        "fp": int(cm.get("FP", 0)),
        "tn": int(cm.get("TN", 0)),
        "fn": int(cm.get("FN", 0)),
    }


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
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="跳过缺失的结果文件（baseline 必须存在）",
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

    missing = [m for m, p in method_to_output.items() if not (ROOT / p).exists()]
    if "baseline" in missing:
        raise FileNotFoundError("缺少 baseline 结果，无法汇总")
    if missing and not args.allow_missing:
        raise FileNotFoundError(
            f"缺少实验结果文件（或加 --allow-missing）: {missing}"
        )

    base = load_summary(ROOT / method_to_output["baseline"])
    summary = {
        "eval_dataset": cfg["files"]["eval_prompts"],
        "baseline_sql_injection_rate": base["sql_injection_rate"],
        "baseline_safe_code_generation_rate": base["safe_code_generation_rate"],
    }
    per_model: dict[str, dict] = {}
    for method in METHODS:
        rel = method_to_output[method]
        if not (ROOT / rel).exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(ROOT / rel)
        s = load_summary(ROOT / rel)
        inj = float(s["sql_injection_rate"])
        safe = float(s["safe_code_generation_rate"])
        summary[f"{method}_sql_injection_rate"] = inj
        summary[f"{method}_safe_code_generation_rate"] = safe
        summary[f"{method}_sql_injection_reduction_vs_baseline_pct"] = pct_drop(
            float(base["sql_injection_rate"]), inj
        )
        per_model[method] = metrics_block_from_eval_json(ROOT / rel)
    summary["per_model"] = per_model

    out_file = ROOT / outs.get("comparison_summary", "outputs/comparison_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] wrote {out_file}")

    compare_alt = outs.get("compare_results")
    if compare_alt:
        alt_path = ROOT / compare_alt
        alt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(alt_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote {alt_path}")


if __name__ == "__main__":
    main()
