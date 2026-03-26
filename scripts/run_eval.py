"""兼容入口：建议改用 evaluation/evaluate.py --model <method>。"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import yaml
from evaluation.evaluator import run_eval_on_prompts, save_results
from evaluation.prompt_loader import load_eval_prompts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--adapter", required=True, help="LoRA 适配器目录")
    p.add_argument("--output", required=True)
    p.add_argument("--tag", default="custom_adapter_eval", help="记录到 meta 中")
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument(
        "--disable-fallback-detector",
        action="store_true",
        help="仅使用 Bandit，不启用轻量规则回退检测",
    )
    args = p.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    files = cfg["files"]
    gen = cfg["generation"]
    out_path = args.output

    eval_samples = load_eval_prompts(ROOT / files["eval_prompts"])
    bundle = run_eval_on_prompts(
        samples=eval_samples,
        base_model=cfg["model"]["base_model"],
        max_new_tokens=gen["max_new_tokens"],
        temperature=gen["temperature"],
        top_p=gen["top_p"],
        load_in_4bit=bool(args.load_in_4bit),
        adapter_path=args.adapter,
        enable_fallback_detector=not bool(args.disable_fallback_detector),
    )
    meta = {
        "mode": args.tag,
        "base_model": cfg["model"]["base_model"],
        "adapter_path": args.adapter,
        "config": args.config,
        "enable_fallback_detector": not bool(args.disable_fallback_detector),
    }
    save_results(ROOT / out_path, bundle, meta)
    print(f"[OK] wrote {ROOT / out_path}")


if __name__ == "__main__":
    main()
