"""生成 dataset/ 下 JSONL：SFT、DPO、评测 prompts。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from dataset.synthetic_sql import build_synthetic_splits


def save_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args()

    with open(Path(ROOT) / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = cfg["dataset"]
    files = cfg["files"]
    data = build_synthetic_splits(
        train_n=ds["train_sft_n"],
        val_n=ds["val_sft_n"],
        eval_prompts_n=ds["eval_prompts_n"],
        seed=ds["seed"],
    )

    base = Path(ROOT)
    save_jsonl(base / files["train_sft"], data["train_sft"])
    save_jsonl(base / files["val_sft"], data["val_sft"])
    save_jsonl(base / files["train_dpo"], data["train_dpo"])
    save_jsonl(base / files["eval_prompts"], data["eval_prompts"])

    print("[OK] wrote:")
    for k in files:
        print(" ", base / files[k])


if __name__ == "__main__":
    main()
