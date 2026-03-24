"""一键顺序执行 7 组实验并生成统一对比。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, check=False, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def run_nonblocking(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, check=False, cwd=str(ROOT))
    if r.returncode != 0:
        print(f"[WARN] command failed but pipeline continues: {' '.join(cmd)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--skip-lora-dpo", action="store_true")
    p.add_argument("--skip-qlora-dpo", action="store_true")
    args = p.parse_args()

    py = sys.executable
    cfg = args.config

    run([py, "dataset/generate_expanded_dataset.py"])
    run([py, "scripts/build_dataset.py", "--config", cfg])
    run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "baseline"])

    run([py, "training/train_lora_only.py", "--config", cfg])
    run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "lora_only"])
    run([py, "training/train_lora_sft.py", "--config", cfg])
    run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "lora_sft"])

    if not args.skip_lora_dpo:
        run([py, "training/dpo_train.py", "--config", "configs/dpo.yaml"])
        run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "lora_dpo"])

    run([py, "training/train_qlora_only.py", "--config", cfg])
    run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "qlora_only"])

    run([py, "training/train_qlora_sft.py", "--config", cfg])
    run([py, "evaluation/evaluate.py", "--config", cfg, "--model", "qlora_sft"])

    if not args.skip_qlora_dpo:
        run_nonblocking([py, "training/train_qlora_dpo.py", "--config", "configs/dpo.yaml"])
        run_nonblocking(
            [
                py,
                "evaluation/evaluate.py",
                "--config",
                cfg,
                "--model",
                "qlora_dpo",
                "--allow-missing-adapter",
            ]
        )

    run([py, "scripts/compare_results.py", "--config", cfg])


if __name__ == "__main__":
    main()
