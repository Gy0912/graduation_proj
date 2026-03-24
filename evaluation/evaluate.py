from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluator import run_eval_on_prompts, save_results
from evaluation.prompt_loader import load_eval_prompts


SUPPORTED_MODELS = (
    "baseline",
    "lora_only",
    "lora_sft",
    "lora_dpo",
    "qlora_only",
    "qlora_sft",
    "qlora_dpo",
)


def resolve_eval_plan(cfg: dict, model_name: str) -> tuple[str | None, bool, str]:
    paths = cfg["paths"]
    outs = cfg["outputs"]
    if model_name == "baseline":
        return None, False, outs["baseline_results"]
    if model_name == "lora_only":
        return paths["lora_only_dir"], False, outs["lora_only_results"]
    if model_name == "lora_sft":
        return paths["lora_sft_dir"], False, outs["lora_sft_results"]
    if model_name == "lora_dpo":
        return paths["dpo_lora_dir"], False, outs["lora_dpo_results"]
    if model_name == "qlora_only":
        return paths["qlora_only_dir"], True, outs["qlora_only_results"]
    if model_name == "qlora_sft":
        return paths["qlora_sft_dir"], True, outs["qlora_sft_results"]
    if model_name == "qlora_dpo":
        return paths["qlora_dpo_dir"], True, outs["qlora_dpo_results"]
    raise ValueError(f"unsupported model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="统一评测入口（7种实验设置）")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="评测 batch size（覆盖配置 eval.per_device_eval_batch_size）",
    )
    parser.add_argument(
        "--allow-missing-adapter",
        action="store_true",
        help="若适配器不存在则给出警告并退出（码 0）",
    )
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    adapter_path, load_in_4bit, output_path = resolve_eval_plan(cfg, args.model)
    if adapter_path is not None and not (ROOT / adapter_path).exists():
        msg = (
            f"[WARN] adapter not found for {args.model}: {ROOT / adapter_path}. "
            "Skip evaluation."
        )
        if args.allow_missing_adapter:
            print(msg)
            return
        raise FileNotFoundError(msg)

    files = cfg["files"]
    gen = cfg["generation"]
    ev_cfg = cfg.get("eval", {})
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else ev_cfg.get("per_device_eval_batch_size", 4)
    )
    num_workers = int(ev_cfg.get("dataloader_num_workers", 2))
    pin_memory = bool(ev_cfg.get("dataloader_pin_memory", True))
    prompts = load_eval_prompts(ROOT / files["eval_prompts"])
    bundle = run_eval_on_prompts(
        prompts=prompts,
        base_model=cfg["model"]["base_model"],
        max_new_tokens=gen["max_new_tokens"],
        temperature=gen["temperature"],
        top_p=gen["top_p"],
        load_in_4bit=load_in_4bit,
        adapter_path=str(ROOT / adapter_path) if adapter_path else None,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=pin_memory,
        debug_timing=True,
    )
    meta = {
        "mode": args.model,
        "base_model": cfg["model"]["base_model"],
        "adapter_path": str(ROOT / adapter_path) if adapter_path else None,
        "load_in_4bit": load_in_4bit,
        "per_device_eval_batch_size": batch_size,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": pin_memory,
        "config": args.config,
        "eval_dataset": files["eval_prompts"],
    }
    save_results(ROOT / output_path, bundle, meta)
    print(f"[OK] wrote {ROOT / output_path}")


if __name__ == "__main__":
    main()
