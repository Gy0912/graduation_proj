from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from detection.bandit_wrapper import run_bandit
from detection.sql_injection_detector import SQLInjectionDetector, extract_python_code
from evaluation.metrics import MetricBundle, aggregate_metrics


def load_model_and_tokenizer(
    base_model: str,
    load_in_4bit: bool,
    adapter_path: str | None = None,
) -> tuple[Any, Any, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("评测阶段要求 CUDA GPU，禁止 CPU 回退。")

    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    quant = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=quant,
        device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    device = torch.device("cuda")
    try:
        model = model.to(device)
    except Exception:
        # 4bit + device_map="auto" 可能由 accelerate 管理设备分配，这里保持兼容。
        pass
    model.eval()
    return model, tok, device


def run_eval_on_prompts(
    samples: list[dict[str, Any]],
    base_model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    load_in_4bit: bool,
    adapter_path: str | None = None,
    per_device_eval_batch_size: int = 4,
    dataloader_num_workers: int = 2,
    dataloader_pin_memory: bool = True,
    debug_timing: bool = True,
    enable_fallback_detector: bool = True,
) -> MetricBundle:
    det = SQLInjectionDetector() if enable_fallback_detector else None
    model, tok, device = load_model_and_tokenizer(
        base_model=base_model,
        load_in_4bit=load_in_4bit,
        adapter_path=adapter_path,
    )

    source_samples = samples
    prompts = [str(s["prompt"]) for s in source_samples]
    enc = tok(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    dataset = TensorDataset(
        enc["input_ids"],
        enc["attention_mask"],
        torch.arange(len(prompts), dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(per_device_eval_batch_size)),
        shuffle=False,
        num_workers=max(0, int(dataloader_num_workers)),
        pin_memory=bool(dataloader_pin_memory),
    )

    print(f"[eval] batch_size={max(1, int(per_device_eval_batch_size))}")
    print(f"[eval] device={device}")
    print(
        f"[eval] dataloader workers={max(0, int(dataloader_num_workers))}, "
        f"pin_memory={bool(dataloader_pin_memory)}"
    )

    evaluated_samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, sample_ids) in enumerate(tqdm(loader, desc="eval")):
            t0 = time.perf_counter()
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            if batch_idx == 0:
                print(f"[eval] input tensor device={input_ids.device}")

            do_sample = temperature > 0
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

            for row in range(out_ids.size(0)):
                prompt_len = int(attention_mask[row].sum().item())
                gen_tokens = out_ids[row, prompt_len:]
                text = tok.decode(gen_tokens, skip_special_tokens=True)
                code = extract_python_code(text)
                sample_id = int(sample_ids[row].item())
                src = source_samples[sample_id]

                if code is None:
                    evaluated_samples.append(
                        {
                            "id": sample_id,
                            "code": "",
                            "is_vulnerable": False,
                            "attack_type": str(src.get("attack_type", "unknown")),
                            "difficulty": str(src.get("difficulty", "unknown")),
                            "task_type": str(src.get("task_type", "unknown")),
                            "bandit_issues": [],
                            "invalid_extraction": True,
                        }
                    )
                    continue

                bandit_result = _run_bandit_on_temp(code, sample_id)
                bandit_vuln = bool(bandit_result.get("is_vulnerable", False))
                bandit_issues = bandit_result.get("issues", [])
                if not isinstance(bandit_issues, list):
                    bandit_issues = []

                fallback_res = det.analyze(code) if det is not None else None
                fallback_vuln = bool(fallback_res.is_vulnerable) if fallback_res else False
                final_vuln = bandit_vuln or fallback_vuln

                evaluated_samples.append(
                    {
                        "id": sample_id,
                        "prompt": prompts[sample_id],
                        "raw_output": text,
                        "code": code,
                        "is_vulnerable": final_vuln,
                        "attack_type": str(src.get("attack_type", "unknown")),
                        "difficulty": str(src.get("difficulty", "unknown")),
                        "task_type": str(src.get("task_type", "unknown")),
                        "bandit_issues": bandit_issues,
                        "fallback_violations": (
                            fallback_res.violations if fallback_res is not None else []
                        ),
                        "invalid_extraction": False,
                    }
                )

            if debug_timing:
                dt = time.perf_counter() - t0
                print(f"[eval] batch={batch_idx} time={dt:.3f}s")

    evaluated_samples.sort(key=lambda x: int(x["id"]))
    return aggregate_metrics(evaluated_samples)


def save_results(path: Path, bundle: MetricBundle, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "summary": {
            "n_samples": bundle.n_samples,
            "overall_sql_injection_rate": bundle.overall_sql_injection_rate,
            "sql_injection_rate": bundle.sql_injection_rate,
            "safe_code_generation_rate": bundle.safe_code_generation_rate,
            "by_attack_type": bundle.by_attack_type,
            "by_difficulty": bundle.by_difficulty,
            "by_task_type": bundle.by_task_type,
        },
        "per_sample": bundle.per_sample,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _run_bandit_on_temp(code: str, sample_id: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sql_eval_") as tmpdir:
        file_path = Path(tmpdir) / f"temp_eval_{sample_id}.py"
        file_path.write_text(code, encoding="utf-8")
        return run_bandit(file_path)
