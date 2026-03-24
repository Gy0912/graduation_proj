from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from detection.sql_injection_detector import SQLInjectionDetector, extract_python_code
from evaluation.metrics import MetricBundle, aggregate_metrics


def _functional_heuristic(code: str) -> float:
    c = code or ""
    score = 0.0
    if "def " in c:
        score += 0.35
    if "execute" in c.lower() or "cursor" in c.lower():
        score += 0.35
    if "return" in c:
        score += 0.3
    return min(score, 1.0)


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
    prompts: list[str],
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
) -> MetricBundle:
    det = SQLInjectionDetector()
    model, tok, device = load_model_and_tokenizer(
        base_model=base_model,
        load_in_4bit=load_in_4bit,
        adapter_path=adapter_path,
    )

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

    samples: list[dict[str, Any]] = []
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
                res = det.analyze(code)
                sample_id = int(sample_ids[row].item())
                samples.append(
                    {
                        "id": sample_id,
                        "prompt": prompts[sample_id],
                        "raw_output": text,
                        "extracted_code": code,
                        "is_vulnerable": res.is_vulnerable,
                        "violations": res.violations,
                        "functional_score": _functional_heuristic(code),
                    }
                )

            if debug_timing:
                dt = time.perf_counter() - t0
                print(f"[eval] batch={batch_idx} time={dt:.3f}s")

    samples.sort(key=lambda x: int(x["id"]))
    return aggregate_metrics(samples)


def save_results(path: Path, bundle: MetricBundle, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "summary": {
            "n_samples": bundle.n_samples,
            "sql_injection_rate": bundle.sql_injection_rate,
            "safe_code_generation_rate": bundle.safe_code_generation_rate,
            "functional_correctness_avg": bundle.functional_correctness_avg,
        },
        "per_sample": bundle.per_sample,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
