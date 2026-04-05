from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from detection.detector import detect_vulnerability
from detection.sql_injection_detector import extract_python_code
from evaluation.metrics import MetricBundle, aggregate_metrics

# 对照实验：恒输出参数化安全片段（不加载大模型）
ALWAYS_SAFE_SYNTHETIC_OUTPUT = '''import pymysql


def fetch_rows(cur, value: str):
    sql = "SELECT * FROM users WHERE username = %s"
    cur.execute(sql, (value,))
    return cur.fetchall()
'''


def _per_sample_from_detection(
    det: dict[str, Any],
    *,
    src: dict[str, Any],
    sample_id: int,
    prompt: str,
    raw_output: str | None,
    code: str,
    invalid_extraction: bool,
    merge_mode: str,
    always_safe_stub: bool = False,
) -> dict[str, Any]:
    bandit_block = det.get("bandit", {})
    issues = bandit_block.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    bandit_confidence_levels = [
        str(i.get("confidence", "UNKNOWN")).upper()
        for i in issues
        if isinstance(i, dict)
    ]
    b608 = bool(bandit_block.get("b608_hit"))
    bandit_any = bool(bandit_block.get("has_issue"))
    rb = det.get("rule_based", {})
    tt = det.get("taint", {})
    taint_on = not bool(tt.get("skipped", True))
    sid = src.get("id")
    return {
        "id": sid if sid is not None else sample_id,
        "sample_index": sample_id,
        "prompt": prompt,
        "raw_output": raw_output,
        "code": code,
        "is_vulnerable": bool(det.get("is_vulnerable")),
        "attack_type": str(src.get("attack_type", "unknown")),
        "vulnerability_type": str(
            src.get("vulnerability_type", src.get("attack_type", "unknown"))
        ),
        "difficulty": str(src.get("difficulty", "unknown")),
        "task_type": str(src.get("task_type", "unknown")),
        "expected_vulnerable": bool(src.get("expected_vulnerable", False)),
        "merge_mode": merge_mode,
        "detection_sources": list(det.get("detection_sources", [])),
        "bandit_issues": issues,
        "bandit_detected": bandit_any,
        "bandit_b608": b608,
        "bandit_issue_count": len(issues),
        "bandit_confidence_levels": bandit_confidence_levels,
        "bandit_has_B608": b608,
        "rule_based_detected": bool(rb.get("is_vulnerable")),
        "rule_based_violations": rb.get("violations", []),
        "taint_detected": bool(taint_on and tt.get("is_vulnerable")),
        "taint_flows_detected": int(tt.get("taint_flows_detected", 0)) if taint_on else 0,
        "fallback_violations": rb.get("violations", []),
        "invalid_extraction": invalid_extraction,
        "always_safe_stub": always_safe_stub,
    }


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


def run_eval_always_safe(
    samples: list[dict[str, Any]],
    *,
    merge_mode: str = "or",
    enable_rule_based: bool = True,
    enable_taint: bool = False,
) -> MetricBundle:
    """
    基线健全性检查：不调用 LLM，每条样本恒返回同一参数化安全代码。
    预期：相对 expected_vulnerable 的 recall_vulnerable≈0（几乎抓不到「应为漏洞」类），
    用于验证标签与分类指标是否合理。
    """
    prompts = [str(s["prompt"]) for s in samples]
    evaluated_samples: list[dict[str, Any]] = []
    code = ALWAYS_SAFE_SYNTHETIC_OUTPUT.strip()
    text = code

    for sample_id, src in enumerate(samples):
        dres = detect_vulnerability(
            code,
            sample_id=sample_id,
            merge_mode=merge_mode,  # type: ignore[arg-type]
            enable_rule_based=enable_rule_based,
            enable_taint=enable_taint,
        )
        evaluated_samples.append(
            _per_sample_from_detection(
                dres,
                src=src,
                sample_id=sample_id,
                prompt=prompts[sample_id],
                raw_output=text,
                code=code,
                invalid_extraction=False,
                merge_mode=merge_mode,
                always_safe_stub=True,
            )
        )

    return aggregate_metrics(evaluated_samples)


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
    merge_mode: str = "or",
    enable_rule_based: bool = True,
    enable_taint: bool = False,
) -> MetricBundle:
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
                            "id": src.get("id", sample_id),
                            "sample_index": sample_id,
                            "prompt": prompts[sample_id],
                            "raw_output": text,
                            "code": "",
                            "is_vulnerable": False,
                            "attack_type": str(src.get("attack_type", "unknown")),
                            "vulnerability_type": str(
                                src.get("vulnerability_type", src.get("attack_type", "unknown"))
                            ),
                            "difficulty": str(src.get("difficulty", "unknown")),
                            "task_type": str(src.get("task_type", "unknown")),
                            "expected_vulnerable": bool(src.get("expected_vulnerable", False)),
                            "merge_mode": merge_mode,
                            "detection_sources": [],
                            "bandit_issues": [],
                            "bandit_detected": False,
                            "bandit_b608": False,
                            "bandit_issue_count": 0,
                            "bandit_confidence_levels": [],
                            "bandit_has_B608": False,
                            "rule_based_detected": False,
                            "rule_based_violations": [],
                            "taint_detected": False,
                            "taint_flows_detected": 0,
                            "fallback_violations": [],
                            "invalid_extraction": True,
                        }
                    )
                    continue

                dres = detect_vulnerability(
                    code,
                    sample_id=sample_id,
                    merge_mode=merge_mode,  # type: ignore[arg-type]
                    enable_rule_based=enable_rule_based,
                    enable_taint=enable_taint,
                )
                evaluated_samples.append(
                    _per_sample_from_detection(
                        dres,
                        src=src,
                        sample_id=sample_id,
                        prompt=prompts[sample_id],
                        raw_output=text,
                        code=code,
                        invalid_extraction=False,
                        merge_mode=merge_mode,
                    )
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
            "bandit_total_detections": bundle.bandit_total_detections,
            "bandit_detection_rate": bundle.bandit_detection_rate,
            "bandit_b608_rate": bundle.bandit_b608_rate,
            "bandit_low_confidence_count": bundle.bandit_low_confidence_count,
            "bandit_medium_confidence_count": bundle.bandit_medium_confidence_count,
            "bandit_high_confidence_count": bundle.bandit_high_confidence_count,
            "bandit_confidence_distribution": bundle.bandit_confidence_distribution,
            "bandit_risk_score": bundle.bandit_risk_score,
            "b608_detection_rate": bundle.b608_detection_rate,
            "by_attack_type": bundle.by_attack_type,
            "by_difficulty": bundle.by_difficulty,
            "by_task_type": bundle.by_task_type,
            "classification_vs_expected": bundle.classification_vs_expected,
            "detection_layer_stats": bundle.detection_layer_stats,
            "detection_source_breakdown": bundle.detection_source_breakdown,
            "per_detector_vs_expected": bundle.per_detector_vs_expected,
            "by_attack_type_metrics": bundle.by_attack_type_metrics,
        },
        "per_sample": bundle.per_sample,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


