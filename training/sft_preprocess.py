"""
SFT 数据预处理：在训练循环外完成分词（dataset.map batched=True），产出 TRL 期望的
input_ids + completion_mask，供 DataCollatorForLanguageModeling 计算 completion-only loss。
"""
from __future__ import annotations

from typing import Any

from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def row_to_prompt_completion(
    instruction: str,
    input_text: str,
    output: str,
    template: str = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
) -> tuple[str, str]:
    """将 instruction/input/output 拼成 prompt + completion（completion 不含 prompt）。"""
    prompt = template.format(instruction=instruction.strip(), input=(input_text or "").strip())
    completion = output.strip()
    if completion and not completion.endswith("\n"):
        completion += "\n"
    return prompt, completion


def tokenize_prompt_completion_batched(
    examples: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, list]:
    """
    batched=True 时调用。对每个样本：
    - 对 prompt 与 prompt+completion 分别编码（与 TRL 默认行为一致：add_special_tokens=False）
    - 构造 completion_mask：prompt 段为 0，completion 段为 1
    - 超长则从右侧截断，并同步截断 completion_mask
    """
    prompts = examples["prompt"]
    completions = examples["completion"]
    all_input_ids: list[list[int]] = []
    all_masks: list[list[int]] = []

    for prompt, completion in zip(prompts, completions):
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]

        if len(full_ids) < len(p_ids):
            # 极端 tokenizer 行为：保底用全长当 completion
            p_ids = full_ids[: max(1, len(full_ids) // 2)]

        if full_ids[: len(p_ids)] != p_ids:
            # 对齐：以最长公共前缀为准（避免空格/特殊符号导致不一致）
            common = 0
            for i, (a, b) in enumerate(zip(p_ids, full_ids)):
                if a == b:
                    common = i + 1
                else:
                    break
            p_ids = full_ids[:common]

        comp_mask = [0] * len(p_ids) + [1] * (len(full_ids) - len(p_ids))

        # 右截断
        if len(full_ids) > max_length:
            overflow = len(full_ids) - max_length
            full_ids = full_ids[-max_length:]
            comp_mask = comp_mask[-max_length:]
            # 若截断吃掉全部 prompt，至少保留最后一个 token 的监督（避免全 -100）
            if sum(comp_mask) == 0:
                comp_mask[-1] = 1

        all_input_ids.append(full_ids)
        all_masks.append(comp_mask)

    return {"input_ids": all_input_ids, "completion_mask": all_masks}


def build_sft_dataset_from_records(
    records: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dataset:
    """records 含 instruction, input, output。"""
    prompts: list[str] = []
    completions: list[str] = []
    for r in records:
        p, c = row_to_prompt_completion(
            str(r.get("instruction", "")),
            str(r.get("input", "")),
            str(r.get("output", "")),
        )
        prompts.append(p)
        completions.append(c)

    ds: Dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
    remove_cols = [c for c in ds.column_names if c not in ("prompt", "completion")]
    if remove_cols:
        ds = ds.remove_columns(remove_cols)

    ds = ds.map(
        lambda batch: tokenize_prompt_completion_batched(batch, tokenizer, max_length),
        batched=True,
        remove_columns=["prompt", "completion"],
        desc="Tokenizing (batched)",
    )
    return ds


def train_val_split(records: list[dict], val_ratio: float, seed: int) -> tuple[list, list]:
    import random

    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n_val = max(1, int(len(records) * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    train = [records[i] for i in train_idx]
    val = [records[i] for i in val_idx]
    if not train:
        train, val = val[:-1], val[-1:]
    return train, val
