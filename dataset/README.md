# dataset/

## 当前主数据流程

- 主入口脚本：`dataset/generate_expanded_dataset.py`
- 主要输出位置：
  - `data/combined/train.json`
  - `data/combined/eval.json`
  - `data/combined/eval.json`（已整合基础评测样本与新增高难评测样本）
  - `data/eval_expanded.json`
  - `data/train_expanded.json`
  - `data/dpo_pairs.json`

## 兼容数据流程

- `scripts/build_dataset.py`：按配置生成 `dataset/*.jsonl`（如 `sft_train.jsonl`、`sft_val.jsonl`、`dpo_train.jsonl`）。
- `dataset/generate_sql_security_dataset.py`：旧版小规模生成器，输出 `dataset/sql_security_dataset.json`，当前不作为主流程默认入口。

## 与训练/评测衔接

- `training/train_lora_sft.py`：读取配置中的 `files.train_sft_json`（默认 `data/combined/train.json`）。
- `training/dpo_train.py`：读取配置中的 `files.dpo_pairs`（默认 `data/dpo_pairs.json`）。
- `evaluation/evaluate.py`：读取配置中的 `files.eval_prompts`（默认 `data/combined/eval.json`）。
