# 操作手册（中期版本）

## 1. 实验目标

- 针对 SQL 注入问题，验证大模型在：
  - 代码生成任务中的漏洞率变化
  - 代码修复任务中的修复成功率变化
  - 微调后基础能力保留情况

## 2. 环境准备

```powershell
powershell -ExecutionPolicy Bypass -File scripts/00_prepare_env.ps1
```

## 3. 数据集构建

```powershell
.\.venv\Scripts\python.exe scripts/01_build_dataset.py --config configs/experiment.yaml
```

输出：
- `data/train.jsonl`
- `data/val.jsonl`
- `data/test_generate.jsonl`
- `data/test_repair.jsonl`
- `data/test_retain.jsonl`

## 4. 基线模型评测（微调前）

```powershell
.\.venv\Scripts\python.exe scripts/02_eval_baseline.py --config configs/experiment.yaml --model_path Qwen/Qwen2.5-Coder-0.5B-Instruct --tag baseline
```

## 5. 微调训练

```powershell
.\.venv\Scripts\python.exe scripts/03_train_lora.py --config configs/experiment.yaml
```

模型输出目录：
- `outputs/lora_sqlfix_qwen05b`

## 6. 微调后评测

```powershell
.\.venv\Scripts\python.exe scripts/04_eval_finetuned.py --config configs/experiment.yaml --model_path outputs/lora_sqlfix_qwen05b --tag finetuned
```

## 7. 图表与对比导出

```powershell
.\.venv\Scripts\python.exe scripts/05_plot_results.py --config configs/experiment.yaml --before_tag baseline --after_tag finetuned
```

说明：如果 PowerShell 禁止执行脚本，直接调用 `.\.venv\Scripts\python.exe` 是最稳定方案，无需 `Activate.ps1`。

输出图：
- `outputs/figures/vulnerable_rate_compare.png`
- `outputs/figures/repair_success_compare.png`
- `outputs/figures/functional_compare.png`

## 8. 指标解读

- `vulnerable_rate` 下降：说明 SQL 注入风险降低
- `repair_success_rate` 上升：说明修复能力增强
- `functional_avg` 变化小：说明基础能力保留较好

## 9. 迁移到约 8B 模型

1. 在 `configs/experiment.yaml` 中替换模型名称
2. 若本机显存不足，建议迁移到服务器复现相同步骤
3. 保持数据划分、评测脚本、图表脚本不变，以保证可比性
