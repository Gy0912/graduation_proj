# 项目结构说明

完整目录树、各文件职责与**入口脚本**说明见根目录 **`README.md` →「4. Project Structure（项目结构）」**。

本文仅保留数据流速查：

```text
dataset/generate_expanded_dataset.py
  → data/* 与 dataset/*.jsonl
  → training/* 与 outputs/models/*
  → evaluation/evaluate.py
  → outputs/*_results.json
  → scripts/compare_results.py → outputs/comparison_summary.json
```
