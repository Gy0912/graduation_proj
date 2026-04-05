# SQL 注入相关 LLM 代码安全评测

## 项目概述

面向 Python **SQL 注入**的代码大模型研究：安全代码生成与漏洞修复；管线包含数据准备、LoRA/QLoRA/SFT/DPO 训练，以及对生成代码的**规则层 + Bandit** 静态检测，以及可选的**动态污点追踪**；并与数据集中的 `expected_vulnerable` 对齐计算 Precision、Recall、F1、FPR、FNR 与混淆矩阵。

## 目录结构

```
.
├── configs/               # 训练与评测 YAML 配置
├── data/                  # 训练/评测/DPO 等 JSON 数据与 schema
├── dataset/               # 数据集生成脚本与合成逻辑
├── detection/             # 规则、Bandit、污点追踪与统一 detect_vulnerability
├── evaluation/            # 评测入口、指标聚合、prompt 加载
├── logs/                  # 实验与错误日志（可选）
├── models/                # 本地模型或说明占位
├── outputs/               # 评测 JSON 与训练产出适配器路径
├── reports/               # 报告与操作说明（若有）
├── scripts/               # 数据集构建、评测兼容入口、结果对比等
├── visualization/         # 汇总对比指标绘图（matplotlib）
├── training/              # LoRA/QLoRA/SFT/DPO 训练脚本与工具
├── pyproject.toml         # 项目元数据与工具配置
├── requirements.txt       # Python 依赖
└── README.md              # 本说明
```

## 如何运行

### 安装依赖

```powershell
Set-Location e:\graduation_proj
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 生成运行用配置

```powershell
.\.venv\Scripts\python.exe scripts/prepare_default_run.py
.\.venv\Scripts\python.exe scripts/prepare_bandit_only_run.py
```

### 生成数据集

```powershell
.\.venv\Scripts\python.exe dataset/generate_expanded_dataset.py --num_samples 2500 --eval_ratio 0.12 --seed 42
.\.venv\Scripts\python.exe scripts/build_dataset.py --config configs/default_run.yaml
```

### 训练（6 个适配器；baseline 不参与训练，仅评测）

```powershell
.\.venv\Scripts\python.exe training/train_lora_only.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe training/train_lora_sft.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe training/dpo_train.py --config configs/dpo.yaml
.\.venv\Scripts\python.exe training/train_qlora_only.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe training/train_qlora_sft.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe training/train_qlora_dpo.py --config configs/dpo.yaml
```

### 评测（各模型输出独立 JSON，格式不变）

```powershell
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model baseline
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model lora_only
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model lora_sft
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model lora_dpo
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model qlora_only
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model qlora_sft
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model qlora_dpo --allow-missing-adapter
```

### 汇总对比（聚合各模型评测 JSON，写入 `comparison_summary` 与 `compare_results`）

```powershell
.\.venv\Scripts\python.exe scripts/compare_results.py --config configs/default_run.yaml
```

### 可视化（由汇总 JSON 生成柱状图）

```powershell
.\.venv\Scripts\python.exe visualization/plot_compare_metrics.py --input outputs/compare_results.json --output-dir outputs/plots
```

## 评测方法说明

1. **规则层**：基于模式的 SQL 注入相关启发式检测，对模型输出的 Python 源码进行分析。
2. **Bandit**：对抽取出的源码运行 Bandit，默认合并模式下以 B608（SQL 注入相关）作为 Bandit 侧主信号；`or_bandit_any` 模式下任意 Bandit issue 可参与合并。
3. **合并**：由 `eval.merge_mode` 控制——`or`（B608 或规则或污点命中）、`or_bandit_any`（任意 Bandit issue 或规则或污点）、`weighted`（多信号加权阈值）。最终 `is_vulnerable` 与 `expected_vulnerable` 对比，得到 `classification_vs_expected` 及 `per_detector_vs_expected` 中各子层指标。

## Dynamic Analysis (Taint Tracking)

**作用**：在沙箱内对片段执行 `exec`，用带标记的 `TaintedStr` 与（仅允许 `import sqlite3` 的）包装层，观察污点是否流入 `Connection.execute` / `Cursor.execute` 的 SQL 字符串，用于补充纯静态规则与 Bandit。

**原理简述**：对用户代码做 f-string 的 AST 重写，将 `JoinedStr` 转为 `TaintedStr` 拼接链以保留污点；`taint_input()` 与 `input()` 注入为污点源；`sqlite3` 由沙箱模块替换，`execute` 在污点 SQL 上记一条 sink 并不再调用真实数据库（避免缺表噪声）。

**如何运行**：

- 评测：`python evaluation/evaluate.py --model baseline --enable-taint`（或配置 `eval.enable_taint: true`）。
- 单测：`python -m unittest tests.test_taint_tracker -v`。
- 直接 API：`detection.taint_tracker.run_taint_analysis(code)` 或 `detect_vulnerability(..., enable_taint=True)`。

**局限**：仅覆盖标准库 `sqlite3` 与沙箱内允许的语法；污点经未包装 API（如其它 DB 驱动）或 C 扩展会丢失；`str` 字面量与 `TaintedStr` 的 `str + TaintedStr` 在部分解释器上无法打补丁，故依赖 AST 重写 f-string 与显式 `TaintedStr` 运算；任意 `exec` 仍有理论滥用面，仅用于受控评测片段。
