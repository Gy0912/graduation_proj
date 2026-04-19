# SQL 注入相关 LLM 代码安全评测

## 项目概述

面向 Python **SQL 注入**的代码大模型研究：安全代码生成与漏洞修复；管线包含数据准备、LoRA/QLoRA/SFT/DPO 训练，以及对生成代码的**规则层 + Bandit** 静态检测，以及可选的**动态污点追踪**；并与数据集中的 `expected_vulnerable` 对齐计算 Precision、Recall、F1、FPR、FNR 与混淆矩阵。

## 项目结构

```
project_root/
├── configs/                         # 配置文件（default/default_run/dpo）
├── data/                            # 训练/评测/DPO 数据（JSON/JSONL）
│   ├── combined/                    # 主流程 train/eval（研究 schema）
│   ├── generation/                  # 按任务拆分数据
│   ├── fix/                         # 按任务拆分数据
│   ├── dpo_pairs.json               # DPO 偏好对（默认配置引用）
│   ├── train_expanded.json          # 兼容旧流程的扩展训练集
│   └── eval_expanded.json           # 兼容旧流程的扩展评测集
├── dataset/                         # 数据生成脚本（主入口见 generate_expanded_dataset.py）
├── detection/                       # 漏洞检测（规则/Bandit/污点）
├── evaluation/                      # 统一评测入口与指标聚合
├── training/                        # 统一训练入口脚本
├── scripts/                         # 配置准备、数据构建、汇总与管线脚本
├── visualization/                   # 结果可视化脚本
├── outputs/                         # 评测结果 JSON 与训练产物
├── logs/                            # 运行日志与变更日志
├── models/                          # 模型路径说明（实际适配器默认在 outputs/models）
├── reports/                         # 历史分析与操作说明
├── requirements.txt                 # Python 依赖
└── README.md                        # 项目总说明
```

## 统一入口（已精简）

- 训练入口（唯一）：
  - `training/train_lora_only.py`
  - `training/train_lora_sft.py`
  - `training/dpo_train.py`
  - `training/train_qlora_only.py`
  - `training/train_qlora_sft.py`
  - `training/train_qlora_dpo.py`
- 评测入口（唯一）：`evaluation/evaluate.py`
- 推理/输出入口（统一到评测生成与落盘）：`evaluation/evaluate.py`

说明：已移除重复的 DPO/Baseline 兼容转发脚本，保留单一主入口，训练与评测逻辑不变。

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

### 校验整合后的评测集（统一入口）

```powershell
Set-Location e:\graduation_proj
python -c "import json,pathlib; p=pathlib.Path(r'data/combined/eval.json'); rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]; print('total=',len(rows)); print('max_id=',max(r['id'] for r in rows))"
```

说明：
- `dataset/generate_expanded_dataset.py`：生成主流程研究数据（`data/combined/*.json`、`data/eval_expanded.json`、DPO 偏好对等）。
- `scripts/build_dataset.py`：按配置生成 `dataset/*.jsonl`（兼容/补充流程）。
- 新增高难评测样本已合并到 `data/combined/eval.json`，保持评测加载路径不变。

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

## 执行流程（数据流）

1. **数据准备**  
   `dataset/generate_expanded_dataset.py` 生成训练/评测样本与 DPO 偏好对；可选执行 `scripts/build_dataset.py` 生成 `dataset/*.jsonl`。

2. **训练流程**  
   使用 `training/*` 入口训练 LoRA/QLoRA/SFT/DPO 适配器，产物写入 `outputs/models/*`。

3. **推理与输出生成**  
   统一通过 `evaluation/evaluate.py --model <name>` 加载基座模型与对应适配器，生成代码并写入 `outputs/*_results.json`。

4. **指标汇总与可视化**  
   `scripts/compare_results.py` 聚合多模型结果；`visualization/plot_compare_metrics.py` 读取汇总结果绘图。

## 手动验收（行为保持不变）

按下面顺序执行，可验证入口清理后功能一致：

```powershell
Set-Location e:\graduation_proj
.\.venv\Scripts\python.exe training/train_lora_sft.py --config configs/default_run.yaml
.\.venv\Scripts\python.exe training/dpo_train.py --config configs/dpo.yaml
.\.venv\Scripts\python.exe evaluation/evaluate.py --config configs/default_run.yaml --model lora_dpo
.\.venv\Scripts\python.exe scripts/compare_results.py --config configs/default_run.yaml
```

期望结果：

- 训练脚本正常结束，并在 `outputs/models/` 下生成对应适配器目录。
- 评测脚本输出 `[OK] wrote ...`，并写入 `outputs/lora_dpo_results.json`。
- 汇总脚本输出 `[OK] wrote ...`，并更新 `outputs/comparison_summary.json` 与 `outputs/compare_results.json`。

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
