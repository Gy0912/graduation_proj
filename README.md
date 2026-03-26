# 面向 SQL 注入安全的代码大模型微调（LoRA/QLoRA 7 组实验）

## 1. 实验目标

本项目面向「代码生成中的 SQL 注入风险」进行可复现实验，统一对比 7 组设置，并固定同一评测集进行公平比较。

- 保留已有稳定流程：`LoRA + SFT`、`LoRA + DPO`
- 旧版「前缀微调」实验已移除（代码与配置中无残留）
- 新增 LoRA-only 与 QLoRA 系列实验
- 在单卡 8GB GPU 上优先保证稳定（FP16 + CUDA 强制）

## 2. 七种实验设置

1. `baseline`
   - 原始基座模型
   - 不挂载 LoRA，不训练
2. `lora_only`
   - 挂载 LoRA 适配器
   - 不训练
3. `lora_sft`
   - LoRA + SFT（已有稳定实现）
4. `lora_dpo`
   - LoRA + DPO（已有稳定实现）
5. `qlora_only`
   - 4bit 量化加载 + 挂载 LoRA
   - 不训练
6. `qlora_sft`
   - QLoRA + SFT
7. `qlora_dpo`
   - QLoRA + DPO
   - 若训练不稳定可跳过，脚本会给 warning，不阻塞全流程

## 3. QLoRA 约束

QLoRA 相关脚本统一使用 bitsandbytes 4bit：

- `load_in_4bit=True`
- `bnb_4bit_compute_dtype=torch.float16`
- `bnb_4bit_use_double_quant=True`

并保持：

- 强制使用 GPU（不允许静默 CPU 回退）
- 避免 BF16 路径，统一以 FP16 稳定为主

## 4. Project Structure（项目结构）

> **入口脚本**：标注为「入口」的文件是日常实验应运行的命令入口；**核心逻辑**为训练/检测/推理实现；**工具**为配置合并、dtype 修正等辅助模块。

```text
.
├── README.md                              # 本说明：实验、命令、结构、清理步骤
├── PROJECT_STRUCTURE.md                   # 简短结构索引（与本文互补）
├── requirements.txt                       # Python 依赖列表
├── pyproject.toml                         # 可选打包元数据（当前以脚本式仓库为主）
│
├── configs/
│   ├── default.yaml                       # 主配置：模型、路径、训练/评测、输出 JSON 路径
│   └── dpo.yaml                           # DPO 覆盖项（与 default.yaml 合并）
│
├── data/                                  # **最终数据**（论文主用；勿删）
│   ├── train_expanded.json                # 扩展 SFT（instruction/input/output + 元数据字段）
│   ├── eval_expanded.json                 # 统一评测（prompt + attack_type/difficulty/task_type）
│   └── dpo_pairs.jsonl                    # DPO 偏好对：prompt / chosen / rejected
│
├── dataset/                               # 数据构建与中间 jsonl
│   ├── generate_expanded_dataset.py       # **入口**：生成 data/*.json（--num_samples）
│   ├── generate_sql_security_dataset.py   # 旧版小数据集生成器
│   ├── sql_security_dataset.json          # 旧版小训练集（无扩展数据时的回退）
│   ├── synthetic_sql.py                   # 合成 SQL 片段等工具
│   ├── eval_prompts.jsonl                 # 由 build_dataset 等生成的评测 jsonl（若使用）
│   ├── sft_train.jsonl / sft_val.jsonl    # SFT 分词用 jsonl（由流水线生成）
│   ├── dpo_train.jsonl                    # DPO 训练用 jsonl（由流水线生成）
│   ├── examples/                          # 示例 json/jsonl
│   └── README.md
│
├── detection/                             # **核心**：Bandit 封装 + 轻量 fallback 检测
│   ├── bandit_wrapper.py                  # Bandit JSON 调用与结果解析
│   ├── sql_injection_detector.py          # 代码抽取与可选轻量规则检查
│   └── __init__.py
│
├── evaluation/                            # **核心**：生成 + 指标
│   ├── evaluate.py                        # **入口**：7 组实验统一评测（--model / --batch_size）
│   ├── evaluator.py                       # 批量推理、预分词、指标聚合
│   ├── metrics.py                         # sql_injection_rate 等指标定义
│   ├── prompt_loader.py                   # 从 JSON/JSONL 加载评测 prompts
│   └── __init__.py
│
├── training/                              # **核心** + 工具
│   ├── train_lora_sft.py                  # **入口**：LoRA + SFT（QLoRA 配置下为 4bit）
│   ├── dpo_train.py                       # **入口**：LoRA + DPO 实现（TRL DPOTrainer）
│   ├── train_dpo.py                       # 兼容转发到 dpo_train.py
│   ├── train_lora_dpo.py                  # 兼容转发到 dpo_train.py
│   ├── train_lora_only.py                 # **入口**：仅挂载 LoRA，不训练
│   ├── train_qlora_only.py                # **入口**：4bit + LoRA，不训练
│   ├── train_qlora_sft.py                 # **入口**：QLoRA + SFT
│   ├── train_qlora_dpo.py                 # **入口**：QLoRA + DPO（失败时告警可跳过）
│   ├── lora_utils.py                      # **工具**：LoRA target_modules 自动推断
│   ├── sft_preprocess.py                  # **工具**：SFT 分词与 train/val 划分
│   ├── config_utils.py                    # **工具**：YAML 深度合并
│   ├── dtype_utils.py                     # **工具**：bf16→fp16 修正与 dtype 诊断
│   ├── amp_grad_debug.py                  # **工具**：SFT 训练时 AMP/梯度诊断回调
│   ├── gpu_debug.py                       # **工具**：训练时 GPU 调试回调
│   ├── common.py                          # **工具**：TrainingArguments 兼容构造
│   └── __init__.py
│
├── scripts/                               # 流水线与辅助
│   ├── run_thesis_pipeline.py             # **入口**：数据→训练→评测→对比（可选跳过 DPO）
│   ├── compare_results.py                 # **入口**：汇总 7 组 outputs/*_results.json
│   ├── build_dataset.py                   # 生成 dataset/*.jsonl（供训练脚本使用）
│   ├── run_baseline.py                    # 兼容：转发 evaluate.py --model baseline
│   ├── run_eval.py                        # 兼容：自定义 adapter 路径评测
│   ├── 00_prepare_env.ps1               # Windows：创建 venv / 安装依赖
│   └── README.md
│
├── models/
│   └── README.md                          # 说明：真实适配器在 outputs/models/ 下
│
├── outputs/                               # 可复现产出（可定期清理中间 checkpoint）
│   ├── baseline_results.json              # 各实验评测 JSON（文件名见 configs）
│   ├── lora_*_results.json / qlora_*_results.json
│   ├── comparison_summary.json            # compare_results.py 汇总
│   └── models/                            # LoRA/QLoRA 适配器与各 run 的 checkpoint-*
│
└── reports/                               # 报告与分析稿（非运行必需）
    ├── ANALYSIS_0.5B_FAILURE.md
    └── operation_manual.md
```

### 4.1 已删除 / 不建议保留的内容（本次清理）

| 类型 | 说明 |
|------|------|
| `scripts/legacy/*` | 旧版流水线脚本，已由 `training/` + `evaluation/` + `scripts/run_thesis_pipeline.py` 替代 |
| `scripts/debug_print_lora_targets.py` | 一次性调试 LoRA 目标的脚本 |
| `cuda_test.py` | 临时 CUDA 自检脚本 |
| `__pycache__/` | Python 缓存，可随时删除，运行时会再生成 |

**未自动删除（请按需手动处理）**

- `outputs/models/*/checkpoint-*`：中间 checkpoint，若只需最终 `adapter_config.json` 等可删以省磁盘。
- 根目录下非英文命名的零散 `.txt`：若为误放笔记，请自行确认后删除。

### 4.2 Prefix / 前缀微调相关

- 仓库内 **已无** `PrefixTuningConfig`、`train_alt_method`、前缀专用配置键或 `finetuned_alt_results` 等引用。
- Tokenizer 词汇表中的 `fim_prefix` 等为 **StarCoder 系列内置符号**，与已删除的 PEFT 前缀微调无关。

## How to clean project (manual steps)

----------------------------------
Step 1: Remove cache files  
Command (PowerShell，项目根目录):

```powershell
Get-ChildItem -Path . -Recurse -Directory -Filter __pycache__ | Where-Object { $_.FullName -notmatch '\\.venv\\' } | Remove-Item -Recurse -Force
Remove-Item -Recurse -Force .ipynb_checkpoints -ErrorAction SilentlyContinue
```

Command（Linux / macOS）:

```bash
find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
rm -rf .ipynb_checkpoints
```

Explanation:  
删除 Python 缓存与 Jupyter 检查点，不影响源代码与数据。

----------------------------------
Step 2: Remove prefix-related files  
Command:  
无需额外操作；若本地仍有旧版 `finetuned_alt_results.json` 或 `prefix_*` 目录，请手动删除以免混淆。

Explanation:  
当前主线已不再使用前缀微调；残留文件仅为历史产物。

----------------------------------
Step 3: Verify structure  
Command（若已安装 Git Bash 或 tree）:

```bash
tree .
```

PowerShell 简易替代:

```powershell
Get-ChildItem -Recurse -Name | Select-Object -First 80
```

Explanation:  
确认目录与 `configs/default.yaml`、`README` 中结构一致。

## 5. 数据集设计（Dataset Design）

### 5.1 样本格式

每条训练/评测样本包含以下字段（由 `dataset/generate_expanded_dataset.py` 生成）：

```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "attack_type": "...",
  "difficulty": "easy | medium | hard",
  "task_type": "generation | fix"
}
```

评测文件 `data/eval_expanded.json` 中每条另含 `prompt`（由 instruction + input 拼成，与训练模板一致）。

### 5.2 attack_type（攻击模式）

| 取值 | 含义 |
|------|------|
| `string_concat` | 字符串拼接构造 SQL |
| `fstring` | f-string 插值拼入查询 |
| `format_string` | `str.format` / `%` 等格式化拼入 SQL |
| `fake_sanitization` | 表面清洗（如 `replace`）后仍拼接 |
| `orm_misuse` | SQLAlchemy `text()` 等误用导致拼接注入 |
| `parameterized_query` | 安全范式（占位符 / 绑定参数）；亦可用于「修复错误参数用法」类题目 |

### 5.3 difficulty（难度）

- **easy**：直观注入（如明显拼接、`OR 1=1` 等）。
- **medium**：轻度混淆（注释、多段拼接、拆分）。
- **hard**：间接注入（小函数封装拼接、ORM 多层组合、多步构造 SQL）。

### 5.4 task_type（任务类型）

- **generation**：根据描述**新写**数据库访问代码；对抗性措辞下仍要求输出参数化安全实现。
- **fix**：给定**含漏洞**的代码片段，输出修复后的安全实现。

### 5.5 规模与划分

- 建议总样本量 **2000–3000**（`--num_samples` 为训练与评测之和）。
- 生成器在 36 个桶（6×3×2）间均衡分配；评测集按 `--eval_ratio`（默认 0.12）分层划分，与训练集 **prompt 去重、无交集**。

---

## 6. 评测方法（Evaluation Method）

当前评测以 **Bandit** 为主、**轻量规则**为可选回退。

- **主检测**：`detection/bandit_wrapper.py` 调用 `bandit <file> -f json`；`results` 非空则计为存在问题（含典型 SQL 相关规则如 B608）。
- **可选回退**：`detection/sql_injection_detector.py` 对 `execute`/f-string 等做补充匹配。
- **最终标签**：Bandit 或回退任一命中则 `is_vulnerable=true`。

**为何使用外部工具**：Bandit 为广泛使用的 Python 安全扫描器，输出格式稳定，便于在论文中说明评测不依赖项目私有启发式；回退用于减少极端漏报。

---

## 7. 评测流水线（Evaluation Pipeline）

1. 模型生成 `raw_output`。
2. `extract_python_code` 抽取可 `ast.parse` 的 Python 代码。
3. 抽取成功：写入临时 `.py`，对该文件运行 Bandit。
4. 抽取失败：`invalid_extraction=true`。
5. `prompt_loader` 透传 `attack_type`、`difficulty`、`task_type`；`metrics` 汇总总体与分组（含 `by_task_type`）。

---

## 8. 指标（Metrics）

- 总体：`overall_sql_injection_rate`（与 `sql_injection_rate` 同义）
- `safe_code_generation_rate = 1 - overall_sql_injection_rate`
- 分组：`by_attack_type`、`by_difficulty`、`by_task_type`

```json
{
  "overall_sql_injection_rate": 0.02,
  "by_attack_type": { "string_concat": 0.15 },
  "by_difficulty": { "easy": 0.01, "medium": 0.05, "hard": 0.20 },
  "by_task_type": { "generation": 0.03, "fix": 0.04 }
}
```

---

## 9. 输出文件说明

- `outputs/baseline_results.json` … `outputs/qlora_dpo_results.json`（七种实验各一份）
- `outputs/comparison_summary.json`（`scripts/compare_results.py`）

---

## 10. 环境准备

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install bandit
```

随后须先按 **第 12 节** 生成 `data/train_expanded.json` 等，再训练与评测。

---

## 11. 评测性能优化（GPU）

### 11.1 现象

逐条 batch=1、循环内重复分词时，易出现 GPU 等待 CPU、利用率锯齿。

### 11.2 已实现优化

- `evaluation/evaluate.py --batch_size` 覆盖 `per_device_eval_batch_size`
- 预分词 + `DataLoader`（`num_workers`、`pin_memory`）

### 11.3 调参建议

- 约 8GB 显存：自 `--batch_size 4` 起试；OOM 则降至 2。

---

## 12. 如何生成数据集

```powershell
python dataset/generate_expanded_dataset.py --num_samples 2500
```

可选：`--eval_ratio 0.12`、`--seed 42`。

**作用**：均衡生成并写出 `data/train_expanded.json`、`data/eval_expanded.json`、`data/dpo_pairs.jsonl`；训练与评测无 prompt 重叠。`--num_samples` 建议 **2000–3000**（脚本下限 720；论文建议 ≥2000）。

---

## 13. 手动运行全流程（推荐顺序）

### Step 1：生成数据集

```powershell
python dataset/generate_expanded_dataset.py --num_samples 2500
```

### Step 2：训练模型

示例（LoRA + SFT）：`python training/train_lora_sft.py --config configs/default.yaml`（数据路径见 `configs/default.yaml` 中 `train_sft_json`）。

### Step 3：运行评测（Bandit + 扩展评测集）

```powershell
python evaluation/evaluate.py --model lora_sft --batch_size 4
```

仅 Bandit、关闭回退：`--disable-fallback-detector`。

---

## 14. 各实验命令（详细）

以下为逐步展开的补充说明；**日常复现以第 13 节三步为主**。

### 14.1 按实验类型（7 组）

**前置**：已完成第 12 节数据生成，并已 `pip install bandit`（见第 10 节）。

----------------------------------
Step A: Run baseline
Command:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model baseline --batch_size 4`
Explanation:
Runs evaluation with larger batch size to improve GPU utilization.

----------------------------------
Step B: Adjust batch size if needed
Command:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model baseline --batch_size 8`
Explanation:
Try higher batch size if GPU memory allows.

----------------------------------
Step C: Monitor GPU usage
Command:
`nvidia-smi`
Explanation:
Check GPU utilization and memory usage in real time.

----------------------------------
Step D: Run LoRA only
Command:
`.\.venv\Scripts\python.exe training\train_lora_only.py`
Explanation:
Loads base model, attaches LoRA adapters, does NOT train, and saves adapter.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model lora_only --batch_size 4`

----------------------------------
Step E: Run LoRA + SFT
Command:
`.\.venv\Scripts\python.exe training\train_lora_sft.py --config configs\default.yaml`
Explanation:
Runs existing stable LoRA supervised fine-tuning pipeline.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model lora_sft --batch_size 4`

----------------------------------
Step F: Run LoRA + DPO
Command:
`.\.venv\Scripts\python.exe training\dpo_train.py --config configs\dpo.yaml`
Explanation:
Runs existing stable DPO training on top of LoRA SFT adapter.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model lora_dpo --batch_size 4`

----------------------------------
Step G: Run QLoRA only
Command:
`.\.venv\Scripts\python.exe training\train_qlora_only.py`
Explanation:
Loads model in 4-bit, attaches LoRA adapters, does NOT train, then saves adapter.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model qlora_only --batch_size 4`

----------------------------------
Step H: Run QLoRA + SFT
Command:
`.\.venv\Scripts\python.exe training\train_qlora_sft.py --config configs\default.yaml`
Explanation:
Runs QLoRA supervised fine-tuning with 4-bit quantized base model.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model qlora_sft --batch_size 4`

----------------------------------
Step I: Run QLoRA + DPO
Command:
`.\.venv\Scripts\python.exe training\train_qlora_dpo.py --config configs/dpo.yaml`
Explanation:
Runs QLoRA DPO on top of QLoRA SFT adapter; if unstable, script logs warning and exits without crashing the whole pipeline.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model qlora_dpo --allow-missing-adapter --batch_size 4`

----------------------------------
Step J: Build comparison summary
Command:
`.\.venv\Scripts\python.exe scripts\compare_results.py --config configs\default.yaml`
Explanation:
Recomputes and writes `outputs/comparison_summary.json` from all experiment result JSONs.
