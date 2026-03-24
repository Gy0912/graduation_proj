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
│   ├── train_expanded.json                # 扩展 SFT 训练集（instruction/input/output）
│   ├── eval_expanded.json                 # 统一评测 prompts（与训练去重）
│   └── dpo_pairs.jsonl                    # DPO 偏好对：prompt / chosen / rejected
│
├── dataset/                               # 数据构建与中间 jsonl
│   ├── generate_expanded_dataset.py       # **入口**：生成 data/ 与 dataset/*.jsonl
│   ├── generate_sql_security_dataset.py   # 旧版小数据集生成器
│   ├── sql_security_dataset.json          # 旧版小训练集（无扩展数据时的回退）
│   ├── synthetic_sql.py                   # 合成 SQL 片段等工具
│   ├── eval_prompts.jsonl                 # 由 build_dataset 等生成的评测 jsonl（若使用）
│   ├── sft_train.jsonl / sft_val.jsonl    # SFT 分词用 jsonl（由流水线生成）
│   ├── dpo_train.jsonl                    # DPO 训练用 jsonl（由流水线生成）
│   ├── examples/                          # 示例 json/jsonl
│   └── README.md
│
├── detection/                             # **核心**：SQL 注入规则检测（评测指标依赖）
│   ├── sql_injection_detector.py          # 检测器与代码抽取
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

## 5. 统一评测与指标

所有方法使用完全相同评测数据集：`configs/default.yaml` 中 `files.eval_prompts`（默认 `data/eval_expanded.json`）。

核心指标：

- `sql_injection_rate`（越低越好）
- `safe_code_generation_rate`（越高越好）
- `reduction vs baseline`（相对 baseline 的注入率下降百分比）

## 6. 输出文件说明

每个实验都会输出一个独立 JSON：

- `outputs/baseline_results.json`
- `outputs/lora_only_results.json`
- `outputs/lora_sft_results.json`
- `outputs/lora_dpo_results.json`
- `outputs/qlora_only_results.json`
- `outputs/qlora_sft_results.json`
- `outputs/qlora_dpo_results.json`

统一汇总文件：

- `outputs/comparison_summary.json`
  - 包含全部 7 种方法
  - 所有指标基于当前结果文件重新计算

## 7. 环境准备

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe dataset\generate_expanded_dataset.py
```

## 8. Evaluation Optimization

### 8.1 为什么 GPU 利用率会波动

评测阶段如果逐条生成（batch=1）且在循环中频繁做分词，会出现以下现象：

- GPU 等待 CPU 准备输入，利用率出现锯齿
- CPU-GPU 频繁小包传输，吞吐偏低
- DataLoader 配置过保守时，GPU 空转时间增加

### 8.2 已做的优化

- 增加可配置评测批大小：`per_device_eval_batch_size`（默认 4）
- 评测前一次性预分词，避免在推理循环中再做 tokenization
- DataLoader 使用 `num_workers=2`（Windows 友好）和 `pin_memory=True`
- 推理明确使用 `model.eval()` + `torch.no_grad()`
- 输入张量显式搬运到 CUDA，并打印设备信息
- 增加调试日志：batch size、device、输入 tensor device、每 batch 用时

### 8.3 如何调参

- 显存约 8GB 建议从 `--batch_size 4` 开始
- 若显存充足可尝试 `--batch_size 8`
- 若遇到 OOM，请降回 `--batch_size 2` 或 `4`

## How to run (manual steps)

----------------------------------
Step 1: Run baseline
Command:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model baseline --batch_size 4`
Explanation:
Runs evaluation with larger batch size to improve GPU utilization.

----------------------------------
Step 2: Adjust batch size if needed
Command:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model baseline --batch_size 8`
Explanation:
Try higher batch size if GPU memory allows.

----------------------------------
Step 3: Monitor GPU usage
Command:
`nvidia-smi`
Explanation:
Check GPU utilization and memory usage in real time.

----------------------------------
Step 4: Run LoRA only
Command:
`.\.venv\Scripts\python.exe training\train_lora_only.py`
Explanation:
Loads base model, attaches LoRA adapters, does NOT train, and saves adapter.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model lora_only --batch_size 4`

----------------------------------
Step 5: Run LoRA + SFT
Command:
`.\.venv\Scripts\python.exe training\train_lora_sft.py --config configs\default.yaml`
Explanation:
Runs existing stable LoRA supervised fine-tuning pipeline.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model lora_sft --batch_size 4`

----------------------------------
Step 6: Run LoRA + DPO
Command:
`.\.venv\Scripts\python.exe training\dpo_train.py --config configs\dpo.yaml`
Explanation:
Runs existing stable DPO training on top of LoRA SFT adapter.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model lora_dpo --batch_size 4`

----------------------------------
Step 7: Run QLoRA only
Command:
`.\.venv\Scripts\python.exe training\train_qlora_only.py`
Explanation:
Loads model in 4-bit, attaches LoRA adapters, does NOT train, then saves adapter.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model qlora_only --batch_size 4`

----------------------------------
Step 8: Run QLoRA + SFT
Command:
`.\.venv\Scripts\python.exe training\train_qlora_sft.py --config configs\default.yaml`
Explanation:
Runs QLoRA supervised fine-tuning with 4-bit quantized base model.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model qlora_sft --batch_size 4`

----------------------------------
Step 9: Run QLoRA + DPO
Command:
`.\.venv\Scripts\python.exe training\train_qlora_dpo.py --config configs/dpo.yaml`
Explanation:
Runs QLoRA DPO on top of QLoRA SFT adapter; if unstable, script logs warning and exits without crashing the whole pipeline.

Then evaluate:
`.\.venv\Scripts\python.exe evaluation\evaluate.py --model qlora_dpo --allow-missing-adapter --batch_size 4`

----------------------------------
Step 10: Build comparison summary
Command:
`.\.venv\Scripts\python.exe scripts\compare_results.py --config configs\default.yaml`
Explanation:
Recomputes and writes `outputs/comparison_summary.json` from all experiment result JSONs.
