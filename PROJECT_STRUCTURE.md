# 项目目录与文件说明

以下为仓库内**源码与配置**的完整树（不含 `.git`、`__pycache__`、以及 `outputs/models/` 下体积较大的 checkpoint 权重文件；评测 JSON 位于 `outputs/*.json`，由实验产生）。

```
.
├── README.md
├── PROJECT_STRUCTURE.md
├── requirements.txt
├── 任务报告书.txt
├── configs/
│   ├── default.yaml
│   └── dpo.yaml
├── data/
│   ├── combined/
│   │   ├── eval.json
│   │   └── train.json
│   ├── fix/
│   │   ├── eval.json
│   │   └── train.json
│   ├── generation/
│   │   ├── eval.json
│   │   └── train.json
│   ├── schema/
│   │   └── dataset_sample.schema.json
│   ├── samples/
│   │   └── examples_research_schema.json
│   ├── dpo_pairs.json
│   ├── eval_expanded.json
│   └── train_expanded.json
├── dataset/
│   ├── __init__.py
│   ├── generate_expanded_dataset.py
│   ├── generate_sql_security_dataset.py
│   ├── research_schema.py
│   ├── sql_security_dataset.json
│   └── synthetic_sql.py
├── detection/
│   ├── __init__.py
│   ├── bandit_wrapper.py
│   ├── detector.py
│   ├── rule_based.py
│   └── sql_injection_detector.py
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── evaluator.py
│   ├── experiment_log.py
│   ├── metrics.py
│   └── prompt_loader.py
├── logs/
│   ├── README.md
│   ├── dataset/
│   │   └── (运行 generate_expanded_dataset 时生成 *.log)
│   ├── errors/
│   │   └── .gitkeep
│   └── experiments/
│       └── .gitkeep
├── models/
│   └── README.md
├── outputs/
│   ├── baseline_results.json
│   ├── lora_dpo_results.json
│   ├── lora_only_results.json
│   ├── lora_sft_results.json
│   ├── qlora_dpo_results.json
│   ├── qlora_only_results.json
│   ├── qlora_sft_results.json
│   ├── comparison_summary.json
│   └── models/
│       └── (各 LoRA 适配器目录与 checkpoint，由训练产生)
├── reports/
│   ├── ANALYSIS_0.5B_FAILURE.md
│   └── operation_manual.md
├── scripts/
│   ├── README.md
│   ├── build_dataset.py
│   ├── compare_results.py
│   ├── migrate_dataset_to_research_schema.py
│   ├── plot_results.py
│   ├── run_baseline.py
│   ├── run_eval.py
│   └── run_thesis_pipeline.py
└── training/
    ├── __init__.py
    ├── amp_grad_debug.py
    ├── common.py
    ├── config_utils.py
    ├── dpo_train.py
    ├── dtype_utils.py
    ├── gpu_debug.py
    ├── lora_utils.py
    ├── sft_preprocess.py
    ├── train_dpo.py
    ├── train_lora_dpo.py
    ├── train_lora_only.py
    ├── train_lora_sft.py
    ├── train_qlora_dpo.py
    ├── train_qlora_only.py
    └── train_qlora_sft.py
```

## 根目录

| 文件 | 作用 |
|------|------|
| `README.md` | 项目概述、架构、环境、端到端命令、指标说明 |
| `PROJECT_STRUCTURE.md` | 本文件：目录树与模块职责 |
| `requirements.txt` | Python 依赖版本下限 |
| `任务报告书.txt` | 用户自备文档（非代码逻辑） |

## configs/

| 文件 | 作用 |
|------|------|
| `default.yaml` | 基座模型路径、数据路径、训练/评测参数、`outputs` 结果文件名、`eval.merge_mode` |
| `dpo.yaml` | 与 default 合并：DPO 学习率、输出目录等覆盖项 |

## data/

| 路径 | 作用 |
|------|------|
| `combined/train.json` / `eval.json` | 研究 schema：含 `task_type`、`expected_vulnerable`、`vulnerability_type`、`difficulty`、`input_code`；训练含 `output` |
| `generation/`、`fix/` | 按 `task_type` 拆分的 train/eval |
| `train_expanded.json` / `eval_expanded.json` | 生成器写出的扁平格式（兼容旧流程） |
| `dpo_pairs.json` | DPO 用 JSONL 行格式（prompt/chosen/rejected） |
| `schema/dataset_sample.schema.json` | 样本 JSON Schema |
| `samples/examples_research_schema.json` | 小样本示例 |

## dataset/

| 文件 | 作用 |
|------|------|
| `generate_expanded_dataset.py` | **主数据入口**：生成扩展集、DPO 对、`data/combined` 与拆分；日志写入 `logs/dataset/` |
| `research_schema.py` | `stable_sample_id`、`to_research_record`、`write_research_splits` |
| `generate_sql_security_dataset.py` | 旧版小规模生成器 |
| `synthetic_sql.py` | 合成片段工具（供 `scripts/build_dataset.py` 等） |
| `sql_security_dataset.json` | 无扩展集时的回退训练数据 |

## detection/

| 文件 | 作用 |
|------|------|
| `sql_injection_detector.py` | **统一检测**：`detect_vulnerability`（Bandit+规则）、`extract_python_code`、合并逻辑 |
| `detector.py` | 向后兼容 re-export |
| `bandit_wrapper.py` | 子进程调用 Bandit JSON |
| `rule_based.py` | 正则/启发式 SQL 构造检测 |
| `__init__.py` | 包导出 |

## evaluation/

| 文件 | 作用 |
|------|------|
| `evaluate.py` | 评测 CLI：多模型模式、`--merge-mode`、日志目录 |
| `evaluator.py` | 加载模型批量生成、调用 `detect_vulnerability`、写结果 JSON |
| `metrics.py` | 聚合：混淆矩阵、FPR/FNR、分层统计、`per_detector_vs_expected`、`by_attack_type_metrics` |
| `prompt_loader.py` | 加载 JSON/JSONL，拼 Instruction/Input 模板 |
| `experiment_log.py` | 文件日志初始化 |

## logs/

| 路径 | 作用 |
|------|------|
| `README.md` | 日志目录约定 |
| `dataset/*.log` | 数据集构建日志 |
| `experiments/` | 评测/实验运行日志（`evaluate.py --log-dir`） |
| `errors/` | 建议存放未捕获异常导出 |

## scripts/

| 文件 | 作用 |
|------|------|
| `migrate_dataset_to_research_schema.py` | 将 `train_expanded`/`eval_expanded` 迁移为 `data/combined` 等 |
| `plot_results.py` | 读取各模型结果 JSON，生成 SQL 注入率与 FPR/FNR 柱状图（matplotlib） |
| `compare_results.py` | 汇总多模型 `outputs/*_results.json` 为 `comparison_summary.json` |
| `build_dataset.py` | 从合成配置写 `dataset/*.jsonl`（小数据/旧流水线） |
| `run_eval.py` | 指定 adapter 路径的评测入口 |
| `run_baseline.py` | 转发 baseline 评测 |
| `run_thesis_pipeline.py` | 可选端到端流水线 |

## training/

| 文件 | 作用 |
|------|------|
| `train_lora_sft.py` | LoRA + SFT 主训练 |
| `dpo_train.py` | LoRA + DPO（需已有 SFT 适配器） |
| `train_lora_only.py` | 仅挂载 LoRA 并保存 |
| `train_qlora_only.py` | 4bit + LoRA 仅挂载 |
| `train_qlora_sft.py` | QLoRA + SFT |
| `train_qlora_dpo.py` | QLoRA + DPO |
| `train_lora_dpo.py` / `train_dpo.py` | 转发到 `dpo_train` |
| `sft_preprocess.py` | instruction/input/output 分词与 completion mask |
| `lora_utils.py` | target_modules 解析 |
| `config_utils.py` | YAML 深度合并（DPO 配置） |
| `dtype_utils.py` / `amp_grad_debug.py` / `gpu_debug.py` / `common.py` | 训练稳定性与诊断 |

## outputs/

| 文件 | 作用 |
|------|------|
| `*_results.json` | 各设置评测汇总 + `per_sample` |
| `comparison_summary.json` | `compare_results.py` 输出 |
| `models/*` | LoRA 权重与 tokenizer 副本 |

## models/

| 文件 | 作用 |
|------|------|
| `README.md` | 说明真实权重位于 `outputs/models/` |

## reports/

| 文件 | 作用 |
|------|------|
| `*.md` | 历史/操作说明（非运行时代码依赖） |
