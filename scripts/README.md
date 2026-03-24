# scripts/

## 新流水线（论文主用）

| 脚本 | 作用 |
|------|------|
| `build_dataset.py` | 生成 `dataset/*.jsonl` |
| `compare_results.py` | 汇总 7 组实验对比 → `outputs/comparison_summary.json` |
| `run_thesis_pipeline.py` | 顺序跑全流程（支持 `--skip-lora-dpo/--skip-qlora-dpo`） |
| `run_baseline.py` | 兼容入口（内部转发到 `evaluation/evaluate.py --model baseline`） |
| `run_eval.py` | 兼容入口（自定义 adapter 评测） |
| `00_prepare_env.ps1` | Windows 创建 venv 与依赖 |

旧版 `scripts/legacy/*` 已移除；请使用根目录 `README.md` 中的主流水线命令。
