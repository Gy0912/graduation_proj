"""
兼容入口：转发到 `dpo_train.py`。

真实 DPO 流程：在 **已训练 SFT LoRA** 上继续优化，偏好数据为 `data/dpo_pairs.jsonl`。
请优先使用：

  python training/dpo_train.py --config configs/dpo.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.dpo_train import main

if __name__ == "__main__":
    main()
