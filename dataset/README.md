# 数据集说明

## 主文件：`sql_security_dataset.json`

- **生成**：`python dataset/generate_sql_security_dataset.py`
- **规模**：默认 100 条（四类各约 25%）
- **字段**：
  - `instruction`：任务说明（可含对抗/模糊表述）
  - `input`：补充上下文（如表名、脆弱代码片段）
  - `output`：**必须为安全实现**（参数化查询等），即使 instruction 诱导不安全写法
  - `category`：`normal` | `ambiguous` | `adversarial` | `repair`

## 设计动机（论文可用）

1. **对抗样本（adversarial）**  
   模拟用户或恶意提示「不要用预处理语句」等。基线模型易生成危险代码；微调数据**只给安全答案**，让模型学会拒绝危险实现。

2. **修复样本（repair）**  
   提供典型拼接 SQL 漏洞片段，输出改为参数化查询，覆盖「代码修改」场景。

3. **模糊样本（ambiguous）**  
   提高在指令不完整时的**默认安全**行为。

4. **正常样本（normal）**  
   稳定主任务分布，减轻灾难性遗忘。

## 与训练代码的衔接

`training/train_lora_sft.py` 读取该 JSON，按 `val_ratio` 划分 train/val，在 **`training/sft_preprocess.py`** 中 `dataset.map(batched=True)` 完成分词，再交给 `SFTTrainer`。
