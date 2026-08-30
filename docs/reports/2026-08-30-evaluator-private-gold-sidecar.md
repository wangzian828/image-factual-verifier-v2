# 2026-08-30 统一数据集 evaluator-private gold

## 已生成位置

数据集：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/datasets/route-aware-hrc-stage2-10563-final-organized-20260824-r2/unified-dataset
```

private gold 目录：

```text
evaluator_private/private-gold-v1/
```

文件：

- `test-private-gold.jsonl`：测试集 1,684 条；
- `train-private-gold.jsonl`：训练集 8,490 条；
- `private-gold.jsonl`：两部分合并的 10,174 条；
- `case-alias-index.jsonl`：只保留无歧义别名；
- `summary.json`：输入文件 hash、补全统计和完整性结果。

## 完整性

- 10,174 条都有稳定 case ID 和可用 private target；
- 测试集缺失的 182 条构造字段已从不可变 archive 补齐；
- 训练集缺失的 92 条构造字段已从不可变 archive 补齐；
- train/test stable case ID 无重复；
- runtime rollout 不读取这些文件；
- Direct QA、Agent private-gold audit 和 SFT 审计只在 rollout 完成后读取对应 split。

训练 autopilot 会优先使用：

```text
<unified-dataset>/evaluator_private/private-gold-v1/train-private-gold.jsonl
```

测试审计会优先使用：

```text
<unified-dataset>/evaluator_private/private-gold-v1/test-private-gold.jsonl
```

这样不再依赖临时从 manifest/archive 猜测 case ID，也不会把 private gold 放进
Agent 的 runtime 三字段输入。
