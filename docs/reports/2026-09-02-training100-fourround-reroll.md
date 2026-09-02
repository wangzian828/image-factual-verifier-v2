# 训练集 100 条四轮 rollout 与 SFT 收尾记录

日期：2026 年 9 月 2 日

本批只用于验证 unified ReAct、SFT eligibility 和 reroll 流程，不进入正式训练，
也没有启动全量 8,490 条教师 rollout。

## 运行范围

- 数据集：统一数据集训练清单中的 100 条固定 case；
- 训练清单：
  `/gsdata/home/wza/image-factual-verifier-v2-data/datasets/route-aware-hrc-stage2-10563-final-organized-20260824-r2/unified-dataset/train-manifest.jsonl`
- pipeline：
  `/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/unified-react-v1-gemini37-train100-fourround-20260902-2e64147/`
- 模型：Gemini 3.7 Flash；
- rollout 并发：10；
- SFT judge 并发：10；
- 质量重跑：初始轮后最多追加 3 轮，每轮只接收上一轮未通过的 case；
- 工程失败：按现有 attempt 队列自动补跑，未把外部失败当作语义通过。

## 四轮结果

| 轮次 | 本轮目标 | terminal success | SFT 通过 | 下一轮 |
| --- | ---: | ---: | ---: | ---: |
| initial | 100 | 100 | 56 | 44 |
| quality-reroll-01 | 44 | 44 | 6 | 38 |
| quality-reroll-02 | 38 | 38 | 5 | 33 |
| quality-reroll-03 | 33 | 33 | 1 | 32 hard case |

最终结果：

- 100 个 case 均获得成功候选；
- 68 个 case 选中一条通过 SFT eligibility 的候选；
- 32 个 case 四轮均未通过，记为 hard case；
- 未解决工程错误：0；
- 所有轮次的 trace、SFT judge、分类和候选选择记录均保留。

最终分类文件：

`classification/final/classification.json`

最终发布源：

`quality-reroll-release/`

## 导出结果

最终 release 选中 68 条，但 policy SFT 与 perception SFT 的准入条件不同：

- reasoning policy SFT：67 条；
- action-only：1 条，case 为 `main-05344`；
- independent perception SFT：68 条；
- RL candidate 清单：68 条；
- trajectory catalog：68 条；
- 质量桶：high 65、usable 3；
- 长度索引：32K 内 51 条，32K–128K 17 条，超过 128K 为 0 条。

`action_only` 没有可读 provider thought，因此不能伪造 `<think>` 混入 reasoning
SFT；但它仍有有效的图像 perception 目标，不能从 perception 数据中一起删掉。

## 导出器问题与修复

旧导出器在 accepted release 遇到 action-only 时直接 `continue`，因此同时跳过了
该条独立 perception 样本，造成 package 中 policy 和 perception 都显示为 67 条。
这不是 rollout 或 judge 丢数据，而是两个训练入口被错误耦合。

现已修复：

1. action-only 继续保留在独立 `action_only.jsonl`；
2. action-only 不进入 reasoning policy `train/validation/test.jsonl`；
3. action-only 对应的有效 perception 继续进入 `perception.*.jsonl` 和
   `ms-swift-perception/`；
4. package manifest 分开记录 selected release、policy、action-only 和 perception
   数量；
5. 新增回归测试覆盖该场景。

## 历史分类字段

`classification/quality-reroll-01/classification.json` 是修复 reroll 目标范围前生成
的历史产物，其中 `case_count=100`、`incomplete_case_count=56` 是旧范围推断错误。
它没有参与最终 68 条选择；后续复核以每轮 `target-case-list.txt` 和按当前代码重算的
分类结果为准。原始文件不覆盖，避免破坏实验审计链。

## 验证状态

- 本地默认回归：481 passed；
- `compileall`：通过；
- `git diff --check`：通过；
- 服务器当前无 rollout、judge 或遗留 Jupyter kernel 进程；
- CLOSE-WAIT：0；
- 全量 8,490 条教师 rollout：未启动。
