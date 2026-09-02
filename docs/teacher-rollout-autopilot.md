# Teacher Rollout 自动流程

当前流程只服务 `unified-react-v1`：

```text
读取 case
  -> 并发启动 rollout
  -> 每条 trace 实时落盘
  -> 工程错误自动回队列重跑
  -> strict trace audit
  -> SFT eligibility / LLM judge
  -> accepted、holdout、action-only、RL candidate 分桶
```

重跑只针对未成功的 case，不覆盖已经成功的 episode。每条 trace 的原始请求、响应、错误和
重试关系都保留；分桶只建立清单，不删除轨迹。

质量重跑在初始 SFT 审计之后最多追加三轮：

```text
初始 rollout
  -> SFT judge
  -> 只把 SFT rejected / 未产出 terminal trace 的 case 放入 quality-reroll-01
  -> SFT judge
  -> 只把上一轮仍 rejected 的 case 放入 quality-reroll-02
  -> SFT judge
  -> 只把上一轮仍 rejected 的 case 放入 quality-reroll-03
  -> 每个 case 选择第一条通过 SFT 的轨迹
```

已通过 SFT 的 case 不会再次 rollout；`final_only_judgment` 虽然是可接受桶，也不会进入
质量重跑队列。三轮后仍未通过的 case 记录为 `hard case`。每轮独立保存：

- `rollouts/quality-reroll-XX/`：该轮所有尝试和完整 trace；
- `sft-eligibility/quality-reroll-XX/`：该轮 SFT 审计；
- `classification/quality-reroll-XX/`：该轮 accepted/rejected 清单。

合并后的 `run_manifest.json` 会继承 attempt 的 `git_commit`、Agent、benchmark
和 `source_access_policy` 元数据。这样 merged trace 在 SFT judge 或独立 strict
audit 中仍使用与 rollout 相同的来源黑名单，不会因合并丢失 policy 而误报普通查询。

最终合并结果写入 `classification/final/`，质量重跑后的训练发布包写入
`quality-reroll-release/` 和 `quality-reroll-training-package/`。已完成的初始 pipeline 可以用
`--reroll-from <pipeline-dir> --quality-reroll-rounds 3` 继续，不会重复初始成功轨迹。

建议生产前先做 10 条并发 10 的真实 Gemini smoke，检查：成功率、工程错误、每条是否有 scene/OCR
action、thought 捕获率、工具调用顺序和最终审计结果。通过后再启动全量 rollout。
