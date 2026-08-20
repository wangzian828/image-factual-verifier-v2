# SFT canonical accepted release workflow

从 2026 年 8 月 17 日起，正式 SFT 不直接消费原始 teacher rollout 目录。

唯一训练链路是：

```text
raw rollout
  + frozen sft_eligibility
  -> stage_accepted_teacher_release.py
  -> accepted_release/
  -> export_dataset.py --accepted-release
  -> ifv-training convert-policy
  -> ifv-training convert-accepted-perception
  -> ms-swift / run_sft.sh
```

`accepted_release/` 是训练前的 canonical source，至少包含：

- `accepted_release_manifest.json`
- `selected_episodes.jsonl`
- `traces/`
- `eligibility/`
- `trajectory_sft.jsonl`
- `perception_trajectories.jsonl`

staging 阶段完成唯一的 teacher acceptance：

- trace 正常结束并有二元 verdict；
- `engineering_valid=true`；
- frozen `sft_eligibility_pass=true`；
- source trace SHA-256 一致；
- deterministic fatal reason 拒绝；
- 完整 episode SFT 行从 canonical trace 生成并写入 `trajectory_sft.jsonl`。

允许通过 judge 的非 fatal 轨迹保留 deterministic red flags。最终 exporter
只验证 canonical release 的完整性、SHA、split 和 model-visible 数据契约，不重新
读取 source run 中可能按旧标准生成的 step-level `policy_trajectories.jsonl`。

这样可以避免两套 SFT 标准再次分叉：source run 的旧 policy 导出失败不会覆盖
canonical release 中已经通过 frozen judge 的完整轨迹行。

训练前固定检查：

```bash
python -m ifv_training convert-policy \
  --input "$ACCEPTED_DATASET" \
  --output "$MS_SWIFT_POLICY"
python -m ifv_training convert-accepted-perception \
  --input "$ACCEPTED_DATASET" \
  --output "$MS_SWIFT_PERCEPTION"
python -m ifv_training audit --strict --input "$MS_SWIFT_POLICY"
python -m ifv_training audit --strict --input "$MS_SWIFT_PERCEPTION"
```
