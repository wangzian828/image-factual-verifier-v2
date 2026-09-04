# 2026-09-04 可移植交付与 SFT 验证计划

## 目标

完成当前小批轨迹的正式 Qwen/ms-swift 发布验证，并把项目整理成可在另一台服务器
部署、检查和小规模运行的交付状态。停止点位于全量教师 rollout 启动之前。

## 待办与验收

1. `[x]` 重建 3 条 accepted release 的正式训练包。
   - reasoning policy：2 条；
   - perception：3 条；
   - action-only：1 条；
   - strict JSON 审计通过。
2. `[x]` 修正真实 processor 探针。
   - 按 Qwen 实际 `<function>` / `<parameter>` 模板核验；
   - 不再要求源 JSON 在编码文本中逐字出现。
3. `[x]` 使用目标 Qwen processor 重新编码正式包并确认 `passed=true`。
4. `[x]` 下载一条完整 reasoning SFT 样本到本地人工复核。
5. `[x]` 核对 `text_image_search`。
   - Serper 图片搜索；
   - ReAct schema 与英文 prompt；
   - Discovery 边界；
   - 下一轮候选图注入；
   - SFT 图片投影。
6. `[x]` 增加通用服务器环境、doctor、bootstrap、run、start、poll 入口。
7. `[x]` 移除 active Python runtime 中当前服务器模型/数据路径默认值。
8. `[x]` 将训练数据根目录、模型 profile 和 GPU allowlist 改为可配置。
9. `[x]` 更新 Agent、Prompt、SFT、Qwen、轨迹与运维文档。
10. `[x]` 完整本地测试、Git 提交并推送。
11. `[x]` gpu-13 fast-forward 后重复结构审计和真实 processor 验证。
12. `[x]` 在正式提交上做小规模全链路 smoke，检查每条轨迹。
13. `[x]` 记录最终 commit、包路径、审计报告、样本路径和已知边界。
14. `[x]` 停在全量 8,490 条教师 rollout 启动之前。

## 2026-09-04 真实 Smoke 收尾

- 发布提交：`414cb07`；gpu-13 已 fast-forward，主仓定向回归 70 项、training 定向回归
  10 项通过。
- 新的教师 rollout 入口使用 `src.eval.run_cases`，不会误要求测试评测专用的
  `evaluation_gold`。
- 并发 10 的真实 10 条 Agent smoke 初始完成 9 条；`main-02730` 的瞬时 SSL
  transport 失败按现有工程重跑逻辑单条续跑，最终 10/10 terminal trace 和 strict
  trace audit 通过，worker `CLOSE-WAIT=0`。
- frozen SFT judge 通过 3 条；新 ms-swift 包包含 3 条 policy、3 条 perception、0 条
  action-only。目标 Qwen3.5 processor 的最长 policy 输入为 28,064 tokens，远低于
  128K。
- 逐条质量复核发现当前最大语义问题是：纯图片 runtime 没有传播 claim，模型会在图片的
  底图、叠字、转发帖文和视频来源之间自行选择一个可调查对象。旧版 SFT judge 曾将
  部分“相关但漏掉关键条件”的错误粗略标为 `different_image_fact`；v7 重审时会以
  `target_scope` 区分它和真正无关的调查，但不会在 rollout 时把 private target 泄漏给模型。
- 详细运行路径、SFT 分桶、长度解释、文搜图真实注入和逐条质量分析见：
  [`2026-09-04-portable-handoff-smoke10.md`](../reports/2026-09-04-portable-handoff-smoke10.md)。
- 停止点保持不变：不启动 8,490 条全量教师 rollout。
