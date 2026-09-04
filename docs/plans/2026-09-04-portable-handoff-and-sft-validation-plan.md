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
12. `[ ]` 在正式提交上做小规模全链路 smoke，检查每条轨迹。
13. `[ ]` 记录最终 commit、包路径、审计报告、样本路径和已知边界。
14. `[ ]` 停在全量 8,490 条教师 rollout 启动之前。
