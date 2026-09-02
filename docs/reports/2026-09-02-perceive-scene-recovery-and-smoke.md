# `perceive_scene` 恢复修复与真实 smoke 记录

日期：2026 年 9 月 2 日

## 问题

当前 smoke 使用 `gemini-3.1-pro-preview` 时，10 条轨迹中有 3 次
`perceive_scene` 工具错误：

- 2 次 Gemini HTTP 400：`invalid_request`；
- 1 次在 150 秒工具边界内超时。

这些错误被记录在轨迹中，没有伪装成成功观察，但会让视觉记忆缺失。

## 修复

- 视觉工具默认使用受控 JPEG，而不是把原始大图直接转成 base64；
- Gemini 视觉 structured-output 请求使用 API 兼容 schema 投影，移除
  `maxLength`、`maxItems` 等 provider 不需要的校验约束；
- `perceive_scene` 对可恢复的 400、传输失败和超时最多追加一次恢复请求：
  使用更小的压缩图和宽松对象 schema；
- 成功恢复时写入 `perception_attempts` 与 `perception_recovery`；
  两次都失败时仍返回 `status=error`；
- active ReAct 工具动作默认保留 210 秒边界，为一次恢复和清理留出空间。

成熟工具的公开名称、任务语义和本地输出归一化契约没有改变。

## 复核

本地定向回归：34/34 通过。全量真实 smoke 在部署后分别使用：

- `gemini-3.1-pro-preview`；
- `gemini-3.7-flash`。

部署 commit、run 路径、每个模型的成功数、恢复次数和剩余工程错误，
在真实 smoke 完成后补记于本文件。
