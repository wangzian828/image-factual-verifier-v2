# 2026-08-28 Gemini 3.7 API 与 compare_with_reference 跟进

## 结论

- 服务器上的 Gemini key 未过期：加载 `/gs/home/wza/.config/image-factual-verifier/runtime.env` 后，带 key 请求 `GET /v1beta/models` 返回 HTTP 200。
- Gemini 3.7 Interactions 生成请求不是 401/403，而是在 90 秒读取窗口后 `ReadTimeout`；因此当前问题是生成链路长尾或代理/服务端响应迟滞，不是认证失效。
- `unified-react-v1-gemini37-train-smoke-20260828-p10-r2` 已完成 9 条正常轨迹、1 条工程错误。160 个 Gemini 请求均收到 HTTP 层响应，仅 2 个请求发生有限重试。
- `main-02754` 的工程错误根因是 `compare_with_reference` 的模型输出字段互相重复约束：`differences` 没有编辑差异，但额外字段给出了不一致的编辑摘要。

## 代码修复

提交：`a556407`  
分支：`codex/gpu13-canary-20260804-plan-relaxation-01`  
GPU13 已更新到该提交。

`compare_with_reference` 现在：

1. 在线 Gemini schema 不再要求输出 `edit_evidence_present` 和 `edit_evidence_strength`。
2. 两个字段统一由 `differences` 的类型和显著性确定。
3. 回放旧响应时仍接受旧字段，但会归一化并记录 `contract_repairs`、`raw_edit_evidence_summary`。
4. 新增回归测试，远端 focused tests 为 11/11 通过。

## 实际 API 检查

| 检查 | 结果 |
|---|---|
| key 是否存在 | 是；仅核验长度，未输出内容 |
| `GET /v1beta/models` | HTTP 200，约 0.7 秒 |
| `gemini-3.7-flash` Interactions 最小生成 | 约 92 秒 `ReadTimeout` |
| 10 条真实 r2 smoke 的 Gemini 请求 | 160 次均有 HTTP 响应；2 次有限重试 |

因此不能把当前 Gemini 描述成“完全不可用”。更准确的描述是：认证有效，但生成接口在当前代理/服务链路上存在明显长尾，低负载单请求也可能超过 90 秒。
