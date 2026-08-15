# Agent 性能优化总计划

状态：第一批优化已实现，等待真实 canary 基线与下一批改动  
日期：2026-08-15  
适用分支：`codex/gpu13-canary-20260804-plan-relaxation-01`

当前已实现：

- context ledger 的组件级大小、request snapshot hash、provider token 和 retry 记录；
- `compare_with_reference` 的参考图缓存、线程级 HTTP session 复用；
- 本地图像 base64/JPEG 序列化缓存；
- 反向搜索上传 URL 的图片 SHA-256 缓存；
- 现有 `visit_many` 页面并发、PaddleOCR 实例池和 OCR perception cache 保持启用。

## 1. 目标

在不改变图像事实判断语义、Evidence reducer、strict audit 和“一次
Agent action 一个工具”协议的前提下，降低：

- Gemini 实际输入 token；
- Gemini 调用延迟和尾延迟；
- 图片、网页和 OCR 的重复处理；
- 无效重试、协议纠正和超时造成的额外等待；
- 并发 rollout 下的本地 I/O 开销。

所有优化都必须以真实 request snapshot 和代表性轨迹的回归结果为依据，
不能只根据 archive 文件大小或单次运行时间判断有效。

## 2. 已有基础

以下优化已经完成，不在本计划中重复实现：

- VLM client 与 Gemini vision transport 复用；
- 统一 `PersistentAsyncRuntime`；
- Jina 网页抽取复用 Gemini transport；
- Jina `visit_many` 最多并发访问 3 个页面；
- 页面线程池、HTTP session、Jina Reader、搜索客户端和缓存复用；
- CPU PaddleOCR 实例池；
- 相关 transport、线程池测试以及服务器测试。

后续工作应建立在这些基础上，避免重新引入每个 action 独立初始化
client、线程池或 OCR 实例的问题。

## 3. 阶段 0：建立性能基线

在修改前选取 3--5 个代表性样本，至少覆盖：

- 需要视觉复查的样本；
- 需要反向搜索的样本；
- 文本搜索后访问多个页面的样本；
- OCR 有结果和无结果的样本；
- 出现 retry、timeout 或 protocol correction 的样本。

每次运行记录：

- 总耗时和各阶段耗时；
- Agent action 数和 LLM 调用数；
- 每个工具的调用次数、耗时和错误；
- Gemini 实际输入 token、输出 token 和请求大小；
- system prompt、工具 schema、Evidence、历史动作各自的大小；
- OCR、Jina、Serper、反向搜索和参考图比较的耗时；
- retry、timeout、protocol correction 次数；
- 图片、页面和 OCR 缓存命中率。

基线结果应写入运行报告，作为后续优化的对照，而不是依赖主观体感。

## 4. 阶段 1：优化真正发送给 provider 的上下文

### 4.1 原则

压缩 workspace archive 不等于压缩 provider 输入。必须从
`context ledger` 的实际 request snapshot 验证输入 token 是否下降。

运行时上下文分为三层：

```text
canonical state      保证事实判断不丢信息
runtime context      当前请求发送给模型的精简视图
workspace archive    完整历史、原文和调试材料
```

### 4.2 第一版策略

- 保留当前目标、canonical state、Evidence 和关键 Finding；
- 历史 action 从完整工具返回改成结构化短记录；
- 重复搜索结果按 URL/hash 去重；
- 失败路线保留“尝试过什么、结果是什么、是否值得重试”的摘要；
- 原始网页文本、图片和完整轨迹继续保存在 archive；
- 只有模型确实需要时，才从 archive 回注入原始材料；
- 不对 Evidence 做不可逆的 LLM 语义摘要；
- 不删除 attempted routes，不改变 reducer 的事实语义。

### 4.3 验收

- 实际 provider input token 下降；
- verdict、Evidence、strict audit 不变；
- 工具调用数不增加；
- 压缩前后相同样本的关键状态一致。

## 5. 阶段 2：优化 `compare_with_reference`

这是优先排查的单次大延迟来源。

- 缓存参考图下载结果；
- 复用 HTTP client 和连接；
- 按图片 hash 缓存 base64 编码；
- 避免 URL fallback 重复下载；
- 对多张参考图进行批量处理；
- 只将确实需要比较的参考图送入 Gemini；
- 记录下载、编码和 VLM 比较三个子阶段的耗时。

不得因为缓存而复用过期或内容不一致的图片。缓存 key 至少应包含
规范化 URL 或内容 hash，以及必要的处理参数。

## 6. 阶段 3：反向搜索上传缓存

以当前图片内容 hash 作为反向搜索缓存 key：

```text
image_hash -> reverse-search result
```

同一张图片的重复反搜直接复用结果，避免重复上传。缓存内容应区分：

- 搜索请求是否成功；
- 搜索结果为空；
- provider 或网络失败。

失败结果不能永久阻塞后续重试；需要区分可重试失败和确定性空结果。

## 7. 阶段 4：搜索候选批处理

Agent 层继续保持：

```text
一次 Agent action = 一个工具调用
```

只在工具内部做安全批处理：

- 文本搜索一次返回候选集合；
- 在候选筛选阶段使用 Jina 获取搜索结果的页面预览；
- 先判断页面是否值得访问，再访问正文；
- 对相互独立的页面最多并发访问 3 个；
- 将批量访问结果统一转换为 Evidence/Discovery；
- 不把 Jina Search 暴露成新的 Agent action。

这样可以减少页面逐个访问造成的串行等待，同时不引入多 action
协议、顺序合并和 reducer 并发一致性问题。

## 8. 阶段 5：OCR 优化

在现有 PaddleOCR 实例池基础上增加结果复用：

- 按原图 hash 和 crop hash 缓存 OCR；
- 合并同一张图的重复 OCR 请求；
- 对相同处理参数直接复用结果；
- 对确认没有文字区域的图片避免重复 OCR；
- 只在视觉复查或文字证据确实需要时触发 OCR；
- 以服务器 CPU 运行方式为基准，不为通用兼容性保留低效 fallback。

OCR 缓存结果必须携带图片 hash、裁剪区域和处理参数，避免不同 crop
之间发生误复用。

## 9. 阶段 6：重试、超时和尾延迟

统一梳理 provider 和工具层的失败分类：

- 临时网络错误：有限次数重试；
- 429 或服务繁忙：指数退避；
- schema、参数和协议错误：不要无意义重复发送相同请求；
- Gemini 超时：及时取消或回收底层任务；
- 页面访问失败：区分抓取失败、无内容和内容确实不存在；
- protocol correction：统计触发率，减少可由本地校验直接解决的纠正调用。

验收时不仅看平均耗时，还要看 P95/P99。一次超时后底层同步线程继续
占用连接或线程池，会在并发 rollout 下放大尾延迟。

## 10. 阶段 7：本地媒体和 I/O

- 缓存图片 hash、缩放结果和 base64；
- trace/artifact 写入异步化或批量化；
- 避免每轮复制完整 Evidence；
- 大型网页原文优先落盘，不反复复制进模型上下文；
- 检查并发 rollout 下的磁盘竞争；
- 对临时文件使用明确的生命周期和清理策略。

本阶段不得牺牲 trace 完整性。优化后的归档仍必须支持严格复盘和
失败诊断。

## 11. 阶段 8：评估 provider 级缓存

在前述优化完成后，再评估 Gemini 是否适合复用：

- 固定 system prompt；
- 工具 schema；
- 稳定的基础上下文。

只有确认当前 provider 接口的支持方式、实际命中率、费用和失效行为后
才接入。不能仅凭请求体看起来重复，就假设 provider 一定会自动缓存。

## 12. 实施顺序

建议按照以下顺序落地：

```text
性能基线
  -> compare_with_reference
  -> 反向搜索上传缓存
  -> retry/timeout 尾延迟
  -> 搜索候选批处理
  -> OCR 缓存与任务合并
  -> 本地媒体和 I/O
  -> runtime context 压缩
  -> provider 级缓存评估
```

每完成一个阶段，都先运行小样本回归，再决定是否进入下一阶段。
不要把多个高风险改动合并后才测试，否则无法判断收益和回归来源。

## 13. 统一验收标准

优化版本必须同时满足：

- 最终 verdict 不变；
- Evidence 不丢失、不改变来源和 stance；
- strict audit 不下降；
- 不增加无效工具调用；
- 实际 Gemini input token 下降，或明确证明该阶段不影响 token；
- 总耗时和 P95/P99 延迟有改善；
- retry、timeout 和 protocol correction 没有异常增加；
- 完整 trace 仍可复盘；
- 服务器环境下测试通过。

## 14. 暂不纳入本计划

以下改动暂不做：

- Agent 一次返回多个 action；
- 放宽或重写 Evidence reducer 语义；
- 删除 attempted routes 或失败证据；
- 对 Evidence 做不可逆语义摘要；
- 为了速度改变事实判断标准；
- 通过更高并发掩盖 provider、网络或缓存问题。
