# IFV ReAct 工具对齐与可见性重构计划

日期：2026-08-31

状态：已完成到人工复核边界；已完成入口、会话链路、候选图回传、Jina 输入边界、
工具生命周期和下游 SFT 适配。

## 1. 目标

把 IFV 的图像事实核查 Agent 整理成一条真正可工作的 ReAct 链路：

```text
模型请求
  -> 一个工具调用
  -> 执行工具
  -> 完整、受限、可审计的工具结果
  -> 下一次模型请求读取该结果
```

原图在需要时重新附加到下一次请求，但不把原图或完整累计状态反复写入文本历史。

本次不再继续维护旧 v4 主流程，也不把旧字段或旧工具名重新暴露到新的 Agent schema。旧归档只通过离线读取映射处理。

## 2. 参考实现已确认的行为

参考项目 `D:\DeepResearch-main\DeepResearch-main\WebAgent\WebWatcher` 的关键行为：

1. Agent 保持一条连续的消息历史。
2. 模型输出 `assistant tool_call` 后执行工具。
3. 工具结果立即以 `tool_response` / function result 追加到历史。
4. 下一次模型请求直接读取刚刚的工具结果。
5. 图片搜索结果除了文字描述，还把候选图片作为多模态输入追加到下一轮。
6. `text_search` 只负责搜索，不在工具内部自动访问网页。
7. `visit` 负责抓取网页并生成目标相关摘要。
8. ReAct 的重新搜索通过下一轮模型选择新的动作完成，不需要独立的“replan 工具”。

## 3. 工具对照与命名决定

| 当前 IFV 工具 | 参考项目对应物 | 是否等价 | 公开名称 | 处理 |
|---|---|---:|---|---|
| `text_search` | `web_search` | 是 | `text_search` | 保留现有名称；每次一个 query、最多 10 条结果 |
| `visit` | `visit` | 是 | `visit` | 保留名称；统一网页结果和消息回传 |
| `reverse_image_search` 的 Lens 分支 | `VLSearchImage` | 核心等价 | `reverse_image_search` | 保留现有名称；保留 Lens 分支及候选图回传 |
| `reverse_image_search` 的 semantic 分支 | `VLSearchText` | 核心等价 | `reverse_image_search` | 保留现有名称；由 Agent 选择 semantic 分支，不在工具内隐藏额外路线 |
| `crop_and_search` | 无 | 否 | 不作为主动 Agent 工具 | 保留成熟实现，但从主动工具集合移除；需要时使用现有显式工具 |
| `perceive_scene` | 无 | 否 | `perceive_scene` | IFV 专有视觉观察工具，保留 |
| `ocr_with_position` | 无 | 否 | `ocr_with_position` | IFV 专有 OCR 工具，保留 |
| `focused_visual_inspection` | 无 | 否 | `focused_visual_inspection` | IFV 专有视觉复查工具，保留 |
| `crop_and_inspect` | 无 | 否 | `crop_and_inspect` | IFV 专有局部观察工具，保留 |
| `compare_with_reference` | 无 | 否 | `compare_with_reference` | IFV 专有参考图比较工具，保留 |
| `check_consistency` | 无 | 否 | `check_consistency` | 作为显式一致性诊断，保留 |
| `analyze_visual_anomalies` | 无 | 否 | `analyze_visual_anomalies` | 作为显式异常诊断，保留 |
| `count_objects` | 无 | 否 | `count_objects` | 只产生视觉观察，不直接产生事实结论 |
| `current_time` | 无 | 否 | `current_time` | 保留为通用工具，但不放入图像核查主动工具集合 |
| `recall_evidence` / `read_evidence` | 无 | 否 | 内部能力 | 不暴露给当前主 ReAct，避免把状态管理伪装成调查动作 |
| `finish_investigation` | 无 | 否 | `finish_investigation` | 保留为运行时控制动作，不产生最终标签 |

说明：

- 工具公开名称保持 IFV 现有名称，不因参考项目而改名。
- 不把 IFV 的视觉专有工具伪装成参考项目已有工具。
- `reverse_image_search` 当前把上传、Lens、VLM 查询生成和语义图搜揉成一次动作；这次拆开后，每个公开动作只表达一种调查意图。
- 参考项目中的 `code_interpreter` 不属于 IFV 当前事实核查核心能力，本次不新增。

## 4. 主流程重构

### 4.1 只保留一个主动 ReAct 入口

当前 `run()` 进入统一的 `_run_react_runtime_policy`。动态工具 schema 可以按轮重建，
但同一个 episode 复用一个 `InteractionSession`，保证工具结果可靠进入下一轮模型输入。

处理方式：

1. `run()` 统一进入新的主 ReAct runtime。
2. 不再让旧 v4 / 旧 runtime 作为主入口。
3. 每个 episode 只建立一个持久的交互 session。
4. `max_rounds=1` 只表示“一次外层调度选择一个动作”，不表示丢弃上一个工具结果。
5. 下一次请求必须带上：
   - 上一次模型的 function call；
   - 对应的完整 canonical function result；
   - 当前紧凑状态；
   - 必要时重新附加的原图。

### 4.2 Gemini Interactions 消息链路

每轮固定为：

```text
初始 user_input（任务 + 原图）
  -> model function_call
  -> function_result（文本结果，必要时附带候选图片）
  -> 下一次请求的 user_input（紧凑状态 + 原图）
```

约束：

- `function_result` 与 `user_input` 作为合法的 Interactions Step 分开提交。
- 不把 function result 再复制成一份完整文本塞进累计状态。
- 原图可以每轮重新附加，但只作为当前请求的独立 image item。
- 工具返回的候选图只在当前结果对应的下一轮附加，最多附加受控数量。
- 最终归档保存完整原始结果；模型看到的是同一个 canonical 结果的有界版本，不能出现“reducer 看到全部、模型只看到一句摘要”的分叉。

### 4.3 非 Gemini / Chat Completions 回退链路

如果使用 Chat Completions：

```text
assistant tool_call
  -> tool/function result
  -> assistant 下一轮
```

必须把相同的 canonical result 放入 `tool` / function result message。不能仅通过重新拼装状态摘要代替工具结果。

## 5. 统一工具结果边界

新增一个运行时统一的 canonical observation 层，所有工具都经过同一层：

```json
{
  "status": "success|error",
  "tool": "公开工具名",
  "observation": {},
  "subcalls": [],
  "timings": {},
  "errors": [],
  "artifacts": []
}
```

规则：

1. 归档、reducer、下一轮模型输入使用同一份规范化结果。
2. 原始 provider 返回保存到 artifact，不直接无限塞进 prompt。
3. `status=error` 的结果不能产生 Discovery、Evidence 或路线进展。
4. 部分 provider 成功必须显式记录在 `subcalls` 和 `observation.partial`，不能一边顶层 error、一边让 reducer 无条件消费成功行。
5. 外部不可访问、SSL、验证码、超时等属于工具结果中的 `external_unavailable`，不是 Agent 协议错误。
6. schema 不一致、未知工具、非法参数、字段互相矛盾属于 `engineering_error`。
7. 每个工具都记录实际子调用数量、provider、耗时和错误，不把一次复合动作伪装成一次外部请求。

## 6. 各工具具体整改

### 6.1 `text_search`

参考项目的搜索职责与当前 `text_search` 对齐：

1. 保持当前 `text_search` 公开名称。
2. 一次动作只接受一个 query。
3. 返回最多 10 条有序结果，每条包含 title、URL、snippet、source、rank。
4. 不自动访问网页，不调用网页抽取模型。
5. Serper source policy 在发请求前检查；被黑名单 query 直接拒绝，不先搜索再过滤。
6. URL 去重和黑名单过滤发生在 canonical result 边界。
7. `answer_box`、`knowledge_graph` 等没有稳定来源 URL 的聚合信息不作为模型证据。

### 6.2 `visit`

保留当前 IFV 较强的精确段落抽取能力，但对齐参考项目的动作语义：

1. 一次动作只处理受控数量的页面，逐页保留独立 visit record。
2. 每个页面独立记录 fetch、抽取、失败和安全检测结果。
3. 模型看到的页面结果与 reducer 看到的页面结果完全一致。
4. 只返回从页面原文恢复的 passage，不信任抽取模型自行编造的 evidence 文本。
5. 保留 exact span、文档 hash、来源 URL 和抽取状态。
6. Jina/direct fallback 是否启用由当前配置决定，但所有尝试必须进入 `subcalls`。
7. 验证码、SSL、访问拒绝、空页面在结果中标记 `external_unavailable`，后续 Agent 可以换页面，不重复访问同一失败候选。

### 6.3 `reverse_image_search` 的 Lens 分支

由 `reverse_image_search` 的 Lens 分支拆出：

1. 一次只对一个当前图像或一个明确裁剪图做图搜。
2. 保留当前 top 3 图片候选数量。
3. 结果保留 provider 原生的候选类型、页面 URL、图片 URL、标题和摘要。
4. 上传、图搜请求、候选下载分别记录为子调用。
5. 结果作为 Discovery，不直接变成 Evidence。
6. 下一轮模型请求把最多 3 张可验证的候选图片作为多模态输入追加；下载失败的候选只保留文字和失败原因。
7. 对象上传的生命周期、缓存命中和清理状态写入 artifact/ledger。

### 6.4 `reverse_image_search` 的 semantic 分支

由原 semantic 分支拆出：

1. 接受 Agent 产生的一个文本 query。
2. 通过图像搜索服务返回候选图，保留当前受控数量。
3. 不在工具内部再次调用 Gemini 生成 query。
4. 这样 query 的产生、修改和重试都是真实 ReAct thought/action 的一部分。
5. 候选图同样以文字 + 多模态图片进入下一轮。
6. 与 Lens 分支分开记录子调用成本，但不新增公开工具名称。

### 6.5 `perceive_scene`

1. 作为普通 ReAct 工具，不固定为流程第一步。
2. 修复 schema 与 prompt 的矛盾：bbox 允许为空，非空时必须是合法归一化 XYXY。
3. 保留整图场景描述、实体、关系、文字布局角色和不确定性。
4. 只记录像素可见内容，不从外观猜人物身份、来源或真假。
5. 返回的关系和文本区域进入紧凑视觉记忆，但不生成最终标签。

### 6.6 `ocr_with_position`

1. 保留 Baidu/EasyOCR 两个明确 backend，不做静默切换。
2. 只有合法位置和达到配置置信度的文本进入 canonical text regions。
3. rejected regions 进入诊断字段，不进入事实记忆。
4. 保留 OCR 输入尺寸、压缩方式、backend 请求和耗时。
5. OCR 结果必须通过同一消息链路进入下一轮模型，不能只写状态快照。

### 6.7 `focused_visual_inspection`

1. 每次请求都发送完整原图和必要的局部视图。
2. 原图和局部图只属于当前请求，不重复追加到历史文本。
3. 保留 `view_index`、区域和回答状态。
4. Reflection 或最终 Judgment 需要看图时，也使用同一原图附加机制。
5. 视觉观察只回答具体可见问题，不替代网页事实核查。

### 6.8 `crop_and_inspect`

1. 只做“裁剪 + 视觉观察”，不自动搜索、不访问网页。
2. 每次只接受一个合法 bbox 和一个具体问题。
3. 临时文件使用唯一文件名并保证 finally 清理，避免并发冲突。
4. 结果归入视觉 observation，不直接归入网页 Evidence。
5. 修正 bbox schema、绝对像素/归一化坐标约定和测试契约。

### 6.9 `crop_and_search`

当前是隐藏复合 Agent：

```text
crop -> upload -> Lens -> VLM query -> image search -> visit -> extraction
```

本次从主动 Agent 工具集合移除。原有实现不删除，但只能作为显式内部诊断或后续拆分的参考；主 Agent 必须分别调用：

```text
crop_and_inspect
   -> reverse_image_search（Lens 或 semantic 分支）
  -> visit
  -> compare_with_reference（如有候选图）
```

### 6.10 `compare_with_reference`

1. 只接受前面图搜产生的候选图片。
2. 下载后必须验证 HTTP 内容类型、图片头和实际解码。
3. 登录页、验证码页、SVG 页面或 HTML 不能作为成功参考图。
4. 无效候选返回 `invalid_reference` / `external_unavailable`，不生成比较证据。
5. 保留当前已有的确定性 exact match、URL fallback、redirect policy 和缓存诊断。
6. `edit_evidence_present` 与 strength 继续由 differences 确定性派生，避免模型重复输出造成字段矛盾。
7. 比较结果是视觉比较观察，不自动等于事实核查结论。

### 6.11 `check_consistency`

1. 只检查调用方指定的一个可见方面或关系。
2. 结果必须描述具体位置、观察和不确定性。
3. 不能因为“没有发现异常”就支持图片真实。
4. 物理世界不合理与图像像素篡改分开表达。
5. 作为诊断观察进入下一轮，不能绕过 ReAct 直接改变 verdict。

### 6.12 `analyze_visual_anomalies`

1. 保留异常列表、位置、类型、严重度和目标关系状态。
2. 只报告清晰、可定位、被像素支持的异常。
3. 不把 `likely_ai`、`likely_manipulated` 作为直接最终结论。
4. 清洁扫描不能作为真实性正证据。
5. 与 `check_consistency` 的职责在公开描述中分开，避免模型把两个工具当成同一个动作。

### 6.13 `count_objects`

1. 保留为纯视觉计数观察。
2. 将成功结果接入 canonical observation/reducer，避免“工具成功但状态没有落点”。
3. 数量、置信度和位置必须经过严格校验。
4. 不能单独产生事实结论或停止调查。

### 6.14 `current_time`

保留实现，不放入当前图像事实核查的主动工具 schema；未来有明确时间截点字段时再启用。

## 7. 状态与上下文

主 ReAct 不再让模型维护一个可变 Claim/Hypothesis 图。模型每轮读取：

```text
任务要求
原图
最近一次工具结果
最近的有效观察和搜索候选
已访问/已尝试记录
剩余预算
```

状态只保存：

- 原图视觉记忆；
- 搜索候选；
- 已访问 URL；
- 工具观察；
- 外部不可用和工程错误；
- 当前动作历史；
- 剩余动作预算。

Replan 不再是独立工具。模型在下一次 `<think>` 中根据刚刚收到的结果选择新的 query、页面、视觉检查或结束动作。

## 8. 测试与验收

### 8.1 单元测试

必须新增或更新：

1. `run()` 使用唯一主动 ReAct 入口。
2. 第二次 Gemini 请求的 `previous_interaction_id` 正确继承。
3. 第二次请求包含上一次 `function_result`。
4. 工具结果只进入一次文本上下文，不产生重复累计状态。
5. 原图在后续请求重新附加，但不会写入历史文本。
6. `reverse_image_search` 候选图确实进入下一轮多模态输入。
7. reducer 和模型使用同一 canonical payload。
8. `text_search` 不触发隐藏 visit。
9. `crop_and_search` 不出现在主动工具 schema。
10. 所有工具的 success/error、外部不可用、工程错误语义一致。
11. bbox、OCR、比较图解码和临时文件清理契约通过。
12. 所有 provider session 在 episode 完成或失败后关闭，不增长 CLOSE-WAIT。

### 8.2 静态门禁

```text
pytest -q
python -m compileall src
git diff --check
```

另外检查：

- 主流程不再调用旧 `_run_react_runtime_policy`；
- 主 Agent prompt 不再出现旧 `text_search` / `reverse_image_search` 公开名字；
- 旧 v4 Planning / Claim / route-local-replan 不进入新的运行时工具 schema；
- 没有工具把完整 raw HTML、base64 或重复原图写进状态文本。

### 8.3 真实 smoke

代码门禁通过后：

1. 用真实 Gemini 跑 10 条。
2. 检查每条轨迹是否出现：
   - 工具调用；
   - 紧接着的工具结果；
   - 下一轮模型读取工具结果；
   - 原图重新附加但上下文不膨胀；
   - 图搜候选图真正进入模型输入。
3. 检查工程错误、外部不可用、重复访问和工具调用数量。
4. 跑最新统一 private-gold 审计。
5. 分析轨迹，不立即启动 8,490 条教师 rollout。

## 9. 实施顺序

1. 新增 canonical observation 与消息链路测试。
2. 切换 `run()` 到唯一连续 ReAct runtime。
3. 保持 `text_search`、`reverse_image_search` 等现有公开名称。
4. 清楚记录 `reverse_image_search` 的 Lens/semantic 分支，避免隐藏重复动作。
5. 从主动工具集合移除 `crop_and_search`。
6. 接入候选图片的多模态回传。
7. 修正各视觉工具 schema、错误状态和 reducer 落点。
8. 统一工具生命周期、缓存和连接关闭。
9. 更新英文运行时 prompt、schema、训练适配器和中文说明文档。
10. 跑完整本地测试。
11. 提交 GitHub 并更新 gpu13 checkout。
12. 真实跑 10 条 smoke 和 private-gold 审计。
13. 在大规模教师 rollout 启动前停止，等待复核。

## 10. 预期结果

完成后，一条正常轨迹应满足：

```text
原图
  -> Agent 选择 perceive_scene / OCR / text_search /
     reverse_image_search / visit / 视觉工具
  -> 工具结果真实进入下一轮
  -> Agent 根据结果继续或换路线
  -> 必要时重新看原图
  -> finish_investigation
  -> 最终 Judgment
```

不再出现：

- 工具实际成功但模型下一轮看不到结果；
- reducer 消费了模型没有看到的隐藏结果；
- 一个工具内部偷偷完成多次搜索、访问和抽取；
- 图搜有候选但候选图片没有进入模型；
- `text_search` / `reverse_image_search` 与参考 Agent 的同类工具使用不同公开名称；
- bbox、比较字段或工具状态互相矛盾；
- 旧 Claim/Hypothesis 状态反复膨胀上下文。

## 11. 最终验收记录

- 本地当前 unified-react 相关门禁通过；服务器当前 checkout 定向门禁 103 项通过。
- 真实 Gemini 3.7 smoke：10/10 完成，0 工程错误，strict trace audit 10/10。
- 当前提交 `62f6b9c` 只补充 claimless unified-react canary 验收，不改变已 smoke 的
  ReAct runtime、工具、prompt 或导出逻辑；服务器已用当前 checkout 复核既有 smoke。
- `trajectory_sft.jsonl` 保留完整 episode 的 canonical 导出；缺少 provider thought 的
  轨迹进入 `action_only`，不伪造 `<think>`；超出 128K 的轨迹进入 holdout。
- 新版 10 条 smoke 未达到整体质量改善门槛，因此不启动新的 Agent-100，不启动
  8,490 条大规模教师 rollout。

## 12. 2026-09-02 续做状态

本轮实现、部署和真实 smoke 已完成，当前停在人工作质量复核边界：

- Jina 页面提取增加 bounded evidence context，保留完整来源段落和必要前置上下文；
- StageRunner 对 canonical tool result 做结构化裁剪，移除二进制/HTML 和历史 Claim 控制字段，
  不再用 JSON 字符串前缀冒充工具观察；
- unified ReAct 接收多页面 evidence、拒绝空 `success` payload 和无效参考图 Evidence；
- compare 结果增加 `comparison_status`，参考图片先做真实栅格解码校验；
- perception entity attributes 使用固定 schema，避免 Gemini schema 400；
- `src.orchestrator` 对活动 ReAct 状态延迟加载，避免工具导入循环；
- 新增回归测试后，服务器全量 `468 passed`，`compileall`、`git diff --check` 通过。

运行时代码提交为 `55ef0b3`， canary 验收提交为 `62f6b9c`，当前文档收尾提交为
`10bb1e1`；gpu-13 已同步且工作树干净。3.1 Pro 与 3.7 Flash 各完成 10 条真实
smoke，工具结果前递、页面证据、空成功结果、候选图输入、工程错误和轨迹产物均已记录。

如果 Gemini 3.7 当前不可用，不再把新的 3.7 smoke 当作文档完成条件；只暂停依赖
3.7 的新运行。质量问题进入人工复核清单，大规模教师 rollout 仍不启动。
