# 2026-09-04 可移植交付真实 Smoke（10 条）

## 范围

- 代码提交：`414cb07`
- 服务器：`gpu-13 / wza`
- 模型：`gemini-3.7-flash`
- 运行输入：
  `/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/unified-react-v1-gemini37-train10-sftformat-20260904-6f2de3f/runtime-release/runtime_input/cases.jsonl`
- 初始运行：
  `/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/portable-handoff-smoke10-20260904-414cb07`
- 工程补跑：
  `/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/portable-handoff-smoke10-retry-main02730-20260904-414cb07`

本次只验证可移植启动入口、真实 Agent 调查、SFT judge、SFT 导出和 Qwen processor。
没有启动 8,490 条全量教师 rollout，也没有修改训练/测试数据集。

## 启动与工程稳定性

新的 `scripts/server/start_teacher_rollout.sh` 将教师轨迹输入明确路由到
`src.eval.run_cases`。因此它不要求测试评测 release 才有的 `evaluation_gold`。

- 初始 10 条：9 条成功，1 条在第 7 次 Gemini 请求发生瞬时
  `SSLError: [SSL] record layer failure`。
- `main-02730` 单条续跑时复用了初始运行已有的成功工具结果，随后成功完成。
- 最终：10/10 有完整 terminal trace，10/10 strict trace audit 通过。
- 运行均值：9.4 个 ReAct 动作、10.4 次工具调用、16.3 次 LLM 请求、246.12 秒/条。
- 没有 protocol rejection、scheduler rejection、route-control rejection 或工具参数格式错误。
- 运行期间 worker 的 `CLOSE-WAIT=0`；运行结束后服务器总数也为 0。

唯一 SSL 失败属于代理/上游 TLS 链路的外部瞬时错误，不是 key、模型、Agent 逻辑或 socket
泄漏。补跑完成后无遗留 rollout/judge 进程。

## 工具观察与图片上下文

10 条完成轨迹共调用：

- `text_search`：28；
- `reverse_image_search`：12；
- `crop_and_inspect`：20；
- `visit`：11；
- `text_image_search`：3；
- `check_consistency`：6；
- `analyze_visual_anomalies`：4；
- `compare_with_reference`：2。

所有已调用工具都向下一轮传入完整 JSON 文本结果；本批没有只返回
`{"status":"success"}` 的空观察，也没有空 tool result。

`text_image_search` 的 Serper 调用均真实返回三个候选图和候选页面。真实注入结果分两类：

- `main-02730`、`main-02731`：候选图已下载并作为图片附在紧邻的下一轮请求；
  下一轮分别携带 2、3 张新图片。
- `main-02735`：文本结果已进入下一轮，但三张 Instagram `lookaside` 候选图没有进入
  请求媒体。当前实现会在候选图下载/验图失败时静默省略图片，只保留文本候选；这不是
  `text_image_search` 未启用，但缺少可观测的候选图片下载失败记录，后续应单独修复。

初始原图继续通过同一 Gemini Interaction session 保留在 provider 历史中，不会在每轮重复
上传。当前请求的 `image_count=0` 仅表示该轮没有额外附图，不表示原图或前一轮完整工具结果
从 session 中丢失。

## 私有 Gold 对照与范围漂移

本批标签正确为 3/10：`main-02728`、`main-02731`、`main-02732`。这只是固定小样本，
不能外推整体准确率。

旧版 v6 frozen SFT judge 将下列 6 条标记为 `different_image_fact`。该标签把
“真正调查无关事实”和“调查相关、但漏掉 target 关键条件或错误解释证据”混在了一起；
下表仅保留为旧版审计记录：

| Case | Agent 实际核查对象 | private target 中应核查的关键事实 |
| --- | --- | --- |
| `main-02729` | 花园里石盆/底图是否真实 | 石盆是否无支撑悬浮 |
| `main-02730` | 图中铁路是否为伪造的已完工工程 | Corcoran grade separation 是否已经完工 |
| `main-02733` | 伊朗驻南非使馆是否发过配文 | 视频中的军人葬礼男孩画面是否为真实事件 |
| `main-02734` | 海地 2010 年救援底图是否真实 | 叠字“哥伦比亚地震救出男孩”的归因是否真实 |
| `main-02735` | 狗被埋住的底图是否为真实救援 | “2026 哥伦比亚地震救援”的传播归因是否真实 |
| `main-02736` | 摩洛哥警察和水炮场景是否存在 | “2026-07-15 Ceuta 第二波”的具体传播语境是否真实 |

最后两项在旧版 judge 中也属于 `different_image_fact`。

### 2026-09-04 v7 target-scope 重审

在提交 `cc1ac5a` 与 `76bc97f` 后，使用同一 private gold、同一批完整 trace
（`main-02730` 使用其成功的 SSL 补跑 trace）对 10 条进行一次全新的 Gemini 3.7 Flash
SFT judge 调用。新结果单独保存，不覆盖旧 v6 artifact：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/
portable-handoff-smoke10-20260904-414cb07/sft-eligibility/v7-target-scope/
```

通过数仍为 3/10，说明这次改动没有放宽 SFT 准入；它只纠正 scope 的诊断口径。

| Case | v7 target scope | 仍被拒绝的核心原因 |
| --- | --- | --- |
| `main-02729` | `direct_target` | 直接调查石盆是否悬浮，但忽略视觉一致性工具指出的无支撑悬浮，错误判为 real |
| `main-02730` | `direct_target` | 直接调查 Corcoran grade separation；轨迹判 fake、private gold 为 real，属于 verdict/gold 证据解释冲突，不是查错对象 |
| `main-02733` | `direct_target` | 调查帖子和视频真实性，但没有证实底层视频画面真实，证据不足且错误判 real |
| `main-02734` | `decisive_subfact` | 找到 Haiti 2010 原图，足以反驳 Colombia 2026 归因；但最终把“底图真实”外推成 real |
| `main-02735` | `direct_target` | 直接调查救援图片和 Colombia 归因，但未找到 Brazil 原始出处，错误判 real |
| `main-02736` | `related_but_incomplete` | 核到摩洛哥水炮的一般事件，却未核具体日期、“第二波”与 Ceuta 传播归因 |

因此，旧版 6 个 `different_image_fact` 中没有一个在 v7 被判为
`unrelated_fact`。`main-02736` 是“相关但漏关键条件”；`main-02734` 是可决定
target 的子事实；其余四条均是直接调查 target。最终是否可进入 SFT 仍由 verdict
正确性、decision support、decisive Evidence、retrieval quality 和行为质量共同决定。

根因不在最终 Judgment 的新造结论，也不是 reducer 改写了一个已有 target：

1. runtime `cases.jsonl` 只有 `case_id`、`image_path`、`image_sha256`，不含传播
   claim 或 private target；
2. `UnifiedReactState.objective` 是泛化的“核查图片表达的事实并输出 real/fake”；
3. 第一轮 ReAct 自己通过搜索目标和 `investigation_progress.basis` 选择“图片中的哪一层
   事实”作为调查范围；
4. final Judgment 继承完整 Interaction 历史和同一泛化 objective，只把已经选定的范围写成
   `claim_under_review`；
5. private gold 只在 trace 结束后由 SFT judge 读取，因此只能拒绝错误范围，不能在运行时
   纠偏。

`main-02733` 最具代表性：模型从首轮开始调查“@IraninSA 是否发布帖文”，中间尝试访问
“视频是否真实”的事实核查页但遇到 SSL 失败，之后仍回到帖文真实性；最终 report 因而忠实地
写成“使馆是否发帖”，而不是 private target 所需的“视频是否为合成”。这说明当前问题是
**输入任务没有提供可冻结的传播断言，而模型在多层图片内容之间自行选了一个可验证子问题**。

这与“纯图片开放调查”可以并存，但它不能直接等同于当前数据集中每条 private target 的
二元标签。后续结构调整应先决定：产品/训练任务是评估开放式 image account，还是评估图片
承载的特定传播断言；在未决定前，不应把本批 3/10 归因成单纯的检索能力差或靠更严 prompt
强行修正。

## SFT judge 与导出

冻结 SFT judge 使用相同 private gold：

- 初始 10 条通过：3 条；
- 工程补跑的 `main-02730`：未通过；
- 通过并进入 accepted release：`main-02728`、`main-02731`、`main-02732`；
- v7 重审仍只通过上述 3 条；其余拒绝主要来自 verdict 错误、证据不足、过度外推、
  retrieval 质量或未核关键 target 条件，而非“查了无关事实”；
- 这一拒绝发生在 rollout 后，不会把 private target 写入模型可见轨迹。

训练包：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/
portable-handoff-smoke10-20260904-414cb07/sft-training-package/
```

- policy reasoning：3 条；
- perception：3 条；
- action-only：0；
- long holdout：0；
- policy/perception 严格 JSON 审计：通过。

真实 Qwen3.5/ms-swift processor：

```text
audits/processor-verification-qwen35-414cb07.json
```

- `passed=true`；
- 37 个 `<think>` 目标位于训练 labels；
- 37 个 tool call、37 个 tool response 均保留；
- 三条 policy 行实际输入 token 为 26,083、26,188、28,064；
- 最大值 28,064，即约 28K，占 128K 上限约 21.4%，没有已接收训练行超限。

trajectory catalog 中另有 6 条 `over_128k_estimate`。它们全部已被 SFT judge 拒绝，
只保留为 audit-only 原始轨迹，不进入上述三条训练行。该标记来自 exporter 的保守
`UTF-8 bytes / 3` 估算，不是 Qwen processor 的实测值；本批对应原始 JSON 约
0.48–1.19 MB，估算约 161K–398K。若未来有这类轨迹通过质量 gate，必须在接收前跑真实
processor；当前没有把它们混进训练。

## 2026-09-04 四轮质量重跑

针对上述 10 条 case，按“首轮完成后，仅对未通过 SFT 的 case 逐轮重跑”的规则完成了
四个候选阶段。首轮使用既有完整 trace 作为候选 1；后面三轮使用提交
`99bd323` 的当前 Agent 代码。

运行目录：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/
portable-handoff-smoke10-20260904-414cb07-four-candidate-reroll-20260904-99bd323/
```

每轮结果：

| 阶段 | 本轮候选数 | SFT 通过 | 继续重跑 |
| --- | ---: | ---: | ---: |
| 首轮候选 1 | 10 | 3 | 7 |
| quality-reroll-01 候选 2 | 7 | 1 | 6 |
| quality-reroll-02 候选 3 | 6 | 0 | 6 |
| quality-reroll-03 候选 4 | 6 | 0 | 0 |

总计 29 条候选全部正常完成，0 工程错误、0 未完成候选。最终结果：

- 选中可用轨迹：4/10；
- 四轮均未通过、标记为 hard case：6/10；
- 选中的 case：`main-02728`、`main-02731`、`main-02732`、
  `main-02733`；
- `main-02733` 在第一轮重跑中首次通过 SFT；
- 其余 6 个 case 在最多四个候选后仍未通过。

最终 release 与训练包：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/
portable-handoff-smoke10-20260904-414cb07-four-candidate-reroll-20260904-99bd323/
quality-reroll-release/

/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/
portable-handoff-smoke10-20260904-414cb07-four-candidate-reroll-20260904-99bd323/
quality-reroll-training-package/
```

训练包包含 4 条 policy、4 条 perception、0 条 action-only；所有轮次的原始 trace、
SFT artifact 和候选 provenance 均保留。运行结束后 autopilot、rollout、SFT judge
进程均已退出，服务器 `CLOSE-WAIT=0`。

## 停止点

已完成本次真实 smoke、结构审计、SFT judge、四轮质量重跑、SFT 导出和真实 processor 验证。全量
8,490 条教师 rollout 仍未启动，等待对“开放图片事实调查”和“传播 claim 核查”任务边界的
人工决定。
