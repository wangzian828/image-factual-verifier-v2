# unified ReAct / SFT judge / 训练集 100 条四轮重跑总计划

更新时间：2026-09-02  
当前分支：`main`
当前基线提交：`a398578`
当前本地 checkout：`C:\Users\wangza\ifv-gpu13-canary-20260804-01`  
明确停止点：本计划全部完成后，停在启动全量 8,490 条教师 rollout 之前。

## 1. 范围和不变约束

1. 当前生产主流程是 `unified-react-v1` 的单一连续 ReAct loop：

   ```text
   原图 + 固定任务
     -> thought
     -> 一个公开工具调用
     -> 完整工具结果
     -> 下一轮 thought / action
     -> final Judgment + fact-check report
   ```

2. 不恢复旧 v3/v4 主流程，不为旧路径重新堆兼容层。旧代码和旧测试只用于历史
   回放时单独识别，不得影响当前 Agent、SFT judge 或新训练数据导出。
3. 成熟工具内部实现保持不动；只修改当前 runtime 的接入、公开 schema、主 Agent
   prompt、SFT judge 输入投影和必要的测试/文档。
4. 测试集和训练集严格隔离：

   - 统一训练集：`/gsdata/home/wza/image-factual-verifier-v2-data/datasets/route-aware-hrc-stage2-10563-final-organized-20260824-r2/unified-dataset/train-manifest.jsonl`
   - 训练集总量：8,490；
   - 测试集：1,684，禁止进入本次训练集 rollout；
   - 本次验证用 100 条只用于实验和统计，不进入正式训练集。

5. `private-gold judge`、`SFT eligibility judge` 和 `semantic reward` 是三套不同用途
   的流程，报告、缓存和指标不得混称。
6. 外部 SSL、验证码、页面不可访问、图片下载失败和 429 等按现有规则记录为
   `external_unavailable`；schema、工具契约、非法参数和启动错误才是工程错误。
7. `AI 痕迹、画质、乱码、风格异常、没有搜到来源` 不能单独支持 `fake`；独立成立
   的图片事实矛盾、关系矛盾或有效来源正文可以作为事实依据。

## 2. 当前基线核对结果

### 2.1 当前主流程

当前入口：

```text
src/orchestrator/pipeline.py::Orchestrator.run()
  -> _run_react_runtime_policy()
  -> src/orchestrator/react_runtime.py::UnifiedReactState
  -> StageRunner + Gemini native Interaction
  -> _run_react_judgment()
```

当前 runtime 状态只保留有限的：

- `objective`
- `visual_memory`
- `discoveries`
- `evidence`
- `failures`
- `attempted_queries`
- `visited_urls`
- `attempted_actions`
- `recent_actions`
- `open_questions`
- `current_focus`
- `investigation_progress`
- `action_count`
- `stop_reason`

不把 `target_facts`、`search_hypotheses`、`claim_assessments` 或旧 graph reducer
重新引入当前主流程。

### 2.2 文搜图工具现状

代码核对过的接入点：

- 实现：`src/tools/text_image_search.py`
- Serper client：`src/integrations/search/serper.py::SerperImageSearchClient`
- 注册：`src/orchestrator/tool_registry.py`
- ReAct 工具列表和预算：`src/orchestrator/react_runtime.py`
- pipeline 工具预算和缓存集合：`src/orchestrator/pipeline.py`
- 当前已有单测：`test_text_image_search.py`、`test_raw_history_audit.py`

因此工具本身已经注册并可执行，但主 Agent prompt 之前没有单独说明：

- 它是“文字 query -> 图片候选”的 Serper 图像搜索；
- 适用于已知专名、事件、地点、人物、物体或图片文字，需要寻找对应图片/页面时；
- 返回的是未验证 Discovery；
- 不能把候选图直接当作同图、同源或事实证据；
- 下一步应由 Agent 选择 `visit` 或 `compare_with_reference` 继续验证。

### 2.3 最近 16 条真实 trace 的核对

来源：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-image-observation-smoke-20260903-p10-0a76001
```

已核对结果：

- 16/16 条 trace 完成；
- `text_image_search`：4/16 条实际使用；
- 共 6 次调用，每次最多 3 个候选，共 18 个候选图结果；
- 工具总调用中还出现了 `reverse_image_search`、`text_search`、`visit`、比较和视觉工具；
- 不是工具未注册或未执行；
- 4 条使用文搜图的轨迹中，工具结果进入了下一次 Gemini Interaction；
- 下一请求带 `parent_interaction_id`，上一轮完整 function result 在输入 payload 中；
- 候选图片作为受控多模态输入追加，最多 3 张；
- `text_image_search` 当前未自动升级为 Evidence，符合证据边界。

当前发现的真正问题是 prompt 引导不足，而不是 Serper 工具失效。此次修复后仍需
重新做真实 smoke，确认模型在有明确图片事实线索时能正确选择该工具。

## 3. 100 条 rollout 前必须完成的工作

### A. 固定代码和运行边界

- [x] 确认本地分支、工作树、基线 commit；
- [x] 确认服务器 checkout 是同一分支，且只接受 GitHub commit；
- [x] 确认没有正在运行的旧 Agent/rollout 进程污染新实验；
- [x] 确认没有固定的 `GEMINI_EVAL_MAX_CONCURRENCY=4` 等环境变量偷偷覆盖命令行并发；
- [x] 确认当前唯一训练输入是统一数据集的 `train-manifest.jsonl`；
- [x] 固定本次验证 100 条 case ID，写入不可变 preparation artifact；
- [x] 在本阶段完成前不启动全量 8,490 条教师 rollout。

### B. 正确接入并说明文搜图

- [x] 保持公开工具名 `text_image_search` 不变；
- [x] 检查其公开 schema 只暴露 query、语言和地区参数，不暴露内部字段；
- [x] 检查 Serper 请求前的 source policy 黑名单拦截；
- [x] 检查空结果明确为 `empty_results`，不伪装为成功证据；
- [x] 检查候选图和候选页面进入 Discovery，不自动进入 Evidence；
- [x] 在英文 Unified ReAct prompt 中增加独立的文搜图规则和使用时机；
- [x] 同步更新英文 prompt 备份、中文阅读版、Agent 结构文档和工具清单；
- [x] 加单测覆盖：可用工具列表、schema、query 校验、候选 Discovery、空结果、
  下一轮候选图注入和禁止直接当 Evidence。

### C. 修复 SFT judge 与新版 unified ReAct 的适配

主要文件：

- `src/trajectory/sft_eligibility.py`
- `src/eval/score_sft_eligibility.py`
- 相关 schema、缓存版本和测试

输入投影必须让 judge 看到“有限但完整”的真实调查过程：

- ReAct accepted action 的时间顺序；
- 每个动作的工具名和公开参数；
- 每个工具的完整文本观察，去掉图片二进制、base64、raw HTML 和内部请求快照；
- `text_search` 和 `text_image_search` 的 query、结果数量、标题、摘要、候选 URL/图片 URL；
- `visit` 的正文证据段和上下文段；
- `reverse_image_search` 的未验证候选，不升级为 Evidence；
- compare、OCR、perceive、crop、consistency 等实际观察；
- runtime 的 state delta、调查进度、动作预算和 stop reason；
- rejected policy turn 的有限结构化记录；
- final report 仅作为待核对摘要，不能创造 Evidence。

投影规则：

1. 新版 trace 没有 `target_facts` 时，不报字段缺失；
2. `visual_memory`、`discoveries`、`evidence` 和 action history 分开表达；
3. Discovery、Evidence、背景信息、外部失败、工程错误分开表达；
4. 不能因为 judge packet 漏掉过程而制造
   `poor_retrieval_quality`、`no_decisive_evidence` 或 `major_overclaiming`；
5. 仍只允许 judge 引用 packet 中真实存在的 Evidence ID；
6. `AI 痕迹/乱码/画质/未找到来源` 单独不能成为 fake 依据；
7. 图内独立事实或关系矛盾可作为有效依据；
8. SFT eligibility 仍是独立 LLM judge，不与 semantic reward 合并；
9. 导出后的 `high/usable/rejected/engineering_error` 仍由现有确定性分桶负责，
   不新增第二个隐式 LLM 判定。

### D. 回归、真实 smoke 和轨迹检查

- [x] 补充新版 ReAct、SFT judge、文搜图和工具结果前递相关单测；
- [x] 跑相关 pytest；
- [x] 跑 `compileall` 和 `git diff --check`；
- [x] 提交前检查所有测试是否使用当前主流程，旧 v3/v4 残留失败单独记录；
- [x] 提交并 push 后更新服务器 checkout；
- [x] 服务器先跑小规模真实 smoke；
- [x] 如果 Gemini 3.7 可用，使用 3.7 验证；
- [x] 如果 3.7 完全不可用，可用 `gemini-3.1-pro-preview` 做工程和调查链路 smoke，
  但不得把它的质量结果写成 3.7 结果；
- [x] 检查每条新 trace 是否：
  - 只产生一个动作；
  - 真的读取上一轮工具结果；
  - 文搜图结果有候选内容时能看到候选；
  - 空 `status=success` 不被当作有效搜索结果；
  - 不把外部访问失败当工程错误；
  - report 不创造未记录事实；
  - 最终 Judgment 能读到有效 Evidence 和调查过程。

### E. 文档和实验记录

必须同步更新：

- `docs/plans/2026-09-02-sft-judge-unified-react-training100-reroll-plan.md`
- `docs/agent-structure.md`
- `docs/agent-prompt-and-runtime-guide.md`
- `docs/agent-prompt-and-runtime-guide-zh.md`
- `docs/active-agent-system-prompts.md`
- `docs/active-agent-system-prompts-zh.md`
- `docs/sft-training-and-data-construction.md`
- `docs/trajectory-artifact-guide.md`
- 本次真实 smoke 和 100 条实验记录

文档必须明确：

- 当前主流程没有独立 Planning/Replan/Reflection 请求；
- `perceive_scene`、`ocr_with_position`、`text_search`、`text_image_search`、
  `reverse_image_search`、`visit`、视觉工具和 `finish_investigation` 都是同一个
  ReAct loop 的动作；
- 文搜图是文字到图片候选，不是反向搜图，也不是证据；
- 原图通过 Interaction 会话复用，不在历史文本中重复叠加；
- 每轮只把上一轮完整工具观察送入下一轮；
- SFT 导出和 judge 如何读取新版 trace；
- 所有 Agent 改动、prompt 版本、schema 版本、测试结果和 commit。

## 4. 训练集 100 条四轮 rollout

只有第 3 节全部通过后才执行。

### 4.1 四轮定义

“roll 四轮”是初始轮加最多三轮串行 reroll，不是四个候选并行生成：

```text
第 1 轮：100 个 case 各 rollout 1 条
       -> 本轮全部完成后跑 SFT eligibility judge
       -> 通过者冻结为当前最佳候选
       -> 未通过者进入第 2 轮

第 2 轮：只 rollout 第 1 轮未通过者
       -> 本轮 judge
       -> 新通过者冻结；仍未通过者进入第 3 轮

第 3 轮：只 rollout 第 2 轮未通过者
       -> 本轮 judge
       -> 新通过者冻结；仍未通过者进入第 4 轮

第 4 轮：只 rollout 第 3 轮未通过者
       -> 本轮 judge
       -> 新通过者冻结；仍未通过者标记 hard case
```

补充约束：

- 每轮必须先完成 rollout，再启动该轮 SFT judge；
- 每轮只把未通过 case 放入下一轮；
- 工程/Provider 失败按现有自动重试队列处理，不当作通过；
- 每个 case 的全部轮次和 judge 结果都保留；
- 最终每个 case 只选择一个最佳通过候选；
- 四轮都没通过的 case 只进入 hard case，不进入正式训练候选；
- 这 100 条不进入正式训练数据；
- 不启动全量 8,490 条教师 rollout。

### 4.2 每轮固定产物

每轮保存：

- rollout run manifest；
- 本轮 case list；
- canonical trace；
- provider/工程失败和重试记录；
- SFT eligibility judge artifact；
- `sft_eligibility_summary.json`；
- 本轮通过/未通过 case list；
- 下一轮 case list；
- candidate selection；
- 模型、commit、并发、超时、thinking 配置；
- 运行日志和路径索引。

## 5. 最终统计和逐条检查

### 5.1 四轮统计

- 初始轮通过数；
- 第 2、3、4 轮新增通过数；
- 四轮均未通过的 hard case 数；
- 工程错误和外部失败数；
- 每轮平均动作数、工具分布和实际耗时；
- `text_image_search` 使用率、有效候选率和后续验证率。

### 5.2 SFT judge 原因分布

统计并抽样核对：

- `different_image_fact`
- `major_overclaiming`
- `poor_retrieval_quality`
- `no_decisive_evidence`
- `trajectory_conduct_*`
- `engineering_error`
- 外部不可访问
- 其它 warning

每个原因都要检查是真实轨迹问题，还是 Agent -> judge packet 投影缺失。

### 5.3 导出质量桶

按现有导出器确定性分桶：

- `high`
- `usable`
- `rejected`
- `engineering_error`

同时记录：

- case 数；
- 完整 episode 数；
- `trajectory_sft` 数；
- `action_only` 数；
- `rl_candidate` 数；
- 32K/128K 长度分布；
- 超过 128K 的 holdout 数。

### 5.4 每条轨迹检查

至少检查：

- 通过的高质量轨迹；
- 通过但被分到 usable 的轨迹；
- 四轮仍拒绝的 hard case；
- 工程错误；
- 最短和最长轨迹；
- 有明显证据但被 judge 拒绝的轨迹；
- judge 认为充分但疑似 overclaiming 的轨迹。

重点看：

- 是否围绕原图事实调查；
- 是否真正读取上一轮工具结果；
- 是否会在工具返回空结果时继续走具体替代路线；
- 是否把文搜图/反向搜图候选误当事实；
- 是否出现无证据套话或莫名早停；
- 是否把 AI 痕迹、画质、乱码当成 fake 依据；
- 是否让 report 创造了工具没有给出的事实；
- 是否最终二分类只是现有收尾，而不是伪造新的调查阶段。

## 6. 提交、部署和停止条件

提交前：

- [x] 所有源代码、测试、prompt 备份、文档和实验记录纳入 commit；
- [x] `git diff --check` 通过；
- [x] 相关 pytest 通过；
- [x] 记录最终 commit 和文件变更；
- [x] push 到当前 canary branch。

服务器：

- [x] 只 fast-forward 到已验证 commit；
- [x] 核对 branch、status、commit、hostname、账号和 `OMP_NUM_THREADS=1`；
- [x] 真实 smoke 只从服务器 checkout 运行；
- [x] 记录 run path、模型、并发、prompt/schema 版本；
- [x] 监控当前 worker 的 CLOSE-WAIT 是否随 trace 无界增长；
- [x] 不处理不属于本次 worker 的历史/他人进程。

最终停止条件：

1. 文搜图接入和 prompt 已验证；
2. SFT judge 已适配新版 unified ReAct；
3. 单测、本地检查和服务器 smoke 已完成；
4. 训练集 100 条已完成初始轮加最多三轮 reroll；
5. 每轮 SFT judge、最终分桶、路径和失败分析已记录；
6. 所有 Agent 改动和 prompt/schema 版本已记录；
7. 不启动全量 8,490 条教师 rollout，只保留可复核的启动方案。

## 7. 本轮实际执行状态

- [x] 当前分支、服务器 checkout、数据集边界和进程状态已核对；
- [x] `text_image_search` 的 schema、Serper 前置黑名单、空结果、
  Discovery 前递和 prompt 规则已完成验证；
- [x] 新版 3.7 smoke、source-policy strict audit 和工具结果前递检查已完成；
- [x] 训练集 100 条 preparation 已创建，runtime/private 各 100 条，gold 全齐；
- [x] 100 条初始 rollout 全部完成，100 条均有 terminal success；
- [x] 初始轮 SFT judge：56 条通过，44 条进入质量 reroll；
- [x] quality-reroll-01：44 条中新增通过 6 条，剩余 38 条；
- [x] quality-reroll-02：38 条中新增通过 5 条，剩余 33 条；
- [x] quality-reroll-03：33 条中新增通过 1 条，剩余 32 条 hard case；
- [x] 最终选中 68 条，hard case 32 条，未解决工程错误 0 条；
- [x] 所有轮次的 trace、judge、分类和失败记录均已保留；
- [x] 发现并修复 package 导出耦合：action-only 不进入 reasoning policy SFT，
  但其有效 perception 样本仍进入独立 perception SFT；
- [x] 默认回归测试 481 passed，`compileall` 和 `git diff --check` 通过；
- [x] 在上述依赖完成前不启动全量 8,490 条教师 rollout。

本轮真实记录：

- `docs/reports/2026-09-02-unified-react-smoke10-718ac2f.md`
- `docs/reports/2026-09-02-training100-fourround-reroll.md`

训练 100 条 preparation：

`/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/unified-react-v1-gemini37-train100-fourround-20260902-2e64147/`

本轮 SFT 训练包的口径为：

- accepted release：68 条；
- reasoning policy SFT：67 条；
- action-only：1 条，单独保留，不混入 reasoning SFT；
- independent perception SFT：68 条；
- 质量目录：high 65、usable 3；
- 100 条实验数据不进入正式训练。

旧的 `classification/quality-reroll-01/classification.json` 保留了修复前的
`incomplete_case_count=56` 历史字段；它不影响最终选择，但已在新的复核产物中
明确标记为 superseded，后续只使用按本轮 target-case-list 重算的结果。

当前代码提交：`a398578`，已 push GitHub 并 fast-forward 到 gpu-13。
修正版训练包：

`/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/unified-react-v1-gemini37-train100-fourround-20260902-2e64147/quality-reroll-training-package-r2/`

历史分类复核：

`/gsdata/home/wza/image-factual-verifier-v2-data/generated/teacher-rollouts/unified-react-v1-gemini37-train100-fourround-20260902-2e64147/classification/recomputed-a398578/recompute-manifest.json`
