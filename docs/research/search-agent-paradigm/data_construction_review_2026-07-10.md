# Benchmark 初案审计与主动视觉调查数据构造建议

日期：2026-07-10

审计对象：`benchmark构建方案(2).md`

> 本文是研究设计讨论，不替代项目的当前运行时契约。

## 1. 一句话结论

原方案中的**语义槽位、单变量扰动、三分类和 evidence requirement** 都值得保留；但“所有 real/fake 都用 AI 生图”与“训练 Agent 通过开放搜索核查真实世界事实”之间存在根本错位。

如果不改造，数据集更可能测到：

- Gemini 生图与图生图的生成器指纹；
- prompt 模板、字幕样式和图像质量差异；
- 某类假图的视觉异常；
- 教师知道标签后写出的事后合理化轨迹；

而不是测到：

- 搜索是否产生新的视觉观察任务；
- Agent 是否找到真实、独立、可访问的外部证据；
- 证据如何更新候选假设；
- 何时停止或正确输出 `unverifiable`。

建议把全合成部分重定位为：

> **具有完整世界状态和证据图的可控 Multimodal Investigation Gym**，主要用于 SFT、动作排序、RL、故障注入和反事实评测。

另建一套严格隔离的真实世界 benchmark，用于最终能力声明。

## 2. 原方案中应保留的部分

### 2.1 语义槽位

`subject / location / event / attribute / relation / time / text_in_image / uncertainty` 是很好的生成与标注中间层。它比直接从标签生成图片更容易控制变量，也便于构造 hypothesis state。

建议新增：

```json
{
  "claim_surface": "图片向用户明确传达的主张",
  "claim_atoms": [],
  "world_facts": [],
  "visible_cues": [],
  "hidden_facts": [],
  "decisive_atoms": [],
  "alternative_hypotheses": []
}
```

### 2.2 单变量反事实思想

“一次只改变一个关键槽位”很适合因果消融和成对比较，尤其可构造：

- 同一视觉背景、不同文字主张；
- 同一文字主张、不同地点线索；
- 同一事件、不同时间；
- 相同搜索状态下两个不同工具动作。

但必须区分：

1. **world-state intervention**：世界事实改变；
2. **claim intervention**：图片表达的主张改变；
3. **rendering intervention**：图片像素/布局改变；
4. **evidence intervention**：外部网页证据空间改变。

原方案目前主要改变 rendering prompt 中的槽位，但没有同步定义真实 world state 和可搜索 evidence，因此严格说还不是真正的事实反事实。

### 2.3 `real / fake / unverifiable`

三分类适合作为产品输出，但内部标签应拆开：

```text
claim_truth         = supported / refuted / unresolved
content_origin      = generated / edited / camera / unknown
context_alignment   = aligned / displaced / unknown
evidence_sufficiency= sufficient / insufficient / conflicting
```

最终 verdict 再由版本化规则映射。否则“AI 生成图片”会被误等同于 `fake`。

### 2.4 evidence requirement

`internal / external / provenance` 是重要维度，但不应互斥。一个样本可能同时需要：

```json
"evidence_requirements": [
  "visual_internal",
  "text_ocr",
  "open_web_fact",
  "image_provenance",
  "source_independence",
  "temporal_reasoning"
]
```

## 3. 根本问题一：没有显式 claim，单图真假不可判定

单张无配文图片通常不携带唯一命题。例如一张机场照片本身不能确定是在表达：

- “机场今天永久关闭”；
- “机场因天气短暂停运”；
- “这是机场宣传图”；
- “这是 AI 概念图”。

因此“图片传达的信息”必须有可观察载体，例如：

- 图内字幕/标题/公告文字；
- 截图中的账号、帖子和正文；
- 信息图/新闻卡片；
- 水印或来源标签；
- 明确的组合视觉命题。

建议将任务单元定义为：

> 给定一张自包含的视觉帖子/信息卡，核验其中**可从像素中恢复的显式或受控隐式 claim**。

数据中必须保存 `claim_surface` 和 `claim_atoms`，评测 Agent 是否正确从图中恢复 claim。不能让教师看到一个图内根本未表达的隐藏 prompt，然后把它当作“图片传达的信息”。

### 推荐 claim schema

```json
{
  "claim_id": "C1",
  "surface_source": {
    "type": "embedded_caption",
    "region": [0.05, 0.72, 0.95, 0.96]
  },
  "surface_text": "本周暴雨导致 X 大桥坍塌",
  "atoms": [
    {"id": "C1a", "predicate": "depicts", "object": "X 大桥"},
    {"id": "C1b", "predicate": "event", "object": "坍塌"},
    {"id": "C1c", "predicate": "time", "object": "本周"},
    {"id": "C1d", "predicate": "cause", "object": "暴雨"}
  ],
  "decisive_atoms": ["C1a", "C1b", "C1c"]
}
```

## 4. 根本问题二：全合成图片没有天然可搜索的现实 provenance

对真实世界搜索 Agent 来说，网页和反向搜图结果是环境的一部分。纯新生成图片通常：

- 没有真实首发来源；
- 不会被搜索引擎索引；
- RIS 无结果是数据生成方式决定的，而不是图片真假证据；
- 不存在可核查的历史传播时间线；
- 对“从未发生”的事件，开放网页中的缺席不能证明不存在。

这尤其破坏：EF1、EF2、EF3、EF4、CD2、CD3 等依赖开放域事实或来源溯源的类别。

解决办法不是放弃生图，而是为每个合成 case 同时生成一个**封闭、可控、可索引的证据世界**。

## 5. 建议的数据基本单元：Investigation World，而不是 Image

```json
{
  "world_id": "W00042",
  "world_state": {
    "entities": [],
    "events": [],
    "timeline": [],
    "ground_truth_facts": []
  },
  "claim": {},
  "image_assets": [],
  "evidence_graph": {
    "documents": [],
    "source_families": [],
    "supports": [],
    "refutes": [],
    "dependencies": []
  },
  "search_index": {
    "queries": [],
    "ranked_results": []
  },
  "tool_fault_profile": {},
  "gold_investigation": {
    "required_slots": [],
    "valid_action_frontier": [],
    "sufficient_evidence_sets": [],
    "stop_states": []
  }
}
```

### 每个世界至少包含

```text
1 个主 claim
2–5 个原子 claim
2 个以上合理候选假设
1–3 条决定性视觉线索
2 个独立支持/反驳来源族
若干转载、无关结果和困难负例
可选冲突来源
可选恶意或提示注入网页
明确的最小充分证据集
```

## 6. 如何利用不限量 Gemini 生图 API

不限量生图最有价值的不是无限扩大 IID 图片数量，而是扩大**受控的反事实和观察动作空间**。

### 6.1 生成多视图而非单图独立样本

对同一 world 生成：

- 原始全图；
- 不同裁剪/缩放/压缩/截图版本；
- 隐藏或暴露关键路牌的版本；
- 字幕覆盖关键线索的版本；
- 相同事件的不同相机视角；
- 相似但属于另一事件的 hard negative；
- 供网页证据使用的配套图片。

这样可训练“整图看不清 → 决定裁剪哪里 → 新证据可见”的主动感知。

### 6.2 生成最小视觉对照组

同一个 seed/world 下生成配对或成组样本：

```text
G1：只改路牌文字
G2：只改旗帜或 logo
G3：只改字幕时间
G4：只改背景地标
G5：保持像素主体，改变新闻卡片 claim
```

要求统一：分辨率、压缩、布局、字体、图像生成路径和后处理。否则模型可能通过画质或模板识别标签。

### 6.3 生成“需要重看”的视觉线索

专门控制关键线索的可观测层级：

```text
Level 0：全图即可读取
Level 1：需要 crop/zoom
Level 2：需要 crop + OCR/增强
Level 3：视觉线索只生成候选，必须外部证据确认
```

评测可以测：Agent 是否在外部搜索得知区分特征后，返回正确区域做进一步观察。

### 6.4 生成证据网页中的图像，而不是只生成待核查图

同一 world 的图像应出现在冻结网页环境中：

- 原始事件报道；
- 官方公告；
- 转载页面；
- 错配谣言页面；
- 事实核查页面；
- 视觉相似但无关页面。

本地 visual search 工具可对图像 embedding 建索引，模拟 RIS；训练时由 gold provenance 图提供可验证 reward。

### 6.5 生成器多样性与反指纹

即使当前只有 Gemini 不限量，也应：

- 混合不同模型版本、prompt 风格、seed、长宽比和后处理；
- 对 real/fake 使用完全相同的生成、编辑和编码流程；
- 加入相机照片、许可真实图或其他生成器作为外部测试；
- 用 generator-held-out、template-held-out、world-held-out split；
- 训练简单像素分类器/频域分类器作为 leakage probe。

如果一个不使用搜索的图像分类器在“external/provenance”子集上表现很高，说明数据存在严重 shortcut。

## 7. Real、Fake、Unverifiable 应怎样构造

### 7.1 Real / Supported

不能定义为“生成 prompt 与图片一致”。应定义为：

```text
图片中恢复出的 decisive claims
被 world state 与最小充分证据集支持，
且未存在同等级未解决反证。
```

AI 生成的配图可以是 supported，例如图内明确写“概念示意图”。如果它冒充新闻现场，则可能 fake。

### 7.2 Fake / Refuted

至少一个 decisive claim 被 evidence graph 中可访问的高质量证据反驳。`manipulated_slot` 只是生成 provenance，不自动等于事实上的决定性矛盾。

### 7.3 Unverifiable

不能只靠“长尾/私人场景”描述生成。必须让它由证据世界结构产生：

- 缺少决定性 evidence slot；
- 支持和反驳证据同等级冲突；
- 只有同一来源族的重复说法；
- 关键页面不可访问且无等价替代；
- claim 超出公开证据边界；
- 视觉线索不足以区分两个候选。

最好从同一个 world 构造三联组：

```text
supported：补入独立、决定性支持证据
refuted：补入独立、决定性反驳证据
unverifiable：移除/冲突决定性证据
```

三者图片可完全相同，仅 evidence world 不同，用于验证 Agent 是否真的读取证据状态，而非从像素猜标签。

## 8. Taxonomy 重构建议

原 EF/AF/CD/VI 混合了四个不同轴：真假机制、被改动语义槽、所需证据和像素生成方式，类别不正交。

建议改成多轴标签，而非强制单一 `fake_type`：

### Axis A：Claim failure mechanism

```text
nonexistent_event
wrong_entity
wrong_relation
wrong_attribute
wrong_time
wrong_location
wrong_source_or_platform
out_of_context_reuse
physically_inconsistent
```

### Axis B：Surface construction

```text
generated_whole_image
image_edit
text_overlay_edit
composite
authentic_image_recontextualized
synthetic_ui_screenshot
```

### Axis C：Required investigation

```text
internal_visual
ocr
web_fact
visual_search
provenance
temporal
source_independence
conflict_resolution
```

### Axis D：Decisive manipulated slot

```text
subject / relation / event / attribute / location / time / source / none
```

### Axis E：Evidence topology

```text
single_source_sufficient
multi_source_conjunctive
source_dependency_trap
conflicting_sources
missing_decisive_evidence
```

这比 EF1/AF2 等单标签更适合训练动作策略和分层分析。

### 与“无脸识别”约束的冲突

原 AF2、EF2、EF4 多次要求识别真实人物或“人脸反搜”。这与项目禁止人脸识别和身份匹配的硬约束冲突。建议：

- 只通过图内文字、官方 caption、服饰徽标、上下文或网页图片整体匹配核验公开角色；
- 不对裸人脸做身份识别或 embedding matching；
- 将必须依赖人脸身份才能解决的 case 排除，或标为 unsupported task；
- EF4 不使用“人脸反搜无结果”证明人物不存在；搜索缺席本身不能证明不存在。

## 9. 教师轨迹的标签泄漏问题

原方案让教师看到：

```text
verdict / fake_type / context_description /
manipulated_slot / key_contradiction / evidence_requirement
```

再让教师生成“最优核查轨迹”。这会产生**答案条件化的事后合理化**：教师知道假在哪里，直接选择命中 manipulated slot 的工具，不代表从图片和观察状态可以发现该路径。

建议分成两种角色：

### Blind Investigator

- 只见运行时可见输入与工具；
- 生成候选动作并实际 rollout；
- 不知道标签、manipulated slot 和 gold contradiction。

### Privileged Verifier / Annotator

- 看到完整 world state 和 evidence graph；
- 只负责给已有 blind trajectories 标注：动作价值、状态更新是否正确、哪里出现首个有害错误；
- 不能伪造工具结果或重写成“完美轨迹”。

可以另用 privileged planner 计算 oracle/near-oracle action，但必须明确标为合成规划监督，不能与真实调查轨迹混在一起。

## 10. 面向 SFT / RL 的轨迹数据

每一步至少记录：

```json
{
  "state_id": "S4",
  "belief_state": {},
  "open_claims": ["C2"],
  "missing_slots": ["original_event"],
  "candidate_actions": [],
  "chosen_action": {},
  "observation": {
    "tool_call_id": "T17",
    "status": "success",
    "raw_artifact_hash": "..."
  },
  "validated_evidence_delta": [],
  "hypothesis_update": [],
  "coverage_before": {},
  "coverage_after": {},
  "terminal": false
}
```

### 必须保存的分支

- 高信息动作与低信息动作；
- 支持搜索与反证搜索；
- 整图 RIS 与关键 crop RIS；
- 新独立来源与重复转载来源；
- 正确继续与过早停止；
- 正确停止与无意义过搜；
- 工具失败后的替代、降级或拒答。

## 11. 数据生成管线建议

### Stage 0：定义 world schema 和版本

先锁定 claim、world、evidence、source-family、tool observation 和 stop-state schema。

### Stage 1：生成世界与证据图

从结构化 world facts 生成 claim、候选假设、决定性证据槽和最小充分证据集。先有真值和证据拓扑，再生成图片。

### Stage 2：生成冻结文档世界

生成或收集 official、primary report、secondary report、UGC、转载、反证和无关文档；保存来源依赖和发布时间。全文可被本地 search/visit 工具访问。

### Stage 3：生成视觉资产

使用 Gemini 为同一 world 生成全图、多视图、困难负例、网页配图和受控局部线索。不要让 label 决定图像的总体画质或模板。

### Stage 4：自动一致性与泄漏审计

- OCR 与预期文字一致；
- 关键 region 可见性达到目标难度；
- 图像/claim/world 一致；
- sufficient evidence set 真的蕴含/反驳 claim；
- 简单 image-only、OCR-only、template-only baseline 不能轻易预测标签；
- 近重复与 world 泄漏检查。

### Stage 5：blind rollout

多个 Investigator 在不知道标签的情况下运行，保存所有真实工具结果和失败。

### Stage 6：privileged annotation

用 world graph 计算过程标签，配合人工抽检：

- 动作是否取得新有效证据；
- 是否属于新的来源族；
- 对 decisive slot 的贡献；
- 是否排除候选；
- stop 是否正确；
- 首个有害错误在哪里。

### Stage 7：反事实分支补采样

在关键状态强制执行多个候选动作，获得真实 branch outcome，而不是让教师凭想象给动作排序。

### Stage 8：严格拆分

至少同时按以下键 group split：

```text
world / event template / entity set / image seed /
document template / source graph / generation model version
```

最终保留：

- IID validation；
- template-held-out；
- world-held-out；
- generator-held-out；
- tool-failure-held-out；
- real-world frozen；
- live rolling test。

## 12. 质量门槛与泄漏探针

在宣称数据可用前，建议跑：

1. **image-only classifier**：external/provenance 子集应接近机会水平，否则有视觉标签捷径；
2. **OCR-only classifier**：检查 fake 常用措辞、日期和模板泄漏；
3. **metadata-only classifier**：文件名、尺寸、编码、生成批次不得预测标签；
4. **no-tool LLM/VLM**：若在必须搜索子集上很高，可能是世界知识泄漏或模板捷径；
5. **random-search agent**：检测证据世界是否太容易；
6. **oracle evidence agent**：测 synthesis 上限；
7. **gold-action oracle**：测工具与 evidence graph 是否足以解决；
8. **source-shuffled test**：验证模型是否关注来源关系；
9. **same-image/different-world triplets**：验证 verdict 是否随证据充分性而非像素改变；
10. **human audit**：抽检 claim 可见性、最小充分证据和 `unverifiable` 合理性。

## 13. 推荐规模：先追求世界深度，不追求图片总量

第一阶段不建议直接生成数千个相互独立的图片。更有价值的 MVP：

```text
100–200 个 Investigation Worlds
每 world：
  3 个 verdict/evidence variants
  4–10 个视觉 assets
  8–20 个文档
  2–5 个 source families
  5–15 个关键 branch states
```

这会自然形成数千图片、数千文档和大量轨迹分支，但 gold world state 仍可管理。

只有在 shortcut audit、oracle solvability、blind trajectory 和人工抽检都通过后，再扩大 world 数量。

## 14. 建议的最小实验

### 实验 A：证明搜索驱动重新观察有价值

比较：

```text
VLM only
固定 OCR + RIS + web 流水线
普通 ReAct
显式 belief state + active revisit
```

指标：决定性线索 region 命中率、有效证据 recall、相同预算 verdict accuracy。

### 实验 B：证明不是生成器捷径

在 generator/template/world held-out 和真实冻结集上测试；同时报告 image-only/OCR-only shortcut baselines。

### 实验 C：证明动作学习有价值

比较 SFT、pairwise ranker/contextual bandit 与 hierarchical RL，在相同工具预算下测：

- decisive evidence discovery；
- independent source-family coverage；
- premature stop / over-search；
- cost per supported verdict。

### 实验 D：证明 `unverifiable` 来自证据边界

对同一图片构造 supported/refuted/unverifiable evidence-world triplet，检查 Agent 是否随 evidence topology 改变 verdict。

## 15. 当前建议的研究故事

最有潜力的故事不是：

> 我们用 Gemini 生成了大量真假图片，并训练 Agent 模仿教师工具轨迹。

而是：

> 我们提出一个可控的多模态调查环境。每个任务包含显式世界状态、图像中的可恢复 claim、来源依赖图、可搜索证据和反事实行动分支。它用于训练和评测 Agent 如何让外部证据指导下一次视觉观察，并在决定性证据充分、冲突或缺失时正确裁决或拒答。

这能同时承载：数据集贡献、active visual investigation 方法、coverage/stopping 和 RL 策略学习。
