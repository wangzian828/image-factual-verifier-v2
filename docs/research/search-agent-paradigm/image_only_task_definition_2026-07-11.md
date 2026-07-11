# 单图发起的开放互联网调查：任务定义、重合与风险

日期：2026-07-11

## 1. 核心判断

当前纠结不是简单的输入格式选择，而是三种不同科学任务：

| 任务 | 输入 | 输出 | 与已有工作的重合 | 根本问题 |
|---|---|---|---:|---|
| Image + Claim Fact Checking | 图片 + 外部 claim | support/refute/insufficient | 高 | 任务清晰，但与 AVerImaTeC、OOC、DEFAME 等方向接近 |
| Image-only Authenticity Classification | 任意图片 | real/fake/unverifiable | 高且定义风险大 | 易退化为生成图/篡改检测；无语境图片没有整体真假值 |
| Image-only Open-world Investigation | 图片 | 经证据验证的事实档案、冲突与未知项 | 中等，组合空间较新 | 需解决开放式目标诱导与评测问题 |

推荐第三条路径：**不给外部 claim，但允许并要求 Agent 使用开放互联网；Agent 从图像中诱导少量有决策价值、可由外部世界核验的 investigation targets，并通过 RIS/Web 搜索、来源访问和重新观察建立一个可引用的 verified image account。**

这不是普通 image understanding，也不应被表述为对任意图片做二元真假分类。

需要特别强调：`image-only` 只描述**查询侧输入接口**，不描述任务环境。完整设置是：

```text
single-image query
+ open Internet environment
+ multimodal retrieval and browsing tools
+ evidence-grounded factual output
```

因此更准确的论文表述不是 `pure image understanding`，而是 **single-image-initiated open-web investigation**。

## 2. 为什么任意纯图片不能直接判真假

一张没有文字或传播语境的机场照片可能对应许多命题：

- 这是机场 A；
- 这是 2025 年的事件 B；
- 机场已经关闭；
- 图片是宣传图；
- 图片经过生成或编辑。

图像像素并没有断言其中任意一个命题。因此，对任意裸图输出单一 `real/fake` 会把以下变量混在一起：

```text
像素/文件来源
场景中对象是否存在
地点与时间归属
事件归属
传播配文是否一致
外部事实是否支持
```

没有配文时尤其不存在“错配语境”，因为待比较的当前语境本身缺失。

## 3. 推荐任务：Single-Image-Initiated Open-Web Investigation

暂定名称候选：

- Single-Image-Initiated Open-Web Investigation；
- Internet-Grounded Image Investigation；
- Image-Centric Open-World Investigation；
- Search-Driven Image Investigation；
- Claim-Inducing Visual Investigation；
- Pixels-to-Evidence Investigation。

### 输入

一张来自公共、可搜索互联网世界的图片，不提供额外文字问题；Agent 被明确授予 RIS、网页搜索和来源访问工具。图片可以包含图内文字，但不依赖单独给定 claim。

### 输出

不是自由 caption，也不是单一真假标签，而是 `Verified Image Account`：

```json
{
  "direct_observations": [],
  "explicit_claims_in_pixels": [],
  "investigation_targets": [],
  "attribution": {
    "source_or_earliest_context": "identified | partial | unknown",
    "event": "supported | conflicting | unresolved",
    "location": "supported | conflicting | unresolved",
    "time": "supported | conflicting | unresolved",
    "content_origin": "camera | synthetic | edited | unknown"
  },
  "evidence": [],
  "unresolved_questions": [],
  "coverage": {},
  "overall_resolution": "resolved | partially_resolved | unresolved"
}
```

其中需要区分三类目标：

1. `explicit_claim`：图片像素中的帖子文字、新闻卡片、公告等明确断言；可以 support/refute。
2. `visual_proposition`：可见对象、文字和空间关系；通常由视觉 observation 验证。
3. `attribution_hypothesis`：Agent 提出的地点、时间、事件或来源候选；只能被外部证据支持、排除或保持未知，不能假装是图片原本断言的 claim。

### 核心过程

```text
image
→ direct visual observations
→ induce checkable investigation targets
→ prioritize targets by decision relevance
→ RIS/Web/OCR/crop
→ retrieved evidence creates discriminative visual questions
→ targeted reinspection
→ update attribution hypotheses
→ stop with verified account and unresolved fields
```

## 4. 它与已有方向的区别

| 已有方向 | 典型形式 | 本任务增加的核心要求 |
|---|---|---|
| Image Captioning / Image Understanding | `image → description` | 不接受无证据描述；要求外部核验、来源与未知项 |
| Knowledge VQA / Multimodal Search | `image + given question → answer` | 问题不是给定的；Agent 必须选择值得核查的 targets |
| Image Geolocation / Landmark Recognition | 预测固定地点或实体 | 多假设、跨字段证据、来源访问、冲突与拒答 |
| Image Provenance / RIS | 找近重复和传播关系 | 相关图只是候选；还需网页语义、重观察和裁决 |
| Image + Claim Fact Checking | 验证用户提供的 proposition | 从图片中发现 explicit claims 或生成 attribution hypotheses |
| AI-generated Image Detection | 判断像素来源 | 只是 content-origin 的一个字段，不能替代整体事实调查 |

最有防御力的任务创新不是“输入少了一个 claim”，而是：

> **模型必须决定什么值得核查。** 它需要从大量视觉描述中选择少量可搜索、可区分候选、可能改变最终归因的目标，再通过真实工具结果更新状态。

## 5. 如何避免退化成普通图片理解

Benchmark 只纳入满足以下条件的图片：

- 来自公共、可搜索、外部有记录的世界；
- 至少有一个可恢复的检索钩子：文字、地标、独特对象、事件结构或视觉邻居；
- 正确完成需要至少一次外部工具，而不是仅凭像素常识；
- 至少一个检索结果能够触发新的局部视觉检查；
- 有可审计的来源、事件或事实 gold；
- 部分样本应因 data void 正确输出 unresolved。

不应纳入大量普通私人照片、通用风景或没有外部记录的场景，否则任务既无法形成稳定 gold，也无法区分搜索能力与自由猜测。

必要的基线探针：

```text
image-only VLM/no tools
caption → text search pipeline
single-shot RIS
fixed-slot geolocation/event classifier
ordinary ReAct
active reinspection Agent
```

如果 no-tool VLM 已能高分，Benchmark 测到的仍是图片理解，不是开放调查。

## 6. 开放式 claim/target induction 如何评测

不能按生成文本逐字匹配。每个 case 应有一个 `Investigation Rubric Graph`：

```text
decisive slots
├─ event identity
│  ├─ visual marker A
│  └─ source evidence B
├─ location
│  ├─ sign/landmark region C
│  └─ independent page D
└─ time/context
   ├─ visual difference E
   └─ timeline source F
```

Agent 生成的 target 映射到 rubric leaves，评估：

- `target utility`：目标是否能区分主要候选；
- `target coverage`：决定性 rubric leaves 覆盖；
- `checkability`：能否由允许工具核验；
- `region grounding`：目标是否指向正确区域；
- `evidence soundness`：引用是否支持更新；
- `update validity`：新 belief 是否由 observation 推出；
- `stopping calibration`：充分、冲突或 data void 时是否正确停止。

这样既允许不同 Agent 用不同语言提出问题，又能稳定评分。

## 7. 对两篇论文的影响

### Paper 1

任务可重新定位为：

> 从单张图片出发，自动发现可核查目标，并通过搜索驱动的重新观察构建一个有证据的图像事实档案。

Paper 1 贡献：

- image-only open-world investigation 定义；
- 自动/半自动生成 Investigation Rubric Graph 的 builder；
- 真实 benchmark；
- ReInspect 参考 Agent；
- 证明现有 VLM、RIS、image+search Agent 在 target induction、false bridge 和 stopping 上的缺口。

### Paper 2

第二篇的学习问题反而更强：

```text
不仅学习 next action
还学习从哪些视觉不确定性诱导 investigation targets
```

利用 internet-grounded synthetic worlds 提供：

- 哪些 targets 是决定性的；
- 哪个 visual region 对应 target；
- 哪类搜索结果能区分候选；
- 哪些动作只是重复信息；
- 何时停止或保持 unresolved。

## 8. 不建议立即做二选一：30×3 Pilot

在最终定任务前，建议构造三个各 30 例的小 pilot：

```text
A  image + external claim
B  single claim-bearing image（文字在像素内）
C  claim-free public-world image → verified account
```

统一测量：

- 双人标注对“待核查目标”的一致率；
- 建立 gold evidence 的分钟/case；
- no-tool VLM 表现；
- 外部搜索的必要性；
- RIS/Web 后产生新视觉问题的比例；
- 最终输出的可评分性；
- 各方向与现有 benchmark 的可解释差异。

决策标准：如果 C 的 target agreement 和 evidence coverage 太低，先采用 B；如果 B 被 OCR-only baseline 轻易解决，则使用 B+C 混合但按子类分别报告；A 可以保留为外部迁移轨道，而不是主任务。

## 9. 当前建议

最值得继续推进的是 C，但要接受两个改变：

1. 论文不再把任意裸图片称为具有单一真假值，而是研究 **verified image account construction**；
2. 产品层的 `real/fake/unverifiable` 只在图片中有明确 claim 或用户后来提供 claim 时映射，研究内部保留多字段状态。

这既不会自我放弃文本模态，也比“图片 + 已给 claim”多出一个真正的 Agent 决策问题：**从图像中发现下一步应该核查什么。**
