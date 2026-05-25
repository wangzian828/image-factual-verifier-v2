# 4-Stage Orchestrator 架构重构方案

## Context

当前系统是单 agent ReAct loop（HarnessAgentLoop），所有 15 个工具平铺给一个 agent，存在：
- 认知过载：一个 LLM 同时负责感知、搜索、推理、判定
- 无阶段保证：可能跳过感知直接判定
- 工具选择困难：工具太多，选择准确率低
- 上下文膨胀：工具结果原始文本堆砌
- 视觉分析 = VLM 再看一遍图（无增量价值）
- OCR 无位置信息、无人脸识别能力

目标：一步到位重构为 **固定骨架 + 动态内部** 的 4 阶段 orchestrator，每阶段内部是 mini-ReAct（模型自由选择该阶段的 3-5 个工具），同时集成 PaddleOCR 和 InsightFace。

---

## 输入

**永远只有一张图片，没有 caption/配文。** Agent 必须自己从图片中发现需要验证的事实声明。

---

## 架构总览

```
orchestrator.run(image_path)
  │
  ├─ Stage 1: PERCEPTION (StageRunner, max_rounds=3)
  │    tools: [perceive_scene, ocr_with_position, face_detect]
  │    output: PerceptionReport
  │    任务: 提取图中所有实体、文字、人脸、场景信息
  │
  ├─ Stage 2: PLANNING (单次 LLM 调用, 无工具)
  │    input: PerceptionReport + image
  │    output: VerificationPlan
  │    任务: 规划需要调查的问题 + 建议工具调用序列
  │    例: 图有马斯克+抖音 → questions: ["这个人是否是马斯克?", "马斯克是否真的在抖音直播过?"]
  │        suggested_actions: ["reverse_image_search", "text_search('马斯克 抖音直播')"]
  │
  ├─ Stage 3: VERIFICATION (StageRunner, max_rounds=8)
  │    tools: [crop_and_inspect, count_objects, text_search, news_search,
  │            reverse_image_search, visit, compare_with_reference,
  │            check_consistency, analyze_visual_anomalies, verify_face_identity]
  │    input: PerceptionReport + VerificationPlan
  │    output: VerificationResult (收集到的证据 + 初步发现)
  │    注: 模型可以偏离计划（ReAct 自由度），但计划提供初始方向
  │
  ├─ Stage 4: JUDGMENT (单次 LLM 调用, 无工具)
  │    input: 全部前序输出
  │    output: FinalJudgment
  │
  └─ return FinalJudgment + VerificationState (trajectory/eval)
```

---

## 核心组件设计

### 1. StageRunner（通用 mini-ReAct 引擎）

简化版 HarnessAgentLoop，每个阶段复用：

```python
class StageRunner:
    def __init__(self, llm, system_prompt, tools, output_schema, max_rounds, image_path):
        ...
    async def run(self, input_context: str) -> Tuple[BaseModel, List[Step]]:
        # 1. Build messages: [system, user(image + input_context)]
        # 2. Loop: LLM → parse → tool_call or output
        # 3. If max_rounds exhausted: force output
        # 4. Return (parsed_output, steps)
```

与 HarnessAgentLoop 的区别：
- 无 AgentState（太重），用轻量 stage-local context
- 无 phase transitions（直接跑到输出或 max_rounds）
- 无 ToolLayer（dedup/diversification 在 orchestrator 层处理）
- 输出验证用 pydantic schema（不是多层 validator）
- Context: system + input_context + last 1 round（不是 hybrid mode）

### 2. 阶段间数据契约（Pydantic Models）

```python
# Stage 1 输出
class Entity(BaseModel):
    name: str                     # "person in blue shirt", "Eiffel Tower"
    entity_type: str              # "person"|"object"|"building"|"text"|"logo"|"animal"
    bbox: List[float] = []        # [x1, y1, x2, y2] normalized 0-1
    confidence: float = 1.0
    attributes: Dict[str, str] = {}

class TextRegion(BaseModel):
    text: str
    bbox_quad: List[List[float]] = []
    confidence: float = 0.0
    language: str = "unknown"

class FaceDetection(BaseModel):
    bbox: List[float] = []
    embedding: List[float] = []   # 512-d vector
    age: int = 0
    gender: str = ""
    confidence: float = 0.0

class PerceptionReport(BaseModel):
    entities: List[Entity] = []
    text_regions: List[TextRegion] = []
    faces: List[FaceDetection] = []
    scene_description: str = ""
    image_type: str = "photo"     # "photo"|"screenshot"|"document"|"illustration"|"meme"

# Stage 2 输出 — 验证计划
class InvestigationQuestion(BaseModel):
    question_id: str              # "q0", "q1", ...
    question: str                 # "这个人是否是马斯克？"
    why: str                      # 为什么需要调查这个问题
    suggested_tools: List[str]    # ["reverse_image_search", "text_search"]
    suggested_queries: List[str]  # ["马斯克 抖音直播", "Elon Musk Douyin"]
    related_entities: List[str]   # 关联的 PerceptionReport 实体名
    priority: int = 1             # 1=必须, 2=建议, 3=可选

class VerificationPlan(BaseModel):
    questions: List[InvestigationQuestion] = []
    image_intent: str = ""        # 图片试图传达什么信息（一句话）
    is_trying_to_be_real: bool = True  # 这张图是否试图让观众相信它是真实的
    risk_assessment: str = ""     # 初步风险判断："可能是AI生成"/"可能是篡改"/"可能是真实但误导"

# Stage 3 输出
class EvidenceItem(BaseModel):
    source: str                   # URL or tool name
    summary: str                  # 1-2 sentence
    direction: str                # "supports"|"refutes"|"neutral"
    quality: str                  # "strong"|"moderate"|"weak"
    tool_used: str = ""
    related_question: str = ""    # 对应哪个 question_id

class VerificationResult(BaseModel):
    evidence: List[EvidenceItem] = []
    visual_anomalies: List[Dict[str, Any]] = []
    authenticity_assessment: str = "uncertain"  # "authentic"|"likely_ai"|"likely_manipulated"|"uncertain"
    key_findings: List[str] = []  # 关键发现的一句话总结

# Stage 4 输出
class FinalJudgment(BaseModel):
    verdict: str                  # "real"|"fake"|"misleading"|"unverifiable"
    confidence: float = 0.5
    reasoning_chain: str = ""     # 完整推理链
    key_evidence: List[str] = []  # 支撑判定的关键证据
    anomalies: List[str] = []     # 发现的异常
    overall_assessment: str = ""  # 一段话总结
```

### 3. 上下文管理

每阶段的 context rendering 策略：

| 阶段 | 输入 context | 工具结果处理 | 窗口 |
|------|-------------|-------------|------|
| Perception | image | 保留全部（小且结构化） | system + image + last 1 round |
| Planning | PerceptionReport JSON（压缩：top-10 entities） + image | 无 | 单次调用（带图） |
| Verification | 压缩 PerceptionReport + VerificationPlan + 已收集证据 | search: top-5 结果各 1 句摘要; visit: 截断 2000 chars; VLM 工具: 保留异常列表 | system + input + last 2 rounds |
| Judgment | 全部 VerificationResult + 压缩 PerceptionReport + VerificationPlan | 无 | 单次调用 |

### 4. 工具管理

每阶段只暴露该阶段的工具子集。依赖规则：
- `crop_and_inspect` 需要 perception 阶段的 entity bbox
- `verify_face_identity` 需要 perception 阶段的 face embedding
- `compare_reference` 需要先通过搜索找到参考图 URL

依赖不满足时返回友好错误信息（不是硬拒绝）。

---

## 新工具实现

### PaddleOCR (`ocr_with_position`)

```python
# src/tools/ocr_with_position.py
class OCRWithPositionTool(BaseTool):
    name = "ocr_with_position"
    # PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
    # 返回: [{text, bbox_quad, confidence, language}]
    # CPU 推理 ~0.5-2s/image
```

### InsightFace (`face_detect`)

```python
# src/tools/face_detect.py
class FaceDetectTool(BaseTool):
    name = "face_detect"
    # FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    # 返回: [{bbox, embedding(512-d), age, gender, confidence}]
    # CPU 推理 ~0.3-1s/image
```

### 其他新工具

| 工具 | 实现方式 | 输入 | 输出 |
|------|---------|------|------|
| `perceive_scene` | VLM 结构化 prompt | image | {entities, scene_type, description} |
| `crop_and_inspect` | PIL crop + VLM 精细分析 | image + bbox + question | 详细描述 + 发现 |
| `check_consistency` | VLM prompt | image + aspect | {consistent: bool, details} |
| `verify_face_identity` | InsightFace cosine similarity + 跨图比对 | face_index + (face_index_b / reference_embedding / reference_image_url) | {match: bool, similarity} |
| `count_objects` | VLM 结构化 prompt（P2 接 GroundingDINO） | image + target | {count, confidence} |

---

## 文件结构

```
src/orchestrator/                    # 新建
├── __init__.py
├── pipeline.py                      # Orchestrator 主类（~200 行）
├── stage_runner.py                  # 通用 StageRunner（~250 行）
├── state.py                         # 所有 Pydantic 数据契约（~150 行）
├── context.py                       # 阶段 context 渲染/压缩（~120 行）
├── tool_registry.py                 # 阶段工具子集构建（~80 行）
└── stages/
    ├── __init__.py
    ├── perception.py                # Stage 1 配置：prompt + tools + max_rounds
    ├── planning.py                  # Stage 2 配置：prompt template
    ├── verification.py              # Stage 3 配置：prompt + tools + max_rounds
    └── judgment.py                  # Stage 4 配置：prompt template

src/tools/                           # 新增文件
├── ocr_with_position.py             # PaddleOCR（~80 行）
├── face_detect.py                   # InsightFace（~70 行）
├── perceive_scene.py                # VLM 结构化感知（~90 行）
├── crop_and_inspect.py              # 裁剪 + VLM 分析（~100 行）
├── check_consistency.py             # VLM 一致性检查（~80 行）
├── verify_face_identity.py          # 嵌入比对（~60 行）
└── count_objects.py                 # VLM 计数（~70 行）
```

---

## 与现有系统的集成

### workflow.py 重写

直接只有 orchestrator，无模式切换：

```python
@dataclass
class WorkflowConfig:
    provider: str = "necodex"
    model_name: str = "gpt-5.5"
    temperature: float = 0.0
    max_tokens: int = 8192
    max_rounds_perception: int = 3
    max_rounds_verification: int = 8
    timeout: float = 300.0
    tool_config_path: str = "configs/tools.yaml"
    output_dir: str = "outputs/traces"

class VerificationWorkflow:
    async def run_single(self, image_path, image_id=""):
        result = await self._orchestrator.run(image_path, image_id)
        return result
```

### 旧代码处理

全部删除，不保留：
- `src/react/harness/` — 删除
- `src/react/agent.py` — 删除
- `src/react/prompts.py` — 删除
- `src/agent/` — 整个目录删除
- `src/assertions/` — 删除（功能由 Stage 2 Planning 替代）
- `src/claiming/` — 删除
- `src/verification/` — 删除（功能由 Stage 4 Judgment 替代）
- `src/retrieval/` — 删除（功能由 Stage 3 内部管理）

保留并移入新结构：
- `src/react/llm_backend.py` → `src/orchestrator/llm_backend.py`
- `src/react/tool_executor.py` → `src/orchestrator/tool_executor.py`
- `src/react/trajectory.py` → `src/orchestrator/trajectory.py`
- `src/react/image_utils.py` → `src/orchestrator/image_utils.py`
- `src/tools/` — 保留所有工具文件
- `src/integrations/` — 保留所有集成客户端

### 输出格式

```json
{"image_id": "...", "verdict": "fake", "confidence": 0.85, "reasoning": "...", "evidence": [...]}
```

eval pipeline (`src/eval/`) 适配新格式。

---

## 复用的现有基础设施

| 文件 | 复用内容 |
|------|---------|
| `src/react/llm_backend.py` | APIBackend, LLMResponse — 所有 LLM 调用 |
| `src/react/tool_executor.py` | ToolRegistry, SyncToolWrapper — 工具执行 |
| `src/tools/base.py` | BaseTool 接口 |
| `src/tools/factory.py` | build_external_tools（Serper/Jina 等） |
| `src/tools/visual_anomaly.py` | VisualAnomalyTool |
| `src/tools/compare_reference.py` | CompareWithReferenceTool |
| `src/tools/text_search.py` | TextSearchTool |
| `src/tools/news_search.py` | NewsSearchTool |
| `src/tools/reverse_image_search.py` | ReverseImageSearchTool |
| `src/tools/visit.py` | VisitTool |
| `src/react/trajectory.py` | Step, Trajectory — 轨迹记录 |
| `src/react/image_utils.py` | 图片处理工具 |
| `src/integrations/` | 所有集成客户端 |

---

## 服务器依赖安装

```bash
# gpu14 服务器上执行
pip install paddlepaddle==3.0.0 paddleocr==2.9.1
pip install insightface==0.7.3 onnxruntime==1.19.0
pip install opencv-python-headless
pip install pydantic>=2.0  # 数据契约
```

---

## 错误处理与容错

- Stage 1 VLM 失败 → 重试一次，仍失败则用 minimal report（空实体列表 + scene_description="unknown"）
- Stage 2 产出 0 questions → 注入默认 question: "这张图片是否是真实的？"
- Stage 3 单个 question 耗尽轮次 → 标记为 "insufficient evidence"，继续下一个
- 总时间超过 300s → 强制进入 Stage 4，用已有证据判定
- LLM 完全不可用 → 从已有 state 构建 fallback answer

---

## 验证方案

1. 单元测试：每个新工具独立测试（PaddleOCR 识别准确率、InsightFace 检测率）
2. 集成测试：用 `datasample_images/` 中的样例图片跑完整 pipeline
3. Benchmark 测试：在测试集上跑准确率/token 用量/延迟

具体验证命令：
```bash
# 单工具测试
python -c "from src.tools.ocr_with_position import OCRWithPositionTool; t = OCRWithPositionTool(); print(t.call({'image_input': 'test.jpg'}))"

# 完整 pipeline
python -m src.eval.run_predict --config configs/orchestrator.yaml --samples datasample_images/

# Benchmark
python -m src.eval.run_eval --benchmark benchmarks/test_set.json
```

---

## 实施顺序

1. `src/orchestrator/state.py` — 数据契约（无依赖）
2. `src/tools/ocr_with_position.py` — PaddleOCR
3. `src/tools/face_detect.py` — InsightFace
4. `src/tools/perceive_scene.py` — VLM 感知
5. `src/tools/crop_and_inspect.py` — 裁剪分析
6. `src/tools/check_consistency.py` — 一致性检查
7. `src/tools/verify_face_identity.py` — 人脸比对
8. `src/tools/count_objects.py` — 计数
9. `src/orchestrator/stage_runner.py` — 通用引擎
10. `src/orchestrator/context.py` — context 管理
11. `src/orchestrator/tool_registry.py` — 工具子集
12. `src/orchestrator/stages/` — 4 个阶段配置
13. `src/orchestrator/pipeline.py` — 主 orchestrator
14. 重写 workflow.py + eval 适配
15. 删除旧代码：`src/react/harness/`、`src/react/agent.py`、`src/agent/`、`src/assertions/`、`src/claiming/`、`src/verification/`、`src/retrieval/`
