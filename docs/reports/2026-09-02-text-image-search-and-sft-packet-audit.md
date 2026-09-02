# 文搜图接入与 SFT judge 输入审计

日期：2026-09-02

## 结论

`text_image_search` 已经是当前统一 ReAct 的正式工具，不是未注册的试验代码。
本次修复补充了主 Agent prompt 对它的使用时机，并让 SFT eligibility judge
看到新版 ReAct 的有序动作、工具观察和图片候选，同时保持候选与 Evidence 分离。

## 16 条真实 trace

运行目录：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/runs/eval/unified-react-v1-gemini37-image-observation-smoke-20260903-p10-0a76001
```

核对结果：

- 16/16 条 trace 完成；
- `text_image_search` 出现在 4/16 条轨迹；
- 共调用 6 次；
- 每次最多返回 3 个图片候选，共 18 个候选图；
- 结果进入下一次 Gemini Interaction 的完整 function result；
- 下一请求带上一轮的 `parent_interaction_id`；
- 候选图按上限追加为下一轮的多模态输入；
- 文搜图候选仍只进入 Discovery，不自动变成 Evidence。

这说明此前问题是主 prompt 对工具职责说明不足，不是工具没有启用。

## 代码核对

当前接入点：

- `src/tools/text_image_search.py`
- `src/integrations/search/serper.py`
- `src/orchestrator/tool_registry.py`
- `src/orchestrator/react_runtime.py`
- `src/orchestrator/stage_runner.py`

当前职责：

- `text_search`：文字到网页候选；
- `text_image_search`：文字到图片/图片页面候选；
- `reverse_image_search`：上传当前图片寻找图像对应候选。

三者都只产生未验证 Discovery。只有后续 `visit`、`compare_with_reference` 或
其他成功观察才能形成 Evidence。

## 本次改动

1. Unified ReAct prompt 增加文搜图与反向搜图的区别、适用时机和候选边界。
2. SFT judge 输入版本升级为 `ifv-sft-eligibility-input-v9`。
3. SFT judge prompt 版本升级为 `ifv-sft-private-image-fact-gate-v6`。
4. SFT judge 新增有界的：
   - accepted ReAct 动作顺序；
   - thought、公开参数和工具观察；
   - 网页/图片搜索候选；
   - runtime state delta；
   - 嵌套网页访问中的 `visits[].evidence_records`。
5. 图片二进制、base64、raw HTML 和内部媒体字段不进入 judge packet。

## 验证

```text
60 passed, 6 skipped
python -m compileall -q src scripts training
git diff --check
```

旧 v3/v4 测试中依赖已删除的 `src.orchestrator.coverage` 的收集错误仍属于历史
测试残留，不恢复旧主流程。
