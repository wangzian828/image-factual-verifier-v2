# OSS 上传 session 生命周期事故 — 2026-08-25

## 现象

Gemini 教师 rollout 的 `attempt-07` 在并发 24 下运行约一小时后，当前
`run_cases` worker 对统一出网代理 `100.10.1.210:47899` 累积了大量
`CLOSE-WAIT`：

- 2026-08-25 05:54 UTC：85 个属于 worker；
- 2026-08-25 06:00 UTC：82 个属于 worker；
- 90 秒观察期间，75 个中有 59 个仍未关闭，而只完成了 4 条 trace。

这不是 GPU、图像大小或单纯 provider 延迟：`CLOSE-WAIT` 表示代理已关闭 TCP 连接，
但本地进程仍持有 socket 文件描述符。

## 根因

反向搜图的图片上传使用：

```text
reverse_image_search
→ ImageUploadClient._upload_to_oss
→ oss2.Bucket(...)
→ oss2.http.Session()
→ requests.Session()
→ 出网代理
```

`oss2.Bucket` 未传入 `session` 时会隐式创建 `oss2.http.Session`，其内部持有
`requests.Session`。OSS SDK 没有公开 close 方法；原代码没有保存这个对象，因此
`Orchestrator.aclose()` 无法关闭其中的代理连接。

这条路径未被此前的 `close_tracked_sessions()` 修复覆盖。此前修复覆盖的是项目直接创建的
Serper、Jina、Baidu OCR、参考图下载等 `requests.Session`。

在当时已完成的 912 条成功轨迹中，504 条使用过 `reverse_image_search`，共执行
723 次，因此该漏点足以持续积压连接。

## 修复

提交 `0ec1419`：

- `ImageUploadClient` 为每个工具线程创建并登记一个 `oss2.http.Session`；
- 创建 `oss2.Bucket` 时显式注入该 session；
- `ImageUploadClient.close()` 除关闭项目 tracked sessions 外，也会关闭每个 OSS session
  内嵌的 `requests.Session`；
- 已关闭的 upload client 不允许创建新的 OSS session。

`Orchestrator.aclose()` 已经能够沿着
`ReverseImageSearchTool → VisualReverseSearchClient → ImageUploadClient` 的对象图调用
`close()`，因此上述 session 会在每个隔离 rollout 终止后释放。

## 验证与恢复

本地与 gpu-13 `ifv-agent` 环境均通过：

```text
test_visual_search_oss_lifecycle.py
test_visual_search_cache.py
test_workflow_lifecycle.py

6 passed
```

旧 `attempt-07` 先正常终止，已落盘的 115 条 trace 保留。终止后
`CLOSE-WAIT` 从约 94 回落到 3。

修复版本以相同输出目录恢复为 `attempt-08`，并发保持 24。恢复后启动期
`CLOSE-WAIT` 维持在个位数；仍须在有一批 case 完成后继续确认其不会随成功 trace 持续增长。

## 后续排查原则

1. 每新增一个 SDK，都必须检查它是否在内部自建 HTTP session / connection pool。
2. 仅调用项目自己的 `close_tracked_sessions()` 不足以关闭第三方 SDK 的嵌套 client。
3. 对每次并发 rollout，至少同时记录：
   - 终端 trace 增量；
   - worker 的 `CLOSE-WAIT`；
   - worker 线程数；
   - 连接远端地址和进程 PID。
4. 若 `CLOSE-WAIT` 随成功 trace 单调累积，应按 socket → 文件描述符 → 线程栈 → SDK
   构造路径定位，不能直接归因于 Gemini 或网络。
