# Flash 3.7 / 3.8 小规模配对探针

2026-09-17 21:31–21:33（北京时间），用户要求检查3.8是否也拥堵。

两模型各发一个短请求和一个相同的真实PSD定位请求。均使用当前服务器相同凭据／接口、Interactions API、high、8192输出上限、不自动重试、不用缓存；不修改正在运行的PSD模型和产物。真实请求来自main-06509的冻结失败轨迹，199568字符、2张归档图片；这不是仅API ping。两个模型配对并发发送，同一模型最多1个探针请求在途。线上PSD负载继续存在，不能据此归因到具体限额维度。

| 请求 | gemini-3.7-flash | gemini-3.8-flash |
| --- | --- | --- |
| 仅回复OK | 51.39秒，completed且内容OK | 18.66秒，HTTP500／api_error／high demand |
| 完整PSD定位 | 1.81秒，HTTP429／too_many_requests／exceeded current quota | 69.83秒，HTTP500／api_error／high demand |

该小样本中3.8两次均未成功，尚无证据支持切换后改善。3.7既观察到额度错误，也在此前独立真实请求中遇到80.11秒后的high demand／500；不能将全部失败归为本地并发或长上下文。provider未给出具体RPM／TPM／每日配额字段，不猜测具体限额。

探针代码：`scripts/server/probe_psd_flash_latency.py`。脱敏结果保存在服务器`/volume/ybo/wza/runs/psd-flash-latency-probe-20260917/results.jsonl`。只记录耗时、用量和筛选过的错误说明，不复制图、轨迹、参考答案或凭据到Git。样本仅每模型2次，不是稳定性／吞吐量基准，也没有自动切换生产模型。

## 21:36–21:37追加3.6与3.5

按用户要求，固定相同case输入目录`017e8abd6e6c1d73`、同一完整PSD定位正文和归档图片，配置不变。相较3.7/3.8是在后续时段请求，并非四模型同时的控制实验。

| 请求 | gemini-3.6-flash | gemini-3.5-flash |
| --- | --- | --- |
| 仅回复OK | 5.72秒，completed且内容OK | 5.01秒，completed且内容OK |
| 完整PSD定位 | 51.65秒，completed，JSON必需字段齐全 | 53.06秒，completed，JSON必需字段齐全 |

两份长请求均计77186输入token（含2128图片token），实际返回model字段与请求一致；thought分别7864/7860，输出256/187。本轮4次调用无429或500，未自动重试。这里只验证接口完成、JSON解析与必需字段存在，没有将探针结果纳入PSD，也没有验证定位正确率或持续高并发稳定性。正式PSD仍使用3.7，未根据单次探针自动换模型。
