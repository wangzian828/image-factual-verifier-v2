# Canonical SFT 重建报告（2026-09-12）

## 结论

本次最终数据侧状态为 **可用于四卡 H20 SP4 重新训练**。正式目录：

`/volume/ybo/wza/data/ifv-initial-accepted2607-canonical-rebuild-20260912-v2`

2,579 条完整 reasoning 轨迹全部保留，未删除任何一行。28 条只有动作而没有完整
reasoning 轨迹的样本单独保存在 canonical 层，不进入“仅完整轨迹”的 policy SFT。
最终 policy 为 2,578 train + 1 validation；外部冻结的 1,527 条测试集仍是模型效果
评估口径，不能用于训练或 PSD target 构造。

该目录取代旧的 H20 policy 转换包和 `canonical-rebuild-20260912-v1`。不得把旧包、
SP8 processor 报告或历史 checkpoint 当成这份数据的启动依据。

## 来源与不可变绑定

- 官方 canonical raw-trace 归档：793,413,471 bytes，SHA-256
  `6460fe4f55f5ace2eb4ad43d811531fa1d5405a663ff8f0fa59e4b8823ff18d2`；
  2,607/2,607 个 trace 的索引和逐文件哈希通过。
- 冻结参考交付包 SHA-256
  `53cff959437147d1abd755731cdd774ad68b07ce4bce069ecb1282735a1c121d`；
  13,872 个 `SHA256SUMS` 条目重新计算后全部一致。
- raw trace 是 thought、工具调用、真实工具结果、观察 locator 和最终答案的唯一来源。
  参考交付包只提供媒体文件及 `<image>` 标记位置，不能覆盖任何训练 target。
- 共 11,277 次图片引用、11,253 个实际不同媒体文件，平均每条完整轨迹约 4.37 次
  图片引用。媒体不复制到新包，逐文件 SHA 绑定既节省空间又防止路径漂移。

顶层 manifest SHA-256：
`c9fb8d4517074d77afe3914d267b5886d25925138dd0b98ba10baefeea2b2060`。

## 发现的问题与处理边界

1. 旧转换剥离 transport envelope 后仍原样保留最终 `verdict_observation_ids`，导致大量
   ID 对模型不可见。新转换从 canonical 原始结果重建 locator；14,815/14,815 个 policy
   最终 ID 现在都字面出现于先前成功、未掩码的观察，重复和丢失均为 0。
2. 所有 2,579 行携带的工具 schema 都与当前 live schema SHA-256 完全一致，未发现
   schema 版本漂移。问题来自历史模型生成的多余字段、`parameters`/`properties`
   包装和少量旧运行时可执行等价形式，而不是工具定义被改过。
3. 32,784 次 policy 工具调用中，29,268 次原样合法；3,180 次只移除运行时忽略的未知
   字段；118 次依据配对的真实执行结果恢复等价参数；148 次做安全值规范化。恢复时以
   **实际工具结果**为准，不采信包装字段中未被执行的“意图”。
4. 70 次历史调用本来就执行失败且无法合法恢复：58 次类型错误、3 次值错误、9 次缺
   必填字段。另有 1 次语义重复动作。这 71 个调用及对应 thought 写入 `loss=false`，
   但整条轨迹、工具错误观察、后续恢复过程和最终答案全部保留。掩码仅占调用的约
   0.22%，不是删样本。
5. 其余 1,574 个合法调用的 error 主要是参考图不可访问、验证码/Cloudflare、Jina 或
   代理超时、429 等外部环境失败。它们保留为后续重试/改道的真实上下文；最终答案不
   引用失败观察。
6. 训练门禁曾使用 `splitlines()` 读取 JSONL，错误地把 JSON 字符串中的 U+0085、
   U+2028、U+2029 当成记录边界。当前 canonical/accepted-release 导出链、processor
   probe、checkpoint 指标读取和 SFT 门禁均已改为 LF-only/流式读取并加入回归测试。
7. 文档里的旧 SP8 示例已纠正。processor 报告必须和目标训练 profile 完全一致；
   当前四卡 H20 使用 SP4，不能用八卡 SP8 报告替代。

## 数据特征与真实 processor 结果

Qwen3.5-9B `Qwen3VLProcessor` + ms-swift `Qwen3_5Template` 对 2,579 行进行了全量
编码，参数为 max length 131,072、`truncation_strategy=raise`、max pixels 262,144、
image token cap 1,024、padding-free、SP4、`ignore_empty_think`，无自动 thinking
prefix。结果 2,579/2,579 通过、0 错误、0 截断：

| 指标 | min | p50 | p95 | max | mean |
|---|---:|---:|---:|---:|---:|
| 输入 token | 7,433 | 23,992 | 39,188 | 57,111 | 24,851.04 |
| 可训练 token | 1,269 | 4,750 | 8,037 | 15,224 | 5,000.98 |

- 385 条 train 轨迹达到或超过 32K；0 条达到 64K。
- 最长轨迹 57,111 token、23 次工具调用；最多图片样本含 18 张图。
- 32,784 个工具调用和 32,784 个工具响应全部出现在编码输入中。
- 32,713 个合法调用受监督；71 个异常调用和 71 个对应 thought 被精确掩码。
- 35,289 个正常 `<think>` target 受监督。

因此 max length 仍保留 128K，以兼容后续更长新增数据并禁止静默截断；当前批次可以
使用已验证的 64K packing + SP4 提高吞吐，而不是把模型能力上限降到 64K。

## 独立验收证据

- 严格 policy audit：通过，0 production blocker；SHA-256
  `bed755f435d06f6d2991ed621ec2f9d0671a95083779d44b696b5d10a6abf57e`。
- 独立 canonical 重建审计：绕过导出器逐轮重构并复算两个来源归档哈希，通过、
  0 错误；SHA-256
  `c5fb837ae5230faa4af80aa6775b75293b3ec50566f1c6d0d99103547d7e6c82`。
- 真实 processor 报告：通过、0 错误；SHA-256
  `c8625ec6566e4c3a3779de1cd5dbe53d9f291dd5b681531972f59652abbdaa1b`。
- SFT raw-data 启动门禁：文件、manifest、模型、SP4 模板和 loss 计数全部交叉匹配，
  通过、0 错误；SHA-256
  `ccafe76c29eccf90c58b4dc0020c477675c30de7b417ed3ac2f06c067216b61a`。
- 最长、最短、最多图片、含掩码异常和 validation 五类人工分层样本再次真实编码；
  图片存在、调用/响应成对、最终 ID 因果审计均通过。
- 安全扫描未发现高置信密钥标记。

重建和独立审计的可复现命令见
[SFT canonical release 工作流](sft-canonical-release-workflow.md)。正式训练启动器必须
绑定本目录的 `processor-verification.json` 与 `sft-data-gate.json`，并拒绝任何 SHA
变化。
