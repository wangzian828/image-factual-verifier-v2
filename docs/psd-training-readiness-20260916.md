# PSD 实机验收续记（2026-09-16）

本记录优先于较早的 v12/v14 状态段。用户要求当前任务持续修复；定时任务保持暂停。
**正式 400×8 源采集尚未启动，正式 PSD optimizer step 为 0。**
以下更新、checkpoint 和压测都是独立工程验收，不是正式训练结果。
原三轮 SFT 模型和完整 checkpoint 保持只读，冻结 Agent `src` 未改。

## 新确认的两类训练问题

### 1. 语法屏蔽后的概率不能当成原始 teacher 分布

vLLM 0.18.1 先执行 grammar bitmask，再进入 Sampler 的 `raw_logprobs`。
真实请求有些位置的 top20 为 `[0, -9999, ...]`；仅检查有限值不能识别这种
已经屏蔽、近似 one-hot 的分布。它与原方法的 forced-token 原模型评分不同。
上游参考：[River top-k 收集器](https://github.com/essamsleiman/psd/blob/778be78bdac582b51a975ff819046583aad383e0/psd/targets/river_topk.py)。
此处只参考评分位置/分布语义，配比仍以 Qwen3.5 发表配置为准，不采用 River 的 1:1 配比。

修复边界：

- 独立 RawTeacherWorker 保存语法屏蔽前 logits；原 sampler 仍按原工具约束、温度、预算选 token。
  只替换返回的 top-k/采样 token logprob，不改变采样结果或屏蔽规则。
- 无 grammar 的请求使用原实现；有 grammar 时的原始 logits 只消费一次，异常清理，禁止跨 batch 复用。
- 新 metadata `pre_grammar_unprocessed_v1` 从受验证的服务，经 PSD-only capture 适配器、候选、目标传到 datum。
  网关只有在所有实际路由的 worker 命令、PID、代码哈希匹配时才标注；混用旧副本不得标注。
- 未带来源的旧 capture 保留 token/轨迹，但 top20 变成 pending；强制按原 teacher prompt、原 completion、原图片补算。
  不删除训练样本，不重采调查，不把旧值换个标签继续用。新 datum/preflight 拒绝无来源的旧“complete”包。

实测：四张 H20、每卡 10 并发、3 波共 **120/120 原始真实请求成功**，非有限值 0，`-9999` 哨兵值 0。
请求不执行工具，不当作训练数据。此候选还显式关闭 async scheduling，仍保留 128K、图片、think8192、out32768。
它不是“NaN 永久消失”的证明；更早 eager 通过后仍曾复现。完整真实 Agent 仍需继续验证。

### 2. 原生 FSDP LoRA 保存成功，但恢复包装错误

四卡原生入口在旧 199-target 工程银行连续执行两步并成功保存完整 DCP，每个约
1,248,153,248 bytes（约 1.16 GiB），包含模型、optimizer、scheduler、四 rank RNG。
该银行现已知道使用了未绑定语义的旧 top20，**只能证明机械训练/存储路径，不是合格 PSD 训练目标**。

真实恢复失败：ms-swift 4.4.2 在缺少 PEFT `adapter_config.json` 的 FSDP DCP 目录调用
`Swift.from_pretrained`，得到零可训练参数的 SwiftModel，然后 Accelerate 的 `assign=True` 加载失败。
未执行错误优化步骤。仅为 `load_state_dict` 加一个参数会掩盖 LoRA 根本没建起来的问题。

新增 PSD-only bridge：对已通过 resume 绑定的 FSDP2 LoRA，先按原配方重建 PEFT 结构，随后让原生 Trainer
加载 DCP 参数和 optimizer/scheduler/RNG。恢复参数在结构构建结束后原样还原；拒绝零 trainable 参数或非 LoRA
trainable 参数，不影响其他 SFT/普通 PEFT 路径。**代码和定向测试通过，四卡实际恢复对照仍待执行。**

## 已验证的训练数值边界

真实 9B、完整图片和 repair/preserve 两个目标的单卡 update/save/resume：

- 默认 FlashAttention 反向：恢复参数最大差 `6.888226e-5`，严格门槛未通过。
- `FLASH_ATTENTION_DETERMINISTIC=1`：恢复 loss、梯度范数和 adapter 参数逐位相同，参数最大差 0。
  没有放宽容差。该设置已加入 DP4/SP4 PSD 配方和 resume 身份。
- 单卡通过不能替代四卡 FSDP 原生恢复验收。

## 当前产物和验收边界

- 已审核真实工程银行：1 repair + 198 preservation targets，最长输入 49,546 token，完整图片和 token 无删减。
- 旧银行：`runs/psd-sft3084-captured-canary4x8-20260916/psd-grounded-review-canary-v15`。
- 新独立银行：同采集根下 `psd-raw-teacher-canary-v18`，199/199 待原模型重新评分，0 样本拒绝。
- 模型服务：`inference/psd-sft3084-20260916/raw-teacher-noasync-v1`；逐卡当前 PID 必须读取服务根 `replica-*.json`。
- 40 并发证据：服务根 `nan-raw-teacher-concurrency40-v1/summary.json`。
- 全量修复候选代码：`training-artifacts/psd-raw-teacher-resume-20260916-v20/code`。
- `psd-fsdp-resume-bridge-20260916-v19` 仅用于同旧银行的机械恢复对照，不能启动正式采样/训练。
- v18 的 434 项训练测试通过；安装的真实 vLLM 环境下，50 项网关/协议/概率检查通过。
  v20 新增 4 项 FSDP bridge 测试。真实恢复及新银行 GPU 更新不得用这些单测替代。

所有上述服务器路径均以 `/volume/ybo/wza/` 为根，不在服务器提交 Git。
早期 FlashInfer 诊断曾触及其默认 `/root/.cache`，已告知用户并停止该分支；未删除或改动该根外目录。
后续显式指定所有相关缓存在允许根内。不能把共享磁盘 `df` 当作个人配额。

## 尚需现场完成

1. 199 个旧目标完成原 token 多模态 frozen-teacher 补算，重新验证并做真实 GPU 更新。
2. 修复后的四卡 native DCP 恢复，与同一 horizon 不间断第二步比较参数/梯度/训练数据顺序。
3. 新 raw gateway 的完整 Agent capture 链路验收；保留原修复预算、完整轨迹，不用有利重采结果遮蔽失败。
4. 门槛通过后启动已授权 400×8、T0.7、并发40的正式源采集、全部目标构建、5 epoch PSD。
   大规模时关闭重复 raw wire 诊断归档，保留必须的 native token/top20/工具轨迹，避免额外空间及预留预算瓶颈。
