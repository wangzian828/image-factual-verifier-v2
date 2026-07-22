# RL 语义评分工具

## 定位

Qwen 独立完成 on-policy rollout，Gemini 只在轨迹结束后离线评分。评分器不参与搜索，
不建议下一步动作，不生成 Qwen 轨迹，也不读取 evaluator-private gold。

```text
Qwen rollout
  -> canonical trace + strict audit
  -> 冻结 Gemini 语义审计
       verdict-blind 独立判断
       recorded verdict 充分性
       verdict swap 反事实
       Evidence dropout 反事实
  -> ifv-semantic-reward-v1
  -> Training reward ledger
  -> rLLM / veRL terminal reward
```

## Gemini 实际看到的内容

- 一次受控分辨率原图；
- ImageClaims；
- 精确 Evidence 摘录、来源、artifact hash；
- Finding 与 Evidence 的引用关系；
- 第二阶段才出现最终 verdict 与 basis。

不会发送 Qwen thinking、旧聊天历史、完整网页、搜索 snippet、private gold 或 20 例标签。

第一阶段还会隐藏 Agent 的 verdict、Claim status 和 Evidence stance，避免教师只复述已有判断。

## 评分维度

- `verdict_blind_agreement`
- `claim_label_agreement`
- `claim_entailment`
- `evidence_citation_fidelity`
- `verdict_sufficiency`
- `verdict_swap_rejection`
- `evidence_dropout_sensitivity`
- `rubber_stamp_risk`

多维结果完整写入 artifact。阈值只决定 `semantic_audit_pass`，训练标量权重由训练仓库的
版本化 profile 管理。

## 调用

对一个已落盘 trace：

```powershell
python -m src.eval.score_semantic_reward `
  --trace D:\runs\traces\case_x.json `
  --image D:\release\runtime_input\assets\sha256\xx\image.jpg `
  --output-dir D:\runs\semantic_rewards `
  --cache-dir D:\runs\semantic_reward_cache
```

对 run 目录中的已有 canonical traces：

```powershell
python -m src.eval.score_semantic_reward `
  --run-dir D:\runs\qwen-rollouts `
  --output-dir D:\runs\qwen-rollouts\semantic_rewards
```

每条未命中缓存的轨迹固定产生两个 standalone Gemini 请求。缓存键绑定 trace hash、评分输入
hash、模型和两版 prompt；修改任一项都会重新评分。artifact 保存模型、prompt 版本、调用 ID、
token 用量和请求/响应 hash，但不保存图片 base64。

## 训练门禁

正向训练样本需要同时满足：

```text
strict trace audit
+ engineering-valid
+ semantic audit
+ classification correctness（仅 rollout 后处理）
```

provider、工具或运行时 fatal error 使用 mask，不给策略错误惩罚。语义判断错误、证据不足或
反事实探针失败保留低 reward，可用于 RL；它们不能进入正向轨迹池。冻结 20 例的评分 artifact
只用于开发评估，仍不得进入 SFT、RL 或教师数据。
