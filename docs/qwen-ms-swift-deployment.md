# Qwen / ms-swift 部署说明

这份文档给另一台服务器部署 IFV SFT 数据链路使用。模型 checkpoint 可以是不同
规模的 Qwen，格式是否可训练必须以实际 checkpoint 的 processor 验证结果为准。

## 1. 建议环境

```text
Python       3.12
ms-swift     4.4.2
transformers 5.12.1
torch        2.10.0
datasets     4.8.4
```

安装时固定版本并记录：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "ms-swift==4.4.2" "transformers==5.12.1" \
  "datasets==4.8.4"
python -m pip check
```

`torch` 和 CUDA 版本应按目标服务器驱动及 ms-swift 官方支持矩阵安装；不要把
gpu-13 的 CUDA wheel 直接复制到另一台机器。

## 2. 安装项目适配器

```bash
python -m pip install -e ./training
python -m pytest -q ./training
```

## 3. 数据转换

```bash
python -m ifv_training convert-policy \
  --input <accepted-dataset> \
  --output <ms-swift-policy>

python -m ifv_training convert-accepted-perception \
  --input <accepted-dataset> \
  --output <ms-swift-perception>
```

policy 的一行对应完整 episode，保留：

```text
assistant <think>...</think>
tool_call {"name":"...","arguments":"{...}"}
tool_response ...
```

不要给 assistant 消息添加 `loss=true`。ms-swift template 会根据角色和训练模式
生成 labels。`IFV_ADD_NON_THINKING_PREFIX` 在 reasoning SFT 中保持关闭。

## 4. 必须做的真实验证

```bash
python training/scripts/probe/verify_ms_swift_agent_dataset.py \
  --model <实际 Qwen checkpoint> \
  --policy-dir <ms-swift-policy> \
  --perception-dir <ms-swift-perception> \
  --output <processor-verification.json>
```

只有以下检查全部通过才可以启动训练：

1. 每行能被真实 processor 编码；
2. policy 的 `<think>` token 出现在 assistant labels；
3. tool call 和 tool response 出现在编码后的输入；
4. 图片数量和图片 marker 对齐；
5. 每行至少有一个可训练 assistant token；
6. 编码长度不超过目标上下文上限。

## 5. gpu-13 上线验收顺序

转发恢复后，先确认实际落到 gpu-13，再执行发布和验证；跳板机不执行项目命令：

```bash
hostname
test "$(hostname)" = "gpu-13"
git status --short --branch
bash scripts/server/update_gpu13_checkout.sh
```

更新脚本会校验 canonical 分支、服务器工作树干净以及 fast-forward 结果。随后在
gpu-13 的目标 Conda 环境中执行：

```bash
source scripts/server/gpu13_env.sh
scripts/server/run_gpu13.sh python -m pytest -q
(
  cd training
  ../scripts/server/run_gpu13.sh python -m pytest -q
)
scripts/server/run_gpu13.sh python -m compileall -q src scripts training/scripts training/ifv_training
```

最后针对实际发布包和实际 Qwen checkpoint 运行第 4 节的 processor 探针。只有
`update_gpu13_checkout.sh`、回归测试、结构审计和真实 processor 验证全部通过，才把
该发布记为服务器验收版本。转发未恢复时，状态只能记为“本地通过、服务器待验收”。

## 6. 训练输入边界

- `ms-swift-policy/` 只放 reasoning policy 数据。
- `ms-swift-perception/` 只放图片观察数据。
- `action_only` 不混入 policy reasoning SFT。
- 不把 `trajectory.json`、judge 输出、private gold 或内部 runtime 状态直接交给
  ms-swift。
- 训练日志、checkpoint 和 processor 验证报告单独保存，并记录代码 commit 和数据
  manifest 哈希。
