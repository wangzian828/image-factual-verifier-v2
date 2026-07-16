# Image Factual Verifier Training

Qwen-VL 学生模型的独立训练工程。Gemini 运行时继续留在
`image-factual-verifier-v3`；本仓库不导入运行时或数据管线 Python 包。

训练、LoRA、DeepSpeed、GRPO、导出和部署均使用固定版本的
`ms-swift==4.4.1`。本仓库只负责：

- 将 v3 的 provider-neutral teacher dataset 转成 ms-swift 官方数据格式；
- 校验模型可见数据、图片、阶段和数据清单；
- 保存可复现的实验、checkpoint 和 serving manifest；
- 提供调用 `swift sft / export / deploy / rlhf` 的薄启动脚本。

## 数据流

```text
v3 accepted teacher traces
  -> ifv-policy-dataset-v2
  -> python -m ifv_training convert-policy
  -> ms-swift messages / tools JSONL
  -> swift sft
  -> swift export
  -> swift deploy
  -> v3 Qwen provider canary
```

Perception 数据单独从质量门禁通过的 canonical trace 导出：

```text
image + fixed instruction -> canonical PerceptionReport JSON
```

## 本地契约验证

```powershell
python -m pip install -e ".[dev]"
pytest -q
```

转换当前 v3 policy dataset：

```powershell
python -m ifv_training convert-policy `
  --input D:\path\to\policy_dataset `
  --output D:\path\to\derived\qwen-policy-v1
```

导出 perception：

```powershell
python -m ifv_training convert-perception `
  --run-dir D:\path\to\accepted-run `
  --split-map D:\path\to\policy_dataset\episode_metadata.jsonl `
  --output D:\path\to\derived\qwen-perception-v1
```

## gpu-13

源代码：

```text
/gs/home/wza/projects/image-factual-verifier-training
```

大文件：

```text
/gsdata/home/wza/image-factual-verifier-v2-data/training
```

环境和双卡命令见 [docs/gpu13.md](docs/gpu13.md)。启动脚本不会选择具体卡号；
必须通过 `CUDA_VISIBLE_DEVICES=<空闲卡1>,<空闲卡2>` 显式传入。

当前稳定基线是服务器已有的：

```text
/gsdata/home/wza/models/Qwen3-VL-8B-Thinking
```

Qwen3.5 是候选门禁，不会替代基线，直到 processor、100-step LoRA、恢复、
导出、服务和真实 Agent canary 全部通过。

阶段 smoke 通过后，`scripts/train/run_curriculum_sft.sh` 直接把五个
stage-specific JSONL 交给 ms-swift，并使用：

```text
perception 25% / planning 10% / react 45% / reflection 15% / judgment 5%
```

采样、DDP、DeepSpeed、LoRA 和 checkpoint 均由 ms-swift 实现。
