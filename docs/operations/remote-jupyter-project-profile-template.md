# `<project-name>` 远程 Jupyter 项目配置档

本文件只保存项目特定配置。共享访问链路、安全规则和故障定位方法见
[`remote-jupyter-operations.md`](remote-jupyter-operations.md)。

禁止在本文件中保存密码、API key、私钥或其他凭据值。

## 项目标识

| 字段 | 值 |
|---|---|
| 项目名 | `<project-name>` |
| 本地 checkout | `<absolute-windows-path>` |
| Git remote | `<remote-name-and-url>` |
| branch | `<branch>` |
| 服务器 clean checkout | `<absolute-server-path>` |
| 数据根目录 | `<absolute-data-root>` |
| Conda/venv | `<environment-name-or-path>` |
| Python | `<version>` |
| Jupyter kernel | `<kernel-name>` |
| 凭据文件 | `~/.config/<project>/runtime.env` |

## 代码依赖

| 文件/组件 | 用途 | 缺失时影响 |
|---|---|---|
| `<local-jupyter-client>` | 非交互 Jupyter 控制 | 可改用浏览器 |
| `<environment-script>` | proxy、线程、缓存、凭据路径 | 项目命令不得运行 |
| `<run-wrapper>` | 统一执行环境 | 必须手工提供等价环境 |
| `<checkout-updater>` | clean + fast-forward 更新 | 可按通用流程手工更新 |
| `<bootstrap-script>` | 创建环境、安装项目和 kernel | 首次部署/依赖变化受阻 |
| `<kernel-wrapper>` | 启动项目 kernel | Jupyter 项目执行受阻 |
| `<background-launcher>` | 日志、PID、信号管理 | 只能前台运行 |

## 环境变量

```bash
export OMP_NUM_THREADS=<project-thread-limit>
export http_proxy=<validated-proxy>
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy="$NO_PROXY"
export <PROJECT_DATA_ROOT>=<absolute-data-root>
export <PROJECT_ENV_FILE>=~/.config/<project>/runtime.env
```

说明每个变量的来源、默认值和允许的 override。不要依赖 `.bashrc` 中的偶然值。

## 首次部署

```bash
set -euo pipefail
export OMP_NUM_THREADS=<project-thread-limit>
mkdir -p <checkout-parent> <data-root>
cd <checkout-parent>
git clone --branch <branch> <repository-url> <checkout-name>
cd <checkout-name>
<bootstrap-command>
```

验收：

```bash
hostname
id -un
git status --short
git rev-parse HEAD
<kernel-verification-command>
<deterministic-test-command>
<provider-smoke-test-command>
```

## 日常更新

```bash
set -euo pipefail
cd <server-checkout>
test -z "$(git status --porcelain)"
<fast-forward-update-command>
<optional-bootstrap-command>
git status --short --branch
```

## 前台运行

```bash
cd <server-checkout>
<run-wrapper> <command>
```

## 后台运行

记录：

```text
launcher:
worker:
log root:
PID root:
stop command:
resume/checkpoint behavior:
output directory rule:
retention rule:
```

## 数据与产物

```text
datasets:
artifacts:
benchmarks:
model cache:
tool cache:
logs:
traces:
evaluation outputs:
generated data:
```

所有路径必须位于 checkout 外。

## 真实服务与健康检查

```text
service:
model ID:
health endpoint:
NO_PROXY requirements:
GPU allocation policy:
startup command:
shutdown command:
probe command:
```

## 验收门禁

```text
deterministic tests:
compile/lint:
provider probe:
small real replay/evaluation:
artifact audit:
expected success criteria:
```

## 已知平台约束

记录驱动、CUDA、NCCL、代理、文件系统、并发和 GPU 使用限制。每项必须注明
验证日期和复现/验收命令。

## 故障处理

只记录该项目特有故障。通用 SSH、Jupyter、PowerShell 引用、loopback proxy、
Git tracking ref 和超时问题统一引用
[`remote-jupyter-operations.md`](remote-jupyter-operations.md)。
