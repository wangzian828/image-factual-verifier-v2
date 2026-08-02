# 通用远程 Jupyter 运维指南

本文是可对外交付的项目无关 runbook，适用于通过 Windows 控制端、SSH
端口转发和 Jupyter 在远程计算服务器上部署、更新、验证和运行任意项目。

项目仓库、分支、Conda 环境、Jupyter kernel、数据目录、启动脚本和验收命令
不得写死在本文件中；这些内容必须放在项目配置档中。

- 新项目模板：
  [`remote-jupyter-project-profile-template.md`](remote-jupyter-project-profile-template.md)

本文不得包含真实主机地址、用户名、密码、项目私有路径、代理地址或内部服务信息。

## 1. 参考拓扑

典型控制链路：

```text
Windows 控制端
  -> SSH <control-host>:<control-port>，用户 <remote-user>
  -> 本地 127.0.0.1:<local-port> 转发到远端 Jupyter
  -> Jupyter kernel 以 <remote-user> 身份在 <expected-hostname> 执行
```

控制入口通常只用于运维命令，不应用于传输大型代码包、数据集或模型。代码应由
服务器从 Git 远端拉取，数据应由服务器从批准的数据源下载或从服务器存储读取。

## 2. 不可破坏的通用规则

1. 源码只在本地工作区修改。
2. 本地提交并推送后，服务器只能 clone、fetch、fast-forward、安装和运行。
3. 不得直接编辑、覆盖或 reset 服务器项目源码。
4. 服务器工作树不干净时停止更新，先确认脏文件归属。
5. 每次执行前验证 `hostname`、`id -un`、项目 kernel 和关键环境变量。
6. 项目 profile 必须声明线程、GPU 和并发限制；若平台要求
   `OMP_NUM_THREADS=1`，所有项目入口必须统一执行。
7. 密码、API key、私钥只能通过交互输入、进程环境或权限为 `600` 的未跟踪
   配置文件提供。
8. 数据集、模型缓存、日志、trace 和生成产物必须位于仓库之外。
9. 长任务必须声明日志、PID、输出目录、停止方式和完成验收方式。
10. 项目 profile 必须记录真实 checkout、branch、kernel、环境和数据根目录。

## 3. 控制端参数

PowerShell 会话中可统一定义：

```powershell
$RemoteControlHost = "<control-host>"
$RemoteControlPort = <control-port>
$RemoteUser = "<remote-user>"
$LocalJupyterPort = <local-port>
$RemoteJupyterPort = <remote-port>
$ExpectedHostname = "<expected-hostname>"
```

项目参数由项目 profile 提供：

```powershell
$ProjectRepo = "<absolute-local-checkout>"
$ProjectKernel = "<project-kernel>"
$ProjectCheckout = "/absolute/server/checkout"
$ProjectRunWrapper = "<optional project run wrapper>"
```

不要把密码放入这些变量示例、PowerShell profile、Git 配置或脚本。

## 4. 依赖边界

| 运维能力 | 必需依赖 | 是否依赖项目代码 |
|---|---|---|
| 建立 SSH 隧道 | Windows OpenSSH `ssh.exe` | 否 |
| 浏览器访问 Jupyter | 浏览器、Jupyter 密码 | 否 |
| 非交互执行 Jupyter 命令 | Python、`requests`、`websocket-client`、Jupyter 控制客户端 | 依赖通用客户端代码，不依赖业务代码 |
| 启动项目 kernel | 已安装 kernelspec 及其 wrapper | 通常是 |
| 注入项目环境变量 | 项目环境脚本或等价配置 | 通常是 |
| 安装项目环境 | 项目 bootstrap 脚本或明确的安装命令 | 通常是 |
| 安全更新 checkout | Git；可选项目 updater | updater 是项目代码 |
| 启动后台评测/训练 | 项目 launcher/worker | 是 |

当前仓库中的通用 Jupyter 客户端是：

```text
scripts/server/jupyter_remote.py
```

它只依赖 Jupyter REST/WebSocket 协议，不导入项目业务模块，可以复制到独立运维
工具目录使用。删除或无法访问该文件时，非交互控制路径不可用；浏览器 Jupyter
路径仍然可用。

项目 profile 必须单独列出所有项目代码依赖。不得把项目 wrapper 描述成宿主机
的固有能力。

## 5. 建立本地 SSH 隧道

在 Windows PowerShell 中运行：

```powershell
ssh -F NUL -N `
  -L "127.0.0.1:${LocalJupyterPort}:127.0.0.1:${RemoteJupyterPort}" `
  -p $RemoteControlPort `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=3 `
  "${RemoteUser}@${RemoteControlHost}"
```

`-F NUL` 用于避免用户 SSH 配置中的无关转发影响本链路。SSH 密码应交互输入。

检查本地监听：

```powershell
Get-NetTCPConnection `
  -LocalAddress 127.0.0.1 `
  -LocalPort $LocalJupyterPort
```

Jupyter 返回登录页、`405` 或其他 Tornado 响应，均可证明 HTTP 已到达服务；
认证和 kernel 执行仍需继续验证。

## 6. Jupyter 控制方式

### 6.1 浏览器方式

打开：

```text
http://127.0.0.1:<local-port>/tree
```

输入 Jupyter 密码。项目任务必须选择项目 profile 指定的 kernel；基础
`python3` kernel 只应用于控制面诊断或项目环境尚未安装时的 bootstrap。

### 6.2 非交互客户端

控制端依赖：

```powershell
python -m pip install requests websocket-client
```

从项目 profile 指定的位置运行客户端：

```powershell
$Client = Join-Path $ProjectRepo "scripts/server/jupyter_remote.py"
$env:JUPYTER_REMOTE_BASE = "http://127.0.0.1:<local-port>"
python $Client --kernel-name $ProjectKernel --shell `
  'hostname; id -un; printf "OMP_NUM_THREADS=%s\n" "$OMP_NUM_THREADS"'
```

默认情况下客户端交互读取密码且不回显。无人值守时可只为当前进程设置
`JUPYTER_REMOTE_PASSWORD`；执行结束后清除该变量，不得持久化。

最低验收：

```text
hostname = <expected-hostname>
id -un   = <remote-user>
required environment variables = project profile values
kernel   = 项目 profile 指定的 kernel
```

列出或停止 kernel：

```powershell
python $Client --list-kernels
python $Client --stop-kernel "<kernel-id>"
```

## 7. PowerShell 到 Bash 的引用规则

传给 `--shell` 的完整 Bash 命令必须作为一个 PowerShell 单引号参数。

正确：

```powershell
python $Client --kernel-name $ProjectKernel --shell `
  'cd /absolute/server/checkout && ./project-runner command --input "/data/a b"'
```

错误：

```powershell
python $Client --kernel-name $ProjectKernel --shell `
  "cd $ProjectCheckout && command --input \"$remote_path\""
```

双引号会让 PowerShell 在命令到达服务器前展开 `$变量` 和 `$(表达式)`。
不要在外层单引号内部写 `\"`；反斜杠会原样到达 Bash。

长命令使用标准输入，降低本地引用风险：

```powershell
@'
set -euo pipefail
cd /absolute/server/checkout
./project-runner command --input /absolute/input --output /absolute/output
'@ | python $Client --kernel-name $ProjectKernel --shell --stdin --timeout 600
```

`--timeout` 是控制客户端等待时间，不应替代项目自身的 action、provider、
stage 或作业超时。

## 8. 项目接入契约

每个项目必须从模板建立 profile，并至少填写：

| 字段 | 要求 |
|---|---|
| 本地 checkout | 唯一允许编辑源码的位置 |
| Git remote 与 branch | 服务器同步来源 |
| 服务器 clean checkout | 不与数据、日志混放 |
| Conda/venv | 明确 Python 版本和安装方式 |
| Jupyter kernel | 名称、安装脚本、wrapper 路径 |
| 环境入口 | proxy、`OMP_NUM_THREADS=1`、缓存和凭据文件 |
| 数据根目录 | 仓库外绝对路径 |
| 更新命令 | 必须拒绝脏工作树并只允许 fast-forward |
| 运行命令 | 前台和后台方式 |
| 日志/PID | 路径、权限、清理规则 |
| 验收门禁 | 测试、健康检查和真实 provider probe |
| 停止/恢复 | 信号、PID、checkpoint 和重启方式 |

如果项目没有 wrapper，也可以使用原生命令，但 profile 必须完整写出环境变量和
验收步骤，不能依赖登录 shell 中的偶然状态。

## 9. 通用部署和更新流程

### 9.1 首次部署

1. 本地完成测试、commit 和 push。
2. 通过 Jupyter 控制面确认主机、用户和线程变量。
3. 由服务器通过 Git HTTPS/SSH clone；不得通过控制入口复制仓库。
4. 在仓库外创建数据、缓存和日志目录。
5. 执行项目 profile 的 bootstrap。
6. 安装并验证项目 kernel。
7. 运行确定性测试，再运行最小真实 provider probe。

### 9.2 日常更新

1. 本地 commit 并 push。
2. 服务器检查 `git status --porcelain` 必须为空。
3. fetch 指定 branch 到明确 tracking ref。
4. 只允许 fast-forward 到远端提交。
5. 必要时重新 bootstrap 或安装依赖。
6. 记录服务器 `HEAD`、测试结果和运行 ID。

禁止：

- 在服务器源码中手工修补；
- 使用 `git reset --hard` 清理未知脏文件；
- 在未验证 branch/tracking ref 时执行模糊的 `git pull`；
- 将数据或运行产物写入 checkout。

## 10. 运行与后台任务

前台命令应通过项目 profile 的环境入口执行：

```bash
cd /absolute/server/checkout
./project-runner python -m pytest -q
```

长任务必须由项目 launcher 管理。launcher 至少应：

- 使用新或空的输出目录；
- 将 stdout/stderr 写入仓库外日志；
- 将 PID 写入当前用户拥有且非符号链接的目录；
- 使用 `umask 077` 保护日志和 PID；
- 转发 `INT`/`TERM` 给子进程；
- 记录开始时间、结束时间和退出码；
- 不把凭据写入日志或命令行快照。

## 11. 数据、缓存和凭据

通用目录原则：

```text
<project-data-root>/
  datasets/
  artifacts/
  benchmarks/
  cache/
  runs/
    _logs/
    traces/
    eval/
  generated/
```

项目可以调整目录名，但必须保持仓库与大数据/运行产物分离。

推荐凭据文件：

```text
~/.config/<project>/runtime.env
```

要求：

```bash
chmod 600 ~/.config/<project>/runtime.env
```

项目 wrapper 应只导出凭据文件路径，不打印、复制或提交凭据内容。

## 12. 故障定位顺序

1. 本地端口是否监听。
2. SSH 进程是否仍在运行。
3. Jupyter 登录是否成功。
4. kernel 名称是否存在。
5. `hostname` 是否与 profile 一致。
6. 用户是否与 profile 一致。
7. 线程、GPU 和并发变量是否与 profile 一致。
8. 项目 checkout 是否 clean、HEAD 是否正确。
9. loopback 是否加入 `NO_PROXY`/`no_proxy`。
10. 项目 proxy、数据根目录、凭据文件和服务健康检查是否符合 profile。

常见现象：

- 本地转发端口监听但 HTTP 卡住：重建 SSH 隧道。
- Jupyter 可登录但命令在错误主机执行：停止错误 kernel，使用项目 kernel。
- `/health` 返回代理生成的 `503`：检查 loopback `NO_PROXY`。
- Git fetch 成功但 branch 未更新：检查单分支 clone 的 refspec，显式 fetch
  tracking ref。
- `unrecognized arguments`：通常是 PowerShell 引用失败，远端命令尚未执行。
- 长命令在约 120 秒停止：提高 Jupyter 客户端 `--timeout`，同时保留项目内部
  有界超时。

## 13. 每次运行前检查单

```text
[ ] SSH/Jupyter 链路可用
[ ] hostname 与项目 profile 一致
[ ] 远程用户与项目 profile 一致
[ ] 使用正确项目 kernel
[ ] 线程、GPU 和并发变量符合 profile
[ ] 服务器 checkout clean
[ ] HEAD 与目标提交一致
[ ] 数据和输出位于 checkout 外
[ ] 凭据只存在于安全输入或未跟踪 600 文件
[ ] 命令使用项目环境入口
[ ] 长任务已定义日志、PID、停止和验收方式
```

## 14. 对外交付边界

可以发送：

```text
docs/operations/remote-jupyter-operations.md
docs/operations/remote-jupyter-project-profile-template.md
scripts/server/jupyter_remote.py
```

发送前再次扫描真实用户名、主机/IP、端口、代理、项目名、仓库地址、目录和凭据。

不要发送：

```text
任何真实主机或项目 profile
.env、runtime.env、PowerShell profile
SSH/Jupyter/API 凭据
内部 HANDOFF、运行日志、trace、数据路径或服务地址
```

SSH/Jupyter 权限必须由管理员通过安全渠道单独授予，不能附在交付包中。
