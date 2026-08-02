# SSH 隧道与远程 Jupyter 通用运维指南

本文可直接提供给使用自己服务器、自己账号和自己凭据的人。它只描述一种通用
控制方式：

```text
本地 Windows
  -> SSH 本地端口转发
  -> 远程 Jupyter
  -> Jupyter kernel 执行 Python 或 shell 命令
```

除管理员指定的共享出网代理外，本文不包含任何真实服务器地址、账号、SSH/Jupyter
端口、密码、项目名或内部目录。

配套客户端：

```text
scripts/server/jupyter_remote.py
```

## 1. 适用前提

远程侧需要满足：

- 用户拥有自己的 SSH 账号；
- SSH 能到达运行 Jupyter 的服务器，或能到达已有 Jupyter 转发的控制节点；
- 用户拥有自己的 Jupyter 登录凭据；
- Jupyter 至少提供一个可用 kernel，例如 `python3`；
- 远程 shell 中可以执行目标命令。

本地 Windows 需要：

- OpenSSH 客户端 `ssh.exe`；
- Python 3；
- Python 包 `requests` 和 `websocket-client`；
- 本文配套的 `jupyter_remote.py`。

安装 Python 依赖：

```powershell
python -m pip install requests websocket-client
```

## 2. 填写自己的连接参数

在 PowerShell 当前会话定义：

```powershell
$ControlHost = "<ssh-host>"
$ControlPort = <ssh-port>
$RemoteUser = "<your-ssh-user>"
$LocalJupyterPort = <unused-local-port>
$RemoteJupyterHost = "127.0.0.1"
$RemoteJupyterPort = <remote-jupyter-port>
$ExpectedHostname = "<expected-server-hostname>"
$KernelName = "python3"
$Client = "<path-to-jupyter_remote.py>"
```

如果 Jupyter 位于 SSH 目标机之外，应将 `$RemoteJupyterHost` 改成 SSH
目标机可访问的 Jupyter 地址。

不要把密码写入本文、脚本、PowerShell profile 或 Git。

## 3. 建立 SSH 隧道

在 Windows PowerShell 中运行：

```powershell
ssh -F NUL -N `
  -L "127.0.0.1:${LocalJupyterPort}:${RemoteJupyterHost}:${RemoteJupyterPort}" `
  -p $ControlPort `
  -o ExitOnForwardFailure=yes `
  -o ServerAliveInterval=30 `
  -o ServerAliveCountMax=3 `
  "${RemoteUser}@${ControlHost}"
```

说明：

- `-F NUL` 避免本机 SSH 配置中的其他转发干扰本次连接；
- `ExitOnForwardFailure=yes` 确保端口转发失败时立即退出；
- SSH 密码应由终端交互读取；
- 保持该 SSH 进程运行，关闭它会同时关闭隧道。

检查本地监听：

```powershell
Get-NetTCPConnection `
  -LocalAddress 127.0.0.1 `
  -LocalPort $LocalJupyterPort
```

也可以在浏览器打开：

```powershell
Start-Process "http://127.0.0.1:${LocalJupyterPort}/tree"
```

看到 Jupyter 登录页只能证明网络链路已通；仍需验证登录、kernel 和实际执行主机。

## 4. 使用通用 Jupyter 客户端

`jupyter_remote.py` 通过 Jupyter REST API 登录、创建或复用 kernel，再通过
WebSocket 执行代码。

先为当前 PowerShell 进程设置隧道地址：

```powershell
$env:JUPYTER_REMOTE_BASE = "http://127.0.0.1:${LocalJupyterPort}"
```

客户端不包含默认服务器地址。也可以在每次调用时显式传入：

```powershell
python $Client `
  --base "http://127.0.0.1:${LocalJupyterPort}" `
  --kernel-name $KernelName `
  --shell 'hostname'
```

未提供密码时，客户端会交互读取 Jupyter 密码且不回显。

无人值守任务可以只在当前进程中注入：

```powershell
$env:JUPYTER_REMOTE_PASSWORD = "<supplied-securely>"
python $Client --kernel-name $KernelName --shell 'hostname'
Remove-Item Env:JUPYTER_REMOTE_PASSWORD
```

不要把真实密码保存在 `.ps1`、`.bat`、shell profile、命令示例或日志中。

## 5. 最低连接验收

```powershell
python $Client --kernel-name $KernelName --shell `
  'hostname; id -un; printf "OMP_NUM_THREADS=%s\n" "$OMP_NUM_THREADS"'
```

确认：

- `hostname` 是预期服务器；
- `id -un` 是对方自己的账号；
- kernel 是预期 kernel；
- `OMP_NUM_THREADS=1`。

`OMP_NUM_THREADS=1` 是通用运行基线，用于避免 OpenMP 在共享服务器上创建过多
CPU 线程；它不是任何具体项目的专属变量。

如果现有 kernel 没有设置，可以在远程命令中显式设置：

```powershell
python $Client --kernel-name $KernelName --shell `
  'export OMP_NUM_THREADS=1; command-to-run'
```

长期使用时，应将该变量放入对方自己的 kernel、环境入口或作业启动器中。

## 6. 服务器出网代理

这组服务器访问 GitHub、PyPI、Hugging Face 等公网服务时使用：

```text
http://100.10.1.210:47899
```

在远程 shell、Jupyter kernel 环境或作业启动器中统一设置：

```bash
export http_proxy=http://100.10.1.210:47899
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy="$NO_PROXY"
export OMP_NUM_THREADS=1
```

旧端口 `47894` 已废弃，不应继承或继续使用。如果登录 shell、Jupyter 服务或
历史配置仍带有旧值，应由当前命令或 kernel 环境显式覆盖。

最低出网检查：

```bash
curl -fsSI --max-time 20 https://github.com/ >/dev/null
python -m pip index versions requests >/dev/null
```

本地或远程 loopback 服务必须由 `NO_PROXY`/`no_proxy` 绕过代理，否则健康检查
可能收到代理生成的 `503`。

## 7. PowerShell 到 Bash 的引用规则

传给 `--shell` 的完整 Bash 命令应作为一个 PowerShell 单引号参数。

正确：

```powershell
python $Client --kernel-name $KernelName --shell `
  'cd /absolute/remote/path && command --input "/data/file with spaces"'
```

错误：

```powershell
python $Client --kernel-name $KernelName --shell `
  "cd $RemotePath && command --input \"$RemoteFile\""
```

外层双引号会让 PowerShell 在命令到达远程 Bash 前展开 `$变量` 和
`$(表达式)`。

在外层单引号内直接使用 Bash 的 `"..."`，不要写成 `\"...\"`。

## 8. 执行长命令

使用标准输入传递多行 shell：

```powershell
@'
set -euo pipefail
export OMP_NUM_THREADS=1
cd /absolute/remote/path
command --input /absolute/input --output /absolute/output
'@ | python $Client `
  --kernel-name $KernelName `
  --shell `
  --stdin `
  --timeout 600
```

`--timeout` 是本地客户端等待 Jupyter kernel 返回结果的时间。长任务还应有自身
的超时、日志、PID 和停止机制。

不要用一个前台 Jupyter 调用无限等待训练或服务进程。长任务应由远程作业启动器、
systemd、调度器或经过验证的后台脚本管理。

## 9. Kernel 管理

列出当前 kernel：

```powershell
python $Client --list-kernels
```

复用已有 kernel：

```powershell
python $Client `
  --kernel "<kernel-id>" `
  --shell `
  'hostname; id -un'
```

停止 kernel：

```powershell
python $Client --stop-kernel "<kernel-id>"
```

默认情况下，客户端创建的临时 kernel 会在命令结束后停止。使用 `--keep` 才会
保留，并输出 kernel ID。

## 10. 小文件上传

客户端支持通过 Jupyter Contents API 上传一个文件：

```powershell
python $Client `
  --upload "<local-file>" "<remote-relative-path>"
```

该功能适合小型配置或辅助文件，不适合大型数据集、模型或仓库同步。大型内容应使用
服务器侧 Git、对象存储、数据挂载或管理员批准的传输方式。

上传路径必须是 Jupyter 根目录下的规范相对路径，不能包含 `..`。

## 11. 安全规则

- 每个人使用自己的 SSH/Jupyter 账号和权限；
- 不共享密码、cookie、API key、私钥或现成登录会话；
- 凭据只通过交互输入、密码管理器或当前进程环境提供；
- 不在命令、日志和截图中打印凭据；
- 除本文明确列出的共享出网代理外，不把真实主机、端口、账号和内部服务信息
  写入通用文档；
- 修改或删除远程文件前，先确认绝对路径和权限边界；
- 不使用 Jupyter 控制链路传输大型敏感数据；
- 使用完毕后停止不再需要的 kernel，并关闭 SSH 隧道。

## 12. 常见故障

### 本地端口未监听

- 检查 SSH 进程是否仍在运行；
- 确认本地端口未被占用；
- 确认 `ExitOnForwardFailure=yes` 没有报错。

### 本地端口监听，但浏览器或客户端超时

- 确认远端 Jupyter host/port；
- 确认 SSH 目标机能访问 Jupyter；
- 重建 SSH 隧道；
- 检查本机代理是否错误处理 loopback。

### Jupyter 登录失败

- 使用对方自己的 Jupyter 密码；
- 确认连接的是正确 Jupyter 实例；
- 不要把 SSH 密码误当成 Jupyter 密码。

### 命令在错误服务器执行

- 立即运行 `hostname` 和 `id -un`；
- 停止错误 kernel；
- 选择或创建正确服务器上的 kernel。

### `unrecognized arguments`

通常是 PowerShell 引用失败，远程命令可能尚未执行。改用外层单引号或
`--stdin`。

### 大约 120 秒后超时

提高客户端 `--timeout`，但同时检查远程任务是否真的有界。

### 本地服务被代理返回 `503`

确保远程环境中的 `NO_PROXY` 和 `no_proxy` 包含：

```text
127.0.0.1,localhost,::1
```

### 公网下载连接到旧代理端口

检查大小写代理变量，覆盖任何指向 `47894` 的旧值，统一使用
`http://100.10.1.210:47899`。

## 13. 可发送文件

只需发送：

```text
remote-jupyter-operations.md
jupyter_remote.py
```

发送前再次确认两个文件中不存在真实账号、SSH/Jupyter 主机和端口、密码、内部
目录、项目名或仓库地址。共享出网代理地址是本文有意保留的服务器环境配置。

SSH/Jupyter 权限由服务器管理员直接授予对方，不通过文档或 Python 文件传递。
