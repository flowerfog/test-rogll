# ROGII Git-Codex-AutoDL 自动实验闭环

若 source GitHub 仓库必须从完全空白开始，请先按 [EMPTY_REPO_RUNBOOK.md](EMPTY_REPO_RUNBOOK.md) 运行一次性 bootstrap。它不会上传当前目录的历史实验代码。

这套实现用两个远端仓库完成控制和结果回传：

```text
本地 Codex -> source repo / experiments 分支 -> AutoDL worker
     ^                                           |
     |                                           v
本地 watcher <- docs/results repo <- 审查、GPU 运行、报告与产物
```

AutoDL 上的 Gitea 是可选缓存镜像，不是必需的协调组件。GitHub 已经是中介时，worker 直接轮询 GitHub 更简单；只有 AutoDL 到 GitHub 不稳定、需要服务器内 UI 或需要保留本地镜像时，才建议加入 Gitea。

## 已实现的约束

- 源提交 SHA 是任务唯一键，SQLite 记录状态，正常重启不会重复跑已完成提交。
- 只处理配置分支的最新提交；提交没有 `autodl/job.json` 或 `enabled=false` 时忽略。
- GPU 命令必须是 JSON 字符串数组，不经过 shell，并受程序白名单、超时和环境变量白名单限制。
- 每个任务使用独立 Git worktree。Codex 以只读 sandbox 和 JSON Schema 输出做执行前审查。
- worker 检查请求的 GPU 编号和显存占用；GPU 忙时保留非终态并在下一轮轮询重试。
- 成功、失败、超时和审查拒绝都会写入 docs 仓库的 `runs/<job_id>/<source_sha>/`。
- 仅上传清单声明的产物，并拒绝符号链接和超过总大小上限的文件；日志过大或磁盘余量低于配置值时终止任务。
- 本地 watcher 支持只分析、留下改动、自动提交三种模式，并有最大轮数、`STOP` 文件和基础设施保护路径。
- 不提供自动竞赛提交。AutoDL 机器不应安装 Kaggle 凭据，最终 submission 保持人工授权。

Codex 审查是质量门禁，不是安全隔离。AutoDL worker 应使用专门的低权限账号或容器运行，并且该账号只拥有两个仓库所需的最小权限。

## 仓库准备

当前项目目录原先只有一个空 `.git` 目录，并不是真正的 Git 仓库。初始化前先确认 `.gitignore`，尤其不要提交 `train/`、`test/`、模型、OOF、CSV、ZIP 和输出目录。

建立两个私有仓库：

1. source repo：代码、配置、`autodl/job.json`，实验分支建议叫 `experiments`。
2. docs repo：初始化 `main` 分支并至少提交一个 `README.md`，用于接收运行报告。

建议分别使用两个 deploy key：source 只读，docs 可写。不要把 token 写进 TOML、Git URL、日志或仓库；优先使用 SSH agent 和 `~/.ssh/config` 主机别名。

任务清单可从 [examples/job.json](examples/job.json) 开始。确认命令和路径后放到源仓库的 `autodl/job.json` 并设为 `enabled=true`。每一轮实验应使用新的 `job_id`，便于报告检索。

## AutoDL 部署

前置条件：Python 3.11+、Git、可工作的 `nvidia-smi`、Codex CLI，以及训练代码自身依赖。先在交互 shell 验证：

```bash
python3 --version
git --version
nvidia-smi
codex --version
codex exec --help
```

无浏览器环境可以用环境中的 API key 初始化登录；不要把 key 放进命令历史或配置文件：

```bash
printenv OPENAI_API_KEY | codex login --with-api-key
```

将 [examples/autodl.example.toml](examples/autodl.example.toml) 复制为一个不入 Git 的 `automation/config/autodl.toml`，填写两个仓库 URL。先做无 GPU、无 Codex、无发布的检查：

```bash
python3 automation/autodl_worker.py \
  --config automation/config/autodl.toml \
  --once --dry-run
```

然后执行一次完整任务：

```bash
python3 automation/autodl_worker.py \
  --config automation/config/autodl.toml \
  --once
```

长期运行可直接使用 [run-autodl-worker.sh](services/run-autodl-worker.sh)，也可按服务器路径修改 [autodl-worker.service](services/autodl-worker.service)。很多 AutoDL 容器没有 systemd，此时使用平台守护机制、`supervisord` 或一个可恢复的 `tmux` 会话。

```bash
chmod +x automation/services/run-autodl-worker.sh
automation/services/run-autodl-worker.sh automation/config/autodl.toml
```

## 本地闭环

从 [examples/local.example.toml](examples/local.example.toml) 建立不入 Git 的 `automation/config/local.toml`。`source.checkout_dir` 必须是专门给 watcher 使用的干净 clone，不要指向日常开发工作区。

```powershell
python automation\local_watcher.py --config automation\config\local.toml --once
```

持续运行：

```powershell
powershell -ExecutionPolicy Bypass -File automation\services\Start-LocalWatcher.ps1
```

推荐按以下顺序启用模式：

1. `analysis`：只生成 `state_dir/proposals/*.json`，不改代码。
2. `edit`：Codex 修改专用 clone，但不提交；人工审查后自行提交。
3. `autopush`：Codex 修改后，watcher 校验路径和任务清单，再提交并推送到 `experiments`。

`autopush` 每个结果只生成一轮新实验。达到 `max_rounds` 后会创建 `state_dir/STOP`；手工创建同名文件也会停止产生新任务。删除前先检查状态和费用。

## 可选 Gitea

[gitea/compose.yaml](gitea/compose.yaml) 只绑定 `127.0.0.1`。启动前显式指定经过审核的固定镜像版本：

```bash
export GITEA_IMAGE='gitea/gitea:<PINNED_VERSION>'
docker compose -f automation/gitea/compose.yaml up -d
```

在 Gitea 中建立 GitHub source repo 的 pull mirror，然后把 AutoDL 配置的 `source.url` 改成服务器本地 Gitea URL。docs 仍直接推到 GitHub，避免 Gitea 再做反向同步。若 AutoDL 平台不支持持久卷或 Docker，不要使用这一层，worker 自身的 bare mirror 已足够。

## 结果格式与故障处理

每次运行目录至少包含 `run.json` 和 `README.md`，通常还包括 `review.json`、`codex.log`、`run.log`、`gpu.txt` 和 `artifacts/`。失败发布时，worker 会保留本地 work 目录并在 SQLite 中记录原因；修复凭据或远端后，建议提交一个新源 SHA 重新触发，不要直接修改 SQLite。

worker 崩溃发生在训练进行中时，重启会恢复该非终态 SHA，存在再次执行的可能。因此这是“终态恰好一次、崩溃恢复至少一次”语义。训练脚本应把写出动作放在 job 专属输出目录，并能够安全覆盖自己的临时文件。
