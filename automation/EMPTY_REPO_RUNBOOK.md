# 从空 GitHub 仓库启动自动实验闭环

本流程假设：

- `https://github.com/flowerfog/test-rogll.git` 完全为空，bootstrap 前不上传任何文件。
- `https://github.com/flowerfog/docs-rogll.git` 只保存结果。
- 当前历史实验目录只提供原始数据和 runner，绝不推送其中的历史实验代码。
- 训练集与测试集只放本地磁盘和 AutoDL 数据盘，不进入 GitHub。

## 1. 初始化 docs 仓库

在 GitHub 的 `docs-rogll` 中创建 `README.md` 并提交到 `main`。`test-rogll` 继续保持空白。

## 2. 准备本地 source 写权限

本地 bootstrap 和 watcher 都要向 `test-rogll` 推送。建议使用 SSH key，并在 `~/.ssh/config` 建立别名：

```sshconfig
Host github-source
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_rogii_source
    IdentitiesOnly yes
```

测试空远端：

```powershell
ssh -T git@github-source
git ls-remote git@github-source:flowerfog/test-rogll.git
```

空输出且退出码为 0 表示可访问且没有分支。

## 3. 从空远端自动生成首次提交

```powershell
cd E:\rogii-wellbore-geology-prediction
New-Item -ItemType Directory -Force automation\config
Copy-Item automation\examples\bootstrap.example.toml automation\config\bootstrap.toml
```

在 `bootstrap.toml` 中确认：

```toml
[bootstrap]
state_dir = "C:/autolab/state/bootstrap"
checkout_dir = "C:/autolab/checkouts/source-controller"
local_data_root = "E:/rogii-wellbore-geology-prediction"
autodl_data_root = "/root/autolab/data"

[source]
url = "git@github-source:flowerfog/test-rogll.git"
branch = "experiments"
```

确认本地环境：

```powershell
codex --version
codex exec --skip-git-repo-check --sandbox read-only --ephemeral "Reply with exactly OK"
git ls-remote git@github-source:flowerfog/test-rogll.git
```

运行一次性 bootstrap：

```powershell
python automation\bootstrap_controller.py --config automation\config\bootstrap.toml
```

bootstrap 会拒绝任何非空 source 远端。成功时它会：

1. 只扫描本地 `train/`、`test/` 和 `sample_submission.csv` 的结构摘要。
2. 调用 Codex 从空目录生成首个 baseline、CPU 测试和 `autodl/job.json`。
3. 阻止数据、模型、submission、密钥和 GitHub workflow 被提交。
4. 自动创建并推送 `experiments` 分支。

至此 `test-rogll` 才第一次出现内容，而且全部是 Codex 从零生成的。

## 4. 上传原始数据到 AutoDL

通过 AutoDL 文件管理器或 SCP 上传，最终保持：

```text
/root/autolab/data/
├── train/
├── test/
└── sample_submission.csv
```

不要上传本地根目录的历史 Python、Markdown、OOF、输出、ZIP 或旧 notebook。

## 5. 独立部署 runner 到 AutoDL

runner 不进入 `test-rogll`，从本地单独复制：

```powershell
scp -P <AUTODL_SSH_PORT> -r automation root@<AUTODL_HOST>:/root/autolab-runner/
```

AutoDL 上应存在：

```text
/root/autolab-runner/automation/autodl_worker.py
```

## 6. 配置 AutoDL 的 docs 写权限

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_docs_rogll -C autodl-docs
cat ~/.ssh/id_ed25519_docs_rogll.pub
```

把公钥添加到 `docs-rogll -> Settings -> Deploy keys` 并勾选写权限。配置：

```sshconfig
Host github-results
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_docs_rogll
    IdentitiesOnly yes
```

## 7. 配置 AutoDL worker

```bash
mkdir -p /root/autolab/{config,state,work,cache}
cp /root/autolab-runner/automation/examples/autodl.example.toml \
   /root/autolab/config/autodl.toml
```

关键配置：

```toml
[source]
url = "https://github.com/flowerfog/test-rogll.git"
branch = "experiments"
mirror_dir = "/root/autolab/cache/source.git"

[results]
url = "git@github-results:flowerfog/docs-rogll.git"
branch = "main"
checkout_dir = "/root/autolab/cache/results"

[execution.env]
PYTHONUNBUFFERED = "1"
ROGII_DATA_ROOT = "/root/autolab/data"
```

确认 AutoDL 环境：

```bash
python3 --version
git --version
nvidia-smi
codex --version
ssh -T git@github-results
```

根据 bootstrap 生成的 `requirements.txt` 在 AutoDL 环境安装依赖。

## 8. dry-run 后执行首个 GPU 任务

```bash
python3 /root/autolab-runner/automation/autodl_worker.py \
  --config /root/autolab/config/autodl.toml \
  --once --dry-run

python3 /root/autolab-runner/automation/autodl_worker.py \
  --config /root/autolab/config/autodl.toml \
  --once
```

成功后 `docs-rogll/main` 会出现：

```text
runs/<job-id>/<source-sha>/
```

## 9. 启动本地自动研究 watcher

```powershell
Copy-Item automation\examples\local.example.toml automation\config\local.toml
```

关键配置：

```toml
[source]
url = "git@github-source:flowerfog/test-rogll.git"
branch = "experiments"
checkout_dir = "C:/autolab/checkouts/source-controller"

[results]
url = "https://github.com/flowerfog/docs-rogll.git"
branch = "main"
checkout_dir = "C:/autolab/checkouts/results-reader"

[proposal]
mode = "autopush"
max_rounds = 3
```

启动：

```powershell
python automation\local_watcher.py --config automation\config\local.toml
```

## 10. 启动 AutoDL 常驻 worker

确认首轮成功后：

```bash
nohup python3 /root/autolab-runner/automation/autodl_worker.py \
  --config /root/autolab/config/autodl.toml \
  > /root/autolab/state/autodl-daemon.log 2>&1 &
```

之后循环为：

```text
docs 新结果 -> 本地 Codex 生成下一实验并推送 source
-> AutoDL Codex 审查 -> GPU 运行 -> docs 新结果
```

达到 `max_rounds` 或创建本地 `STOP` 文件后停止产生新实验。最终竞赛提交始终人工授权。

## 11. 第二阶段再接 Gitea

先验证 GitHub 直连闭环。需要 Gitea 时，只把它作为 `test-rogll` 的 AutoDL 本地 pull mirror；worker 改读本地 Gitea，结果仍直接写入 GitHub `docs-rogll`。
