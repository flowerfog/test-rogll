# 6.781 历史主线 PF 修改说明

> Status: archived on 2026-07-23. The current mainline is the frozen 6.47 pipeline documented in `mainline_6p47_baseline.md`. Do not use this file as the parent for new experiments.

源 notebook：`fork-of-fork-of-fork-of-rogii-wellbore-geol-ecf6db (1).ipynb`

SHA256：`ECA8BF99480B0C6AAD9524455BDB087546F90EEA635CD06D80C41DA8463CA7E2`

所有可直接粘贴的 Cell 都在 `mainline_6p781_pf_hidden_patch_cells.py`。该文件只依赖主线 notebook 已经定义的函数和全局变量，不调用额外 `.py` 文件。

## 已核对的 6.781 路径

1. SP45：128-seed likelihood PF、beam、selector。
2. Ridge/GBDT 与 SP45 按 `0.30 / 0.70` 融合。
3. 四次多项式投影。
4. SP45 与独立 FLE 分支按 `0.55 / 0.45` 融合。
5. 再做四次多项式投影。
6. guarded same-well contact。
7. CP12。
8. Gold visible-prefix calibration。
9. full-progress geometric Neighbor，权重 `0.25`、cap `6 ft`。
10. 输出最终 `submission.csv`。

不要把 `ultimate-pf-config-strategy-a-reproducible-score.ipynb` 的完整推理 Cell 替换进来；它不是这条 6.781 主线。

## Cell 放置顺序

1. 用 `CELL A0` 替换当前最前面的 Gold profile Cell。
2. 在原 SP45 函数 Cell（原 notebook cell 4）后粘贴 `CELL A1`。
3. `CELL A2` 已由 LB 6.806 证实在该主线上无增益；后续 H1/G1 不再粘贴它，保留原 FLE `lik_pf`。
4. 用 `CELL B` 完整替换原 SP45 test inference Cell（原 notebook cell 15）。
5. 主线其余 Cell 保持原顺序运行，包括 FLE、CP12、Gold 和 Neighbor。
6. 在 active Neighbor Cell（原 notebook cell 42）之后粘贴 `CELL C`。
7. 最后运行 `CELL D` 做 submission 审计和快照。

## 已获得的主线证据

`A1 + A2 + B` 已提交，LB 为 `6.806`：

- `A1` 只保存 seed 统计，不改变 SP45 输出。
- `B` 只是让原 SP45 inference 调用 A1 的统计接口；未运行 `CELL C` 时不改变预测。
- 因而该提交唯一的 scoring 改动是 `A2` 的 FLE likelihood `GS x 1.3`。
- 相对 6.781 没有增益，所以 `GS x 1.3` 在含 CP12、Gold、Neighbor 的主线上关闭。

这不否定它在 6.594 那条较短 pipeline 中的作用，只说明增益不能跨 pipeline 直接迁移。

## 仍待单独验证的修改

- PF seed-branch hedge：复用 SP45 已计算的 128 条 seed 轨迹，在最终 Neighbor 之后最多移动 `2 ft`。
- balanced Gold：必须与 seed hedge 分开提交；三口示例井移动为零不能代表 hidden 井。

`model-package correction` 不属于原 6.781 active 路径。没有因为示例井结果而判定它对 hidden 无效，但它应作为独立提交验证，不能混入首个 PF 修改版本，否则无法归因。

## 推荐提交矩阵

为了识别真实增益，下一轮应保留下面三个配置：

| 提交 | Gold | learned GS | seed hedge | 目的 |
|---|---:|---:|---:|---|
| P0 | conservative | `1.00` | 关闭 | 原 6.781 重现/运行一致性 |
| H1 | conservative | `1.00` | 开启 | 只测 seed-branch hedge |
| G1 | balanced | `1.00` | 关闭 | 单独测 balanced Gold |
| H2 | balanced | `1.00` | 开启 | 仅在 H1/G1 至少一项通过后测组合 |

对应修改 `CELL A0` 即可：

```python
# P0
os.environ["ROGII_GOLD_PROFILE"] = "conservative"
LEARNED_LIKPF_GR_SIGMA_MULTIPLIER = 1.00
RUN_PF_SEED_BRANCH_HEDGE = False

# H1
os.environ["ROGII_GOLD_PROFILE"] = "conservative"
LEARNED_LIKPF_GR_SIGMA_MULTIPLIER = 1.00
RUN_PF_SEED_BRANCH_HEDGE = True

# G1
os.environ["ROGII_GOLD_PROFILE"] = "balanced"
LEARNED_LIKPF_GR_SIGMA_MULTIPLIER = 1.00
RUN_PF_SEED_BRANCH_HEDGE = False

# H2
os.environ["ROGII_GOLD_PROFILE"] = "balanced"
LEARNED_LIKPF_GR_SIGMA_MULTIPLIER = 1.00
RUN_PF_SEED_BRANCH_HEDGE = True
```

如果提交预算有限，先跑 `H1`。`G1` 用来单独判断 balanced visible-prefix；不要直接用 H2 混合两个尚未独立通过的改动。

## 输出检查

最终应生成：

- `submission.csv`
- `submission_mainline_6p781_pf_patch.csv`
- `submission_mainline_6p781_pf_patch_audit.json`
- `submission_before_pf_seed_branch.csv`
- `submission_pf_seed_branch_hedge.csv`
- `pf_seed_branch_hedge_report.csv`

`PF SEED-BRANCH HEDGE` 日志里的 `statistics_wells / submission_wells` 应接近 `1.0`。若明显偏低，应先检查 SP45 PF 是否对某些 hidden 井异常回退，不能直接提交。
