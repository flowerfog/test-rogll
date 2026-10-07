# 从 ABCD 范式重置到 S5 的实验总结

## 1. 总结范围与起点

本总结从 `ROGII_5x_paradigm_reset_action_plan_v6_zh.pdf` 开始。该报告提出四条新主干：

- A：完整轨迹 / 地层状态蒸馏。
- B：Raw-window GBDT。
- C：TVT-CorrNet 稠密对应。
- D：全曲线 Energy / 候选路径优化。

当时的出发点不是否定现有成绩，而是重新定义研究目标：

- 已确认的部署安全链为 `SP45/FLE 系 -> CP12 -> Gold -> Neighbor`，Public LB 为 `6.882`。
- 这条链由多层保守修正构成，适合作为安全专家，但很难仅靠 cap、weight 和顺序微调进入 `5.x`。
- 高位选手公开信息表明：不使用邻井，仅用本井 horizontal、typewell 和可见前缀，也能把 pooled 5-fold CV 做到约 `5 ft` 甚至以下。
- 因此主目标改为训练一个新的独立单井模型，而不是继续给旧链增加补丁。

后续统一验证协议逐渐收紧为：

- 随机 GroupKFold by well；fold 1 作为统一 gateway。
- 只在官方评分行计算 pooled RMSE。
- pseudo-PS 必须留在原井 fold 内。
- 单模型输入不得依赖 Neighbor、Gold、CP12 或 hidden/LB 反馈。
- 单折 `RMSE <= 7.0` 才算架构通过，`<= 5.5` 才达到目标。
- no-GR 和 shuffled-typewell 应至少各造成约 `0.30 ft` 退化，才认为模型真正使用了 GR 对应。
- 未通过单折架构门的模型不扩第二 seed 和五折。

## 2. 第一阶段：直接验证 ABCD 四条主干

### A. 完整轨迹 / 地层状态蒸馏

实验：`trajectory_geology_state_gateway_v0`

设计：

- 用完整 MD/X/Y/Z/GR、typewell 特征预测未来轨迹。
- 训练期六个地层面只作为 privileged auxiliary labels。
- 比较 geometry-only、GR/typewell-only、combined 和附加状态头。

结果：

- 外层状态分类 accuracy `0.9785`，但 balanced accuracy 仅 `0.6601`。
- 最佳轨迹模型 `combined_state = 14.109 ft`。
- `geometry = 14.172`，`combined = 14.213`，`geometry_state = 14.540`。
- `gr_typewell` 单独达到 `101.736`，完全不可用。

结论：

- 轨迹和地层状态确实存在可学习信号，状态蒸馏不是空的。
- 但这些低维状态不足以恢复精确 TVT；分类正确不等于累计轨迹误差小。
- A 作为 standalone 主模型关闭，状态特征只保留为潜在辅助监督。

### B. Raw-window GBDT

实验：`raw_window_gbdt_onefold_v0`

设计：

- 在固定物理尺度上铺开 horizontal GR、未来 GR、typewell GR 与 geometry，共 `560` 列。
- 并行比较 `truth_delta`、`truth_s_delta` 和 direct `truth_tvt`。
- 做 no-future-GR 与 no-typewell 消融。

结果：

- 最佳 `truth_delta = 13.408 ft`。
- `truth_s_delta = 13.787`；direct `truth_tvt = 29.054`。
- 移除未来 GR 后为 `13.421`，仅退化 `0.013`。
- 移除 typewell 后为 `13.858`，退化 `0.450`。
- 同折历史参考为 strong `8.981`、strong_bias `8.876`、CP12 `8.796`。

结论：

- typewell 原始窗口含有真实信号，但固定宽度表格表示没有解决全局模式歧义。
- 未来 GR 几乎没有被 GBDT 有效利用，说明“把原始窗口直接摊平”不是赢家所指的关键输入表达。
- B 的当前表示关闭，不继续做三模型五折集成。

### C. TVT-CorrNet 稠密对应

#### C0. 容量与坐标守卫

实验：`tvt_corrnet_capacity_audit_v1`

- teacher decode RMSE `0.572 ft`，标签覆盖 `100%`。
- 1 井过拟合 `0.488`，4 井 `0.744`，24 井 `0.351`。

含义：面板轴、标签映射和网络容量没有基础错误；后续失败主要是跨井泛化，而不是模型连训练样本都拟合不了。

#### C1. 原始 CorrNet v2

实验：`tvt_corrnet_onefold_v2`

- full 模型 `14.286 ft`。
- 独立训练的 no-GR arm 反而为 `14.131`；shuffled-typewell arm 为 `14.469`。
- 对同一 full checkpoint 做输入干预时，no-GR 退化 `0.589`，shuffled-typewell 退化 `0.919`。
- 与 CP12/strong_bias 做 5% 以上融合全部变差；最佳增益仍为负。

含义：

- full checkpoint 内部确实用到了 GR/typewell。
- 但跨井训练时模型也学到了更强的错误 shortcut；因此“因果使用了 GR”与“GR 使用正确”是两回事。
- 原始 C 架构关闭，但 TVT 对应思想保留。

#### C2. multi-PS CorrNet v3

实验：`tvt_corrnet_multips_onefold_v3`

- 训练使用约 `3,090` 个 pseudo-PS 面板，而不是每井一个官方起点。
- 可见 seed 的最佳 fold-1 RMSE 约为 `14.023` 和 `13.642`，仍远离架构门。

含义：扩大锚点状态分布可以降低训练损失，但不能单独修复对应表示与跨井泛化。

#### C3. 中心轨迹审计与 heel-conditioned v4

中心审计：

- 纯 raw-prefix anchor TVT 为 `15.910`。
- 最佳合法单井 PF 中心 `likpf_scale_12 = 10.962`。
- 该中心在 `+/-64 ft` 内覆盖约 `99.778%`，适合做窄状态轴。

实验：`tvt_corrnet_heel_conditioned_onefold_v4`

- PF 中心 `10.944`，模型输出 `10.124`，改善 `0.820`。
- median well RMSE `5.379`，但 p95 为 `18.783`，pooled 仍被尾部井主导。
- no-GR 退化 `0.412`，shuffled-typewell 退化 `0.655`，因果信号通过。

后续 RMSE fine-tune：

- 最优 dose `0.9` 仅到 `10.117`。
- 新训练没有超过原 checkpoint，说明剩余差距不是学习率或 loss 权重的小问题。

结论：v4 是第一个明确证明 GR/typewell 对应能产生独立改善的 CorrNet，但架构精度仍不足。

### D. 全曲线 Energy / 候选路径优化

D 的出发证据来自此前整路径候选实验：235 条路径的 whole-well oracle 可到约 `3.43`，但 heuristic 和 learned ranker 分别约 `11.66` 和 `9.31`。这说明候选生成很强，而部署选择证据很弱。

#### D1. E1 posterior path reranking

实验：`gr_path_posterior_mean_gateway_v1`

- fold 5 校准时，从 strong `10.141` 改善到 `9.520`，增益 `0.620`。
- 冻结配方后应用 fold 1，从 strong `8.981` 变成 `9.037`，反而下降 `0.056`。
- fold 1 正收益井比例仅 `51.6%`。

结论：温度、Top-M 和 dose 能在校准折拟合得很好，但无法跨折迁移。E1 posterior 关闭。

#### D2. Shape-first 候选篮子

实验：`shape_basket_ceiling_audit_v0`

- S 空间插值相对 TVT 插值稳定改善约 `0.05-0.056 ft`，5/5 折同向。
- B8 最佳单体 Neighbor 为 `8.818`，whole-well oracle 为 `7.014`。
- 加 CP12 的 B9 oracle 为 `6.894`。
- Top-40 oracle 改善约 `5.99-6.11 ft`。

结论：候选篮子存在可观 ceiling，但还没有达到 `5.5`，且 oracle 主要依赖逐井真值选择。

#### D3. dSSE pairwise/listwise selector

实验：`dsse_shape_selector_nested_v0`

- pruned B7 的 nested selector 最优选择是完全回退 safe Neighbor，增益 `0`。
- B8+CP12 selector 为 `8.824`，相对 safe `8.818` 下降约 `0.006`。
- 尽管 whole-well oracle 约 `7.09-7.21`，模型无法可靠判断何时换路径。

结论：D 的主要瓶颈被定位为 selector，不是是否能制造不同形状。

## 3. 第二阶段：从 ABCD 收敛到完整单模型矩阵

ABCD 的共同结果是：

- A/B 的输入含有弱信号，但 standalone 在 `13-14 ft`。
- C 能学到真实 GR 对应，但跨井泛化仍在 `10-14 ft`。
- D 有候选 ceiling，却没有可迁移选择器。

因此后续不再围绕旧链做增量，而把 PDF 中仍未严格覆盖的单模型假设整理成 S1-S5。

### S1. 全井 dual encoder + Neural CRF

先测试 full-well 2D U-Net：

- 原 checkpoint 约 `15.665`。
- decoder audit 最好 posterior mean `15.554`；DP 约 `16.89`，argmax `17.18`。
- 说明 posterior 中不存在一个被简单 decoder 隐藏的好路径。

随后测试 `fullwell_neural_crf_onefold_v0`：

- 轴覆盖和转移覆盖守卫通过。
- Viterbi 最佳为 `18.045`。
- exact forward-backward posterior 审计的最优乐观配置为 `14.969`，比 anchor `15.543` 好 `0.574`。

结论：当前 emission encoder、CRF loss 和 posterior decoder 均无法形成可用全井路径，S1 关闭。

### S2. 部署合法的 prefix meta-adaptation

实验：`tvt_corrnet_prefix_meta_adapter_v0`

- 冻结 v4 为 `10.124`。
- 全 dose adapter 为 `10.110`，只改善 `0.014`。
- 同折最优 dose 0.5 为 `10.067`，乐观改善 `0.058`。
- 107/155 井可完成适配；shuffled adapter 变差到 `10.219`。

结论：可见前缀确实能估计少量井内 domain shift，但信号太小，不能把失败架构变成强模型。

### S3. Axial / RAFT 迭代对应

实验：`tvt_corrnet_axial_raft_onefold_v0`

- 中心为 `10.944`，Axial/RAFT 输出 `9.687`，改善 `1.257`。
- median well RMSE `5.728`，p95 `16.172`。
- no-GR 仅退化 `0.237`，shuffled-typewell 仅退化 `0.136`，未通过因果门。

结论：迭代 cost-volume 更新明显优于 U-Net/v4，是这一阶段最好的 standalone 结果；但仍高于 `7.0`，且仍依赖 shortcut。S3 关闭，不扩五折。

### S4. 校准合成预训练

先完成 sim-to-real 审计：

- naive synthetic 的 domain AUC `0.999`，明显失真。
- residual-block 降至约 `0.755`；进一步冻结为 orthogonal residual transfer、scale 1.0。
- 合成 local matcher 对真实样本有小幅 within-8 改善，但不足以独立解码轨迹。

完整模型 A/B：`tvt_corrnet_synthetic_pretrain_gateway_v0`

- matched scratch 为 `9.847`。
- synthetic-pretrained 为 `9.788`，只改善 `0.060`。
- 历史 S3 最佳仍为 `9.687`，因此 S4 没有刷新最佳。
- no-GR 退化 `1.483`，shuffled-typewell 退化 `0.652`，是最强因果读数。

结论：合成预训练能迫使模型真正使用 GR/typewell，并修复 shortcut；但它主要改善少数高 SSE 井，不能解决架构精度。S4 关闭。

### S5. 物理 S 路径库 + 深度 listwise scorer

实验：`fullwell_listwise_path_scorer_s5_onefold_v0`

设计：

- 不围绕 strong/PF 建候选，而从可见 heel 锚定 fold-clean 的物理 S(MD) 模板。
- 训练样本为完整 panel/well state，共 `3,088` 个 cross-fit pseudo-PS。
- 共享序列编码器做候选相对评分，最终输出 posterior mean，不做硬选。

结果：

- full candidate oracle `5.992`，裁剪 64 条后的 oracle `6.142`。
- heuristic top-1 约 `103.95`，说明单纯 GR 代价几乎不会选路径。
- posterior mean `11.883`，优于 hard MAP `14.564`，但离 oracle 仍差约 `5.74 ft`。
- no-GR 退化 `3.448`，shuffled-typewell 退化 `4.129`，因果信号非常强。

结论：模型确实在做 GR matching，posterior mean 也比硬选正确；但候选库自身尚未达到 `5.5`，选择器更远未回收 oracle。S5 关闭。

## 4. 整体证据链

### 已经被充分支持的结论

1. GR/typewell matching 不是无效信息。
   - v4、S4、S5 的 no-GR 与 shuffled-typewell 均有明确退化。

2. “模型使用了 GR”不等于“模型能选对全局模式”。
   - S4/S5 的因果消融很强，但 pooled RMSE 仍为 `9.79/11.88`。

3. 候选生成与候选选择是两个独立瓶颈。
   - strong-centered 老候选库 oracle 可到约 `3.43`。
   - 完全 standalone 的物理 S 菜单 oracle 约 `5.99-6.14`。
   - 实际 learned selector 通常停在 `9-12`。

4. posterior mean 通常优于 hard MAP/argmax，但不是万能补救。
   - S1 posterior 优于 Viterbi；S5 posterior 优于 MAP。
   - posterior 中若没有校准好的模式概率，仍无法接近 oracle。

5. pseudo-PS、meta-adaptation 和 synthetic 都只能增强已有正确表示。
   - multi-PS 没有救活 v3。
   - prefix adapter 只改善 `0.014-0.058`。
   - synthetic 只改善 `0.060`，但显著增强因果依赖。

6. 当前最好的新 standalone 单折仍是 S3 的 `9.687 ft`。
   - 没有任何模型通过 `7.0` 架构门，因此没有第二 seed 或五折扩展对象。

### 被关闭的是哪些范围

- 当前 Raw-window 固定表格表示。
- 当前轨迹/地层状态蒸馏到 TVT 的实现。
- 当前 fixed panel U-Net、full-well U-Net、Neural CRF、heel-conditioned CorrNet、Axial/RAFT recipe。
- 当前低维 prefix adapter。
- 当前 calibrated synthetic + S3 组合。
- 当前 physical-S candidate bank + listwise posterior scorer。
- 当前 shape basket 的 dSSE selector 与 E1 posterior 配方。

### 没有被数学上否定的内容

- 所有可能的 GR matching 网络。
- 高位选手所用但未公开的 complete-well 输入表达与训练 recipe。
- 更强的 target-specific proposal generator。
- 能把候选 oracle 压到 `4.5` 以下并校准模式概率的新方法。

## 5. 当前决策

- 部署保底仍是已被 hidden 验证的 `CP12 -> Gold -> Neighbor`，Public LB `6.882`。
- PDF 中有明确规格的 standalone 单模型路线已经全部完成 gateway 验证。
- 继续追 `5.5` 已不属于“补跑 PDF”，而是新的研究阶段。
- 下一阶段若继续，优先级应是 target-specific 多尺度 GR emission + k-best 路径并集，先把候选 oracle 压到 `<=4.5`；在此之前不应继续扩大 selector 或重复调 loss。

## 6. PDF 方案覆盖核对

| PDF 中的方案 | 对应本地验证 | 状态 |
|---|---|---|
| v6-A 轨迹/地层状态蒸馏 | `trajectory_geology_state_gateway_v0` | 已完成，14.109 |
| v6-B Raw-window GBDT | `raw_window_gbdt_onefold_v0` | 已完成，13.408 |
| v6-C TVT-CorrNet | capacity、v2、v3、v4、S1、S3 | 已完成，最佳 9.687 |
| v6-D 全曲线 Energy/优化 | E1、shape basket、DSSE selector、S5 | 已完成，selector 未回收 oracle |
| Updated-E1 候选曲线 posterior/reranker | `gr_path_posterior_mean_gateway_v1` | 已完成，跨折失败 |
| Updated-E2 Meta-adapted CorrNet | S2 prefix meta-adapter | 已完成，仅 +0.014 至 +0.058 |
| Updated-E3 Raw-window/cost-profile 表格模型 | Raw-window GBDT 与既有 cost-profile 消融 | 已完成，未过门 |
| Shape-first A：候选 S shape basket | `shape_basket_ceiling_audit_v0` | 已完成，oracle 约 7.01 |
| Shape-first B：shape 内 offset posterior | E1 posterior 与后续 S5 posterior | 已完成，不能跨折兑现 |
| Shape-first C：跨 shape pair/listwise selector | `dsse_shape_selector_nested_v0`、S5 | 已完成，选择失败 |
| Shape-first D：S posterior mean/安全收缩 | E1、S1 posterior audit、S5 posterior | 已完成，posterior 优于硬选但仍不足 |
| Heel episodic/meta adapter | S2 | 已完成，信号太小 |
| Synthetic pretraining | domain audit、generator calibration、local matcher、S4 | 已完成，完整模型仅 +0.060 |
| Axial/RAFT 对应 | S3 | 已完成，9.687 |
| Whole-path neural scorer | S5 | 已完成，11.883 vs oracle 6.142 |

以下项目不是遗漏，而是 PDF 自身规定的条件分支，因前置门失败而没有启动：

- 单模型第二 seed 与完整五折：没有模型达到单折 `7.0` 架构门。
- domain-specific head / MoE：共享 backbone 没有先成为强模型。
- 大规模 100k synthetic 预训练：真实 backbone 未通过，bounded S4 已证明边际仅 `0.060`。
- 新模型叠加 Gold/Neighbor 与最终 non-negative fusion：没有合格的新 standalone 专家可进入融合。
- Kaggle SUB-B/SUB-C：没有通过 OOF 门的新机制，不应消耗提交。
