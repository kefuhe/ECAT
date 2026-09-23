# BLSE 固定几何 L-curve

当数据、断层几何、网格和约束已经检查通过，但平滑强度尚未确定时，在**同一个固定几何
模型**上扫描 penalty weight。独立模板是
[`scripts/test_BLSE_L_Curve.py`](../../scripts/test_BLSE_L_Curve.py)。

普通 BLSE、平滑搜索和倾角搜索是不同科研任务。不要为了减少脚本数量而把它们强制合并：

- 普通 BLSE 用于复现一个已选参数模型；
- 平滑搜索用于检查数据拟合与模型粗糙度的权衡；
- 倾角搜索用于在固定平滑强度下比较几何；
- 倾角 × 平滑搜索用于最后检查两者耦合。

## penalty weight 的含义

BLSE 目标中，penalty weight 控制 Laplacian 平滑项。数值越大，平滑约束越强。与
`alpha` 的关系是：

```text
penalty_weight = 1 / alpha
```

若配置使用 `alpha.log_scaled: true`，则 `alpha: -2` 表示：

```text
alpha = 10^-2 = 0.01
penalty_weight = 100
```

搜索模板直接使用 penalty weight，避免同时混入 `alpha`、`log10(alpha)` 和倒数三层
解释。运行时传入 penalty weight 会覆盖当前运行读取的 alpha 初始值，但不会修改 YAML。

## 复制并修改模板

把模板复制到线性反演目录，与 `default_config_BLSE.yml` 和 `bounds_config.yml` 放在一起：

```bash
python test_BLSE_L_Curve.py
```

第一轮需要修改数据路径、固定断层几何和配置文件名。模板中的默认候选范围与既有
`test_slip_inv_BLSE.py --mode loop` 一致：

```python
penalty_weight_candidates = [
    1.0,
    5.0,
    10.0,
    30.0,
    50.0,
    80.0,
    100.0,
    125.0,
    150.0,
    200.0,
    250.0,
    300.0,
    400.0,
    500.0,
    600.0,
    800.0,
    1000.0,
]

preferred_penalty_weight = 100.0
```

`preferred_penalty_weight` 只在图上做参考标记，不代表程序自动选择的最优值。完成第一轮
宽范围扫描后，应把列表缩小到 L-curve 转折附近，再检查较密的局部范围。

主配置必须保持：

```yaml
alpha:
  enabled: true
```

若关闭 alpha，模板会明确报错，而不是运行一组实际相同的无平滑模型。数据权重、poly、
bounds 和 rake 在整个扫描中保持不变。

模板中的 `faults_list` 可以放一个或多个已经完成 mesh 的固定断层，`gpsdata` 与 `insardata`
也会一起进入每个候选。这里扫描的是一个 scalar `penalty_weight`；底层按当前 alpha 布局把
它解析为活动平滑源共同使用的权重。因此它不是“只能单断层”，但也不是多个断层或多个
平滑组的独立多维搜索。需要各组分别变化时，应另行设计候选组合并明确记录每一维含义。

脚本保留了近断层 InSAR 过滤的注释位置。只有已经识别出失相干、解缠错误或无法解析的
破裂带时才启用，并且必须在 inversion 创建前完成；过滤会改变所有候选共同使用的观测覆盖，
不能用作改善拟合的默认步骤。

## 计算和输出

断层几何、GF、Laplacian 和约束只建立一次；规范扫描入口统一负责候选求解、固定计算
复用、统计收集和状态恢复：

```text
固定 fault + geodata + config
  -> 建立一次 G、L、bounds 和 rake constraints
  -> scan_penalty_weights
       -> penalty 1：run(report="none") -> global + dataset statistics
       -> penalty 2：run(report="none") -> global + dataset statistics
       -> 恢复扫描前的活动模型和数据合成结果
```

模板直接调用：

```python
summary, fit_stats = inversion.scan_penalty_weights(
    penalty_weight_candidates,
    include_fit_statistics=True,
    verbose=verbose,
)
```

`summary` 每个候选一行，`fit_stats` 每个候选、每个数据集一行。扫描不会自动激活某个
候选；选定权重后仍须使用普通 BLSE `run()` 显式重跑。

默认输出：

| 文件 | 内容 |
| --- | --- |
| `blse_l_curve_results/blse_l_curve.csv` | penalty、等价 log10(alpha)、按断层顺序解析的实际权重、roughness、全局 RMS/VR 和观测单位 |
| `blse_l_curve_results/blse_l_curve_fit_statistics.csv` | 每个候选的逐数据集 RMS/VR、poly、数据类型及分组信息 |
| `blse_l_curve_results/blse_l_curve.png` | penalty–RMS、penalty–VR 和线性 roughness–RMS |
| `blse_l_curve_results/blse_l_curve_roughness_rms.png` | 单独的经典 roughness–RMS L-curve，便于论文排版 |

CSV 不记录单一 `fault_dip_deg` 或单一 `n_patches`：它们无法准确描述多断层、分段倾角或
倾角剖面。固定几何本身由本次运行的 fault 构建段和配置负责复现。模板仍明确展示
“准备模型 → 扫描 → 保存两张表 → 绘图 → 选定后重跑”；容易错位的状态恢复和缓存复用由
`scan_penalty_weights()` 负责。三联图和单幅 L-curve 分别由
`plot_blse_lcurve_summary()` 与 `plot_blse_roughness_rms()` 根据同一规范摘要生成；两者都不
接收 inversion 对象，也不会触发求解或统计重算。

三联图默认按论文双栏宽度排成三个面板，单幅 L-curve 默认使用论文单栏宽度；两者的轴标签
均为 9 pt、刻度为 8 pt，并以 300 dpi PNG 保存。常用绘图与输出选项集中放在搜索设置之后：

```python
plot_options = dict(
    figsize="double",
    figsize_unit="inch",
    label_fontsize=9,
    tick_fontsize=8,
    style="science",  # 可改为 "science-serif"
)
single_lcurve_options = {
    **plot_options,
    "figsize": "single",
    "legend_loc": "best",
}
figure_formats = ("png",)  # 同时输出矢量图可改为 ("png", "pdf")
figure_dpi = 300
```

`figsize` 也可以写成数值二元组；此时 `figsize_unit="cm"` 可直接使用厘米。PDF 是矢量
输出，`figure_dpi` 主要影响 PNG 等栅格格式。RMS 轴直接使用摘要中的
`observation_unit`，规范绘图层不猜测或转换单位。前两幅只对 penalty 横轴取对数；每个十倍
区间保留 2–9 的 minor tick marks，并在上下边框明确显示，但不显示 minor labels。第三幅
roughness–RMS 使用线性坐标，以保留脚本中直观的权衡曲线。
默认 `science` 风格为无衬线字体，函数会在退出样式上下文前固化实际字体，避免保存阶段受
外部 Matplotlib 全局设置影响。

摘要的 roughness 使用每轮求解发布的未乘 penalty 的 \(L_0\)，因此在固定几何和同一
Laplacian 下可比较不同候选。`resolved_penalty_order` 和
`resolved_penalty_weights` 以 JSON 数组记录按断层顺序解析后的实际权重，可以安全经过
CSV 往返。全局 RMS/VR 直接来自组装后的求解向量，不是逐数据集指标的平均值。

模板使用 `data_poly="config"`，因此拟合统计包含配置中实际求解的 offset/ramp。完整统计
定义见 [Fit Statistics](../reference/fit_statistics.md)。

旧脚本仍可用 `simple_run_loop()` 获得历史四列表和单幅 roughness–RMS 图；模板中保留了
注释调用用于迁移对照。新工作流应直接使用 `scan_penalty_weights()`，避免丢失单位和逐数据集
长表。

## 如何选择

不要只取 RMS 最小值；减弱平滑通常就能降低 RMS。至少同时检查：

- roughness–RMS 的转折区；
- 各 InSAR 轨道或 GPS 数据的逐数据集 RMS/VR；
- 残差中是否仍有 ramp、周跳或局部系统结构；
- 滑动是否过度集中、出现棋盘格或被过度抹平；
- penalty 在转折点附近变化时，主要滑动区是否稳定。

选定 penalty 后，用普通 BLSE 模板重新运行并导出完整滑动和残差结果，不需要为全部候选
保存大批 GMT/PDF 文件。

## 下一步

- 已选好平滑强度，需要比较倾角：进入
  [BLSE 固定拓扑倾角搜索](04b_blse_dip_search.md)。
- 怀疑倾角选择随平滑强度明显变化：进入
  [倾角 × 平滑参数敏感性分析](04c_blse_dip_smoothing_search.md)。
- BLSE 配置、约束和结果解释：回到
  [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md)。
