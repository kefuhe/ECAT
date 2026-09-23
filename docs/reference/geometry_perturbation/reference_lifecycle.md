# 几何参考的建立与生命周期

本页只回答三件事：零扰动几何从哪里来，何时冻结，以及何时允许重新设基线。具体扰动方法
和参数路由见[几何扰动总览](../geometry_perturbation.md)。

## 核心不变量

每个候选都从同一个冻结参考独立生成：

```text
candidate_i = transform(geometry_ref, delta_i)
```

不会从上一候选累积。`geometry_ref` 定义零点；bounds 定义允许的增量范围；
`sample_positions` 定义从全局样本向量取哪一段。这三者不能互相替代。

## 按权威来源选择入口

| 权威参考 | 采样前入口 | 典型用途 |
| --- | --- | --- |
| 已明确构造的 top/bottom | `snapshot(capture_vertices=False, capture_layers=False)` | 迹线加倾角、解析边界、外部边界 |
| 当前 mesh 的实际 top/bottom | `set_edges_for_bayesian_optimization()` | 导入、裁切或人工修整后的 mesh |
| trace 作 top、当前 mesh 作 bottom | `set_edges_for_bayesian_optimization(use_trace=True)` | top 必须沿原 trace |
| 最终 mesh 的 vertices/faces | `snapshot(capture_vertices=True, capture_layers=False)` | 整体 mesh 平移、旋转或固定拓扑变换 |
| 多层边界 | `snapshot(capture_vertices=False, capture_layers=True)` | layer 或 multiLayer 方法 |
| 非分层倾角 profile | `set_dip_profile()`，可选 `set_densification()`，再物化一次零扰动 | 由 top/profile 生成 bottom 的 sampled/fixed/transition 组合 |

入口由权威状态决定，不由断层是直线还是曲线决定。`use_trace=True` 仍要求当前几何能提供
bottom，不是“只有 trace 就自动构造完整断层”。

## `snapshot()`

```python
ref = fault.snapshot(
    capture_vertices=False,
    capture_layers=False,
)
```

它复制当前 top/bottom，并按开关复制 layers 或完整 `Vertices/Faces` pair；已有 dip profile
和 densification 会整体保留。返回值同时保存到 `fault.geometry_ref`，数组是不可写副本。

再次调用会以**当前 fault 状态**替换参考。若 capture 开关为 false，旧 vertices/faces 或
layers 不会被暗中沿用。因此同一 SMC run 的样本循环、target 或 stage 内禁止重新 snapshot。
target 会保存构造时的 reference 对象身份；后续任何 reference 替换都会使旧 target 明确
报错，而不是让后半段样本静默改用新基线。完成 `set_dip_profile()`、`set_densification()`、
重新 snapshot 或其他 reference 演进后，应重新创建 target；若参数布局也改变，则重新创建
`BayesianMultiFaultsInversion`，使 bounds、样本列和报告一并对齐。

## `set_dip_profile()` 的 top 所有权

`set_dip_profile()` 有两种明确状态：对象没有 `geometry_ref` 时，当前 top 被捕获为最小
generator reference；对象已有 `geometry_ref` 时，已有 frozen top 保持不变，setter 只替换
profile。它不会读取并提升某个 current candidate top。这一规则保证重新配置 controls 不会
悄悄移动所有样本的几何零点。

若确实要把当前完整 top/bottom 作为下一轮独立反演的新基线，可在候选循环之外有意调用
`refresh_geometry_baseline()`。该入口保留已经声明的 profile 和 density policy，同时替换
它们的 frozen top/bottom；只有 profile 定义本身也要改变时，才随后重新调用 setter。这个
高级入口要求 current top/bottom 已共同通过科学检查，不属于普通 profile 修改步骤。

### 边界参考的常用准备顺序

```python
fault.snapshot(
    capture_vertices=False,
    capture_layers=False,
)

fault.generate_and_deform_mesh(
    top_size=3.0,
    bottom_size=6.0,
    num_segments=25,
    disct_z=10,
    remap=True,
    bottom_norm_offset=None,
    show=False,
    verbose=0,
)
```

这一路由先冻结权威边界，再建立采样期复用的固定拓扑映射；不需要为了取 reference 先建立
一套临时 mesh。

## `set_edges_for_bayesian_optimization()`

```python
fault.set_edges_for_bayesian_optimization(
    segs=25,
    sort_axis=0,
    sort_order="ascend",
    use_trace=False,
)
```

该入口从当前几何提取并排序边界，然后自动执行边界 snapshot。成功后不要机械地再 snapshot。
`sort_axis/sort_order` 决定正向点序，进而影响走向、下倾侧和节点对应，必须在采样前固定。

## `prepare_for_inversion()`

它是边界提取的便捷封装，也可同时声明 dip profile 和 densification：

```python
fault.prepare_for_inversion(
    segs=25,
    sort_axis=0,
    sort_order="ascend",
    dip_sampled_controls=sampled_controls,
    dip_fixed_controls=fixed_controls,
    dip_interpolation_axis="arc_length",
    # dip_perturbation_groups=["wm", "wm", "east"],  # 可选共享增量
    densify_num_segments=80,
)
```

`dip_perturbation_groups` 与 sampled controls 的声明顺序逐项对齐；省略时保留既有逐点或
标量广播语义。唯一标签数必须与 Bayesian `sample_positions` 的切片长度一致，完整映射表见
[Bayesian sampled、fixed 与 transition 组合](../dip_profile/bayesian_mixed.md#参数角色和顺序)。

复杂场景仍建议分步建立和检查 reference、profile 与 mesh，避免一站式调用隐藏权威来源。

## 稀疏参考与坐标加密

```python
fault.set_densification(interval=1.0)
# 或
fault.set_densification(num_segments=80)
```

加密规则保存在 reference 中，但不会把冻结的稀疏 top/bottom 或样本参数扩成密集节点。
密集坐标只在当前候选的走向、倾角、物理计算或 mesh 消费阶段临时生成，因此 mesh 分辨率
不会直接膨胀 Bayesian 几何维数。

候选密度只有这一个 Python 所有者。配置文件不接受 `geometry.densification`；采用
`set_densification(...)` 后，`update_fault_geometry` 方法参数也不要再设置
`discretization_interval`。双重来源会被拒绝，而不是由调用顺序决定谁覆盖谁。

`set_densification()` 只登记规则，不立即修改 current top/bottom；直接调用
`generate_and_deform_mesh()` 也不会替代候选 pipeline 去应用这条规则。生成型 dip profile
应先完成全部 reference setter，再只物化一次零扰动候选：

```python
profile = fault.set_dip_profile(...)
fault.set_densification(interval=2.0)  # 可选

fault.perturb_dips_with_preset_params(
    np.zeros(profile.perturbation_parameter_count),
)
fault.generate_and_deform_mesh(..., remap=True)
```

这里不调用 snapshot：冻结 top、profile 和 density 是输入，bottom 是候选输出。若准备完成
后又改变 profile 或 density，必须重新物化零扰动候选并重建 mapping；这属于 reference
合同变化，不是同一准备流程中的“第二次零扰动”。

采样中的同一 mesh 方法使用 `remap=False`：会重新解析当前候选、临时加密边界并变形已有
顶点，但不会重建 Gmsh 拓扑。加密增加的是与临时边界节点数线性相关的几何预处理，不增加
sampled controls、滑动参数、patch 或 Faces 数量。完整 transition 欠分辨判断和可复制设置见
[Bayesian sampled/fixed/transition 组合](../dip_profile/bayesian_mixed.md#稀疏-top-与候选加密)。

## 修改或重新设基线

| 目标 | 入口 |
| --- | --- |
| 用当前 top/bottom/layers/mesh 建立新 run | 再次显式 `snapshot(...)` |
| 替换整个非分层 dip profile | `set_dip_profile(...)` |
| 开始新 run，并有意用当前完整 top/bottom 重设 profile 基线 | 高级 `refresh_geometry_baseline()` |
| 改稀疏边界的加密规则 | `set_densification(...)` |

允许在开始一轮新的独立采样前重新设基线；不允许在同一 run 中随候选改 reference。只换
reference 还必须同步复核 `sample_positions`、bounds、方法依赖和固定拓扑映射。

## 旧 reference 入口

`set_top_coords_ref()` 与 `set_bottom_coords_ref()` 是仍可调用的 legacy 入口，会委托给
`snapshot(capture_vertices=False)`。新脚本应先设置 current top/bottom，再显式 snapshot。
它们不是扩展新功能的挂载点。

`_ensure_vertices_ref()` 是内部惰性兼容机制，不是用户 API。whole-mesh 方法应在最终参考
mesh 建好后显式捕获完整 vertices/faces；半套 pair 必须失败。

非分层倾角旧调用另见[旧倾角剖面调用迁移](../dip_profile/migration.md)。
