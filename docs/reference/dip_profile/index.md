# 倾角剖面模式总览

本组页面说明非分层断层沿走向变化倾角时，top、strike、dip 和 bottom 怎样配合。
如果研究的是随深度变化的多层断层，应回到
[倾角随深度变化](../fault_geometry_construction.md#layered-dip)；两者当前不是同一个参数模型。

## 先选择计算生命周期

dip-profile 只回答“怎样从 top、走向和沿走向倾角场生成 bottom”。它不等于固定拓扑，也不
等于 Bayesian 几何扰动。先按实际任务选择 mesh 生命周期，再选择下一节的走向来源：

| 任务 | 推荐入口 | 最短路径 |
| --- | --- | --- |
| 一组确定 controls，只运行一次 BLSE/VCE | [固定几何中的倾角剖面](fixed_geometry.md) | profile → 零扰动物化 bottom → `generate_mesh(...)` |
| 用 BLSE/VCE 比较多个几何且保持 patch 对应 | [固定拓扑倾角搜索](../../workflows/04b_blse_dip_search.md) | reference → `remap=True` → 候选 `remap=False` |
| 在 SMC-FJ 中采样 controls | [sampled/fixed/transition 组合](bayesian_mixed.md) | reference → 可选候选期加密 → mapping → sampler 重放 |

固定几何不需要为了使用 profile 而建立 `snapshot()` 和参数 mapping；固定拓扑和 SMC 才需要
冻结 reference。

## 再选择边界分辨率的所有者

下面四类设置都可能出现“间隔”或“点数”，但它们控制不同对象。按表中优先级选择，不要把
数值相同理解为同一个设置。

| 优先级 | 任务 | 推荐入口 | 生效范围 |
| --- | --- | --- | --- |
| 1 | 希望加密后的迹线就是模型输入 | `discretize_trace(every=...)` | profile 声明前，改变权威 trace/top |
| 2 | 单个固定几何只在建 bottom 时临时加密 | `discretization_interval=...` | 当前一次 dip 方法调用 |
| 3 | 固定拓扑比较或 SMC 的每个候选都要加密 | `set_densification(interval=...)` | 冻结 reference 及全部候选重放 |
| 4 | 已有边界已经确定，只调 mesh/mapping 分辨率 | `top_size/bottom_size`、`num_segments`、`disct_z` | Gmsh 拓扑或参数映射，不重定义 profile 边界 |

通常先选第 1 项；需要保留稀疏权威迹线的单次 BLSE/VCE 才选第 2 项；需要逐候选一致重放时
只选第 3 项。第 2、3 项不能同时启用，系统会直接拒绝双重加密来源。Bayesian 配置文件不
接受 `geometry.densification`；候选加密应在 Python 几何准备段中声明并在建立 mapping 前
物化。

所有间隔均为有限正数，单位 km。节点数应通过明确的 `num_segments` 参数表达；负间隔不再
重载为节点数。可复制的固定几何写法见
[固定几何中的倾角剖面](fixed_geometry.md#按便利性选择一种加密方式)，候选重放写法见
[sampled/fixed/transition 组合](bayesian_mixed.md#稀疏-top-与候选加密)。

## 再选择走向模式

| 目标 | 应选模式 | 核心设置 |
| --- | --- | --- |
| 固定一组沿走向倾角，只生成一次 mesh | [固定几何中的倾角剖面](fixed_geometry.md) | 零扰动物化后使用 `generate_mesh(...)` |
| 曲线 top 的下倾方向跟随局部切线 | [top 局部走向](local_strike.md) | `use_average_strike=False`，不提供 strike controls |
| 近直线断层全部使用同一走向 | [单一代表性走向](representative_strike.md) | `use_average_strike=True`，选择 `user` 或 `pca` |
| 已有逐段地质走向控制 | [控制点走向插值](controlled_strike.md) | 四列 `xydip`，`use_average_strike=False` |
| Bayesian 中混合自由、固定控制点和渐变区 | [sampled/fixed/transition 组合](bayesian_mixed.md) | `set_dip_profile(...)` |
| Bayesian 前检查转折位置和转换带尺度 | [曲率与转换带预分析](transition_preflight.md) | `analyze_reference_top_curvature(...)` |
| 旧 Bayesian 倾角控制脚本升级 | [旧调用迁移](migration.md) | 把位置、角色、轴和过渡区集中进 profile |

单一代表性走向的 `user` 和 `pca` 是同一模式的两个走向来源，不是两套几何协议。
Bayesian `DipProfileSpec` 当前只保存三列位置和倾角；四列 strike-control 插值属于普通
`AdaptiveTriangularPatches` 构模接口，不能直接塞进 `set_dip_profile()`。

Bayesian profile 的每个 sampled/fixed control 和 transition anchor 可以独立使用 lon/lat、
fault-local x/y，或 reference top 的首端/尾端里程；完整协议和混合示例见
[sampled/fixed/transition 组合](bayesian_mixed.md#位置声明协议)。

如果 transition 的中心和半宽尚不确定，先运行
[Bayesian 前的曲率与转换带预分析](transition_preflight.md)。该工具只提供可审阅建议，最终
设置仍使用同一套 `transition_zones` 协议。端点诊断默认使用相对峰值法，也可显式比较累计
转角覆盖率；正式 profile 的区间内形状默认保持线性，可按 transition 选择 `smoothstep`。

## 共同角度约定

strike 是从北顺时针量取的地理方位角。角度按周期解释：

\[
\theta\equiv\theta+360^\circ k,\qquad k\in\mathbb Z.
\]

例如 `-90° == 270°`、`0° == 360°`。局部 top 走向由有序节点计算：

\[
\theta_i=90^\circ-\operatorname{atan2}(\Delta y_i,\Delta x_i).
\]

内部节点使用相邻线段方向的圆周平均，因此 `350° -> 10°` 经由 `0°`，而不是经由
`180°`。top 点序 `top_coords[0] -> top_coords[-1]` 决定正走向。

倾角有效输入域为：

\[
[-90^\circ,0^\circ)\cup(0^\circ,180^\circ).
\]

推荐新脚本使用带符号形式
`[-90°, 0°) ∪ (0°, 90°]`。兼容形式满足：

\[
-\delta\equiv180^\circ-\delta,\qquad 0^\circ<\delta\le90^\circ.
\]

所以 `-30° == 150°`。内部先转换到连续 `(0°, 180°)` 再插值，使左右侧切换经过
垂直 `90°`，不会经过水平 `0°`。`0°`、`180°`、非有限值和越界候选都会被拒绝。

参考 control dip 一律使用度；`angle_unit="degrees"|"radians"` 只解释 Bayesian
候选扰动量。

## top、strike 与 bottom 的几何关系

在 fault-local 坐标中，x 向东、y 向北。走向切向量和右手侧水平法向分别为：

\[
\mathbf t=(\sin\theta,\cos\theta),\qquad
\mathbf r=(\cos\theta,-\sin\theta).
\]

正倾角沿右手侧下倾。设深度差为 \(h\)，则：

\[
\Delta x=h\cot\delta\cos\theta,\qquad
\Delta y=-h\cot\delta\sin\theta,\qquad
\Delta z=h.
\]

因此 top 到 bottom 的连接线本来就应近似垂直于 top 走向；它不应沿 top 延伸。负倾角在
底边生成的局部副本中等价为 `strike + 180°` 与 `abs(dip)`，只翻转一次下倾侧。

## 共同数据和索引约定

- `is_utm=False`：位置是 lon/lat。
- `is_utm=True`：位置是与 fault 相同投影下的 fault-local x/y，单位 km；不是以 m 为单位的
  原始 UTM easting/northing。
- 空间排序只服务于插值；top、bottom 和候选参数布局不会因此重排。
- `bottom_coords[i]` 始终由 `top_coords[i]` 生成。
- `top_strike/top_dip` 是底边构造阶段的逐节点参考元数据；最终 patch 的 canonical
  strike/dip 应从实际顶点和 `getpatchgeometry()` 获取。

## 共同检查

1. 先确认 top 点序与预期正走向一致。
2. 查看 top、bottom 和连接线，确认下倾侧符合 strike/dip 组合。
3. Bayesian profile 使用
   [`plot_dip_profile_diagnostics()`](bayesian_mixed.md#只读诊断)检查输入点投影、sampled/fixed
   角色和过渡区。
4. 若 top 在相邻节点处接近 180° 急转，应先平滑或分段；该位置的局部切向圆周平均没有稳定
   物理方向。
5. 最终使用实际 mesh/patch 做几何、法向与边界检查，不用元数据替代成品几何。

更一般的 strike/dip/rake 转换见
[断层走向、倾角与滑动基底约定](../../concepts/fault_angle_conventions.md)。
