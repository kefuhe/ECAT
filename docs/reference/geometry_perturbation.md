# Bayesian 联合反演中的可扰动断层几何 / Perturbable Fault Geometry

本页是 `BayesianAdaptiveTriangularPatches` 的用户参考，重点说明几何参考从哪里来、
`ref*` 相关接口何时使用，以及几何扰动、网格更新和 YAML 怎样保持一致。第一次理解
状态关系，先读 [Bayesian 联合反演中的几何参考](../concepts/bayesian_geometry_reference.md)；
需要按顺序运行，读
[联合 Bayesian workflow](../workflows/05_joint_bayesian_geometry_slip.md)。

## 阅读路径

- 想先看一个样本怎样从配置流到 mesh、GF、GL 和 likelihood：看
  [一个样本从配置到似然的完整流转](../concepts/bayesian_geometry_reference.md#一个样本从配置到似然的完整流转)。
- 正在把现有曲线断层接入联合反演：看 [按参考来源选择入口](#按参考来源选择入口)。
- 不确定需不需要再调用 `snapshot()`：看 [建立参考的公开接口](#建立参考的公开接口)。
- 旧脚本中有 `set_*_ref`：看 [旧 ref 接口](#旧-ref-接口)。
- 需要修改参考：看 [修改和重新设基线](#修改和重新设基线)。
- 正在写 YAML：看 [YAML 三道开关](#yaml-三道开关) 和 [网格职责](#网格职责)。
- 倾角控制点或稀疏曲线：看 [dip 与 densification](#dip-与-densification)。

## 核心对象与不变量

| 对象 | 作用 |
| --- | --- |
| `BayesianAdaptiveTriangularPatches` | 对用户开放的可扰动三角断层对象 |
| `GeometryReference` | 一次采样运行中不可变的零扰动几何 |
| `GeometryState` | 单个样本从参考复制出的可变工作状态 |
| `DensificationConfig` | 稀疏控制点到密集边界的临时加密规则 |

每个样本满足：

```text
candidate = transform(geometry_ref, current_sample_delta)
```

不会从上一样本累加。`geometry_ref` 也不是先验或 bounds：它定义零点，bounds 定义允许
的增量范围，`sample_positions` 定义从全局样本向量取哪一段，扰动方法定义怎样应用增量。

## 按参考来源选择入口

| 你的权威参考 | 采样前操作 | 推荐入口 | 典型场景 |
| --- | --- | --- | --- |
| 已知或程序化构造的 top/bottom | 写入/构造两条边界并检查点序与下倾侧 | `snapshot(capture_vertices=False, capture_layers=False)` | 外部几何、迹线加倾角、解析构造、只扰动边界后重建 mesh |
| 当前 mesh 的 top/bottom 边 | 先导入、生成并完成必要的 mesh 修整 | `set_edges_for_bayesian_optimization()` | 实际 mesh 边界才是零扰动权威来源 |
| trace 作为 top、当前 mesh 作为 bottom | 先有 trace 和 mesh | `set_edges_for_bayesian_optimization(use_trace=True)` | top 必须严格沿原迹线，但 bottom 取当前 mesh |
| 最终 mesh 顶点和拓扑 | 先完成最终参考 mesh | `snapshot(capture_vertices=True, capture_layers=False)` | 直接整体平移、旋转或变换 vertices |
| 多层边界 | 先建立 `layers` | `snapshot(capture_vertices=False, capture_layers=True)` | layered-dip 或 `_multiLayerMesh` |
| 倾角控制点 | 先写入 dip controls，再完成 top/bottom | `set_dip_control_points*()` + 显式 `snapshot(...)` | 沿走向变化倾角 |

入口由权威状态决定，不由断层是直线还是曲线决定。曲线 top 若来自经过检查的迹线，bottom
若由明确倾角或控制点构造，就属于第一行，不需要先生成临时 mesh。只有导入、裁切、人工
修整或其他 mesh 操作后的实际边界才是基线时，才选择第二行。

`set_edges_for_bayesian_optimization(use_trace=True)` 只让 top 来自 trace；bottom 仍从当前
几何/mesh 提取。因此它不是“只有迹线就能完成准备”的入口。

`sort_axis` 和 `sort_order` 控制边界点序，并由此确定正走向以及依赖点序的下倾侧和控制点
对应关系；应在采样前选定并保持不变。

## 建立参考的公开接口

### `snapshot()`

```python
ref = fault.snapshot(
    capture_vertices=False,
    capture_layers=False,
)
```

把当前 `top_coords` 和 `bottom_coords` 复制进新的不可变 `GeometryReference`。两者必须
已经存在。已有的 dip control points 和 densification 配置会被保留。

| 参数 | 何时设为 `True` | 何时可设为 `False` |
| --- | --- | --- |
| `capture_vertices` | 方法直接变换现有 mesh 顶点 | 方法只改边界/控制点，随后单独生成或变形 mesh |
| `capture_layers` | 方法读取或扰动多层边界 | 单层 top/bottom 断层 |

默认会尝试捕获 vertices 和 layers。为了让依赖清晰，示例中建议显式写出所需值。
当 `capture_vertices=True` 时，应先确认当前 `Vertices` 和 `Faces` 都属于同一个最终 mesh；
只有 vertices 或只有 faces 的半套拓扑不是可用的 whole-mesh reference。

#### 返回值、保存位置和复制规则

`snapshot()` 完成四件事：

1. 从当前 fault 复制 `top_coords` 和 `bottom_coords`。
2. 按两个 capture 开关选择性复制 `layers`、`Vertices/Faces`。
3. 保留已有 reference 中的 dip control points 和 densification 配置。
4. 创建新的只读值对象，同时保存到 `fault.geometry_ref` 并返回它。

```python
ref = fault.snapshot(
    capture_vertices=False,
    capture_layers=False,
)

assert ref is fault.geometry_ref
assert ref.top_coords is not fault.top_coords
assert not ref.top_coords.flags.writeable
```

这一步不会重新生成 mesh，也不会把 reference 写入主 YAML、bounds 或单独的磁盘文件。
主 YAML 保存的是方法、参数切片和 mesh 更新规则，bounds 保存的是相对 reference 的允许
增量；参考坐标仍由建模脚本在创建 inversion 前构建并捕获。可复现工作应保存构建 reference
所需的输入几何、脚本参数和配置，而不是把 YAML 当成几何快照。

再次调用 `snapshot()` 会用**当前** fault 状态替换整个 reference。只有 dip controls 和
densification 会自动沿用；若本次把 `capture_vertices` 或 `capture_layers` 设为 `False`，
新 reference 不会保留旧的 vertices/faces 或 layers。这也是为什么 capture 开关应显式写出，
而且同一次采样运行中不应重新 snapshot。

#### reference 怎样被后续扰动使用

采样过程不直接改 reference。每个样本按以下顺序处理：

```text
fault.geometry_ref
  -> 复制为本样本 GeometryState
  -> 应用 sample_positions 指向的几何增量
  -> 生成或更新候选 mesh
  -> 写入本样本的 current fault 状态
  -> 计算 Green's functions、滑动解和似然
```

因此需要同时对齐四件事：reference 包含的方法所需字段、方法消费的参数个数、
`sample_positions` 的切片，以及 bounds 中对应增量的单位和范围。任何一个只描述其中一部分，
都不能独立定义完整的几何采样问题。

#### 最常用的边界参考流程

top/bottom 已由迹线、倾角或外部几何明确构造时，标准流程是先冻结边界，再从同一组边界
生成一次参数化 mesh：

```python
fault.snapshot(
    capture_vertices=False,
    capture_layers=False,
)

fault.generate_and_deform_mesh(
    top_size=3.0, bottom_size=6.0, num_segments=25, disct_z=10,
    remap=True,                 # 建立采样期复用的参数映射。
    bottom_norm_offset=None,    # 不在构网阶段移动 frozen baseline。
    show=False, verbose=0,
)
```

这里无需为了取得 reference 先调用 `generate_mesh()`，也无需调用
`set_edges_for_bayesian_optimization()`。后者只用于“当前 mesh 的实际边界才是权威基线”
的分支。

### `set_edges_for_bayesian_optimization()`

```python
fault.set_edges_for_bayesian_optimization(
    segs=25,
    sort_axis=0,
    sort_order="ascend",
    use_trace=False,
)
```

该入口从当前几何提取 top/bottom，可选统一离散化，然后自动执行
`snapshot(capture_vertices=False)`。因此调用成功后不需要再机械地 `snapshot()`。

适合 mesh 边界是权威参考的脚本，不是所有曲线断层的通用准备步骤。若后续选择直接依赖
vertices 的方法，应在最终参考 mesh 建好后再显式
`snapshot(capture_vertices=True, capture_layers=False)`。

### `prepare_for_inversion()`

这是对 `set_edges_for_bayesian_optimization()` 的便捷封装，并可同时配置 dip controls
和 densification：

```python
fault.prepare_for_inversion(
    segs=25,
    sort_axis=0,
    sort_order="ascend",
    dip_control_file="dip_control.txt",
    densify_num_segments=80,
)
```

它同样要求当前几何足以提取 bottom；不要把它理解为由空对象自动构建完整断层。简单
场景可以用一站式入口，复杂场景建议分开调用，以便逐步检查 top、bottom、dip controls
和 mesh。

## 修改和重新设基线

参考不可原地修改。正确模式是先修改当前建模状态，再通过公开接口生成一个新参考。

| 修改目标 | 推荐方式 | 何时使用 |
| --- | --- | --- |
| top/bottom/layers/mesh 全部按当前状态重建 | 再次调用 `snapshot(...)` | 修正基线后、开始一轮新的独立采样前 |
| 只替换 dip controls | `set_dip_control_points*()` 或 `update_dip_baseline(x, y, dip)` | 仍使用相同 top/bottom 时 |
| dip controls 不变，只刷新 top/bottom | `refresh_geometry_baseline()` 或无参数 `update_dip_baseline()` | 已有 dip controls 的 dip workflow |
| 改加密规则 | `set_densification(...)` | 已有完整 geometry reference 后 |

`refresh_geometry_baseline()` 当前是 **dip workflow 专用便捷入口**：现有参考必须包含
`dip_control_points`，否则会报错。普通 top/bottom 场景直接调用
`snapshot(capture_vertices=False, capture_layers=False)`。

重新设基线的安全边界：

- 可以：修改输入几何后，重新创建 inversion 或开始另一轮独立采样。
- 可以：把上一轮明确选择的 MAP/median 几何作为下一轮的中心，但要重新解释并记录
  bounds。
- 不可以：在 SMC 样本循环、目标函数或同一 run 的 stage 之间悄悄 snapshot。
- 不可以：只换 reference，不同步检查 `sample_positions`、扰动方法和 geometry bounds。

## dip 与 densification

### 倾角控制点

推荐入口：

```python
fault.set_dip_control_points(lon, lat, dip)
fault.set_dip_control_points_from_coords(coords, dips)
fault.set_dip_control_points_from_file("dip_control.txt")
```

默认输入为经纬度；局部投影坐标以 km 输入时写 `is_utm=True`。内部保存形式始终是
lon/lat。控制点可在完整 snapshot 前设置；此时对象会先创建只含当前可用字段的最小
reference。它是构建期过渡状态，完成 bottom 后仍应 `snapshot()`，让正式采样参考包含
完整 top/bottom；新的 snapshot 会保留 dip controls。

坐标标志只属于**输入边界**：`set_dip_control_points*()` 和直接接收控制点的
`perturb_dips()` 用它说明调用者提供的坐标。两个 preset 方法读取的是已经规范化的
reference，因此不接受 `is_utm`，也不要求用户再次声明坐标系；它们会把 reference 中的
lon/lat 自动投影为 fault 局部 x/y 后再插值。不要直接构造带投影 x/y 的
`DipControlPoints` 并塞入 `geometry_ref`，应通过 setter 建立 reference。

### 倾角插值轴

`perturb_dips()`、`perturb_dips_with_preset_params()` 和
`perturb_DipsPresetParams_SimpleMesh()` 的 `interpolation_axis` 接受四个严格值：

| 值 | 实际插值坐标 | 适用场景 | 主要限制 |
| --- | --- | --- | --- |
| `"x"` | fault 局部投影的 easting，单位 km | 近东西向且 x 单调的 top | 回折或近南北向 trace 可能把不同位置投到同一 x |
| `"y"` | fault 局部投影的 northing，单位 km | 近南北向且 y 单调的 top | 回折或近东西向 trace 可能把不同位置投到同一 y |
| `"auto"` | 对当前 top 做 PCA 后，从 x/y 中选择一个主轴 | 近直线但不想手工判断 x/y | 只会选择 x 或 y，不会自动切换到弧长 |
| `"arc_length"` | 当前 top 折线上的累计弧长，单位 km | 强弯曲或在 x/y 上不单调的 top | 不支持 `buffer_nodes`/`buffer_radius` |

`"arc_length"` 会先把每个控制点投影到当前 top 的最近线段，再按投影位置的累计弧长排序并
进行线性插值。控制点不必正好是 top 节点，但应靠近预期的 trace 分支，并投影到互不重复的
弧长位置。第一个和最后一个控制点之外使用最近端点的倾角；`N` 个控制点形成 `N - 1` 个
线性变化区间。例如三个控制点形成两个区间，不代表三个分段常值区域。

```python
import numpy as np

dip_controls = np.array([
    [100.00, 30.00, 70.0],
    [100.20, 30.15, 78.0],
    [100.45, 30.10, 85.0],
])

fault.set_dip_control_points_from_coords(
    coords=dip_controls[:, :2],
    dips=dip_controls[:, 2],
    is_utm=False,  # 本次输入为 lon/lat；preset 消费端无需再次声明。
)

# 三个零增量用于检查参考倾角场；正式采样时由 geometry sample slice 提供增量。
fault.perturb_dips_with_preset_params(
    perturbations=np.zeros(3),
    interpolation_axis="arc_length",
)
```

这里的弧长总是根据本样本实际参与计算的 top 得到；若 reference 配置了 densification，或
传入 `discretization_interval`，会先临时加密 top，再完成投影和插值。对于自相交、彼此紧邻
的回折分支，最近线段可能不是用户预期分支，应检查控制点投影顺序和生成的 `top_dip`。

本节只描述 Bayesian 几何扰动方法。断层构建阶段的
`interpolate_top_dip_from_relocated_profile()` 是另一套 API，当前只接受 `auto | x | y`；不要
因为名称相近而假定它也支持 `arc_length`。

### 坐标加密

```python
fault.set_densification(interval=1.0)
# 或
fault.set_densification(num_segments=80)
```

`set_densification()` 要求 geometry reference 已存在。加密规则保存于参考中，但冻结的
top/bottom 仍保留稀疏控制点；密集坐标只在走向、倾角插值、物理计算或网格消费时生成。
这使采样维数不随 mesh 分辨率一起膨胀。

## 旧 ref 接口

旧脚本可能包含：

| 兼容入口 | 当前等价行为 | 新文档推荐 |
| --- | --- | --- |
| `set_top_coords_ref(value)` | 可先替换当前 top，再 `snapshot(capture_vertices=False)` | `set_coords(..., coord_type="top")` 后显式 `snapshot(...)` |
| `set_bottom_coords_ref(value)` | 可先替换当前 bottom，再 `snapshot(capture_vertices=False)` | `set_coords(..., coord_type="bottom")` 后显式 `snapshot(...)` |
| `set_xy_dip_ref(...)` | 委托给 `set_dip_control_points(...)` | 使用新名称 |
| `set_xy_dip_ref_from_coords(...)` | 委托给 `set_dip_control_points_from_coords(...)` | 使用新名称 |
| `set_xy_dip_ref_from_file(...)` | 委托给 `set_dip_control_points_from_file(...)` | 使用新名称 |

这些入口用于兼容已有脚本，不适合继续扩展新功能。当前实现把它们标为 legacy，但不会
统一发出 `DeprecationWarning`，所以不能依赖 warning 判断脚本是否仍使用旧接口。

`_ensure_vertices_ref()` 是内部惰性兼容机制，不是用户 API。需要 vertices 时应在最终参考
mesh 建好后显式 `snapshot(capture_vertices=True, capture_layers=False)`，这样参考来源和
时机可审计。内部机制只会成对捕获当前完整 `Vertices/Faces`；半套 reference 会直接失败，
不会用当前 Faces 补 frozen vertices。

## 扰动方法与参考字段

用当前安装版本的注册信息作为方法名称和参数个数的权威来源：

```bash
ecat-list-fault-perturb-methods
```

```python
fault.help()
fault.help("perturb_bottom_coords_along_fixed_direction")
```

按目标判断参考依赖：

| 扰动目标 | 至少需要的参考字段 | 之后的 mesh 处理 |
| --- | --- | --- |
| top/bottom 坐标 | 对应 edge；部分方法同时需要 top 和 bottom | 用独立 `update_mesh` 或方法自带后缀 |
| dip controls | `dip_control_points + top_coords` | 先生成 bottom，再重建简单、变形或多层 mesh |
| 某一 layer | `layers` | 多层 mesh 更新 |
| 整个 geometry/mesh | 同一次 snapshot 的完整 `vertices + faces` pair；个别方法另需 pivot 所用边界 | 方法直接更新顶点并维持已冻结拓扑 |

方法名常见后缀：

| 后缀 | 职责 |
| --- | --- |
| 无 mesh 后缀 | 只更新几何；YAML 需要有效 `update_mesh` |
| `_simpleMesh` | 方法内部重建简单 mesh |
| `_DeformMesh` | 方法内部变形 Gmsh/参数化 mesh |
| `_multiLayerMesh` | 方法内部重建多层 mesh |

不要按旧文档中的固定方法总数写脚本；注册表会随版本扩展，CLI/`fault.help()` 才是当前
版本的发现入口。

绝大多数 Bayesian 扰动从 `GeometryReference` 读取基线。历史
`perturb_geometry_dutta` 是明确登记为 `current_geometry` 的例外，`fault.help()` 会显示该
来源；它不代表其他方法可以把当前候选当成下一候选的参考。

### Whole-mesh pair 协议

直接平移、旋转或逐顶点改变整个 mesh 的方法遵守以下协议：

1. reference vertices 和 faces 必须同时存在，并属于同一次最终 mesh。
2. 当前 mesh 必须保持相同 vertex shape、face rows、顶点编号、连接关系和绕序。
3. 合法 fixed-topology 变形只改变 Vertices；不会每个样本重新提取边界、重建邻接或比较
   整张网格的 hash。
4. 捕获 reference 后若发生 remesh，应重新检查 mesh 并建立新的 reference，再开始新的 run。

检查在 fault 被修改前完成。第一轮会验证完整 pair；同一 reference 和同一 Faces 发布下的
后续候选复用该 reference 检查结果，因此本轮新增保护不会给固定拓扑 Bayesian 循环增加
逐候选的全 Faces 扫描。mesh 发布层仍按其显式 topology contract 独立保证提交正确性。
直接对 `fault.Faces[:]` 原地写值会绕过正式 mesh 发布接口，不属于支持的用法。

## YAML 三道开关

几何更新必须同时满足：

1. 顶层 `nonlinear_inversion: true`。
2. 该断层 `geometry.update: true`。
3. `geometry.sample_positions` 与扰动方法需要的采样参数个数一致。

```yaml
nonlinear_inversion: true

faults:
  MainFault:
    geometry:
      update: true
      sample_positions: [0, 1]
    method_parameters:
      update_fault_geometry:
        method: perturb_bottom_coords_along_fixed_direction
        average_direction: 10.0
        angle_unit: degrees
        perturbation_direction: horizontal
```

`sample_positions` 是全局几何样本向量的半开区间，并且所有启用断层覆盖的位置必须从
0 开始形成连续序列。多个断层可以显式共享同一段位置，表示使用相同几何参数。

`update_fault_geometry` 除 `method` 外的键必须来自该方法的公开签名。配置预检会在采样
开始前拒绝未知键，并列出有效参数；preset 倾角方法中不要写 `is_utm` 或已经移除的
`update_xydip_ref`。坐标只在建模脚本调用 setter 时解释一次。

### 固定和动态参数个数

参数个数不是所有方法共享的常数。预检按方法声明区分两类：

| 契约 | `sample_positions` 长度 | 常见情形 |
| --- | --- | --- |
| 固定 `exact` | 必须等于该方法声明的固定长度 | 单一 offset、旋转+平移、多层组合等 |
| 动态 | 一个广播标量，或每个可移动节点/倾角控制点一个值 | `fixed_nodes`、可变数量 dip controls |

动态方法的长度由冻结 reference 和已解析的 `fixed_nodes` 共同决定。例如 reference 有 3 个
倾角控制点、固定第 0 点时，切片长度可以是 1（广播）或 2（逐可移动控制点）；若 3 点全部
固定，空切片 `[k, k]` 以及长度 1 的广播切片都保持为合法 no-op，因为没有可移动点。
未知的未来动态契约不会被核心猜成固定数；没有新 schema 的扩展方法仍由自身运行时校验。

启用的 geometry 配置还必须完整提供 `method_parameters.update_fault_geometry.method`。
unknown kwargs、缺失 reference、whole-mesh pair 错配和已声明的 cardinality 错误会在采样前
失败；禁用的 geometry 模板不会因保留空 method 块而误报。

主配置定义“怎样应用”，bounds 文件定义“允许多大”：

```yaml
geometry:
  MainFault: [-15.0, 15.0]
```

geometry bounds 的单位取决于具体采样参数，可能是 km、degrees 或混合参数；应在案例
注释和结果报告中逐项写清楚。

统一边界可直接写一个 pair；逐参数边界使用显式 `lb/ub` 数组：

统一边界：

```yaml
geometry:
  MainFault: [-15.0, 15.0]
```

四个参数分别给出边界：

```yaml
geometry:
  MainFault:
    lb: [-10.0, -15.0, -5.0, -5.0]
    ub: [10.0, 15.0, 5.0, 5.0]
```

内部规范化器还接受 `[lower_array, upper_array]`，但公共文档对逐参数情形推荐 `lb/ub`
映射，因为方向不易误读。不要提供 `N x 2` 的逐行 `[lb, ub]` 数组；当前解析器不支持该
表示。数组长度必须等于 `sample_positions[1] - sample_positions[0]`。

## 网格职责

坐标类方法和自带 mesh 的方法不能混为一谈。

```yaml
faults:
  MainFault:
    method_parameters:
      update_fault_geometry:
        method: perturb_bottom_coords_along_fixed_direction
        average_direction: 10.0
      update_mesh:
        method: generate_and_deform_mesh
        top_size: 3.0
        bottom_size: 6.0
        num_segments: 25
        disct_z: 10
```

- 无 mesh 后缀的方法需要独立 `update_mesh`。
- `_simpleMesh`、`_DeformMesh`、`_multiLayerMesh` 方法内部负责 mesh；额外的
  `update_mesh` 不应承担第二次重建。
- `update_mesh` 是样本内重放阶段，不能再偷偷改变几何参考。
- `generate_and_deform_mesh` 的标准初始脚本使用 `remap=True`，并让
  `bottom_norm_offset` 保持默认 `None`；Bayesian `update_mesh` 配置不能使用 `remap`、
  `use_current_mesh`、`bottom_norm_offset` 或平滑坐标类选项。

`generate_and_deform_mesh` 的固定拓扑重放依赖准备阶段生成的 Gmsh vertices/faces 和逐顶点
参数坐标。创建 `FULLSMC` 或 `SMC_FJ` target 时，系统会一次性确认这组映射完整、仍对应
当前 face rows，并与重放配置的 `num_segments`、`disct_z` 相容；当 `disct_z=None` 时还会
核对实际定义竖向网格的 `bias/min_dz`。`projection=None` 表示保留每个顶点准备时记录的
投影；显式 `xy/xz/yz` 只能用于与全部已存投影一致的映射。检查失败时应回到准备阶段重新
执行 `generate_and_deform_mesh(..., remap=True, bottom_norm_offset=None)`，不能让首个样本
静默 remap。该检查只发生在 target 构造阶段，不逐样本扫描 Faces，也不增加 MPI collective。

当前实现中，任何非 `None` 的 `bottom_norm_offset` 都会在 mesh 方法内部调用一次底边扰动；
`0.0` 也会触发 reference 读取和 current bottom 重写，非零值则会让初始 mesh anchor 与
frozen reference 分离。它不是 sampler 初值，也不是后续 sample delta 的固定常数项。

如果 offset 属于物理 baseline，应在正式 snapshot 前，用与采样方法相同的方向、单位和
fixed-node 语义显式修改 current bottom；然后冻结 offset 后的边界，并以
`bottom_norm_offset=None` 构网。如果它只服务于 mesh mapping anchor，则把它视为高级数值
策略，单独验证映射范围、极值候选和离散误差，不作为标准用户流程。

样本内顺序是：

```text
sample slice
  -> update_fault_geometry
  -> update_mesh（仅当方法未自行更新）
  -> update_GFs
  -> update_Laplacian（仅当平滑项参与且当前 GL 已失效）
  -> 获取当前面积（仅当 magnitude/moment 需要）
  -> SMC_FJ 线性滑动求解，或 FULLSMC 滑动似然
```

### 几何变化与派生量刷新

内部会按候选 mesh 的真实变化决定哪些派生量仍可复用。用户不需要在 YAML 中填写变化
类型，也不应在脚本中手工修改缓存状态；这张表用于解释计算量和检查结果是否符合预期。

| 变化 | 几何含义 | GF | Laplacian | 面积 | 邻接与边界关系 |
| --- | --- | --- | --- | --- | --- |
| `none` | 没有发布新的候选 mesh | 保留 | 保留 | 保留 | 保留 |
| `rigid` | 整个 mesh 做同一个平移或旋转，形状和连接关系不变 | 更新 | 保留已有有效值 | 保留已有有效值 | 保留 |
| `deform` | 顶点坐标和相对距离改变，但 Faces、顶点编号和 patch 顺序不变 | 更新 | 实际参与平滑时重建 | magnitude/moment/summary 需要时重算 | 保留索引关系，按当前坐标刷新显示 |
| `remesh` | Faces、连接关系、编号或 patch 对应关系改变 | 更新 | 实际参与平滑时重建 | 需要时重算 | 失效并按需重建 |

`none` 表示没有进入 mesh 发布，不表示“采样值恰好为零”必然触发零成本快路径。当前非线性
候选只要执行了几何更新，就会保守地刷新 GF；代码不会为跳过一次 GF 而逐点比较新旧几何。
`remesh` 还必须检查或重建 slip/patch 参数映射，不能只清理绘图用的边界坐标。

邻接、边界 membership 和 MudPy 边界 stencil 依赖拓扑；固定拓扑的 `deform` 可以复用这些
索引关系，但 Laplacian 的数值还依赖 patch 中心距离，因此仍可能需要重建。边界绘图、writer
和 summary 是消费者，不会反向改变这些状态。

`SMC_FJ` 与 `FULLSMC` 共用上述候选刷新入口。刚体平移或旋转仍会更新 GF，
但可复用现有面积和 Laplacian；倾角、底边或其他非刚性变形会使面积和
Laplacian 失效，只有当前 likelihood/prior 实际需要它们时才重新物化。
这些 validity 状态由扰动和 mesh 发布层维护，不是 YAML 参数，也不应由用户脚本直接赋值。

BLSE/VCE 使用不同的固定几何契约：先完成几何修改，再为该几何构建新的
`BoundLSEMultiFaultsInversion`。同一 BLSE/VCE 实例的 `run()`、平滑循环或 VCE
迭代期间不要原地更换 fault mesh；这些内循环只更新线性解或权重，不会隐式重建整套
GF、Laplacian、参数布局和约束。

## 建议的采样前检查

```python
assert fault.geometry_ref is not None
fault.geometry_summary()
fault.help("perturb_bottom_coords_along_fixed_direction")

inversion.print_parameter_positions()
constraint_state = inversion.get_constraint_snapshot(validate=True)
print(constraint_state["validation"])
```

`fault.snapshot()` 产生几何参考；`inversion.get_constraint_snapshot()` 产生约束诊断副本，
两者名称相似但用途完全不同。

此外应检查：

- 参考 top/bottom 点序、形状和下倾侧正确。
- 所选方法需要的 reference 字段都存在。
- `sample_positions` 长度、方法参数和 geometry bounds 对齐。
- 使用参数化 fixed-topology mesh 时，初始 `remap=True` 的映射参数与 YAML 重放参数一致。
- 候选极值几何仍能生成有效 mesh，patch 数量/顺序满足后端要求。
- GF、Laplacian 和面积项随几何变化按预期更新。

## 相关页面

- [Bayesian 联合反演中的几何参考](../concepts/bayesian_geometry_reference.md)
- [联合 Bayesian 几何参考与配置短例](../examples/joint_bayesian_geometry_setup.md)
- [联合 Bayesian workflow](../workflows/05_joint_bayesian_geometry_slip.md)
- [Bayesian 联合反演参考](bayesian_joint_inversion.md)
- [断层几何构建](fault_geometry_construction.md)
- [CLI 命令参考](cli.md)
