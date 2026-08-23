# 联合 Bayesian 几何参考与配置短例

本例展示一个曲线断层边界扰动场景：地表迹线和显式倾角已经足以确定最终 top/bottom，
因此直接冻结边界参考，再生成一次参数化 Gmsh mesh。采样时只扰动底边坐标，由单独的
`update_mesh` 阶段变形同一套 mesh topology。数据对象 `geodata` 的构造见
[反演前读取 InSAR 与 GNSS](inversion_data_loading.md)。

完整模板位于 `scripts/`：

| 场景 | Python | 主配置 | Bounds |
| --- | --- | --- | --- |
| 标量底边位移示例 | [`test_joint_bayesian_bottom_offset.py`](https://github.com/kefuhe/eqtools/blob/main/scripts/test_joint_bayesian_bottom_offset.py) | [`bottom_offset.yml`](https://github.com/kefuhe/eqtools/blob/main/scripts/configs/joint_bayesian/bottom_offset.yml) | [`bottom_offset_bounds.yml`](https://github.com/kefuhe/eqtools/blob/main/scripts/configs/joint_bayesian/bottom_offset_bounds.yml) |
| 多个倾角控制点示例（模板使用 3 点） | [`test_joint_bayesian_three_dip_controls.py`](https://github.com/kefuhe/eqtools/blob/main/scripts/test_joint_bayesian_three_dip_controls.py) | [`three_dip_controls.yml`](https://github.com/kefuhe/eqtools/blob/main/scripts/configs/joint_bayesian/three_dip_controls.yml) | [`three_dip_controls_bounds.yml`](https://github.com/kefuhe/eqtools/blob/main/scripts/configs/joint_bayesian/three_dip_controls_bounds.yml) |
| 组合扰动示例（当前方法使用 4 个参数） | [`test_joint_bayesian_custom_perturbation.py`](https://github.com/kefuhe/eqtools/blob/main/scripts/test_joint_bayesian_custom_perturbation.py) | [`custom_perturbation.yml`](https://github.com/kefuhe/eqtools/blob/main/scripts/configs/joint_bayesian/custom_perturbation.yml) | [`custom_perturbation_bounds.yml`](https://github.com/kefuhe/eqtools/blob/main/scripts/configs/joint_bayesian/custom_perturbation_bounds.yml) |

下面聚焦 reference、mesh 和配置之间必须一致的接口关系。数据路径、迹线、断层物理参数和
mesh 参数应在使用前核对；`lon0/lat0` 是数据与断层共享的坐标参考，应保持同一来源。

## 1. 构建并冻结参考几何

```python
import numpy as np

from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)

trace = np.loadtxt("fault_trace.txt", comments="#", usecols=(0, 1))

fault = TriFault("MainFault", lon0=lon0, lat0=lat0, verbose=True)
fault.top = 0.0
fault.depth = 25.0
fault.trace(trace[:, 0], trace[:, 1], utm=False)
fault.set_top_coords_from_trace(
    sort_axis=0, sort_order="ascend",  # 定义 top edge 的正走向。
)
fault.generate_bottom_from_single_dip(
    dip_angle=80.0,
    dip_direction=10.0,  # 地理方位角：从北顺时针，单位 degree。
)

# top/bottom 是本例经过检查的权威几何，只冻结边界字段。
fault.snapshot(
    capture_vertices=False,
    capture_layers=False,
)

# 从同一组 top/bottom 建立参数化 Gmsh mesh，只生成一次。
# remap=True 只在采样前建立映射；bottom_norm_offset 保持默认 None。
fault.generate_and_deform_mesh(
    top_size=3.0, bottom_size=6.0, num_segments=25, disct_z=10,
    remap=True,                 # 仅在采样前建立一次参数映射。
    bottom_norm_offset=None,    # 不在构网时改变物理参考底边。
    show=False, verbose=0,
)
fault.initializeslip(values="depth")

# 零扰动参考与初始 mesh 使用同一组边界。
np.testing.assert_allclose(fault.top_coords, fault.geometry_ref.top_coords)
np.testing.assert_allclose(fault.bottom_coords, fault.geometry_ref.bottom_coords)

fault.geometry_summary()
fault.help("perturb_bottom_coords_along_fixed_direction")
```

本例不需要先调用 `generate_mesh()`，也不需要
`set_edges_for_bayesian_optimization()`：曲线形状不决定参考入口，经过科学检查的权威状态
才决定入口。这里的 top/bottom 已由迹线和倾角明确构造，先生成临时 mesh 再把边界提取回来
只会增加一次构网和一次状态转换。`sort_order` 不只是显示顺序：对依赖局部走向和右手侧的
倾角方法，它还确定正走向与倾向侧约定。

如果权威边界只存在于导入、裁切或人工修整后的 mesh 中，才使用另一条路线：

```python
fault.read_mesh_file(...)  # 或生成并完成必要的 mesh 修整
fault.set_edges_for_bayesian_optimization(
    sort_axis=0,
    sort_order="ascend",
)
```

该入口内部已建立 reference，调用成功后不要机械地再 `snapshot()`。

若方法直接平移或旋转整个现有 mesh，应在最终参考 mesh 建好后显式改为：

```python
fault.snapshot(capture_vertices=True, capture_layers=False)
```

## 2. 修改生成的主配置

先生成规范模板，再保留其中的 geodata、sigma/alpha、GF 和 Laplacian 设置：

```bash
ecat-generate-config -o joint_config.yml --gf-method cutde
ecat-generate-boundary -o joint_bounds.yml -f MainFault
```

把主配置中的联合几何相关字段改为：

生成器中的 `ExampleFault` 只是占位项；应删除它或改成与 Python fault object 完全一致的
`MainFault`，不要让占位断层和真实断层同时留在配置中。

```yaml
nonlinear_inversion: true
bayesian_sampling_mode: SMC_FJ
slip_sampling_mode: ss_ds

faults:
  defaults:
    geometry:
      update: false
      sample_positions: [0, 0]
    method_parameters:
      update_GFs:
        method: cutde
        options: {}
      update_Laplacian:
        method: Mudpy
        bounds: [free, locked, free, free]
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
        use_average_strike: false
      update_mesh:
        method: generate_and_deform_mesh
        top_size: 3.0
        bottom_size: 6.0
        num_segments: 25
        disct_z: 10
```

这里 `[0, 1]` 是全局几何样本向量的半开区间。该方法接收一个扰动量，并按方法语义
应用到未固定的底边节点。若要逐节点独立扰动，应先用 `fault.help()` 选择支持相应参数
个数的方法，再同步调整 `sample_positions` 和 bounds。

在 `joint_bounds.yml` 中设置同名断层的几何增量范围：

```yaml
geometry:
  MainFault: [-15.0, 15.0]
```

此处单位由采样参数定义；本例是水平距离，单位为 km。bounds 限制的是相对于
`geometry_ref` 的增量，不是绝对经度、纬度或深度。

## 3. 创建 inversion 前检查

```python
from eqtools.csiExtend.bayesian_multifaults_inversion import (
    BayesianMultiFaultsInversion,
)

assert fault.geometry_ref is not None
assert fault.geometry_ref.top_coords is not None
assert fault.geometry_ref.bottom_coords is not None

inversion = BayesianMultiFaultsInversion(
    config="joint_config.yml",
    faults_list=[fault],
    geodata=geodata,
    bounds_config="joint_bounds.yml",
)

inversion.print_parameter_positions()
constraint_state = inversion.get_constraint_snapshot(validate=True)
print(constraint_state["sampling_mode"])
print(constraint_state["validation"])

# 小样本检查通过后再开始正式采样。
# inversion.walk(nchains=100, chain_length=50)
```

断层参考快照和约束诊断快照是两件不同的事。前者在构造 `fault` 时建立；后者在
`inversion` 初始化后用于检查 bounds 与线性约束。
`walk()` 构造 `FULLSMC`/`SMC_FJ` target 时还会只读检查一次参数化 mesh 映射；若准备
映射、当前 topology 或重放分辨率不一致，会在正式候选循环开始前报错，而不会自动 remap。

## 任意数量倾角控制点（以三个为例）

若用户提供 `N` 个 `lon/lat/dip` 参考点，应先用 `N` 个零增量生成正式参考 bottom，再冻结。
下面以 `N = 3` 为例：

```python
dip_controls = np.array([
    [LON_0, LAT_0, DIP_0],
    [LON_1, LAT_1, DIP_1],
    [LON_2, LAT_2, DIP_2],
])

fault.set_dip_control_points_from_coords(
    coords=dip_controls[:, :2],
    dips=dip_controls[:, 2],
    is_utm=False,  # 只说明本次输入为 lon/lat；reference 内部规范保存。
)
fault.perturb_dips_with_preset_params(
    perturbations=np.zeros(3),       # 零增量生成权威参考 bottom。
    interpolation_axis="arc_length",  # 沿弯曲 top 的累计弧长插值。
    fixed_nodes=None, angle_unit="degrees", use_average_strike=False,
)
fault.snapshot(capture_vertices=False, capture_layers=False)
```

本例的三个增量在 YAML 中使用半开区间 `[0, 3)`：

```yaml
geometry:
  update: true
  sample_positions: [0, 3]

method_parameters:
  update_fault_geometry:
    method: perturb_dips_with_preset_params
    interpolation_axis: arc_length
    fixed_nodes: null
    angle_unit: degrees
    use_average_strike: false
```

逐参数 bounds 使用明确的 `lb/ub` 数组：

```yaml
geometry:
  MainFault:
    lb: [-15.0, -15.0, -15.0]
    ub: [15.0, 15.0, 15.0]
```

参数顺序仍是 `dip_controls` 的输入行顺序；`arc_length` 只把控制点投影到当前 top 并按
空间弧长插值，不重排后验列名或 bounds。`N` 个控制点生成 `N - 1` 个连续插值区间，本例
三个控制点对应两个区间。proposal 可按
用户意图跨过 90°，但绝对倾角必须保持在 `(0°, 180°)`。

## 改成其他扰动方法时检查六处

组合扰动模板使用 `[bottom_offset, rotation, dx, dy]` 这一方法专属的四参数实例。更换方法时
同步检查：

1. Python 中哪份几何是权威 reference；
2. `snapshot()` 是否需要 top/bottom、layers 或 vertices/faces；
3. `update_fault_geometry.method` 及公开关键字；
4. `sample_positions` 的长度和参数顺序；
5. geometry bounds 的单位和逐项顺序；
6. 方法内部是否已经更新 mesh，是否还需要独立 `update_mesh`。

当前四参数实例的受支持 bounds 写法是：

```yaml
geometry:
  MainFault:
    lb: [-10.0, -15.0, -5.0, -5.0]
    ub: [10.0, 15.0, 5.0, 5.0]
```

当前 bounds 解析器把二维数组解释为 `[lower_array, upper_array]`，因此下面这种按参数逐行
排列的 `4 x 2` 形式不受支持，不要使用：

```yaml
# Unsupported
geometry:
  MainFault:
    - [-10.0, 10.0]
    - [-15.0, 15.0]
    - [-5.0, 5.0]
    - [-5.0, 5.0]
```

## 其他参考来源的最短写法

| 参考来源 | 建立方式 | 适用扰动 |
| --- | --- | --- |
| 权威 top/bottom 坐标 | 写入/构造边界后 `snapshot(capture_vertices=False, capture_layers=False)` | 外部坐标、迹线加倾角、解析构造、边界扰动 |
| 当前 mesh 的实际边界 | `set_edges_for_bayesian_optimization(...)` | 导入、裁切或人工修整后的 mesh 边界作为基线 |
| 最终 mesh 顶点 | 建完最终参考 mesh 后 `snapshot(capture_vertices=True, capture_layers=False)` | 整体平移、旋转或直接顶点变换 |
| 多层边界 | 设置 `layers` 后 `snapshot(capture_vertices=False, capture_layers=True)` | `_multiLayerMesh` 和 layered-dip |
| 倾角控制点 | `set_dip_control_points*()`，完成 bottom 后再 `snapshot()` | 沿走向变化倾角 |

完整的接口时机、重新设基线规则和 legacy `ref*` 入口见
[可扰动断层几何参考](../reference/geometry_perturbation.md)。

## 下一步

配置对齐后，使用本页开头的完整模板进行装配检查和采样。代表模型回填、拟合统计、
fault/slip 与 `Modeling/` 导出的状态要求见
[联合反演结果参考](../reference/bayesian_joint_inversion.md#标准结果入口与脚本层导出)。
