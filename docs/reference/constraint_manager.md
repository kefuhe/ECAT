# ECAT 约束管理器

ECAT 反演中的约束管理器统一管理参数边界和线性约束。它覆盖 Bayesian/SMC 反演与 BLSE 线性反演，但不同采样模式能使用的约束类型不同。

## 入口和职责

| 反演入口 | 管理器 | 主要职责 |
| --- | --- | --- |
| `BayesianMultiFaultsInversion` | `ConstraintManagerSMC` | 管理 SMC 参数边界；在 `SMC_F_J + ss_ds` 中管理线性约束。 |
| `BoundLSEMultiFaultsInversion` | `ConstraintManagerBLSE` | 管理 BLSE 的滑动、poly 边界和线性约束。 |
| `multifaultsolve_boundLSE` | `ConstraintManagerBLSE` | 提供较底层的手工约束接口。 |

两类管理器共享同一套基本结构：

```text
ConstraintManagerBase
├── bounds: lb / ub / strikeslip / dipslip / poly / source_bounds
├── inequality constraints: A @ x <= b
├── equality constraints: A @ x = b
├── cache: 合并后的 A, b
└── validate / print_summary / sync_to_solver
```

## 模式矩阵

| 模式 | 滑动参数化 | 超参数处理 | 线性参数处理 | 支持的约束 |
| --- | --- | --- | --- | --- |
| `FULLSMC` | `ss_ds` | SMC 采样 | SMC 采样 | `strikeslip` / `dipslip` 边界。 |
| `FULLSMC` | `magnitude_rake` | SMC 采样 | SMC 采样 | `slip_magnitude` / `rake_angle` 边界，采样后回算 ss/ds。 |
| `FULLSMC` | `rake_fixed` | SMC 采样 | SMC 采样 | `slip_magnitude` 边界，rake 固定后回算 ss/ds。 |
| `SMC_F_J` | `ss_ds` | SMC 采样 | BLSE 求解 | 边界约束、rake 线性约束、Euler 约束、零滑/边界零滑、自定义线性约束。 |
| `BLSE` | `ss_ds` | 无 SMC 超参数采样 | BLSE 求解 | 边界约束、rake 线性约束、Euler 约束、零滑/边界零滑、自定义线性约束。 |

`mag_rake` 是 `magnitude_rake` 的配置别名，读入后会被归一化为 `magnitude_rake`。`SMC_FJ` 是 `SMC_F_J` 的别名。

入门两步走路线通常使用 `BLSE + ss_ds`。`FULLSMC` 和 `SMC_F_J` 的约束支持作为采样模式参考列出，方便阅读配置和调试脚本。

## 参数分层

| 层级 | 参数 | 说明 |
| --- | --- | --- |
| 超参数 | `geometry` | 断层位置、走向、倾角、网格扰动等，只在 Bayesian/SMC 参数块中采样。 |
| 超参数 | `sigmas` | 数据标准差或单位权标准差尺度。 |
| 超参数 | `alpha` | 拉普拉斯平滑尺度。 |
| 线性参数 | `slip` | `ss_ds`、`magnitude_rake` 或 `rake_fixed` 决定参数化。 |
| 线性参数 | `poly` | InSAR ramp、多项式或 Euler rotation 等数据修正参数。 |

`SMC_F_J` 中，约束矩阵 `A` 的列数对应线性参数块，即滑动和 poly 参数，不包含 geometry/sigmas/alpha 超参数。

## Bounds 配置键

`bounds_config.yml` 可包含以下键：

```yaml
lb: -3
ub: 3

geometry:
  FaultA: [-10, 10]

slip_magnitude:
  FaultA: [0, 15]

rake_angle:
  FaultA: [-120, -60]

strikeslip:
  FaultA: [-10, 10]

dipslip:
  FaultA: [-10, 0]

poly:
  FaultA: [-1000, 1000]

sigmas: [-3, 3]
alpha: [-3, 3]
```

边界模板可由 CLI 生成后再修改：

```bash
ecat-generate-boundary -o bounds_config.yml -f FaultA
```

边界值支持三种常用写法：

```yaml
# 统一边界，应用到该参数的所有元素
strikeslip:
  FaultA: [-10, 10]

# 逐元素边界
strikeslip:
  FaultA:
    - [-10, -8, -6]
    - [10, 8, 6]

# 字典写法
strikeslip:
  FaultA:
    lb: [-10, -8, -6]
    ub: [10, 8, 6]
```

`rake_angle` 这个键在不同模式下含义不同：

- `FULLSMC + magnitude_rake`：它是被采样的 rake 参数边界。
- `SMC_F_J + ss_ds` 与 `BLSE + ss_ds`：它生成基于 `strikeslip/dipslip` 的线性 rake 角约束。
- `FULLSMC + ss_ds`：不会生成线性 rake 角约束，只能通过 ss/ds 边界间接控制机制。

## Rake 约束

在 `SMC_F_J + ss_ds` 和 `BLSE + ss_ds` 中，rake 范围会被转换成线性不等式：

```text
ss * sin(rake_min) - ds * cos(rake_min) <= 0
-ss * sin(rake_max) + ds * cos(rake_max) <= 0
```

配置方式：

```yaml
rake_angle:
  FaultA: [-30, 60]
```

Bayesian/SMC 手工更新：

```python
inversion.update_rake_constraints(
    rake_angle={"FaultA": [-30, 60]}
)
```

固定 rake 是等式约束，不要和 `slip_sampling_mode: rake_fixed` 混淆：

```python
inversion.update_rake_constraints(
    fixed_rake={"FaultA": -90.0}
)
```

`rake_fixed` 是 FULLSMC 的滑动参数化方式；`fixed_rake` 是 `SMC_F_J/BLSE` 中作用在 ss/ds 线性参数上的等式约束。

## Euler 约束

Euler 约束用于震间或块体运动约束。当前适用条件：

- `SMC_F_J + ss_ds`
- `BLSE + ss_ds`
- 只作用于 `Fault` 类型源
- 基于 strike-slip 分量和两侧块体相对运动

约束形式为：

```text
motion_sign * slip_strike
+ motion_sign * (euler_block_1_strike - euler_block_2_strike)
<= 0
```

其中 `motion_sign = +1` 表示 `dextral/right_lateral`，`motion_sign = -1` 表示 `sinistral/left_lateral`。

Euler 配置放在主配置 `default_config.yml`，不是 `bounds_config.yml`。建议同时打开顶层开关和配置块：

```yaml
use_euler_constraints: true

euler_constraints:
  enabled: true
  defaults:
    block_types: [dataset, dataset]
    euler_pole_units: [degrees, degrees, degrees_per_myr]
    euler_vector_units: [radians_per_year, radians_per_year, radians_per_year]
    fix_reference_block: null
    apply_to_patches: null
    normalization: false
    regularization: 0.01
  faults:
    FaultA:
      block_types: [dataset, euler_pole]
      blocks: [GPS_Block_A, [lat_deg, lon_deg, omega_deg_per_myr]]
      block_names: [Block_A, Block_B]
      reference_strike: 0.0
      motion_sense: dextral
      apply_to_patches: null
      units:
        euler_pole_units: [degrees, degrees, degrees_per_myr]
```

读入配置后，`euler_pole` 和 `euler_vector` 会被转换到标准单位，并保存为标准化后的 `blocks_standard`。用户配置里仍写 `blocks`。
当前解析器按 `[lat, lon, omega]` 读取 `euler_pole`，单位顺序应与 `euler_pole_units` 保持一致。

Bayesian/SMC 手工更新：

```python
# 如果 default_config.yml 已经包含完整 euler_constraints，
# 这里通常只需要打开或替换部分字段。
inversion.update_euler_constraints({"enabled": True})
```

若在脚本中临时构造完整 Euler 约束，需要传入已标准化的块体参数，或先经过配置解析流程完成单位转换：

```python
inversion.update_euler_constraints({
    "enabled": True,
    "faults": {
        "FaultA": {
            "block_types": ["dataset", "euler_pole"],
            "blocks_standard": ["GPS_Block_A", [lat_rad, lon_rad, omega_rad_per_year]],
            "motion_sense": "dextral",
        }
    }
})
```

如果需要生成带 Euler 示例的主配置模板，可使用线性配置生成 CLI 的 Euler 选项：

```bash
ecat-generate-config --include-euler-constraints
```

## 零滑与边界零滑约束

同震 BLSE/VCE 中常见的零滑约束包括三类：

| 类型 | 作用 | 推荐写法 |
| --- | --- | --- |
| 分量全零 | 某个断层所有 patch 的 `strikeslip` 或 `dipslip` 固定为 0 | `source_constraints` 中写 `strikeslip == 0` 或 `dipslip == 0` |
| 指定 patch 零滑 | 只固定某些 patch 的某个滑动分量 | 脚本中调用 `add_patch_slip_constraint(...)` |
| 边界零滑 | top/bottom/left/right 边界上的 patch 滑动固定为 0 | `source_constraints` 中写 `zero_edge_slip(...)`，或脚本中调用 `add_zero_edge_slip_constraint(...)` |

这些约束都是等式约束，形式为：

```text
A @ x = 0
```

其中 `x` 是线性参数向量中的 `strikeslip/dipslip` 参数块。固定权重 BLSE、smoothing loop 和 VCE 都使用同一套约束矩阵；VCE 只改变数据项和正则化项的权重估计，不改变这些物理约束的写法。

### 配置文件写法

`bounds_config.yml` 中通过 `source_constraints` 写零滑约束：

```yaml
source_constraints:
  FaultA:
    # 整个断层的走滑或倾滑分量固定为 0
    - {name: zero_ss_all, type: equality, rule: "strikeslip == 0"}
    - {name: zero_ds_all, type: equality, rule: "dipslip == 0"}

    # 顶边走滑固定为 0
    - {name: zero_top_ss, type: equality, rule: "zero_edge_slip(top, strikeslip)"}

    # 顶边和底边的走滑、倾滑都固定为 0
    - {name: zero_top_bottom_sd, type: equality, rule: "zero_edge_slip(top+bottom, ss+ds)"}
```

`zero_edge_slip(...)` 支持的写法：

```text
zero_edge_slip(top, strikeslip)
zero_edge_slip(top, dipslip)
zero_edge_slip(top+bottom, ss+ds)
zero_edge_slip(left+right, ds)
zero_edge_slip(top, strikeslip, dipslip)
```

支持的边界名取决于断层对象中的 `edge_triangles_indices`，常见为：

```text
top, bottom, left, right
```

支持的滑动分量别名：

| 标准名 | 可用别名 |
| --- | --- |
| `strikeslip` | `ss`, `s`, `strike`, `strike_slip` |
| `dipslip` | `ds`, `d`, `dip`, `dip_slip` |

边界零滑约束要求断层对象已经有 `edge_triangles_indices`。如果没有，约束管理器会报错并提示需要先完成边界识别。实际脚本中应在网格生成和边界识别完成后再初始化反演或应用约束。

### 脚本接口写法

底层 BLSE 求解器和 `SMC_F_J` 高级接口都提供快捷方法。固定同震 BLSE 中常用：

```python
solver.add_zero_edge_slip_constraint(
    "FaultA",
    edges=["top", "bottom"],
    slip_modes=["strikeslip", "dipslip"],
)

solver.add_patch_slip_constraint(
    {"FaultA": [0, 1, 2]},
    slip_component="dipslip",
    value=0.0,
    constraint_type="equality",
)
```

`SMC_F_J` 脚本接口也有同名方法：

```python
inversion.add_zero_edge_slip_constraint(
    "FaultA",
    edges="top",
    slip_modes=["ss", "ds"],
)

inversion.add_patch_slip_constraint(
    {"FaultA": [0, 1, 2]},
    slip_component=["ss", "ds"],
    value=0.0,
    constraint_type="equality",
)
```

对可复现的同震 BLSE/VCE 案例，优先把零滑和边界零滑写进 `bounds_config.yml` 的 `source_constraints`；脚本接口适合临时试验、自动生成 patch 列表，或根据数据质量动态关闭某些区域滑动。

## 自定义线性约束

自定义约束统一写成矩阵形式：

```text
A @ x <= b
A @ x = b
```

在 `SMC_F_J` 中，`x` 是线性参数块；`A.shape[1]` 必须等于 `inversion.lsq_parameters`。在 BLSE 中，`A.shape[1]` 也必须等于求解器的 `lsq_parameters`。

Bayesian/SMC 接口：

```python
inversion.add_custom_inequality_constraint(A, b, name="my_ineq")
inversion.add_custom_equality_constraint(Aeq, beq, name="my_eq")
```

BLSE 底层接口：

```python
solver.add_inequality_constraint(A, b, name="my_ineq")
solver.add_equality_constraint(Aeq, beq, name="my_eq")
```

也可以一次更新多类约束：

```python
inversion.update_all_constraints(
    rake_angle={"FaultA": [-30, 60]},
    euler_config=euler_config,
    custom_inequality=[
        {"A": A, "b": b, "name": "my_ineq"}
    ],
)
```

## Source Bounds 与 Source Constraints

多源反演中，非 `Fault` 源不要用 `strikeslip/dipslip` 键。应使用 `source_bounds`：

```yaml
source_bounds:
  MyPressureSource:
    pressure: [-1.0e6, 1.0e6]
  MySbarbotSource:
    eps12: [-1.0e-4, 1.0e-4]
    eps13: [-1.0e-4, 1.0e-4]
```

`source_constraints` 通过 source adapter 生成约束矩阵，可写在 `bounds_config.yml`：

```yaml
source_constraints:
  FaultA:
    - {name: ss_positive, type: inequality, rule: "strikeslip >= 0"}
    - {name: ds_negative, type: inequality, rule: "dipslip <= 0"}
    - {name: zero_ds_all, type: equality, rule: "dipslip == 0"}
    - {name: zero_top_ss, type: equality, rule: "zero_edge_slip(top, strikeslip)"}
    - {name: zero_top_ds, type: equality, rule: "zero_edge_slip(top, dipslip)"}
  MyPressureSource:
    - {name: positive_pressure, type: inequality, rule: "pressure >= 0"}
  MySbarbotSource:
    - {name: incompressible, type: equality, rule: "incompressible"}
```

常用快捷接口：

```python
inversion.add_zero_edge_slip_constraint(
    "FaultA",
    edges="top",
    slip_modes=["strikeslip", "dipslip"],
)

inversion.add_patch_slip_constraint(
    {"FaultA": [0, 1, 2]},
    slip_component="dipslip",
    value=0.0,
    constraint_type="equality",
)
```

## 推荐使用顺序

1. 在 `default_config.yml` 中设置模式开关：`slip_sampling_mode`、`bayesian_sampling_mode`、`use_bounds_constraints`、`use_rake_angle_constraints`、`use_euler_constraints`。
2. 在 `bounds_config.yml` 中设置边界：`strikeslip/dipslip`、`slip_magnitude/rake_angle`、`poly`、`sigmas`、`alpha`。
3. 对同震 BLSE/VCE 案例，优先通过 `source_constraints` 管理零滑、边界零滑和符号约束，保证可复现。
4. 对 `SMC_F_J` 高级反演，用 `update_bounds()`、`update_rake_constraints()`、`update_euler_constraints()` 和 `add_custom_*_constraint()` 做脚本级补充。
5. 求解前查看 `constraint_manager.print_summary()`；若使用底层 `multifaultsolve_boundLSE`，也可调用其 `print_constraint_summary()` 封装，确认边界数量、线性约束数量和矩阵维度。

## 注意事项

- 线性 rake、Euler 和自定义 `A @ x` 约束只在 `SMC_F_J + ss_ds` 和 `BLSE + ss_ds` 中生效。
- `FULLSMC` 中滑动参数本身被采样，因此约束管理器只提供边界，不合并线性约束矩阵。
- `rake_angle` 在 `magnitude_rake` 中是采样参数，在 `ss_ds` 的 SMC_F_J/BLSE 中是线性角度约束。
- `source_bounds` 和 `source_constraints` 用于非 `Fault` 源或 adapter 支持的通用源约束。
- `zero_edge_slip(...)` 只适用于 `Fault` 源，并要求断层对象已有 `edge_triangles_indices`。
- `strikeslip == 0` 或 `dipslip == 0` 会固定该断层所有 patch 的对应滑动分量；若只想固定部分 patch，应使用脚本级 `add_patch_slip_constraint(...)`。
- 断层名、数据集名和 source 名必须与 Python 对象名一致。
- 当前文档中的滑动正负号应与案例机制和底层 CSI 约定一致；案例文档需要明确写出符号约定。
