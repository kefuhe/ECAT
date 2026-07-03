# ECAT 约束管理器

ECAT 约束管理器统一管理参数边界、线性不等式约束和线性等式约束。它覆盖 BLSE/VCE 线性反演，也服务于 Bayesian/SMC 中的 `SMC_F_J` 线性子问题。

## 入口和职责

| 反演入口 | 管理器 | 主要职责 |
| --- | --- | --- |
| `BoundLSEMultiFaultsInversion` | `ConstraintManagerBLSE` | 管理 BLSE 的滑动、poly 边界和线性约束 |
| `BayesianMultiFaultsInversion` | `ConstraintManagerSMC` | 管理 SMC 参数边界；在 `SMC_F_J + ss_ds` 中管理线性约束 |
| `multifaultsolve_boundLSE` | `ConstraintManagerBLSE` | 提供较底层的手工约束接口 |

两类管理器共享基本结构：

```text
ConstraintManagerBase
├─ bounds: lb / ub / strikeslip / dipslip / poly / source_bounds
├─ inequality constraints: A @ x <= b
├─ equality constraints: A @ x = b
├─ cache: 合并后的 A, b
└─ validate / print_summary / sync_to_solver
```

## 模式矩阵

| 模式 | 滑动参数化 | 线性参数处理 | 支持的约束 |
| --- | --- | --- | --- |
| `FULLSMC + ss_ds` | `ss_ds` | SMC 采样 | `strikeslip` / `dipslip` 边界 |
| `FULLSMC + magnitude_rake` | magnitude + rake | SMC 采样 | `slip_magnitude` / `rake_angle` 边界 |
| `FULLSMC + rake_fixed` | fixed rake + magnitude | SMC 采样 | `slip_magnitude` 边界 |
| `SMC_F_J + ss_ds` | `ss_ds` | BLSE 线性子问题 | bounds、rake 线性约束、Euler cap、zero-slip、custom matrices |
| `BLSE + ss_ds` | `ss_ds` | BLSE 求解 | bounds、rake 线性约束、Euler cap、zero-slip、custom matrices |

线性矩阵约束只在 `SMC_F_J + ss_ds` 和 `BLSE + ss_ds` 中生效。

## 状态源和更新时间

ECAT 约束的唯一可写状态源是 `inversion.constraint_manager`。用户应通过
配置文件或公开方法更新约束，不应直接改 `solver.bounds`、
`solver.inequality_constraints` 或 `solver.equality_constraints`。

| 内容 | 推荐更新入口 | 只读检查入口 |
| --- | --- | --- |
| 参数上下界 | `set_bounds(...)`、`update_bounds(...)`、`bounds_config.yml` | `constraint_manager.bounds`、`get_constraint_snapshot()` |
| 不等式约束 `A @ x <= b` | `add_inequality_constraint(...)`、`add_custom_inequality_constraint(...)`、配置文件规则 | `constraint_manager.inequality_constraints` |
| 等式约束 `A @ x = b` | `add_equality_constraint(...)`、`add_custom_equality_constraint(...)`、配置文件规则 | `constraint_manager.equality_constraints` |

这些检查入口返回只读视图。尝试直接修改其中的字典或矩阵会报错；这是有意设计，
用来避免“看似改了 solver 属性，但反演仍然使用 manager 旧状态”的混乱。

BLSE/VCE 和 SMC 的生效时间不同：

- **BLSE/VCE**：求解前从 `constraint_manager` 读取最新 bounds、等式约束和不等式约束。
  因此在调用 `run()` 或 VCE 求解前更新约束即可。
- **SMC_F_J**：`make_F_J_target_for_parallel()` 或 `walk_F_J()` 构造 target 时读取并冻结
  当时的线性约束和线性参数 bounds。若 target 已经构造，再修改约束，需要重新构造 target
  或重新调用对应采样入口。

常用脚本写法：

```python
# BLSE/VCE: update before run()
inversion.set_bounds(
    strikeslip_limits={"FaultA": [-5.0, 5.0]},
    dipslip_limits={"FaultA": [-2.0, 2.0]},
)
inversion.add_equality_constraint(Aeq, beq, name="custom_eq")
inversion.add_inequality_constraint(A, b, name="custom_cap")
inversion.run(...)

# SMC_F_J: update before target construction or walk_F_J()
inversion.update_bounds(
    strikeslip={"FaultA": [-5.0, 5.0]},
    dipslip={"FaultA": [-2.0, 2.0]},
)
inversion.add_custom_equality_constraint(Aeq, beq, name="custom_eq")
inversion.add_custom_inequality_constraint(A, b, name="custom_cap")
inversion.walk_F_J(...)
```

快速检查当前状态：

```python
snapshot = inversion.constraint_manager.get_constraint_snapshot()
print(snapshot["bounds"])
print(snapshot["inequality_constraints"].keys())
print(snapshot["equality_constraints"].keys())
```

## Bounds

`bounds_config.yml` 可包含：

```yaml
lb: -3
ub: 3

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

边界值支持统一边界、逐 patch 边界和字典写法：

```yaml
strikeslip:
  FaultA: [-10, 10]

dipslip:
  FaultA:
    lb: [-2, -2, -1]
    ub: [0, 0, 0]
```

`rake_angle` 在不同模式下含义不同：

- `FULLSMC + magnitude_rake`：被采样的 rake 参数边界。
- `SMC_F_J + ss_ds` 和 `BLSE + ss_ds`：转为 `strikeslip/dipslip` 的线性 rake 角约束。
- `FULLSMC + ss_ds`：不生成线性 rake 角约束，只能通过 ss/ds 边界间接控制机制。

## Rake 线性约束

完整数学公式、角度范围限制和多断层矩阵排列见 [Rake Constraints](rake_constraints.md)。

在 `SMC_F_J + ss_ds` 和 `BLSE + ss_ds` 中，rake 范围转换为：

```text
ss * sin(rake_min) - ds * cos(rake_min) <= 0
-ss * sin(rake_max) + ds * cos(rake_max) <= 0
```

配置：

```yaml
rake_angle:
  FaultA: [-30, 60]
```

脚本更新：

```python
inversion.update_rake_constraints(
    rake_angle={"FaultA": [-30, 60]},
)
```

固定 rake 是等式约束：

```python
inversion.update_rake_constraints(
    fixed_rake={"FaultA": -90.0},
)
```

## Zero-Slip 和边界零滑

同震 BLSE/VCE 中常见的零滑约束包括：

| 类型 | 作用 | 推荐写法 |
| --- | --- | --- |
| 分量全零 | 某个断层所有 patch 的 `strikeslip` 或 `dipslip` 固定为 0 | `source_constraints` 中写 `strikeslip == 0` 或 `dipslip == 0` |
| 指定 patch 零滑 | 只固定局部 patch 的某个分量 | 脚本中调用 `add_patch_slip_constraint(...)` |
| 边界零滑 | top/bottom/left/right 边界上的 patch 固定为 0 | `source_constraints` 中写 `zero_edge_slip(...)`，或脚本调用 `add_zero_edge_slip_constraint(...)` |

配置写法：

```yaml
source_constraints:
  FaultA:
    - {name: zero_ds_all, type: equality, rule: "dipslip == 0"}
    - {name: zero_top_ss, type: equality, rule: "zero_edge_slip(top, strikeslip)"}
    - {name: zero_top_bottom_sd, type: equality, rule: "zero_edge_slip(top+bottom, ss+ds)"}
```

脚本写法：

```python
top_ids = get_edge_patch_indices(fault, "top")

inversion.add_zero_edge_slip_constraint(
    "FaultA",
    edges=["top", "bottom"],
    slip_modes=["strikeslip", "dipslip"],
)

inversion.add_patch_slip_constraint(
    {"FaultA": top_ids},
    slip_component="dipslip",
    value=0.0,
    constraint_type="equality",
)
```

`zero_edge_slip(...)` 要求断层对象已有 `edge_triangles_indices`。边界识别见 [断层边界识别](fault_edges.md)。

## 震间 Blocks、Fault Loading、Euler Cap 和 Backslip

震间配置使用独立 `interseismic_config.yml`，不写在 `bounds_config.yml`，也不再写旧主配置 `euler_constraints`。

```yaml
blocks:
  Block_A:
    datasets: [GPS_Block_A]
    euler:
      mode: estimate
  Block_B:
    datasets: [GPS_Block_B]
    euler:
      mode: estimate

fault_loading:
  enabled: true
  faults:
    FaultA:
      blocks: [Block_A, Block_B]
      reference_strike: 90.0
      motion_sense: sinistral

cap_constraints:
  enabled: true
  faults:
    FaultA:
      selector: {depth_range: [0.0, 15.0]}
      max_coupling: 1.0

backslip_constraints:
  - fault: FaultA
    state: full_coupling
    selector: {edge: top}
```

职责分离：

| 区块 | 生成内容 |
| --- | --- |
| `blocks` | 命名块体、对应数据集和 Euler 参数来源；可生成 `interseismic_block_euler_constraints` 等式组 |
| `fault_loading` | 每条断层两侧 block 关系，以及所有 patch 的构造加载率 `b` |
| `cap_constraints` | 可选不等式约束；默认按 `motion_sense` 限制 `q+k*b`，固定 loading 时也可按实际 `loading_sign` 限制 coupling 区间 |
| `backslip_constraints` | 可选等式约束 `q=0`、`q+b=0`、`q+k*b=0`、`q=value` |

在 `fault_loading` 中，`blocks: [Block_A, Block_B]` 的顺序定义 loading 为 `Block_A - Block_B` 投影到 patch 局部走向。`reference_strike` 只选择走向正向分支；`motion_sense` 只决定 cap 方向和诊断期望，不会改变 block 顺序或 loading 符号。推荐让 `Block_A` 位于 `reference_strike` 的右手侧、`Block_B` 位于左手侧；只有在 east side 确实是参考走向右手侧时，`east_side - west_side` 才是安全简写。完整约定见 [震间加载、Backslip 与 Coupling](interseismic_kinematics.md)。

`motion_sense` 的唯一配置来源是 `fault_loading`，或 `fault_loading.loading_regions`
中的局部覆盖。`cap_constraints.mode: motion_sense` 只是说明 cap 使用这些
`fault_loading` 方向来写不等式；不要在 `cap_constraints` 下写
`motion_sense`、`reference_strike`、`blocks` 或 `loading_regions`。这些字段属于
loading 定义层，误放到 cap 层会被配置解析器拒绝。

正式反演前建议运行 `inversion.print_interseismic_preflight_report()`。它会按 fault 紧凑列出 block 顺序、loading 统计、cap patch 数、hard backslip/coupling 行数，以及 cap 与 hard/free selector 的重叠情况。

约束优先级为：

```text
fault_loading          -> 定义 loading b
backslip_constraints   -> hard equality，如 full_coupling / creep
cap_constraints        -> 对仍自由估计的 q 与 b 添加不等式
bounds_config.yml      -> 限制 q 本身的上下界
```

默认 `hard_overlap: skip`，因此 cap 不等式会自动跳过同一 fault、同一 component
上已经由 `backslip_constraints` 固定的 patch。这个默认值避免 `full_coupling`
的 `q+b=0` 与 `max_coupling: 1.0` 的 `q+b<=0` 在同一 patch 上零松弛重合。
专家调试时可设 `hard_overlap: keep` 保留重叠，或设 `hard_overlap: error`
让重叠在构建 cap 时直接报错。

Euler cap 的公式：

```text
dextral:    q + k*b <= 0
sinistral:  q + k*b >= 0
```

其中 `k = cap_constraints.*.max_coupling`，默认 1.0；旧键名 `factor` 会作为输入别名解析为 `max_coupling`。这是默认 `mode: motion_sense`，适合 loading 仍由待估 Euler 参数决定的情况。若两侧 block loading 都是固定值，也可以用 `mode: loading_sign`，由实际投影出的 `sign(b)` 自动生成 `0 <= -q/b <= k`。`loading_sign` 会拒绝待估 Euler loading 和近零 loading，因为那时求解前没有可靠的 `b` 符号。

Cap 的配置有一个容易误用的细节：

```yaml
cap_constraints:
  enabled: true
  defaults:
    selector: null
    mode: motion_sense
    hard_overlap: skip
    max_coupling: 1.0
  faults:
    FaultA:
      selector: null
      mode: motion_sense
```

这里的 `mode: motion_sense` 不是右旋/左旋值，而是 cap 的符号来源模式。真正的
右旋/左旋期望应写在：

```yaml
fault_loading:
  faults:
    FaultA:
      motion_sense: dextral
```

上面会为 `FaultA` 生成 cap 行。若省略 `faults` 键，解析器会对
`fault_loading` 中已有的 fault 使用 defaults。若显式写成 `faults: {}`，
则表示没有任何 fault 使用 cap，约束管理器不会生成 `euler_cap_constraints`
矩阵组。

求解前检查：

```python
inversion.print_interseismic_preflight_report()
inversion.print_interseismic_constraint_report("FaultA")
```

报告中 `Euler-cap rows` 和 `Euler-cap matrix group` 必须有非零行数，才能说明
cap 已实际进入线性约束矩阵。

`bounds_config.yml` 可继续控制 backslip `q` 的基础符号和宽范围。默认 `motion_sense` 模式下，若同时使用符号 bounds 与 `max_coupling: 1.0`，就可以强制常见的 `0 <= coupling <= 1`；若 loading 是固定的，`loading_sign` 模式会直接把符号和量级一起写进不等式。若希望允许 over-coupling，可以把 `max_coupling` 设为更大的值；若希望自由异常 patch 完全不受 cap 限制，则不要对这些 patch 启用 cap。

动态更新 cap：

```python
inversion.update_euler_cap_constraint(
    "FaultA",
    selector={"edge": "top"},
    mode="motion_sense",
    max_coupling=1.5,
    enabled=True,
)
```

添加 hard backslip 约束：

```python
inversion.add_interseismic_backslip_constraint(
    "FaultA",
    state="prescribed_coupling",
    selector={"edge": "bottom"},
    coupling=0.0,
)
```

完整公式和字段见 [震间加载、Backslip 与 Coupling](interseismic_kinematics.md)。

### Deep-slip loading proxy 约束

深部滑动加载代理不写入 `interseismic_config.yml`，也不复用上面的
`backslip_constraints`。它把浅部 slip 参数 `s` 和深部 slip 参数 `b` 直接建立跨
fault 线性关系，例如：

```text
bottom_continuity:  s_i - b_j = 0
full_locking:       s_i = 0
cap:                -sigma*b_j <= 0, -sigma*s_i <= 0,
                    sigma*s_i - kmax*sigma*b_j <= 0
```

这些矩阵仍然注册到同一个 constraint manager，但入口是脚本 API：

```python
mapping = inversion.preview_deep_slip_loading_mapping(
    shallow_fault="ShallowFault",
    deep_faults=["DeepFault"],
    shallow_selector={"edge": "bottom"},
)

inversion.add_deep_slip_loading_constraint(
    mapping=mapping,
    state="bottom_continuity",
)
```

该路径的后处理字段为 `deep_loading_proxy_rate`、`shallow_slip_rate` 和
`coupling_to_deep`，不要与 Euler/block direct-backslip 的 `backslip_rate` 和
`coupling_ratio` 混用。完整说明见 [深部滑动加载代理](deep_slip_loading_proxy.md)。

## 自定义线性约束

自定义约束统一写成矩阵形式：

```text
A @ x <= b
A @ x = b
```

在 `SMC_F_J` 和 BLSE 中，`x` 是线性参数块，`A.shape[1]` 必须等于当前求解器的 `lsq_parameters`。

脚本接口：

```python
inversion.add_custom_inequality_constraint(A, b, name="my_ineq")
inversion.add_custom_equality_constraint(Aeq, beq, name="my_eq")
```

若约束对象是数据改正项，例如让两个 GPS 数据集共享 `translation`，不要手工猜测 poly 列号。
优先使用数据改正参数 resolver：

```python
inversion.add_data_correction_equality(
    [
        {"owner": "MainFault", "dataset": "gps_a", "transform": "translation"},
        {"owner": "MainFault", "dataset": "gps_b", "transform": "translation"},
    ],
    space="raw",
    name="common_gps_translation",
)
```

该接口会解析组合 `geodata.polys` 中的子 transform、处理 BLSE 与 Bayesian 线性参数偏移，并可在
`space="physical"` 下按归一化尺度比较 ramp/strain/Helmert 梯度。完整说明见
[数据改正项与 Frame Transform](data_corrections.md#数据改正参数关系约束)。

底层 BLSE solver：

```python
solver.add_inequality_constraint(A, b, name="my_ineq")
solver.add_equality_constraint(Aeq, beq, name="my_eq")
```

手工矩阵约束必须使用当前求解器的全局参数列。约束管理器在合并 `Aeq @ x = beq`
时会移除完全重复的 equality 行；若相同的 `A` 行对应不同的 `b`，或合并后 equality
矩阵仍然秩亏，会在求解前报错。这样可以避免把退化或互相冲突的等式约束交给底层
BLSE/SMC 求解器。

## Source Bounds 和 Source Constraints

多源反演中，非 `Fault` 源不要使用 `strikeslip/dipslip` 键，应使用 `source_bounds`：

```yaml
source_bounds:
  MyPressureSource:
    pressure: [-1.0e6, 1.0e6]
  MySbarbotSource:
    eps12: [-1.0e-4, 1.0e-4]
    eps13: [-1.0e-4, 1.0e-4]
```

`source_constraints` 通过 source adapter 生成约束矩阵：

```yaml
source_constraints:
  FaultA:
    - {name: ss_positive, type: inequality, rule: "strikeslip >= 0"}
  MyPressureSource:
    - {name: positive_pressure, type: inequality, rule: "pressure >= 0"}
  MySbarbotSource:
    - {name: incompressible, type: equality, rule: "incompressible"}
```

对 `Fault` 源，`source_constraints` 按分量名定位列。`slipdir` 只表示启用哪些
分量，ECAT/CSI 内部统一按 canonical `sdtc` 顺序排列参数；因此 `slipdir: ds`
与 `slipdir: sd` 等价，启用走滑和倾滑时总是先走滑、后倾滑。若配置引用了
当前 `slipdir` 中不存在的分量，ECAT 会直接报错；若不写该规则，则表示不施加
对应约束。需要手工检查矩阵列或编写自定义 `A @ m` 约束时，再阅读
[Rake Constraints](rake_constraints.md) 中的“线性未知参数排列（高级）”小节。

## 推荐使用顺序

1. 在 `default_config.yml` 中设置数据、GF、Laplacian、sigma/alpha 和 `interseismic_config_file`。
2. 在 `bounds_config.yml` 中设置 `strikeslip/dipslip`、`rake_angle`、`poly`、`sigmas`、`alpha` 和普通 `source_constraints`。
3. 若是震间模型，在 `interseismic_config.yml` 中设置 `blocks` 和 `fault_loading`；只在需要时启用 `cap_constraints` 或 `backslip_constraints`。
4. 用 [Fault Patch Indices](fault_patch_indices.md) helper 处理动态 patch 子集，脚本只传最终 selector 或 patch id。
5. 求解前查看 `constraint_manager.print_summary()`；震间案例再运行 `print_interseismic_constraint_report(...)`。

## 注意事项

- 线性 rake、Euler cap 和自定义 `A @ x` 约束只在 `SMC_F_J + ss_ds` 和 `BLSE + ss_ds` 中生效。
- `FULLSMC` 中滑动参数本身被采样，约束管理器主要提供边界，不合并线性约束矩阵。
- `source_bounds` 和 `source_constraints` 用于非 `Fault` 源或 adapter 支持的通用源约束。
- `zero_edge_slip(...)` 只适用于 `Fault` 源，并要求断层对象已有 `edge_triangles_indices`。
- 旧的 `euler_constraints` 主配置结构已移除；新脚本应使用 `interseismic_config.yml:blocks`、`fault_loading` 和可选 cap/backslip 约束。

## 相关页面

- [线性滑动反演配置](config_linear_slip.md)
- [震间加载、Backslip 与 Coupling](interseismic_kinematics.md)
- [Fault Patch Indices](fault_patch_indices.md)
- [断层边界识别](fault_edges.md)
- [BLSE/VCE 参考](blse_vce.md)
