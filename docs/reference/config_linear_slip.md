# 线性滑动反演配置

本页说明 BLSE/VCE 线性滑动分布反演中配置文件的组织方式。入门流程见 [BLSE/VCE 线性滑动反演](../workflows/04_linear_slip_blse_vce.md)，约束细节见 [ECAT 约束管理器](constraint_manager.md)。

## 文件分工

| 文件 | 职责 | 常改字段 |
| --- | --- | --- |
| `default_config.yml` | 数据顺序、poly、sigma/alpha、Green's functions、Laplacian、DES，以及可选震间配置文件指针 | `geodata`, `alpha`, `faults.defaults.method_parameters`, `des`, `interseismic_config_file` |
| `bounds_config.yml` | 线性参数边界和普通线性约束 | `strikeslip`, `dipslip`, `rake_angle`, `poly`, `sigmas`, `alpha`, `source_constraints` |
| `interseismic_config.yml` | 震间块体运动、断层加载关系、可选 Euler cap、可选 backslip/coupling 等式约束 | `blocks`, `fault_loading`, `cap_constraints`, `backslip_constraints` |
| Python 脚本 | 读取数据、构建断层 mesh、加载配置、运行求解、导出结果 | `geodata = [...]`, `faults_list = [...]`, `BoundLSEMultiFaultsInversion(...)` |

配置文件不是独立运行的。脚本中的 `geodata` 顺序、断层对象名称和配置中的数据/断层设置必须一致。

## 生成模板

```bash
ecat-generate-config -o default_config.yml --gf-method cutde
ecat-generate-boundary -o bounds_config.yml -f MyFault
ecat-generate-interseismic -o interseismic_config.yml -f MyFault
```

如果主配置需要记录震间配置文件指针：

```bash
ecat-generate-config -o default_config.yml --gf-method cutde --interseismic-config interseismic_config.yml
```

模块形式：

```bash
python -m eqtools.cli_tools.generate_config -o default_config.yml --gf-method cutde --interseismic-config interseismic_config.yml
python -m eqtools.cli_tools.generate_bounds_config -o bounds_config.yml -f MyFault
python -m eqtools.cli_tools.generate_interseismic_config -o interseismic_config.yml -f MyFault
```

模板只提供字段结构和默认示例值。正式案例仍需设置数据对象顺序、断层名、Green's function 后端、`polys`、`sigmas`、`alpha`、滑动边界和物理约束。

## 脚本入口

```python
from eqtools.csiExtend.blse_multifaults_inversion import BoundLSEMultiFaultsInversion

geodata = [sar_asc, sar_desc]
faults_list = [fault]

inv = BoundLSEMultiFaultsInversion(
    "linear_slip",
    faults_list,
    geodata,
    config="default_config.yml",
    bounds_config="bounds_config.yml",
    verbose=True,
)
```

如果 `default_config.yml` 中有 `interseismic_config_file`，初始化时会自动读取。也可以显式传入：

```python
inv = BoundLSEMultiFaultsInversion(
    "linear_slip",
    faults_list,
    geodata,
    config="default_config.yml",
    bounds_config="bounds_config.yml",
    interseismic_config="interseismic_config.yml",
)
```

## 主配置

最小可读结构：

```yaml
GLs: null
shear_modulus: 3.0e10
use_bounds_constraints: true
use_rake_angle_constraints: true
interseismic_config_file: null
units:
  observation: m

geodata:
  data: null
  verticals: true
  polys: 3
  faults: null
  sigmas:
    mode: individual
    update: true
    initial_value: [0.0, 0.0]
    log_scaled: true
```

常用字段：

| 字段 | 含义 | 入门建议 |
| --- | --- | --- |
| `GLs` | 自定义 Green's functions 路径；通常由代码自动生成 | 初学先设 `null` |
| `shear_modulus` | 剪切模量，影响地震矩和 Mw | 明确单位为 Pa |
| `use_bounds_constraints` | 是否启用 `bounds_config.yml` 的边界 | 通常 `true` |
| `use_rake_angle_constraints` | 是否把 `bounds_config.yml` 的 `rake_angle` 转成线性约束 | 机制明确时用 `true` |
| `interseismic_config_file` | 可选震间配置文件路径 | 只在震间/块体运动案例设置 |
| `units.observation` | 进入反演矩阵后的统一观测单位 | 同震/余滑累计位移通常 `m`；震间速度通常 `m/yr` 或 `mm/yr` |
| `geodata.verticals` | 每个数据集是否使用垂直分量 | 可写单个布尔值或与 `geodata` 等长列表 |
| `geodata.polys` | 每个数据集的数据改正项。历史字段名叫 poly；InSAR 是 offset/ramp，GPS 是 frame transform | InSAR 常用 `3`，GPS 需要框架平移时用 `translation` |
| `geodata.faults` | 每个数据集关联哪些断层 | 多事件/多断层建议显式写 |

旧的 `use_euler_constraints` 和 `euler_constraints` 已移出主配置。若主配置仍包含这些字段，读取时会直接报错；请改用独立的 `interseismic_config.yml`。

`units.observation` 是 reader/factor 转换之后、进入线性反演之前的全局数值单位。ECAT 假设 Green 函数、滑动参数、约束右端项和观测数据已经在同一数值单位下，因此 `units.observation` 也决定 fault slip、slip rate 和 loading 的解释单位。未写该字段时，普通位移反演按 `m` 解释，震间 loading 相关计算按 `m/yr` 解释。

## 数据改正项

`geodata.polys` 是数据改正项入口，不只表示数学多项式。常见设置：

```yaml
geodata:
  # Python geodata = [asc, desc, gps]
  verticals: [true, true, true]
  polys: [3, 3, translation]
```

| 数据类型 | 常用设置 | 含义 |
| --- | --- | --- |
| InSAR/SAR LOS | `1` | offset |
| InSAR/SAR LOS | `3` | offset, x ramp, y ramp |
| InSAR/SAR LOS | `4` | offset, x ramp, y ramp, xy cross term |
| GPS | `translation` | east/north/up 平移；是否包含 up 取决于垂直分量 |
| GPS | `full`, `strain`, `translationrotation` 等 | 高级 frame transform，容易吸收长波构造信号 |

当前 CSI GPS 实现不支持给 GPS 使用整数 `1/3/4`。GPS 若需要估计框架平移，应写
`translation`，不要写 `3`。完整的 transform 列表、参数数和使用风险见
[数据改正项与 Frame Transform](data_corrections.md)。

## Sigmas

`geodata.sigmas` 控制数据标准差或单位权标准差尺度。`run(...)` 中如果不显式传 `sigma` 或 `data_weight`，会使用配置里的 `initial_value`。

```yaml
geodata:
  sigmas:
    mode: individual
    update: true
    initial_value:
      T012A: 0.0
      T121D: 0.0
    log_scaled: true
```

| `mode` | 含义 |
| --- | --- |
| `single` | 所有数据集共享一个 sigma |
| `individual` | 每个数据集一个 sigma |
| `grouped` | 按用户定义的数据组共享 sigma |

`log_scaled: true` 时，`initial_value: -2.0` 表示实际 sigma 为 `10 ** -2`。完整分组规则见 [Sigmas 和 Alpha](sigmas_alpha.md)。

## Alpha

`alpha` 控制 Laplacian 平滑尺度。在线性求解中通常使用：

```text
penalty_weight = 1 / alpha
```

因此 `alpha` 越小，平滑惩罚越强。

```yaml
alpha:
  enabled: true
  mode: single
  update: true
  initial_value: [-2.0]
  log_scaled: true
  faults: null
```

`alpha.initial_value: -2.0` 且 `log_scaled: true` 时，实际 `alpha = 0.01`，对应 `penalty_weight = 100`。脚本中可等价写：

```python
inv.run(alpha=[-2.0])
# or
inv.run(penalty_weight=[100.0])
```

二者不要同时传入；代码会直接报错。

## Green's Functions 和 Laplacian

`faults.defaults.method_parameters` 控制每个断层的 GF 和 Laplacian 构建方式：

```yaml
faults:
  defaults:
    geometry:
      update: false
    method_parameters:
      update_GFs:
        method: cutde
        geodata: null
        verticals: null
      update_Laplacian:
        method: Mudpy
        bounds: [free, locked, free, free]
        topscale: 0.25
        bottomscale: 0.03
```

常用 GF 后端：

| `update_GFs.method` | 用途 |
| --- | --- |
| `cutde` | 三角断层常用弹性半空间后端 |
| `okada` | 矩形断层常用弹性半空间后端 |
| `pscmp` | PSCMP/PSGRN 层状介质工作流 |
| `edcmp` | EDCMP/EDGRN 层状介质工作流 |

查看层状介质后端选项：

```bash
ecat-generate-config --show-gf-options edcmp
ecat-generate-config --show-gf-options pscmp --format text
```

`update_Laplacian.bounds` 的顺序通常按 `[top, bottom, left, right]` 解释。边界设定应与物理假设和网格边界一致。

## DES

深度均衡平滑是可选项：

```yaml
des:
  enabled: false
  mode: per_patch
  norm: l2
```

入门 BLSE/VCE 教程建议先关闭 DES，除非案例明确需要深度均衡平滑。打开 DES 后，应在报告中说明 `mode`、`norm` 以及是否影响最终滑动深度分布。

## Bounds 配置

典型 `bounds_config.yml`：

```yaml
lb: -3
ub: 3

rake_angle:
  MyFault: [-120, -60]

strikeslip:
  MyFault: [-10, 10]

dipslip:
  MyFault: [-10, 0]

poly:
  MyFault: [-1000, 1000]

sigmas: [-3, 3]
alpha: [-3, 3]
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `lb`, `ub` | 通用兜底边界 |
| `strikeslip` | 走滑分量边界 |
| `dipslip` | 倾滑分量边界 |
| `rake_angle` | rake 角范围；在 BLSE/VCE 中转为 `strikeslip/dipslip` 线性角度约束 |
| `poly` | ramp/poly 参数边界 |
| `sigmas` | sigma 参数边界 |
| `alpha` | alpha 参数边界 |
| `source_bounds` | Pressure、Sbarbot 等非 Fault 源参数边界 |
| `source_constraints` | 用户自定义线性等式/不等式约束，包括零滑和边界零滑 |

当前约定中，左旋走滑为正，逆冲倾滑为正，opening 为正。边界正负号应与断层走向、rake 和案例机制说明一致。

## 普通线性约束

`source_constraints` 写在 `bounds_config.yml` 中：

```yaml
source_constraints:
  MyFault:
    - {name: ss_positive, type: inequality, rule: "strikeslip >= 0"}
    - {name: ds_negative, type: inequality, rule: "dipslip <= 0"}
    - {name: zero_ds_all, type: equality, rule: "dipslip == 0"}
    - {name: zero_top_ss, type: equality, rule: "zero_edge_slip(top, strikeslip)"}
    - {name: zero_top_bottom_sd, type: equality, rule: "zero_edge_slip(top+bottom, ss+ds)"}
```

这些约束由统一约束管理器解析。`strikeslip == 0` 或 `dipslip == 0` 会固定该断层所有 patch 的对应分量；`zero_edge_slip(...)` 只固定指定边界上的 patch。Fault 源的 `slipdir` 只表示启用哪些分量，ECAT/CSI 内部统一按 canonical `sdtc` 顺序排列参数，因此 `slipdir: ds` 与 `slipdir: sd` 等价，启用走滑和倾滑时总是先走滑、后倾滑。脚本和扩展代码应按分量名定位列，不应把用户输入字符串的字符顺序当作参数列顺序。若需要固定非边界的局部 patch 子集，通常在脚本中用 [Fault Patch Indices](fault_patch_indices.md) helper 生成 id，再传给 `add_patch_slip_constraint(...)`。

## 震间配置

震间配置不要写入 `bounds_config.yml` 或旧主配置字段。它使用独立文件：

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
    MyFault:
      blocks: [Block_A, Block_B]
      reference_strike: 90.0
      motion_sense: sinistral

cap_constraints:
  enabled: false
  faults:
    MyFault:
      selector: null
      hard_overlap: skip
      max_coupling: 1.0

backslip_constraints:
  - fault: MyFault
    state: full_coupling
    selector: {edge: top}
```

在 `fault_loading` 中，`blocks: [Block_A, Block_B]` 表示 `Block_A - Block_B` 投影到 patch 局部走向；`reference_strike` 只选择走向正向分支；`motion_sense` 只决定 Euler cap 的不等式方向和诊断期望，不会重排 block 顺序或翻转 loading。为了与 Blocks/celeri 常用走滑符号一致，推荐让 `Block_A` 位于 `reference_strike` 的右手侧、`Block_B` 位于左手侧；`east_side - west_side` 只在该 east side 确实位于参考走向右手侧时才是安全简写。完整公式、字段和导出接口见 [震间加载、Backslip 与 Coupling](interseismic_kinematics.md)。

固定 `fixed_pole`/`fixed_vector` 的输入按物理 Euler 单位解释，ECAT 会根据主配置 `units.observation` 转换到反演矩阵单位。若 `units.observation: mm/yr`，用户仍按 `degrees_per_myr` 或 `radians_per_year` 写固定 Euler；不要手动把固定值乘 1000。

如果长期加载不是 Euler/block pair，而是由一个深部自由滑动 fault 的普通 slip 参数代理，不要把它写入 `interseismic_config.yml`。该模式使用脚本 API 建立浅深映射和跨 fault 线性约束，见 [深部滑动加载代理](deep_slip_loading_proxy.md)。

### 震间 cap 配置排错

`cap_constraints` 的 `faults` 字段要特别区分三种写法：

```yaml
# 推荐：显式配置需要 cap 的 fault
cap_constraints:
  enabled: true
  defaults:
    selector: null
    mode: motion_sense
    hard_overlap: skip
    max_coupling: 1.0
  faults:
    MyFault:
      selector: null
      mode: motion_sense
      hard_overlap: skip
      max_coupling: 1.5

# 可用：省略 faults，让 defaults 应用到 fault_loading 中的所有 fault
cap_constraints:
  enabled: true
  defaults:
    selector: null

# 不要误用：这是显式空集合，不会生成任何 cap 行
cap_constraints:
  enabled: true
  defaults:
    selector: null
  faults: {}
```

若启用了 cap 但输出的 `coupling_ratio` 仍明显超出 1，先确认是否主动设置了
`max_coupling > 1`；例如 `max_coupling: 1.5` 会允许 coupling 到 1.5。若结果超过设置值，再运行
`inversion.print_interseismic_preflight_report()`，确认 Euler-cap rows 是否为
非零。默认 `mode: motion_sense` 只给 `q` 与 `b` 的相对上限，还需要
`bounds_config.yml` 同时给 direct backslip `q` 设置正确的基础符号边界。若
loading 由固定块体给出，可改用 `mode: loading_sign`，它会按实际 `sign(b)`
直接约束 `0 <= -q/b <= max_coupling`。

默认 `hard_overlap: skip` 表示 cap 只作用于仍自由估计的 patch。若某些 patch
已经在 `backslip_constraints` 中被 `full_coupling`、`creep` 或
`prescribed_coupling` 等 hard equality 固定，cap 会自动跳过这些 patch。
这样可以避免 `max_coupling: 1.0` 时 `q+b=0` 与 `q+b<=0` 在同一 patch 上
完全重合造成求解器数值退化。专家调试时才需要把它改成 `keep` 或 `error`。

## 常见误区

- `default_config.yml` 中的 `geodata` 顺序必须与脚本 `geodata = [...]` 一致。
- `bounds_config.yml` 的断层名必须与断层对象 `fault.name` 一致。
- BLSE/VCE 使用 `ss_ds` 线性滑动参数化；即使模板中有 Bayesian 采样字段，BLSE 配置类也会按 `ss_ds` 处理。
- `rake_angle` 在 BLSE/VCE 中不是独立待求参数，而是约束 `strikeslip/dipslip` 的角度范围。
- 零滑和边界零滑优先写在 `bounds_config.yml` 的 `source_constraints` 中，便于固定权重 BLSE、smoothing loop 和 VCE 使用同一套约束。
- 震间 `blocks` 和 `fault_loading` 负责计算所有 patch 的 loading；cap/backslip selector 只控制约束，不应让未约束 patch 的构造速率变为 0。
- 震间 `motion_sense` 不参与 loading 计算；如果 loading 符号不符合右旋/左旋预期，应检查 `blocks` 顺序和 `reference_strike`。推荐规则是 `blocks[0]` 放在 `reference_strike` 右手侧，`blocks[1]` 放在左手侧。
- ECAT 当前震间接口是 fault 级 ordered block pair；若需要 Blocks/celeri 式 segment topology 和长断层不同段自动继承不同 block pair，建议使用专业块体模型程序或先在外部确定 segment/block 关系。
- deep-slip loading proxy 第一版不是 YAML 配置项；它通过脚本显式检查 mapping 后再注册约束，避免把尚需案例检验的浅深几何关系过早固化到配置文件。
- 震间配置完成后可运行 `inversion.print_interseismic_preflight_report()`；它只读配置和当前约束，紧凑打印每条断层的 loading 统计、cap 数量和 hard/free overlap。
- `alpha` 和 `penalty_weight` 是倒数关系，脚本调用时不要同时传入。

## 相关页面

- [BLSE/VCE 线性滑动反演](../workflows/04_linear_slip_blse_vce.md)
- [BLSE/VCE 参考](blse_vce.md)
- [Sigmas 和 Alpha](sigmas_alpha.md)
- [ECAT 约束管理器](constraint_manager.md)
- [震间加载、Backslip 与 Coupling](interseismic_kinematics.md)
- [深部滑动加载代理](deep_slip_loading_proxy.md)
- [CLI 参考](cli.md)
