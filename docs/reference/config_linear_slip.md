# 线性滑动反演配置

本文说明 BLSE/VCE 线性滑动分布反演中 `default_config.yml` 和 `bounds_config.yml` 的组织方式。入门流程见 [BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md)，案例脚本对照见 [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md)。

## 文件分工

| 文件 | 负责内容 | 常改字段 |
| --- | --- | --- |
| `default_config.yml` | 数据顺序、poly、sigma/alpha、Green's functions、Laplacian、Euler/DES 开关 | `geodata`, `alpha`, `faults.defaults.method_parameters`, `des`, `euler_constraints` |
| `bounds_config.yml` | 线性参数边界和线性约束 | `strikeslip`, `dipslip`, `rake_angle`, `poly`, `sigmas`, `alpha`, `source_constraints` |
| Python 脚本 | 读取数据、构造断层网格、加载配置、运行求解、导出结果 | `geodata = [...]`, `faults_list = [...]`, `BoundLSEMultiFaultsInversion(...)` |

配置文件不是独立运行的。脚本中的 `geodata` 顺序、断层对象名称和配置中的数据/断层设置必须一致。

## 生成模板

在新的线性反演目录中，可以先生成模板：

```bash
ecat-generate-config -o default_config.yml --gf-method cutde
ecat-generate-boundary -o bounds_config.yml -f MyFault
```

模块形式：

```bash
python -m eqtools.cli_tools.generate_config -o default_config.yml --gf-method cutde
python -m eqtools.cli_tools.generate_bounds_config -o bounds_config.yml -f MyFault
```

模板只提供字段结构和默认示例值。正式案例需要继续设置数据对象顺序、断层名、Green's function 后端、`polys`、`sigmas`、`alpha`、滑动边界和线性约束。更多命令参数见 [CLI 命令参考](cli.md#线性-blsevce-配置)。

## 脚本如何读入

典型脚本入口如下：

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

构造对象时会读取两个 YAML 文件，随后按配置更新 Green's functions、Laplacian、数据组装和约束矩阵。`run(...)` 负责求解，`returnModel(...)` 把结果写回断层对象。

## 主配置：全局与数据

最小可读结构：

```yaml
GLs: null
shear_modulus: 3.0e10
use_bounds_constraints: true
use_rake_angle_constraints: true
use_euler_constraints: false

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
| `use_bounds_constraints` | 是否启用上下界约束 | 通常 `true` |
| `use_rake_angle_constraints` | 是否把 `bounds_config.yml` 中的 `rake_angle` 转成线性约束 | 机制明确时建议 `true` |
| `use_euler_constraints` | 是否启用 Euler 约束 | 震间或板块运动约束场景再打开 |
| `geodata.verticals` | 每个数据集是否使用垂直分量 | 可写单个布尔值或与 `geodata` 等长列表 |
| `geodata.polys` | 每个数据集的 ramp/poly 参数 | InSAR 常用 `3`，GPS 通常 `null` |
| `geodata.faults` | 每个数据集关联哪些断层 | 单事件可 `null`；多事件/多断层建议显式写 |

`geodata.faults` 与脚本中的数据顺序一一对应。多事件场景中，如果某些数据只覆盖其中一个事件，就应显式指定数据关联的断层子集；这种写法见 [Ridgecrest：GPS+InSAR 非线性几何反演](../casebook/ridgecrest_gps_insar.md#多事件覆盖关系)。

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

`mode` 支持：

| 模式 | 含义 |
| --- | --- |
| `single` | 所有数据集共享一个 sigma |
| `individual` | 每个数据集一个 sigma |
| `grouped` | 按用户定义的数据组共享 sigma |

`log_scaled: true` 时，`initial_value: -2.0` 表示实际 sigma 为 `10 ** -2`。完整分组规则见 [Sigmas 与 Alpha 配置模式](sigmas_alpha.md)。

## Alpha

`alpha` 控制 Laplacian 平滑尺度。在线性求解中，代码通常使用：

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

`alpha.initial_value: -2.0` 且 `log_scaled: true` 时，实际 `alpha = 0.01`，对应 `penalty_weight = 100`。脚本中可以等价地写：

```python
inv.run(alpha=[-2.0])
# 或
inv.run(penalty_weight=[100.0])
```

二者不要同时给；代码会直接报错。多断层平滑分组、`single/individual/grouped` 写法见 [Sigmas 与 Alpha 配置模式](sigmas_alpha.md)。

## Green's Functions 与 Laplacian

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

`pscmp` 和 `edcmp` 需要额外 `options`，可用 CLI 查看：

```bash
ecat-generate-config --show-gf-options edcmp
ecat-generate-config --show-gf-options pscmp --format text
```

`update_Laplacian.bounds` 的顺序通常按 `[top, bottom, left, right]` 解释。边界设定应与物理假设和网格边界一致，不能无解释地沿用模板。

## DES

深度均衡平滑可选：

```yaml
des:
  enabled: false
  mode: per_patch
  norm: l2
```

入门 BLSE/VCE 教程建议先关闭 DES，除非案例明确需要深度均衡平滑。打开 DES 后，应在报告中说明 `mode`、`norm` 以及是否影响最终滑动深度分布。

## Euler 约束

Euler 约束写在主配置中，并通过 `use_euler_constraints: true` 启用：

```yaml
use_euler_constraints: true

euler_constraints:
  enabled: true
  defaults:
    block_types: [dataset, dataset]
    euler_pole_units: [degrees, degrees, degrees_per_myr]
    fix_reference_block: null
  faults:
    MyFault:
      block_types: [dataset, euler_pole]
      blocks: [GPS_data, [100.2, 25.5, 0.45]]
      motion_sense: dextral
```

Euler 约束只适合有板块运动或震间加载含义的模型。它在线性参数 `strikeslip/dipslip` 上形成约束，详细逻辑见 [ECAT 约束管理器](constraint_manager.md#euler-约束)。

## 边界配置

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
| `rake_angle` | rake 角范围；在线性 BLSE/VCE 中转换为 `strikeslip/dipslip` 线性角度约束 |
| `poly` | ramp/poly 参数边界 |
| `sigmas` | sigma 参数边界；通常用于可更新或可估计权重 |
| `alpha` | alpha 参数边界；通常用于可更新或可估计平滑尺度 |
| `source_bounds` | Pressure、Sbarbot 等非 Fault 源参数边界 |
| `source_constraints` | 用户自定义线性等式/不等式约束，包括零滑和边界零滑 |

当前约定中，左旋走滑为正，逆冲倾滑为正，opening 为正。边界正负号应与断层走向、rake 和案例机制说明一致。

## 自定义线性约束

`source_constraints` 可写在 `bounds_config.yml` 中：

```yaml
source_constraints:
  MyFault:
    - {name: ss_positive, type: inequality, rule: 'strikeslip >= 0'}
    - {name: ds_negative, type: inequality, rule: 'dipslip <= 0'}
    - {name: zero_ds_all, type: equality, rule: 'dipslip == 0'}
    - {name: zero_top_ss, type: equality, rule: 'zero_edge_slip(top, strikeslip)'}
    - {name: zero_top_bottom_sd, type: equality, rule: 'zero_edge_slip(top+bottom, ss+ds)'}
```

这些约束由统一约束管理器解析。`strikeslip == 0` 或 `dipslip == 0` 会固定该断层所有 patch 的对应分量；`zero_edge_slip(...)` 只固定指定边界上的 patch，需要断层对象已经识别出 `edge_triangles_indices`。rake、Euler、零滑、边界零滑和自定义约束的支持模式见 [ECAT 约束管理器](constraint_manager.md#零滑与边界零滑约束)。

## 常见误区

- `default_config.yml` 中的 `geodata` 顺序必须与脚本 `geodata = [...]` 一致。
- `bounds_config.yml` 的断层名必须与断层对象 `fault.name` 一致。
- BLSE/VCE 使用 `ss_ds` 线性滑动参数化；即使模板中有 Bayesian 采样字段，BLSE 配置类也会按 `ss_ds` 处理。
- `rake_angle` 在线性 BLSE/VCE 中不是独立待求参数，而是约束 `strikeslip/dipslip` 的角度范围。
- 零滑和边界零滑应优先写在 `bounds_config.yml` 的 `source_constraints` 中，便于固定权重 BLSE、smoothing loop 和 VCE 使用同一套约束。
- `alpha` 和 `penalty_weight` 是倒数关系，脚本调用时不要同时传入。
- InSAR-heavy 反演应显式设置 `polys`，并说明 ramp/poly 是否参与结果导出。
- 最终案例不应无解释地保留模板中的过宽滑动边界、poly 边界或默认 sigma/alpha。
