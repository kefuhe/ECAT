# BLSE/VCE 线性滑动分布反演

BLSE/VCE 是非线性几何反演后的标准第二步。几何固定后，建立断层网格，组装 Green's functions，再把分布式滑动作为约束线性反问题求解。

如果只需要最小可复制脚本，先看 [BLSE/VCE 最小脚本骨架](../examples/blse_minimal_run.md)。如果还不清楚为什么标准流程要先几何、再线性滑动，先读 [标准两步走反演逻辑](../concepts/two_step_inversion.md)。

## 对应案例与参考

| 你要确认的问题 | 推荐入口 | 相关参考 |
| --- | --- | --- |
| 固定几何后如何跑 BLSE 入门例子 | [Dingri 2020: BLSE/VCE](../casebook/dingri_blse_vce.md) | [BLSE/VCE 参考](../reference/blse_vce.md) |
| `default_config.yml`、`bounds_config.yml` 和 `interseismic_config.yml` 怎么分工 | 本页 [配置文件来源](#配置文件来源) | [线性滑动配置](../reference/config_linear_slip.md), [CLI](../reference/cli.md#linear-blse-vce-config) |
| rake、零滑、边界零滑、Euler cap 和自定义约束如何管理 | [约束管理器](../reference/constraint_manager.md) | [Fault Patch Indices](../reference/fault_patch_indices.md) |
| 如何计算 Euler/block 模式的震间 loading/backslip/coupling | [Interseismic Kinematics](../reference/interseismic_kinematics.md) | [线性滑动配置](../reference/config_linear_slip.md#震间配置) |
| 如何用深部自由滑动作为浅部加载代理 | [Deep Slip Loading Proxy](../reference/deep_slip_loading_proxy.md) | [Fault Patch Indices](../reference/fault_patch_indices.md) |
| sigma 和 alpha 如何解释 | [Sigmas and Alpha](../reference/sigmas_alpha.md) | [BLSE/VCE 参考](../reference/blse_vce.md) |

## 目标

这一阶段估计：

- strike-slip 和 dip-slip 分布；
- InSAR 多项式或 ramp 修正；
- 数据拟合、残差、地震矩和震级；
- 可选的震间 loading、backslip、coupling 和 creep 派生字段；也可用深部滑动加载代理导出 `coupling_to_deep`。

## 配置文件来源

线性反演通常至少需要：

```bash
ecat-generate-config -o default_config.yml --gf-method cutde
ecat-generate-boundary -o bounds_config.yml -f MyFault
```

若是震间模型，再生成独立震间配置，并让主配置记录指针：

```bash
ecat-generate-config -o default_config.yml --gf-method cutde --interseismic-config interseismic_config.yml
ecat-generate-interseismic -o interseismic_config.yml -f MyFault
```

三类配置的职责：

| 文件 | 内容 |
| --- | --- |
| `default_config.yml` | 数据顺序、GF、Laplacian、sigma/alpha、poly、DES、`interseismic_config_file` |
| `bounds_config.yml` | 滑动边界、rake 约束、poly/sigma/alpha 边界、普通 `source_constraints` |
| `interseismic_config.yml` | 震间 `blocks`、`fault_loading`、可选 `cap_constraints`、可选 `backslip_constraints` |

旧的主配置 `euler_constraints` 已移除。震间块体运动不要写在 `bounds_config.yml`，也不要通过 cap selector 间接控制 loading。

## 典型脚本流程

```python
import numpy as np
from csi import insar
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import BayesianAdaptiveTriangularPatches as TriFault
from eqtools.csiExtend.blse_multifaults_inversion import BoundLSEMultiFaultsInversion

lon0 = 87.5
lat0 = 28.5

sar_t012a = insar("T012A", lon0=lon0, lat0=lat0, verbose=False)
sar_t012a.read_from_varres("../InSAR/Dingri_2020_T012A/downsampled/S1_T012A_ifg")
sar_t012a.buildDiagCd()

sar_t121d = insar("T121D", lon0=lon0, lat0=lat0, verbose=False)
sar_t121d.read_from_varres("../InSAR/Dingri_2020_T121D/downsampled/S1_T121D_ifg")
sar_t121d.buildDiagCd()

geodata = [sar_t012a, sar_t121d]

fault = TriFault("Dingri_2020", lon0=lon0, lat0=lat0, verbose=False)
fault_top_depth = 0.0
fault_bottom_depth = 8.0
fault.top = fault_top_depth
fault.depth = fault_bottom_depth

# clon/clat/cdepth 是非线性几何步骤得到的顶边中点三维坐标。
fault.generate_top_bottom_from_nonlinear_soln(
    clon=87.39976,
    clat=28.66787,
    cdepth=1.7692,
    strike=332.2241,
    dip=52.0271,
    length=12.0,
    top=fault_top_depth,
    depth=fault_bottom_depth,
)
fault.generate_mesh(top_size=1.0, bottom_size=1.5, show=False, verbose=0)
fault.initializeslip(values="depth")

inv = BoundLSEMultiFaultsInversion(
    "linear_slip",
    [fault],
    geodata,
    config="default_config.yml",
    bounds_config="bounds_config.yml",
    verbose=True,
)

inv.run(penalty_weight=None, alpha=[np.log10(1 / 100.0)])
inv.returnModel(print_stat=True)
inv.print_faults_summary()
inv.extract_and_plot_blse_results(plot_faults=True, plot_data=True)
```

`clon/clat/cdepth` 对应非线性几何结果中的 `lon/lat/depth`，含义是断层顶边中点三维坐标。`fault.top` 和 `fault.depth` 是线性滑动面扩展后的顶部、底部深度，不能混写。

## 求解模式

| 模式 | 方法 | 用途 |
| --- | --- | --- |
| 固定平滑 | `run(alpha=[...])` 或 `run(penalty_weight=[...])` | 复现已选模型 |
| L-curve / smoothing loop | `simple_run_loop(...)` | 诊断平滑与数据拟合权衡 |
| VCE | `run_simple_vce()` | 估计数据和约束权重 |

三种模式共用同一套约束管理器。`bounds_config.yml` 中的边界、rake、零滑、边界零滑和自定义线性约束会在固定权重 BLSE、smoothing loop 和 VCE 中统一生效。震间 cap/backslip 约束来自 `interseismic_config.yml`。

第一个可运行例子建议先用固定平滑 BLSE，确认约束和输出链条正确后，再用 smoothing loop 或 VCE 做权重诊断。

## 震间解释

### Euler/block direct-backslip

如果配置了 `interseismic_config.yml:fault_loading`，反演后可计算 Euler/block 震间字段：

```python
result = inv.calculate_interseismic_fields(
    "MyFault",
    slip_component="strikeslip",
)

inv.print_interseismic_constraint_report("MyFault")

inv.plot_interseismic_field(
    "MyFault",
    field="coupling_ratio",
    cmap="viridis",
    cblabel="Coupling ratio",
)
```

震间接口使用：

```text
q = backslip_rate
b = tectonic_loading_rate
coupling_ratio = -q / b
creep_rate_signed = b + q
```

字段定义、右旋/左旋符号和导出方法见 [震间加载、Backslip 与 Coupling](../reference/interseismic_kinematics.md)。设置 `fault_loading.blocks` 时，推荐让 `blocks[0]` 位于 `reference_strike` 右手侧、`blocks[1]` 位于左手侧；`motion_sense` 只用于诊断和约束方向。

### Deep-slip loading proxy

如果长期加载由深部自由滑动 patch 表达，而不是由 Euler/block pair 表达，则不要使用 `calculate_interseismic_fields()` 解释结果。先建立浅部到底部深部 patch 的几何映射，再添加可选约束并导出 deep proxy 字段：

```python
mapping = inv.preview_deep_slip_loading_mapping(
    shallow_fault="ShallowFault",
    deep_faults=["DeepFault"],
    shallow_selector={"edge": "bottom"},
    component="strikeslip",
)

inv.print_deep_slip_loading_report(mapping)

inv.add_deep_slip_loading_constraint(
    mapping=mapping,
    state="bottom_continuity",
)

result = inv.calculate_deep_slip_loading_fields(mapping=mapping)
coupling = result["fields"]["coupling_to_deep"]
```

该路径使用：

```text
b = matched deep slip
s = shallow_slip_rate
coupling_to_deep = (b - s) / b
creep_fraction_to_deep = s / b
```

完整说明见 [深部滑动加载代理](../reference/deep_slip_loading_proxy.md)。

## 输出

标准输出应包括：

- 滑动平面图和地图图件；
- data/synthetic/residual 文件；
- `output/slip_<FaultName>.gmt`；
- `output/slipdir_<FaultName>.txt`；
- `output/stat_infos/*`；
- 地震矩和震级摘要；
- 断层概览统计，可通过 `inv.print_faults_summary()` 或 `inv.get_faults_summary()` 查看；
- L-curve 或 VCE 诊断结果；
- 若是 Euler/block 震间模型，可额外导出 loading、backslip、coupling、creep patch GMT 和 center text。
- 若是 deep-slip loading proxy，可额外导出 `deep_loading_proxy_rate`、`shallow_slip_rate`、`slip_deficit_to_deep_signed`、`coupling_to_deep` patch GMT 和 center text。

## 检查清单

- 几何来自非线性几何反演或明确的外部模型。
- `inv.print_faults_summary()` 中的 trace 长度、patch/mesh 数、面积和深度范围符合预期。
- `default_config.yml` 的 `geodata` 顺序与脚本 `geodata = [...]` 一致。
- `bounds_config.yml` 的断层名与 `fault.name` 一致。
- bounds 与震源机制和符号约定一致。
- 若使用边界零滑，断层对象已有 `edge_triangles_indices`。
- 若需要局部 patch 子集，优先在脚本中用 [Fault Patch Indices](../reference/fault_patch_indices.md) helper 生成并保存 patch id。
- 若使用 Euler/block 震间模型，`blocks` 和 `fault_loading` 应在所有 patch 上计算 loading；cap/backslip selector 只控制约束范围。
- 若使用 Euler/block 震间模型，`blocks[0] - blocks[1]` 是代数顺序；若 loading 符号异常，优先检查 block 顺序和 `reference_strike` 分支。
- 若使用 Euler/block 震间模型，正式反演前运行 `inv.print_interseismic_preflight_report()`，确认 loading 符号、block 顺序、cap active/configured patch 数和 `skipped_hard`。
- 若使用 deep-slip loading proxy，先运行 `inv.print_deep_slip_loading_report(mapping)`，确认浅深映射距离、unique deep patch 数、分量和 near-zero deep loading 警告。
- 若启用 Euler cap，确认 `interseismic_config.yml:cap_constraints.faults` 不是显式空字典 `{}`；preflight 中 active cap 行应为非零。默认 `hard_overlap: skip` 会让 cap 自动跳过 `full_coupling`、`creep` 等 hard equality patch；默认 `mode: motion_sense` 下，若 cap 行数正常但 `coupling_ratio > max_coupling`，再检查 `bounds_config.yml` 是否同时约束 direct backslip `q` 的符号；固定 loading 场景可用 `mode: loading_sign` 直接按实际 loading 符号约束。
- InSAR `polys` 明确；若包含 GPS，vertical 分量使用方式明确。
- VCE 或 L-curve 结果被保存，而不只是保留最终图。

## 下一步

- 要复现实例，转到 [Dingri 2020: BLSE/VCE](../casebook/dingri_blse_vce.md)。
- 要调整约束，查 [ECAT 约束管理器](../reference/constraint_manager.md)。
- 要计算 Euler/block 震间 loading/backslip/coupling 或导出 GMT，查 [震间加载、Backslip 与 Coupling](../reference/interseismic_kinematics.md)。
- 要用深部自由滑动作为加载代理，查 [深部滑动加载代理](../reference/deep_slip_loading_proxy.md)。
- 要解释 trace 长度、mesh、面积、slip 和 Mw 统计，查 [Fault Summary](../reference/fault_summary.md)。
- 如果固定几何不足以表达滑动分布不确定性，查高级路线 [Bayesian 联合几何-滑动分布反演](05_joint_bayesian_geometry_slip.md)。
