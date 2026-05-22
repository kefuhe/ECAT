# BLSE/VCE 线性滑动分布反演

BLSE/VCE 是非线性几何反演后的标准第二步。几何固定后，建立断层网格，组装 Green's functions，再把分布式滑动作为约束线性反问题求解。

## 对应案例与参考

| 你要确认的问题 | 推荐案例 | 相关参考 |
| --- | --- | --- |
| 固定几何后如何跑一个 BLSE 入门例子 | [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md) | [BLSE/VCE 参考](../reference/blse_vce.md) |
| `default_config.yml` 和 `bounds_config.yml` 怎么写 | [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md#文件来源与生成方式) | [线性滑动反演配置](../reference/config_linear_slip.md), [CLI 命令参考](../reference/cli.md#线性-blsevce-配置) |
| rake、Euler、零滑、边界零滑和自定义线性约束如何管理 | [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md) | [ECAT 约束管理器](../reference/constraint_manager.md) |
| sigma 和 alpha 如何解释 | [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md#4-single-模式用固定平滑参数求解) | [Sigmas 与 Alpha 配置模式](../reference/sigmas_alpha.md) |

## 目标

这一步估计：

- strike-slip 和 dip-slip 分布
- 支持时可扩展 tensile 或 coupling 分量
- InSAR 多项式或 ramp 修正
- 数据拟合、残差、地震矩和震级

## 入口

```python
from eqtools.csiExtend.blse_multifaults_inversion import BoundLSEMultiFaultsInversion
```

## 配置文件来源

线性反演通常需要两个配置文件：`default_config.yml` 和 `bounds_config.yml`。案例目录中一般已经提供了按事件调好的版本；新建反演目录时，可以先用 CLI 在当前目录生成模板：

```bash
ecat-generate-config -o default_config.yml --gf-method cutde
ecat-generate-boundary -o bounds_config.yml -f MyFault
```

随后按案例修改数据顺序、断层名称、Green's function 方法、平滑参数、滑动边界和 rake/Euler 约束。命令细节见 [CLI 命令参考](../reference/cli.md)，配置字段见 [线性滑动反演配置](../reference/config_linear_slip.md)。

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
top_edge_mid_depth = 1.7692
fault_top_depth = 0.0
fault_bottom_depth = 8.0
fault.top = fault_top_depth
fault.depth = fault_bottom_depth
# clon/clat/cdepth 为非线性几何步骤得到的顶边中点三维坐标。
# top/depth 为线性滑动面向上、向下扩展后的顶部和底部深度。
fault.generate_top_bottom_from_nonlinear_soln(
    clon=87.39976,
    clat=28.66787,
    cdepth=top_edge_mid_depth,
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
inv.extract_and_plot_blse_results(plot_faults=True, plot_data=True)
```

## 求解模式

| 模式 | 方法 | 用途 |
| --- | --- | --- |
| 固定平滑 | `run(alpha=[...])` 或 `run(penalty_weight=[...])` | 复现已选模型。 |
| L-curve/smoothing loop | `simple_run_loop(...)` | 诊断平滑与数据拟合权衡。 |
| VCE | `run_simple_vce()` | 估计数据和约束权重。 |

三种模式共用同一个约束管理器。`bounds_config.yml` 中的边界、rake、Euler、零滑、边界零滑和自定义线性约束会在固定权重 BLSE、smoothing loop 和 VCE 中统一生效。第一个可运行例子建议先用固定平滑 BLSE，确认约束和输出链条正确后，再用 smoothing loop 或 VCE 做权重诊断。

从非线性结果进入线性反演时，`clon/clat/cdepth` 对应 `model_results_median.txt` 或等价摘要中的 `lon/lat/depth`，含义是断层顶边中点三维坐标。`fault.top` 和 `fault.depth` 则是线性滑动面扩展后的顶部、底部深度，常用于把紧凑源几何扩展成可分布滑动的断层面。两者不能混写。

线性配置中的 `geodata.sigmas` 控制数据标准差超参数，`alpha` 控制拉普拉斯平滑尺度。两者都支持 `single`、`individual` 和 `grouped` 三种组织模式；`log_scaled: true` 时，配置值是 `log10` 尺度。例如 `alpha.initial_value: -2.0` 表示实际 `alpha = 0.01`，对应较强的平滑惩罚权重。完整写法见 [Sigmas 与 Alpha 配置模式](../reference/sigmas_alpha.md)。

## 输出

标准输出应包括：

- 滑动平面图和地图图件
- data/synthetic/residual 文件
- `output/slip_<FaultName>.gmt`
- `output/slipdir_<FaultName>.txt`
- `output/stat_infos/*`
- 地震矩和震级摘要
- L-curve 或 VCE 诊断结果

## 检查清单

- 几何来自非线性几何反演或明确的外部模型。
- 断层迹线和网格生成可复现。
- bounds 与震源机制和符号约定一致。
- 如使用零滑或边界零滑，`source_constraints` 中的断层名、边界名和滑动分量别名正确；边界零滑需要断层对象已有 `edge_triangles_indices`。
- InSAR `polys` 明确。
- 若包含 GPS，vertical 分量使用方式明确。
- VCE 或 L-curve 结果被保存，而不是只保留最终图。

## 下一步

- 要复现实例，转到 [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md)。
- 要调整约束，查 [ECAT 约束管理器](../reference/constraint_manager.md)。
- 要报告结果，按 [BLSE/VCE 参考](../reference/blse_vce.md#推荐报告内容) 中的字段整理。
