# BLSE/VCE 线性滑动分布反演

BLSE/VCE 是非线性几何反演后的标准第二步。几何固定后，建立断层网格，组装 Green's functions，再把分布式滑动作为约束线性反问题求解。

如果只需要最小可复制脚本，先看 [BLSE/VCE 最小脚本骨架](../examples/blse_minimal_run.md)。如果还不清楚为什么标准流程要先几何、再线性滑动，先读 [标准两步走反演逻辑](../concepts/two_step_inversion.md)。

<a id="linear-inputs"></a>

## 反演前先准备 fault、geodata 和 config

先按手头输入选择路线，不需要把所有构建方法写进同一个反演脚本：

| 手头输入或任务 | 可复制入口 | 完整说明 |
| --- | --- | --- |
| 非线性几何反演的 `lon/lat/depth/strike/dip/length` | [由非线性结果构建断层](../examples/fault_from_nonlinear_geometry.md) | [Fault Geometry Construction](../reference/fault_geometry_construction.md#nonlinear-result) |
| 地表迹线 + 单倾角 | [单倾角平面](../examples/fault_trace_preprocessing.md#single-dip) | [Fault Geometry Construction](../reference/fault_geometry_construction.md#trace-dip-single) |
| 地表迹线 + 多个倾角参考点 | [沿走向变化倾角](../examples/fault_trace_preprocessing.md#multiple-dips) | [Fault Geometry Construction](../reference/fault_geometry_construction.md#trace-dip-varying) |
| 用 BLSE 比较多个倾角且保持 patch 对应 | [固定参考拓扑](../examples/fault_trace_preprocessing.md#fixed-topology) | [固定拓扑倾角搜索](04b_blse_dip_search.md) |
| ECAT 降采样、外部 SAR 点或 GNSS ENU | [反演前读取 InSAR 与 GNSS](../examples/inversion_data_loading.md) | [观测数据读入参考](../reference/observation_data_readers.md) |
| BLSE/VCE 配置与约束 | 本页[配置文件来源](#配置文件来源) | [线性滑动配置](../reference/config_linear_slip.md) |

## 运行与诊断入口

| 你要确认的问题 | 推荐入口 | 相关参考 |
| --- | --- | --- |
| 固定几何后如何跑 BLSE 入门例子 | [BLSE/VCE 最小脚本骨架](../examples/blse_minimal_run.md) | [BLSE/VCE 参考](../reference/blse_vce.md) |
| `default_config.yml`、`bounds_config.yml` 和 `interseismic_config.yml` 怎么分工 | 本页 [配置文件来源](#配置文件来源) | [线性滑动配置](../reference/config_linear_slip.md), [CLI](../reference/cli.md#linear-blse-vce-config) |
| rake、零滑、边界零滑、Euler cap 和自定义约束如何管理 | [约束管理器](../reference/constraint_manager.md) | [Fault Patch Indices](../reference/fault_patch_indices.md) |
| 如何计算 Euler/block 模式的震间 loading/backslip/coupling | [Interseismic Kinematics](../reference/interseismic_kinematics.md) | [线性滑动配置](../reference/config_linear_slip.md#震间配置) |
| 如何用深部自由滑动作为浅部加载代理 | [Deep Slip Loading Proxy](../reference/deep_slip_loading_proxy.md) | [Fault Patch Indices](../reference/fault_patch_indices.md) |
| sigma 和 alpha 如何解释 | [Sigmas and Alpha](../reference/sigmas_alpha.md) | [BLSE/VCE 参考](../reference/blse_vce.md) |
| 固定几何后如何选择平滑强度 | [固定几何平滑搜索](04a_blse_smoothing_search.md) | [平滑模板](../../scripts/test_smoothing_search_BLSE.py), [BLSE/VCE 参考](../reference/blse_vce.md#smoothing-loop) |
| 迹线已定但需要用 BLSE 比较倾角 | [固定拓扑倾角搜索](04b_blse_dip_search.md) | [倾角模板](../../scripts/test_dip_search_BLSE.py), [Fit Statistics](../reference/fit_statistics.md) |
| 如何检查倾角选择是否依赖平滑强度 | [倾角 × 平滑敏感性](04c_blse_dip_smoothing_search.md) | [联合模板](../../scripts/test_dip_smoothing_search_BLSE.py) |

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

### 约束从哪一层开始

普通同震或震后反演先把可复现的 bounds、rake 和零滑规则写入
`bounds_config.yml`。只有当前实验需要临时改变时，才在 inversion 初始化后调用
runtime 接口：

| 当前任务 | 推荐入口 | 下一步 |
| --- | --- | --- |
| 建立可分享的默认约束 | `bounds_config.yml` | 生成模板后修改断层名、边界和 rake |
| 当前对象试验一个 fault/component 边界 | `update_bounds(...)` | 求解前检查 snapshot |
| 试验局部 patch | `add/replace_patch_constraints(...)` | 明确 selector；重叠时显式决定是否覆盖 |
| 临时调整 fault rake | `update/replace_fault_rake_limits(...)` | 需要恢复时调用对应 `clear` |
| 不确定约束最终是否生效 | `get_constraint_snapshot(validate=True)` | 检查 bounds、group 数和 validation |

可直接复制的 YAML 与 Python 片段见
[约束配置与运行时调整短例](../examples/constraint_config_runtime.md)；完整优先级、
FULLSMC/SMC_FJ 差异和高级矩阵接口见
[约束管理器](../reference/constraint_manager.md)。

## 典型脚本流程

```python
import numpy as np
from csi import insar
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import BayesianAdaptiveTriangularPatches as TriFault
from eqtools.csiExtend.blse_multifaults_inversion import BoundLSEMultiFaultsInversion

lon0 = 96.2
lat0 = 21.1

# prefix 对应 <prefix>.txt、<prefix>.rsp 和 <prefix>.cov，不写扩展名。
sar_a = insar("TrackA", lon0=lon0, lat0=lat0, verbose=False)
sar_a.read_from_varres(
    "InSAR/downsample/track_a_ifg",
    triangular=False,
    cov=True,
)

sar_b = insar("TrackB", lon0=lon0, lat0=lat0, verbose=False)
sar_b.read_from_varres(
    "InSAR/downsample/track_b_ifg",
    triangular=False,
    cov=True,
)

geodata = [sar_a, sar_b]

fault = TriFault("MainFault", lon0=lon0, lat0=lat0, verbose=False)
fault_top_depth = 0.0
fault_bottom_depth = 20.0
fault.top = fault_top_depth
fault.depth = fault_bottom_depth

# clon/clat/cdepth 是非线性几何步骤得到的顶边中点三维坐标。
fault.generate_top_bottom_from_nonlinear_soln(
    clon=96.20,
    clat=21.10,
    cdepth=1.5,
    strike=65.0,
    dip=70.0,
    length=30.0,
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

上例读取普通四叉树/矩形 `.rsp`，所以使用 `triangular=False`；trirb 或其他三角形结果必须改为
`triangular=True`。`cov=True` 会读取完整 `.cov`，此时不要再调用 `buildDiagCd()` 覆盖它；若没有
`.cov`，使用 `cov=False`，读入后再调用 `buildDiagCd()`。完整分流见
[反演前读取 InSAR 与 GNSS 数据](../examples/inversion_data_loading.md)。

如果一个脚本里需要反复生成多数据集拟合图、多断层滑动图或震间字段图，可以在保留上述
`extract_and_plot_blse_results()` 主线的基础上使用上层 figure product：

```python
inv.plot_data_fits(outdir="Modeling", file_type="pdf")
inv.plot_fault_fields(fields=("total", "ss"), outdir="output")
```

这些接口复用已有 CSI/ECAT 绘图方法，只负责组织常用图件。完整参数见
[Figure Products](../reference/figure_products.md)。

## 求解模式

| 模式 | 方法 | 用途 |
| --- | --- | --- |
| 固定平滑 | `run(alpha=[...])` 或 `run(penalty_weight=[...])` | 复现已选模型 |
| L-curve / smoothing loop | `simple_run_loop(...)` | 诊断平滑与数据拟合权衡 |
| VCE | `run_simple_vce()` | 估计数据和约束权重 |

三种模式共用同一套约束配置和管理器。`bounds_config.yml` 中的边界、rake、
零滑、边界零滑和自定义线性约束由同一入口组装；震间 cap/backslip 约束来自
`interseismic_config.yml`。VCE 如果输出移除等式或不等式后重试的 warning，
该结果不再代表完整约束模型，应回到固定权重 BLSE 检查约束可行性；详细限制见
[BLSE/VCE 参考](../reference/blse_vce.md#约束检查)。

第一个可运行例子建议先用固定平滑 BLSE，确认约束和输出链条正确后，再用 smoothing loop 或 VCE 做权重诊断。

## 运行后端与单进程线程

普通桌面运行直接执行脚本，Matplotlib 使用当前交互后端；无图形界面的 WSL、服务器
或批处理任务可以临时使用 `Agg`：

```bash
python test_slip_inversion.py
MPLBACKEND=Agg python test_slip_inversion.py
```

这两条命令的反演模式相同。`MPLBACKEND=Agg` 只让图件保存到文件而不弹出交互窗口，
不会把 BLSE 改成另一种求解算法，也不会控制 NumPy、SciPy、CVXOPT 或 CUTDE 的
计算线程。看到 `FigureCanvasAgg is non-interactive` 时，含义只是当前后端不能弹窗。

BLSE/VCE 通常是单 Python 进程内的稠密线性代数。不要把某台电脑的 8、16 或更多
线程写成固定经验值；先运行不带线程变量的默认基线，再确认实际加载的是 MKL、
OpenBLAS 还是两者并存，并在代表性案例上比较 1、4、8、16 线程的总耗时。候选值
不应机械超过物理核心数，绘图后端和输出选项必须相同，结果还应在数值容差内一致。
检测命令、Windows/Linux 临时变量写法和完整测速方法见
[BLSE 初始化、BLAS 后端与线程](../getting_started/troubleshooting.md#5-blse-初始化blas-后端与线程)。
如果需要先区分单进程线程与 MPI rank，读
[进程、MPI Rank、线程与 CPU 亲和性](../concepts/parallel_process_rank_thread.md)。

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
- 做倾角、smoothing 或约束方案循环测试时，按
  [循环统计可复制模式](../examples/script_templates.md#loop-statistics) 在每轮求解后立即保存逐数据集
  RMS/VR 和全局 solver-vector RMS/VR；不要把逐数据集 RMS/VR 的算术平均当作总拟合。
- 做倾角循环时优先使用 [固定拓扑倾角搜索](04b_blse_dip_search.md)，避免把每轮重新剖分造成的 patch 数量和位置差异误当成倾角效应。
- BLSE 的 `run()` 和 VCE 的 `run_simple_vce()` 返回时已分发最新模型；统计应紧跟在该轮求解之后。`fit_statistics_to_dataframe()` 只转换已有 rows，不会重新求解或重建另一套模型。
- VCE 或 L-curve 结果被保存，而不只是保留最终图。

## 下一步

- 要从最短代码开始，转到 [BLSE/VCE 最小脚本骨架](../examples/blse_minimal_run.md)。
- 要调整约束，查 [ECAT 约束管理器](../reference/constraint_manager.md)。
- 要计算 Euler/block 震间 loading/backslip/coupling 或导出 GMT，查 [震间加载、Backslip 与 Coupling](../reference/interseismic_kinematics.md)。
- 要用深部自由滑动作为加载代理，查 [深部滑动加载代理](../reference/deep_slip_loading_proxy.md)。
- 要解释 trace 长度、mesh、面积、slip 和 Mw 统计，查 [Fault Summary](../reference/fault_summary.md)。
- 要确认 RMS/VR 公式、poly include 语义或输出结构化拟合表，查 [Fit Statistics](../reference/fit_statistics.md)。
- 要在固定几何上选择平滑强度，查 [BLSE 固定几何平滑参数搜索](04a_blse_smoothing_search.md)。
- 要在保持 patch 身份一致的条件下比较一组倾角，查 [BLSE 固定拓扑倾角搜索](04b_blse_dip_search.md)。
- 要检查倾角与平滑耦合，查 [倾角 × 平滑参数敏感性分析](04c_blse_dip_smoothing_search.md)。
- 如果固定几何不足以表达滑动分布不确定性，查高级路线 [Bayesian 联合几何-滑动分布反演](05_joint_bayesian_geometry_slip.md)。
