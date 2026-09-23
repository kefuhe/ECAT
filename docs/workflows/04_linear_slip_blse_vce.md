# BLSE/VCE 线性滑动分布反演

BLSE/VCE 是非线性几何反演后的标准第二步。几何固定后，建立断层网格，组装 Green's functions，再把分布式滑动作为约束线性反问题求解。

这里的“几何固定”也限定一个 inversion 对象的生命周期：如果要比较倾角、底边或其他
几何候选，应先修改 fault 并完成 mesh，再为每个候选新建 inversion。一个候选内部可以
复用该对象扫描平滑权重或运行 VCE；不要在对象创建后原地换 mesh 并期待 `run()` 自动刷新
GF、Laplacian 和约束。

如果只需要最小可复制脚本，先看 [BLSE/VCE 最小脚本骨架](../examples/blse_minimal_run.md)。如果还不清楚为什么标准流程要先几何、再线性滑动，先读 [标准两阶段反演逻辑](../concepts/two_step_inversion.md)。

<a id="linear-inputs"></a>

## 反演前先准备 fault、geodata 和 config

先按手头输入选择路线，不需要把所有构建方法写进同一个反演脚本：

| 手头输入或任务 | 可复制入口 | 完整说明 |
| --- | --- | --- |
| 非线性几何反演的 `lon/lat/depth/strike/dip/length` | [由非线性结果构建断层](../examples/fault_from_nonlinear_geometry.md) | [Fault Geometry Construction](../reference/fault_geometry_construction.md#nonlinear-result) |
| 地表迹线 + 单倾角 | [单倾角平面](../examples/fault_trace_preprocessing.md#single-dip) | [Fault Geometry Construction](../reference/fault_geometry_construction.md#trace-dip-single) |
| 地表迹线 + 多个倾角参考点，只构建一个固定几何 | [沿走向变化倾角](../examples/fault_trace_preprocessing.md#multiple-dips) | [固定几何中的倾角剖面](../reference/dip_profile/fixed_geometry.md) |
| 用 BLSE 比较多个倾角且保持 patch 对应 | [固定参考拓扑](../examples/fault_trace_preprocessing.md#fixed-topology) | [固定拓扑倾角搜索](04b_blse_dip_search.md) |
| ECAT 降采样、外部 SAR 点或 GNSS ENU | [反演前读取 InSAR 与 GNSS](../examples/inversion_data_loading.md) | [观测数据读入参考](../reference/observation_data_readers.md) |
| BLSE/VCE 配置与约束 | 本页[配置文件来源](#配置文件来源) | [线性滑动配置](../reference/config_linear_slip.md) |

<a id="input-handoff"></a>

### 把三个输入接成同一个问题

| 已准备的内容 | 在本阶段如何使用 | 创建 inversion 前核对 |
| --- | --- | --- |
| [数据读入短例](../examples/inversion_data_loading.md#assemble-geodata)得到的 `geodata` | 传给构造函数的 `geodata` | 共同投影原点、单位、协方差和配置数据顺序 |
| [几何交接短例](../examples/fault_from_nonlinear_geometry.md#geometry-handoff)得到的 `rect` 或 `tri` | 将选定对象放入 `faults_list`，不重复建一套占位几何 | mesh 已完成，断层名与主配置及 bounds 一致 |
| 本页生成并修改的主配置与 bounds | 分别传给 `config` 和 `bounds_config` | 实际数据与断层名称、GF、平滑和约束均已核对 |

一个或多个断层直接按求解参数块顺序放入列表即可：`faults_list = [fault]`，多断层则写成
`faults_list = [west_fault, east_fault]`。不需要为了保序先建 `OrderedDict` 再转回列表；对象
的 `name` 用于匹配配置 source 名，列表位置用于确定 source/参数块顺序。

第一次装配时，可按下方典型脚本或[最小 BLSE 脚本](../examples/blse_minimal_run.md)的顺序组织。
如果已经建立 `geodata` 和 fault，就替换其中的数据与构网部分，继续使用自己的对象；
不要把示例中的坐标、数据文件名或 `MyFault/MainFault` 当作本项目的默认值。
先完成一次固定权重求解与输出检查，再进入 VCE 或参数搜索。

## 运行与诊断入口

| 你要确认的问题 | 推荐入口 | 相关参考 |
| --- | --- | --- |
| 固定几何后如何跑 BLSE 入门例子 | [BLSE/VCE 最小脚本骨架](../examples/blse_minimal_run.md) | [BLSE/VCE 参考](../reference/blse_vce.md) |
| `default_config.yml`、`bounds_config.yml` 和 `interseismic_config.yml` 怎么分工 | 本页 [配置文件来源](#配置文件来源) | [线性滑动配置](../reference/config_linear_slip.md), [CLI](../reference/cli.md#linear-blse-vce-config) |
| rake、零滑、边界零滑、Euler cap 和自定义约束如何管理 | [约束管理器](../reference/constraint_manager.md) | [Fault Patch Indices](../reference/fault_patch_indices.md) |
| 如何计算 Euler/block 模式的震间 loading/backslip/coupling | [Interseismic Kinematics](../reference/interseismic_kinematics.md) | [线性滑动配置](../reference/config_linear_slip.md#震间配置) |
| 如何用深部自由滑动作为浅部加载代理 | [Deep Slip Loading Proxy](../reference/deep_slip_loading_proxy.md) | [Fault Patch Indices](../reference/fault_patch_indices.md) |
| sigma 和 alpha 如何解释 | [Sigmas and Alpha](../reference/sigmas_alpha.md) | [BLSE/VCE 参考](../reference/blse_vce.md) |
| 固定几何后如何选择平滑强度 | [固定几何 L-curve](04a_blse_l_curve.md) | [L-curve 模板](../../scripts/test_BLSE_L_Curve.py), [BLSE/VCE 参考](../reference/blse_vce.md#smoothing-scan) |
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

# 求解前核对 L 中的 source/slip/poly 列；sigma/alpha 是尺度控制，不占 L。
inv.print_parameter_positions()
inv.run(penalty_weight=None, alpha=[np.log10(1 / 100.0)])
inv.extract_and_plot_blse_results(
    plot_faults=True,
    plot_data=True,
    data_poly="config",
    file_type="png",
    fault_outdir="output",
    data_outdir="Modeling",
    show=False,
)
```

`print_parameter_positions()` 只读 constraint manager 已解析的线性布局。需要程序化检查时
使用 `collect_parameter_layout()` 返回的 rows，不要解析终端表格。求解后的活动
sigma/alpha 数值仍由 scale parameter report 给出；结构表不会从配置初值猜测结果。完整
`S/L` 语义和各诊断入口的职责见
[参数列布局与诊断接口](../concepts/observation_matrix_layout.md#参数列布局与诊断接口)。

`clon/clat/cdepth` 对应非线性几何结果中的 `lon/lat/depth`，含义是断层顶边中点三维坐标。`fault.top` 和 `fault.depth` 是线性滑动面扩展后的顶部、底部深度，不能混写。

上例读取普通四叉树/矩形 `.rsp`，所以使用 `triangular=False`；trirb 或其他三角形结果必须改为
`triangular=True`。`cov=True` 会读取完整 `.cov`，此时不要再调用 `buildDiagCd()` 覆盖它；若没有
`.cov`，使用 `cov=False`，读入后再调用 `buildDiagCd()`。完整分流见
[反演前读取 InSAR 与 GNSS 数据](../examples/inversion_data_loading.md)。

`extract_and_plot_blse_results()` 与下列调用使用相同的合成观测和绘图协议。需要选择
部分数据集、多个滑动字段或震间字段时，可以直接调用：

```python
inv.plot_data_fits(outdir="Modeling", file_type="png")
inv.plot_fault_fields(fields=("total", "ss"), outdir="output", file_type="png")
```

这些接口沿用与正式结果相同的 `buildsynth()` 参数，只统一图件组织、保存目录和格式；
不会改变 Green's functions、协方差、约束、权重或 BLSE/VCE 解。完整参数见
[Figure Products](../reference/figure_products.md)。

`extract_and_plot_blse_results()` 继续处理 GPS、InSAR、leveling 和 cross-fault offset；
Bayesian 结果入口还会处理 opticorr。共享绘图产品不会扩大任一结果入口原有的数据类型
参与范围。

公共 `test_slip_inv_BLSE.py --mode single` 与 `test_slip_inv_VCE.py` 在该入口之后直接复用已经生成的 synthetic，
不再调用第二套 `buildsynth()`。GPS 与没有 corner 的 InSAR/opticorr 输入直接写点表；有
corner 的 raster 默认写降采样多边形，增加 `--export-point-values` 后才在
`Modeling/points/` 额外写中心点表。InSAR 点表包含 ENU 投影向量，opticorr 点表同行包含
east/north。多边形调用的 `triangular=None` 让 CSI 按 corner 形状选择三角形或四边形，
避免脚本误把降采样类型写死。

## 棋盘格分辨率检查

固定几何、约束和数据读入方式已经确认后，可复制
[`test_BLSE_Inv_Checkboard.py`](../../scripts/test_BLSE_Inv_Checkboard.py) 检查空间分辨率：

```bash
python test_BLSE_Inv_Checkboard.py
```

模板按下列顺序运行：

```text
固定 fault/mesh + InSAR/GPS/optical 观测几何与活动分量
  -> 生成 checkerboard truth
  -> 正演 synthetic
  -> 按数据名加噪并更新对角 Cd
  -> synthetic 替换原观测
  -> BLSE 恢复
  -> truth/recovery、拟合图和 data/synth/resid 文件
```

这是一个串行模板，不使用 MPI rank。随机种子只固定加噪复现性，不改变反演公式。
有 corner 的 InSAR/optical 结果保留为降采样多边形；没有 corner 的点模式输入直接写点表，
GPS 写站点 ENU 表，optical 点表同行写 east/north。

启用 GPS 时，取消脚本中 GPS 读取块的注释，把对象加入 `gpsdata`，并在 `noise_config` 中用
完全相同的数据名称设置噪声。配置中的 `geodata.data`、`verticals`、`faults`、`polys` 以及
sigma 顺序必须继续与脚本中的 `geodata` 一致。若 U 为 NaN 或对应
`verticals: false`，checkerboard 只替换并加噪 E、N，按 `direction="en"` 重建 Cd，原 U
保持不变；只有 U 有效且配置允许时才处理 ENU。启用 optical 块时，把对象加入
`opticaldata`，用同名 `noise_config` 项设置标量或 `east/north` 噪声，并把对应
`verticals` 项设为 `false`。

GPS 使用标量噪声时，各站、各活动分量独立抽样，但 E/N/U 共用同一标准差；它不是整条
ENU 向量共享一个随机数。垂向误差明显更大时，可在同一个 `noise_config` 中给 GPS 数据名
配置 `east/north/up` 三个标准差。optical 的 east/north 也独立抽样。可复制调用方式及
对角 `Cd` 的同步规则见
[checkerboard 模板说明](../examples/script_templates.md#非线性正演和分辨率检查模板)。

若明确知道近断层像元存在解缠、失相干或无法解析的破裂带，可以在创建 inversion 前启用
脚本中的过滤示例，但它会改变覆盖范围和最终分辨率结论，不能当作通用预处理默认值。

## 求解模式

| 模式 | 方法 | 用途 |
| --- | --- | --- |
| 固定平滑 | `run(alpha=[...])` 或 `run(penalty_weight=[...])` | 复现已选模型 |
| L-curve / smoothing loop | `scan_penalty_weights(...)` | 返回候选摘要和逐数据集长表，诊断平滑与数据拟合权衡 |
| VCE | `run_simple_vce()` | 估计数据和约束权重 |

三种模式共用同一套约束配置和管理器。`bounds_config.yml` 中的边界、rake、
零滑、边界零滑和自定义线性约束由同一入口组装；震间 cap/backslip 约束来自
`interseismic_config.yml`。VCE 如果输出移除等式或不等式后重试的 warning，
该结果不再代表完整约束模型，应回到固定权重 BLSE 检查约束可行性；详细限制见
[BLSE/VCE 参考](../reference/blse_vce.md#约束检查)。

第一个可运行例子建议先用固定平滑 BLSE，确认约束和输出链条正确后，再用 smoothing loop 或 VCE 做权重诊断。
VCE 对所有可更新有效组统一估计绝对方差尺度，并以
`max(abs(log(update_factor))) < tol` 判断乘法收敛；默认 `tol=1e-4`。固定组仍参与
线性求解，但不参加停止判断。公式、结果字段和单分量情形见
[BLSE/VCE 参考](../reference/blse_vce.md#乘法收敛判据)。
简化 VCE 稳定后，大型连续 QP 可选用
`qp_acceleration="certified_kkt"` 评估活动集复用收益。该设置不改变 VCE 公式，快速路径
未通过证书时会回到中央求解路线；不要为了提速删除具有物理意义的 bounds、rake 或其他
约束。启用条件、诊断字段和回退含义见
[连续 QP 快速路径](../reference/blse_vce.md#连续-qp-快速路径可选)。

BLSE、VCE、L-curve、倾角搜索和 checkerboard 模板均为单 Python 进程任务，直接执行
`python <script>.py`；不要用 `mpiexec` 或 SMC 启动脚本。数值线性代数仍可使用底层线程，
其设置与 MPI rank 是两层不同的并行机制。

Smoothing loop 只返回候选表和权衡图：粗糙度统一按未加权 \(L_0\) 计算，且循环结束后
恢复调用前的活动解。图中的 preferred 点不会自动成为最终模型；选定权重后，用固定平滑
`run(penalty_weight=...)` 重新求解并输出滑动和残差产品。

## 运行后端与单进程线程

普通桌面运行直接执行脚本，Matplotlib 使用当前交互后端；无图形界面的 WSL、服务器
或批处理任务可以临时使用 `Agg`：

```bash
# Linux / WSL / Bash
python test_slip_inversion.py
MPLBACKEND=Agg python test_slip_inversion.py
```

Windows PowerShell 不能使用 `名称=值 command` 的 Bash 前缀语法，应写成：

```powershell
$env:MPLBACKEND = "Agg"
python .\test_slip_inversion.py
Remove-Item Env:MPLBACKEND -ErrorAction SilentlyContinue
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

## 特殊固定几何模型

本页主线面向同震固定几何滑动反演。震间和深部加载模型仍使用同一 BLSE/VCE 求解入口，
但字段定义、符号和前置检查不同：

| 场景 | 应使用的说明 |
| --- | --- |
| Euler/block direct-backslip，输出 loading、coupling 和 creep | [震间加载、Backslip 与 Coupling](../reference/interseismic_kinematics.md) |
| 深部自由滑动 patch 作为浅部长期加载代理 | [深部滑动加载代理](../reference/deep_slip_loading_proxy.md) |

不要把 deep-slip proxy 结果交给 Euler/block 的 `calculate_interseismic_fields()` 解释；进入
相应参考页后按其 preflight 和符号检查执行。

## 输出

标准输出应包括：

- VCE 尺度表中的 `State`、`Variance (v)`、物理 `Scale (s)`、`1/s`、`Qw` 和
  `Approx. red.Q`；固定组继续显示，但不应被解释为已估计分量；
- 滑动平面图和地图图件；
- data/synthetic/residual 文件；
- `output/slip_<FaultName>.gmt`；
- `output/slipdir_<FaultName>.txt`；
- `output/stat_infos/*`；
- 地震矩和震级摘要；
- 断层概览统计，可通过 `inv.print_faults_summary()` 或 `inv.get_faults_summary()` 查看；
- L-curve 或 VCE 诊断结果；
- 特殊震间或深部加载模型的派生字段与导出文件见对应参考页。

## 检查清单

- 几何来自非线性几何反演或明确的外部模型。
- `inv.print_faults_summary()` 中的 trace 长度、patch/mesh 数、面积和深度范围符合预期。
- `default_config.yml` 的 `geodata` 顺序与脚本 `geodata = [...]` 一致。
- `bounds_config.yml` 的断层名与 `fault.name` 一致。
- bounds 与震源机制和符号约定一致。
- 若使用边界零滑，断层对象已有 `edge_triangles_indices`。
- 若需要局部 patch 子集，优先在脚本中用 [Fault Patch Indices](../reference/fault_patch_indices.md) helper 生成并保存 patch id。
- 特殊震间或深部加载模型先运行对应 reference 指定的 preflight，确认符号、patch 选择和几何映射后再反演。
- InSAR `polys` 明确；若包含 GPS，vertical 分量使用方式明确。
- 做倾角、smoothing 或约束方案循环测试时，按
  [循环统计可复制模式](../examples/script_templates.md#loop-statistics) 在每轮求解后立即保存逐数据集
  RMS/VR 和全局 solver-vector RMS/VR；不要把逐数据集 RMS/VR 的算术平均当作总拟合。
- 做倾角循环时优先使用 [固定拓扑倾角搜索](04b_blse_dip_search.md)，避免把每轮重新剖分造成的 patch 数量和位置差异误当成倾角效应。
- BLSE 的 `run()` 和 VCE 的 `run_simple_vce()` 返回时已分发最新模型；统计应紧跟在该轮求解之后。`fit_statistics_to_dataframe()` 只转换已有 rows，不会重新求解或重建另一套模型。
- VCE 或 L-curve 结果被保存，而不只是保留最终图。

## 结果判读与后续分析 {#result-checks}

先确认上述输出和检查清单，再判断滑动特征是否稳定。拟合好坏应结合逐数据集残差、
模型约束和敏感性分析解释，不设跨案例通用的 RMS 合格阈值。

| 观察到的现象 | 优先核对 | 下一步 |
| --- | --- | --- |
| 滑动集中在模型边界 | 网格范围、边界零滑和数据覆盖 | 比较合理的边界与网格方案，记录滑动特征是否稳定 |
| 残差呈大尺度或局部系统结构 | 数据改正项、固定几何、观测投影 | 对照 data/synth/resid，回到对应的数据或几何检查 |
| 结果随平滑强度明显变化 | 拟合与未加权粗糙度的权衡 | 做[固定几何 L-curve](04a_blse_l_curve.md)，选定后显式求解并导出 |
| VCE 权重异常或结果不稳定 | 协方差、分组、固定组和约束 | 对照固定权重 BLSE 基线，保存迭代诊断后再解释权重 |

报告时保留数据、几何、网格、约束与权重设置，并按
[推荐报告内容](../reference/blse_vce.md#recommended-reporting)保存逐数据集统计和模型输出。
进入联合 Bayesian 前，先确认固定几何基线及其局限；固定几何结果本身不包含几何不确定性的传播。

## 下一步

- 要从最短代码开始，转到 [BLSE/VCE 最小脚本骨架](../examples/blse_minimal_run.md)。
- 要调整约束，查 [ECAT 约束管理器](../reference/constraint_manager.md)。
- 要计算 Euler/block 震间 loading/backslip/coupling 或导出 GMT，查 [震间加载、Backslip 与 Coupling](../reference/interseismic_kinematics.md)。
- 要用深部自由滑动作为加载代理，查 [深部滑动加载代理](../reference/deep_slip_loading_proxy.md)。
- 要解释 trace 长度、mesh、面积、slip 和 Mw 统计，查 [Fault Summary](../reference/fault_summary.md)。
- 要确认 RMS/VR 公式、poly include 语义或输出结构化拟合表，查 [Fit Statistics](../reference/fit_statistics.md)。
- 要在固定几何上选择平滑强度，查 [BLSE 固定几何 L-curve](04a_blse_l_curve.md)。
- 要在保持 patch 身份一致的条件下比较一组倾角，查 [BLSE 固定拓扑倾角搜索](04b_blse_dip_search.md)。
- 要检查倾角与平滑耦合，查 [倾角 × 平滑参数敏感性分析](04c_blse_dip_smoothing_search.md)。
- 如果固定几何不足以表达滑动分布不确定性，查高级路线 [Bayesian 联合几何-滑动分布反演](05_joint_bayesian_geometry_slip.md)。
