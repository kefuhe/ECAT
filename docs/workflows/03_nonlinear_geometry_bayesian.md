# Bayesian 非线性几何反演

本工作流说明 Bayesian 非线性几何反演：用紧凑源估计断层几何，并把优选几何传递给后续 BLSE/VCE 分布式滑动反演。

如果已经有 `clon/clat/cdepth/strike/dip/length` 等几何结果，只想看如何构建矩形元或三角元 fault object，直接看 [非线性几何结果到 fault object](../examples/fault_from_nonlinear_geometry.md)。

## 对应案例与参考

| 你要确认的问题 | 推荐案例 | 相关参考 |
| --- | --- | --- |
| InSAR-only 几何搜索如何组织 | [Wushi：InSAR-only 非线性几何反演](../casebook/wushi_nonlinear_geometry.md) | [非线性几何反演配置](../reference/config_nonlinear_geometry.md) |
| GPS+InSAR 多数据几何搜索如何组织 | [Ridgecrest：GPS+InSAR 非线性几何反演](../casebook/ridgecrest_gps_insar.md) | [InSAR 与 GPS 数据读取](01_data_reading_insar_gps.md), [Sigmas 与 Alpha 配置模式](../reference/sigmas_alpha.md) |
| 优选几何如何进入线性滑动反演 | [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md) | [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md), [线性滑动反演配置](../reference/config_linear_slip.md) |

## 目标

这一步估计紧凑源几何。注意，经纬度和深度指**断层顶边中点**，不是断层面几何中心：

- 顶边中点经度、纬度
- 顶边中点深度
- 走向
- 倾角
- 长度
- 宽度
- 平均滑动或震级代理量
- rake

输出应是优选几何、不确定性摘要和拟合诊断。分布式滑动后续用 BLSE/VCE 反演。

## 入口

ECAT 现在保留两套非线性几何入口：

| 入口 | 配置文件 | 生成命令 | 适用场景 |
| --- | --- | --- | --- |
| 旧版 legacy `explorefault` | `default_config.yml` | `ecat-generate-nonlinear` | 复现旧案例或继续使用旧参数组织 |
| 新版 `NonlinearGeometrySMCInversion` | `nonlinear_geometry.yml` | `ecat-generate-nonlinear-geometry` | 新项目推荐入口，参数注册、数据改正和诊断更清晰 |

新建案例目录时，推荐先生成新版配置：

```bash
ecat-generate-nonlinear-geometry -o nonlinear_geometry.yml
```

等价模块形式：

```bash
python -m eqtools.cli_tools.generate_nonlinear_geometry_config -o nonlinear_geometry.yml
```

不指定 `-o` 时，新版命令默认在当前工作目录写出 `nonlinear_geometry.yml`。模板生成后，再按案例修改 `bounds`、`fixed_params`、`geodata.polys`、`geodata.sigmas`、`fault_aliasnames`、`nchains` 和 `chain_length`。

新版参考脚本在：

```text
scripts/test_nonlinear_geometry_smc.py
```

用户通常需要修改脚本中的：

- `lon0/lat0`：CSI 局部投影原点，不是反演参数。
- 数据文件路径和读取方式：例如 `read_from_varres(...)`、`read_from_enu(...)`。
- `geodata = [...]` 的数据集顺序。
- `config_file` 和 HDF5 样本文件名，若案例中使用不同命名。

脚本中的 `geodata` 顺序必须和 `nonlinear_geometry.yml` 中的 `geodata.verticals`、`geodata.faults`、`geodata.polys`、`geodata.sigmas` 顺序一致。

## 典型脚本流程

```python
from mpi4py import MPI
from csi.insar import insar
from eqtools.csiExtend import NonlinearGeometrySMCInversion

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

lon0 = 78.56
lat0 = 41.19

asc = insar("S1T056A_ifg", lon0=lon0, lat0=lat0, verbose=False)
asc.read_from_varres("../InSAR/downsample/S1T056A_ifg", cov=True)

dsc = insar("S1T034D_ifg", lon0=lon0, lat0=lat0, verbose=False)
dsc.read_from_varres("../InSAR/downsample/S1T034D_ifg", cov=True)

geodata = [asc, dsc]

inv = NonlinearGeometrySMCInversion(
    "geometry_search",
    lon0=lon0,
    lat0=lat0,
    config_file="nonlinear_geometry.yml",
    geodata=geodata,
    verbose=rank == 0,
)

inv.setPriors(bounds=None, initialSample=None, datas=None)
inv.setLikelihood(datas=None, verticals=None)

inv.walk(
    nchains=inv.nchains,
    chain_length=inv.chain_length,
    comm=comm,
    filename="samples_mag_rake_multifaults.h5",
)

inv.extract_and_plot_bayesian_results(
    rank=rank,
    filename="samples_mag_rake_multifaults.h5",
    plot_faults=True,
    plot_sigmas=True,
    plot_data=True,
    plot_data_corrections=True,
)

if rank == 0:
    inv.load_samples_from_h5("samples_mag_rake_multifaults.h5")
    inv.plot_fault_parameter_trends(
        save_path="fault_parameter_trends.png",
        show=False,
    )
```

<a id="geometry-results-to-linear-inversion"></a>

## 结果进入线性反演

非线性几何反演的结果通常保留为：

```text
samples_mag_rake_multifaults.h5
samples_final.h5
model_results_median.txt
model_results_median.json
```

两步走路线中，线性 BLSE/VCE 阶段不直接把非线性样本文件当作滑动模型读入。实际做法是：从 `model_results_median.txt`、`model_results_median.json` 或同等摘要中选定一组几何参数，再在线性脚本中生成固定断层网格。摘要中的 `lon/lat/depth` 仍然表示**断层顶边中点**，对应线性脚本中的 `clon/clat/cdepth`。

如果需要在矩形元、三角元、trace、等深线或外部 mesh 等不同输入之间选择构建方式，先读 [Fault Geometry Construction](../reference/fault_geometry_construction.md)。

典型桥接代码如下：

```python
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)

# 来自 model_results_median.txt 或人工筛选后的几何摘要。
geom = {
    "clon": 78.679379,    # 非线性结果 lon：顶边中点经度
    "clat": 41.206628,    # 非线性结果 lat：顶边中点纬度
    "cdepth": 7.063830,   # 非线性结果 depth：顶边中点深度
    "strike": 228.826543,
    "dip": 73.354888,
    "length": 32.662322,
}

# 线性滑动反演阶段的断层面扩展范围。
# top/depth 不是非线性反演中的 cdepth；它们是扩展后滑动面的顶部和底部深度。
fault_top_depth = 0.0
fault_bottom_depth = 15.0

fault = TriFault("fixed_geometry", lon0=lon0, lat0=lat0, verbose=False)
fault.generate_top_bottom_from_nonlinear_soln(
    clon=geom["clon"],
    clat=geom["clat"],
    cdepth=geom["cdepth"],
    strike=geom["strike"],
    dip=geom["dip"],
    length=geom["length"],
    top=fault_top_depth,
    depth=fault_bottom_depth,
)
fault.generate_mesh(top_size=1.0, bottom_size=1.5, show=False, verbose=0)
fault.initializeslip(values="depth")
```

这里最容易混淆的是 `cdepth`、`top` 和 `depth`：

- `clon/clat/cdepth` 是非线性几何反演得到的顶边中点三维坐标。
- `top` 是线性滑动面扩展后的顶部深度，常常比 `cdepth` 更浅。
- `depth` 是线性滑动面扩展后的底部深度，通常比 `cdepth` 更深。

因此，`top/depth` 是线性滑动反演阶段的网格设计参数，不应被解释成非线性反演的 `depth`。从非线性紧凑源进入分布式滑动反演时，需要明确断层面如何向上、向下扩展；如果改用非线性 `width` 控制底部位置，也要和固定 `top/depth` 的做法区分开。

非线性结果里的 `slip` 和 `rake` 是紧凑源搜索中的平均滑动或机制参数，不是 BLSE/VCE 的分布式滑动结果。进入线性阶段时，`rake` 可用于设置 rake 约束或检查机制一致性；分布式滑动仍由 BLSE/VCE 重新求解。

## MPI 运行

```bash
mpiexec -n 4 python test_nonlinear_geometry_smc.py -r
python test_nonlinear_geometry_smc.py
```

第一条命令采样，第二条命令读取已有 HDF5 样本并重新生成摘要、诊断和图件。`-r` 和非 `-r` 后处理都会触发新版收敛诊断；若 HDF5 是旧式结果且不含过程统计，诊断会自动降级。

## 配置概念

新版 `nonlinear_geometry.yml` 通常包含以下内容。下面的 `lon`、`lat`、`depth` 分别表示断层顶边中点的经度、纬度和深度：

```yaml
prior_bounds_format: lower_upper
nchains: 100
chain_length: 50
nfaults: 1
slip_sampling_mode: mag_rake

bounds:
  defaults:
    lon: [Uniform, 78.56, 80.56]
    lat: [Uniform, 41.19, 43.19]
    depth: [Uniform, 5.0, 25.0]
    dip: [Uniform, 45.0, 89.9]
    width: [Uniform, 0.1, 30.0]
    length: [Uniform, 5.0, 50.0]
    strike: [Uniform, 180.0, 360.0]
    slip: [Uniform, 0.0, 10.0]
    rake: [Uniform, -90.0, 90.0]

geodata:
  verticals: [true, true]
  polys: [3, 1]
  poly_bounds: [Uniform, -1000.0, 1000.0]
  sigmas:
    mode: individual
    update: true
    bounds:
      defaults: [Uniform, -3.0, 3.0]
    values: [0.0, 0.0]
    log_scaled: true
```

新版模板默认使用 `prior_bounds_format: lower_upper`，因此 `Uniform` 写法是：

```text
[Uniform, lower, upper]
```

解析后内部会统一转换成底层采样需要的 lower/range 形式。旧版 `default_config.yml` 仍默认使用 `prior_bounds_format: lower_range`，即 `[Uniform, lower, range]`；不要在两个配置文件之间直接复制边界数值而不检查格式。

`geodata.polys` 与脚本里的 `geodata` 顺序一一对应。SAR/InSAR 常用 `1` 表示 offset，`3` 表示 offset + x/y ramp，`4` 表示二阶 ramp。普通用户通常只需要设置 `polys` 和统一的 `poly_bounds`；只有需要逐数据集或逐参数覆盖边界、显示名时，才使用 `data_corrections` 高级段。

`geodata.sigmas` 控制各数据集的标准差超参数。非线性几何入口使用 `values` 字段作为初值，若 `update: true`，`bounds` 给出 sigma 采样范围；当 `log_scaled: true` 时，采样值为 `log10(sigma)`。`mode` 可设为 `single`、`individual` 或 `grouped`，详见 [Sigmas 与 Alpha 配置模式](../reference/sigmas_alpha.md)。本几何工作流不设置 `alpha`；`alpha` 是后续分布式滑动反演中的平滑尺度。

## 案例应保留

- 输入降采样数据
- 几何反演配置
- 运行与绘图脚本
- HDF5 样本文件，若体量可接受
- `samples_mag_rake_multifaults_convergence.txt` 和 `.yml` 诊断报告
- `model_results_median.txt` 或等价摘要
- data/synthetic/residual 图
- 几何参数和 sigma 参数 KDE 图
- `fault_parameter_trends.png` 断层参数 stage 演化图

## 下一步

选定优选几何后，进入 [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md)。如果需要先理解配置字段，查 [非线性几何反演配置](../reference/config_nonlinear_geometry.md)；如果要解释 sigma 参数，查 [Sigmas 与 Alpha 配置模式](../reference/sigmas_alpha.md)。

如果研究目标是把可扰动断层几何和分布式滑动放在同一个后验中，而不是先选一个优选几何再固定求解滑动，转到高级路线 [Bayesian 联合几何-滑动分布反演](05_joint_bayesian_geometry_slip.md)。
