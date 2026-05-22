# Bayesian 非线性几何反演

本工作流说明 Bayesian 非线性几何反演：用紧凑源估计断层几何，并把优选几何传递给后续 BLSE/VCE 分布式滑动反演。

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

非线性几何反演的配置文件可以先用 CLI 在当前目录生成模板：

```bash
ecat-generate-nonlinear -o default_config.yml
```

等价模块形式：

```bash
python -m eqtools.cli_tools.generate_nonlinear_config -o default_config.yml
```

不指定 `-o` 时，默认也会在当前工作目录写出 `default_config.yml`。生成模板后，再参照案例逐项设置 `bounds`、`fixed_params`、`geodata`、`fault_aliasnames`、`nchains` 和 `chain_length` 等参数。

案例脚本主要使用：

```python
from eqtools.csiExtend.exploremultifaults_smc import explorefault
```

该入口适合做低维非线性几何搜索，并使用案例目录中的 `default_config.yml`。

## 典型脚本流程

```python
from mpi4py import MPI
from csi.insar import insar
from eqtools.csiExtend.exploremultifaults_smc import explorefault

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

lon0 = 78.56
lat0 = 41.19

asc = insar("S1T056A_ifg", lon0=lon0, lat0=lat0, verbose=False)
asc.read_from_varres("../InSAR/downsample/S1T056A_ifg", cov=True)

dsc = insar("S1T034D_ifg", lon0=lon0, lat0=lat0, verbose=False)
dsc.read_from_varres("../InSAR/downsample/S1T034D_ifg", cov=True)

geodata = [asc, dsc]

expfault = explorefault(
    "geometry_search",
    lon0=lon0,
    lat0=lat0,
    config_file="default_config.yml",
    geodata=geodata,
    verbose=False,
)

expfault.setPriors(bounds=None, initialSample=None, datas=None)
expfault.setLikelihood(datas=None, verticals=None)
expfault.walk(
    nchains=expfault.nchains,
    chain_length=expfault.chain_length,
    comm=comm,
    filename="samples_mag_rake_multifaults.h5",
)

expfault.extract_and_plot_bayesian_results(
    rank=rank,
    filename="samples_mag_rake_multifaults.h5",
    plot_faults=True,
    plot_sigmas=True,
    plot_data=True,
)
```

## 结果进入线性反演

非线性几何反演的结果通常保留为：

```text
samples_mag_rake_multifaults.h5
samples_final.h5
model_results_median.txt
model_results_median.json
```

两步走路线中，线性 BLSE/VCE 阶段不直接把非线性样本文件当作滑动模型读入。实际做法是：从 `model_results_median.txt`、`model_results_median.json` 或同等摘要中选定一组几何参数，再在线性脚本中生成固定断层网格。摘要中的 `lon/lat/depth` 仍然表示**断层顶边中点**，对应线性脚本中的 `clon/clat/cdepth`。

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

因此，`top/depth` 是线性滑动反演阶段的网格设计参数，不应被解释成非线性反演的 `depth`。文档要说明断层面如何从非线性紧凑源向上、向下扩展；如果改用非线性 `width` 来控制底部位置，也必须写清楚这与固定 `top/depth` 的区别。

非线性结果里的 `slip` 和 `rake` 是紧凑源搜索中的平均滑动或机制参数，不是 BLSE/VCE 的分布式滑动结果。进入线性阶段时，`rake` 可用于设置 rake 约束或检查机制一致性；分布式滑动仍由 BLSE/VCE 重新求解。

## MPI 运行

```bash
mpiexec -n 4 python test_nonlinear_mag_rake.py -r
python test_nonlinear_mag_rake.py
```

第一条命令采样，第二条命令应能读取保存的 HDF5 样本并绘图。

## 配置概念

非线性几何配置通常包含以下内容。下面的 `lon`、`lat`、`depth` 分别表示断层顶边中点的经度、纬度和深度：

```yaml
nchains: 100
chain_length: 50
nfaults: 1
slip_sampling_mode: mag_rake

bounds:
  defaults:
    lon: [Uniform, 78.56, 2.0]
    lat: [Uniform, 41.19, 2.0]
    depth: [Uniform, 5.0, 20.0]
    dip: [Uniform, 45.0, 44.9]
    width: [Uniform, 0.1, 29.9]
    length: [Uniform, 5.0, 45.0]
    strike: [Uniform, 180.0, 180.0]
    magnitude: [Uniform, 0.0, 10.0]
    rake: [Uniform, -90.0, 180.0]

geodata:
  verticals: [true, true]
  polys:
    enabled: true
    boundaries:
      defaults: [Uniform, -200.0, 400.0]
  sigmas:
    mode: individual
    update: true
    bounds:
      defaults: [Uniform, -3.0, 6.0]
      sigma_0: [Uniform, -3.0, 6.0]
    values: [0.0, 0.0]
    log_scaled: true
```

当前配置里的 `Uniform` 不是 `[下界, 上界]`。它沿用 `scipy.stats.uniform` 风格的输入：

```text
[Uniform, start, range]
```

第二个数是起点或 loc，第三个数是范围或 scale；实际采样上界是 `start + range`。例如 `dip: [Uniform, 45.0, 44.9]` 表示倾角从 `45.0` 到 `89.9` 度。这里的 `start` 不等同于 `setPriors(..., initialSample=...)` 里的初始样本。以后配置格式可能改为直接写下界和上界，但当前手册先按这个现有格式说明。

`geodata.sigmas` 控制各数据集的标准差超参数。非线性几何入口使用 `values` 字段作为初值，若 `update: true`，`bounds` 给出 sigma 采样范围；当 `log_scaled: true` 时，采样值为 `log10(sigma)`。`mode` 可设为 `single`、`individual` 或 `grouped`，详见 [Sigmas 与 Alpha 配置模式](../reference/sigmas_alpha.md)。本几何工作流不设置 `alpha`；`alpha` 是后续分布式滑动反演中的平滑尺度。

## 案例应保留

- 输入降采样数据
- 几何反演配置
- 运行与绘图脚本
- HDF5 样本文件，若体量可接受
- `model_results_median.txt` 或等价摘要
- data/synthetic/residual 图
- 几何参数和 sigma 参数 KDE 图

## 下一步

选定优选几何后，进入 [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md)。如果需要先理解配置字段，查 [非线性几何反演配置](../reference/config_nonlinear_geometry.md)；如果要解释 sigma 参数，查 [Sigmas 与 Alpha 配置模式](../reference/sigmas_alpha.md)。
