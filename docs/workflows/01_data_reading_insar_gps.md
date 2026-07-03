# InSAR 与 GPS 数据读取

数据读取是可重复研究的第一道关口。进入反演前，每个数据集都必须明确物理量、单位、正负号约定、投影原点、误差模型和协方差来源。

## 对应案例与参考

| 你要确认的问题 | 推荐案例 | 相关参考 |
| --- | --- | --- |
| 降采样 InSAR `.txt/.rsp/.cov` 如何读入 | [Wushi：InSAR-only 非线性几何反演](../casebook/wushi_nonlinear_geometry.md), [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md) | [InSAR 降采样](02_insar_downsampling.md#read-downsampled-output) |
| 外部 ASCII InSAR 点位格式如何读入 | [Ridgecrest：GPS+InSAR 非线性几何反演](../casebook/ridgecrest_gps_insar.md) | [SAR Reader 参考](../reference/sar_reader.md#external-ascii-point-data) |
| 原始 GAMMA/GeoTIFF/GMTSAR/HyP3/offset 产品如何进入降采样 | [InSAR/Offset 降采样案例](../casebook/insar_downsampling_gamma_geotiff.md) | [SAR Reader 参考](../reference/sar_reader.md), [CLI 命令参考](../reference/cli.md#downsampling-config) |

<a id="insar-data-entry"></a>

## InSAR 数据入口

本手册主要使用三类入口。

第一类是已经降采样后的 CSI 格式数据：

```python
from csi.insar import insar

sar = insar("S1T056A_ifg", lon0=lon0, lat0=lat0, verbose=False)
sar.read_from_varres("../InSAR/downsample/S1T056A_ifg", cov=True)
```

这类数据通常共享同一个文件前缀：

```text
S1T056A_ifg.txt
S1T056A_ifg.rsp
S1T056A_ifg.cov
```

第二类是外部已经整理好的普通文本点位数据，常见于外部软件已经抽样、手工整理或历史案例共享。它不走 `eqtools.csiExtend.sarUtils` 原始产品 reader，而是直接由 CSI 的 `insar.read_from_ascii(...)` 读入：

```python
from csi.insar import insar

sar = insar("coAscsar", lon0=lon0, lat0=lat0, verbose=False)
sar.read_from_ascii("../InSAR/coAscending_los_CSI.dat", factor=1.0, header=1)
sar.buildDiagCd()
```

文本列顺序应明确为：

```text
lon lat data err Elos Nlos Ulos
```

其中 `data` 是 LOS 观测量，在很多脚本代码属性名里也叫 `vel`；`err` 是观测不确定度；`Elos/Nlos/Ulos` 是 LOS 投影向量在 east、north、up 方向的分量，不是 GPS 的 east/north/up 位移。Ridgecrest 案例的文件头写作：

```text
Lon Lat Los(m) wt E N U
```

但 `read_from_ascii` 会把第 4 列读入 `sar.err`。如果外部文件把这一列命名为 `wt`，文档里必须说明它在当前脚本中按误差使用，或先转换成真正的 `err` 后再读入。

第三类是原始或栅格化 SAR 产品，先用 `eqtools.csiExtend.sarUtils` 读取，再做降采样。

```python
from eqtools.csiExtend.sarUtils.readGamma2csisar import GammasarReader

sar = GammasarReader(
    name="S1_range",
    lon0=lon0,
    lat0=lat0,
    directory_name=".",
    mode="range_offset",
)
sar.extract_raw_grd(
    phsname="roff_20250101_20250113.phs",
    rscname="roff_20250101_20250113.phs.rsc",
    azifile="off_20250101_20250113.azi",
    incfile="off_20250101_20250113.inc",
)
sar.read_observation(downsample=1)
```

## SAR Reader 模式

手册中优先使用 `reader + mode`，不要在脚本里手动翻转符号。

| 产品 | reader | mode |
| --- | --- | --- |
| GAMMA 二进制解缠相位 | `gamma` | `unwrapped_phase` |
| GAMMA 二进制 LOS 位移 | `gamma` | `los_displacement` |
| GAMMA range offset | `gamma` | `range_offset` |
| GAMMA azimuth offset | `gamma` | `azimuth_offset` |
| GAMMA GeoTIFF 相位/位移 | `gamma_tiff` | 对应观测模式 |
| GMTSAR-style direct-projection GRD/NetCDF + ENU projection | `gmtsar` | `phase_los`、`los_displacement`、`range_offset` 或 `azimuth_offset` |
| HyP3 GeoTIFF | `hyp3` | `unwrapped_phase` 或 `los_displacement` |

reader 会把产品转换到 CSI 使用的投影形式：

```text
scalar_observation = ENU_displacement dot projection_vector
```

## GPS 数据

GPS 位移通常使用 CSI ENU 格式：

```python
from csi.gps import gps

cogps = gps("cogps", lon0=lon0, lat0=lat0, verbose=False)
cogps.read_from_enu("GPS_ENU_CSI.dat", factor=1.0, minerr=1.0, header=1, checkNaNs=True)
cogps.buildCd(direction="enu")
```

最小要求：

- 站点经纬度
- east、north，必要时包含 up 位移
- 各分量不确定度
- 与 `factor` 一致的单位

## 检查清单

- 同一个案例内所有数据使用相同 `lon0/lat0`。
- InSAR 正负号约定已记录。
- InSAR 协方差存在，或明确说明使用对角阵/单位阵。
- `read_from_ascii` 文本第 4 列的含义已明确，是误差、权重还是需要转换。
- GPS 误差下限明确。
- Python 脚本里的 `geodata` 顺序与配置中的 `verticals`、`polys`、`sigmas`、`faults` 顺序一致。

## 下一步

- 如果手里是原始 SAR/offset 产品，进入 [InSAR 降采样](02_insar_downsampling.md)。
- 如果已有可反演的 InSAR/GPS 数据，进入 [Bayesian 非线性几何反演](03_nonlinear_geometry_bayesian.md) 或 [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md)。
