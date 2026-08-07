# InSAR 与 GNSS 数据读取

数据读取是可重复反演的第一道关口。进入非线性几何或 BLSE/VCE 之前，每个数据集都必须
明确物理量、单位、正负号、投影原点、误差模型和协方差来源。

如果手里的数据已经能直接进入反演，先复制
[反演前读取 InSAR 与 GNSS 数据](../examples/inversion_data_loading.md)；本页负责解释如何选择入口
以及原始产品何时应先经过降采样。

<a id="insar-data-entry"></a>

## 先按输入选择入口

| 手头数据 | 下一步 | 精确格式 |
| --- | --- | --- |
| ECAT `std/data` 四叉树或矩形 `from_rsp` | `read_from_varres(..., triangular=False)` | [观测数据读入参考](../reference/observation_data_readers.md#csi-varres) |
| ECAT `trirb` 或三角 `from_rsp` | `read_from_varres(..., triangular=True)` | [观测数据读入参考](../reference/observation_data_readers.md#csi-varres) |
| 外部抽样后的 SAR 点 | 整理七列后用 `read_from_ascii(...)` | [外部 ASCII SAR](../reference/observation_data_readers.md#external-ascii-sar) |
| GNSS ENU 点 | 整理九列后用 `read_from_enu(...)` | [GNSS ENU](../reference/observation_data_readers.md#gnss-enu) |
| GAMMA、GMTSAR、HyP3、GeoTIFF、NetCDF/HDF5 或 offset 栅格 | 先用 reader 转换并降采样 | [InSAR 降采样](02_insar_downsampling.md) |

## 1. 已降采样的 CSI varres 数据

共同前缀通常对应：

```text
track_ifg.txt
track_ifg.rsp
track_ifg.cov
```

普通四叉树或矩形结果使用：

```python
from csi.insar import insar

sar = insar("TrackA", lon0=lon0, lat0=lat0, verbose=False)
sar.read_from_varres(
    "InSAR/downsample/track_ifg",
    triangular=False,
    cov=True,
)
```

trirb 或三角 `from_rsp` 结果必须改为 `triangular=True`。CSI 的这个 reader 不接受 `triangular=None` 自动分流；
矩形模式只会在两种矩形 `.rsp` 列布局之间自动判断。若 `.cov` 不存在，使用 `cov=False`，
然后调用 `buildDiagCd()`；已经用 `cov=True` 读取完整协方差后不要再覆盖它。

## 2. 外部 SAR 点和 GNSS ENU

外部 SAR 点应整理为：

```text
lon lat data err Elos Nlos Ulos
```

第 4 列会读入 `sar.err`。一些历史文件可能把这一列标为 `weight` 或 `wt`；若实际保存的是
权重，必须按原始定义转换为标准差，不能只改表头。后三列是 projection，不是 ENU 位移：

```text
scalar_observation = ENU_displacement dot projection
```

GNSS ENU 应整理为：

```text
station lon lat east north up sigma_e sigma_n sigma_u
```

完整代码、`factor`、`minerr` 和 `geodata` 顺序见
[反演前读取 InSAR 与 GNSS 数据](../examples/inversion_data_loading.md)。

## 3. 原始或栅格化 SAR/offset 产品

原始产品先由 `eqtools.csiExtend.sarUtils` reader 转换到 CSI 观测对象，再估计协方差和降采样。
下面只是 GAMMA range offset 的入口骨架：

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
    phsname="range_offset.phs",
    rscname="range_offset.phs.rsc",
    azifile="azimuth_angle.azi",
    incfile="incidence_angle.inc",
)
sar.read_observation(downsample=1)
```

GAMMA prefix、GAMMA TIFF、HyP3、GMTSAR direct projection、optical GeoTIFF、标准
`.nc/.h5` 和 CSI varres 的完整脚本统一放在
[观测数据读入参考](../reference/observation_data_readers.md)。首次从原始产品开始时，直接跟随
[InSAR 降采样工作流](02_insar_downsampling.md)。

## 4. SAR reader 模式和统一物理语义

优先通过 `reader + mode` 表达输入物理量，不要在脚本中凭文件名手动翻转符号。

| 产品 | reader | mode |
| --- | --- | --- |
| GAMMA 二进制解缠相位 | `gamma` | `unwrapped_phase` |
| GAMMA 二进制 LOS 位移 | `gamma` | `los_displacement` |
| GAMMA range offset | `gamma` | `range_offset` |
| GAMMA azimuth offset | `gamma` | `azimuth_offset` |
| GAMMA GeoTIFF 相位/位移 | `gamma_tiff` | 对应观测模式 |
| GMTSAR-style NetCDF/GRD + ENU projection | `gmtsar` | 相位、LOS、range 或 azimuth 对应模式 |
| HyP3 GeoTIFF | `hyp3` | `unwrapped_phase` 或 `los_displacement` |

所有标量 SAR/offset 观测最终统一为：

```text
scalar_observation = ENU_displacement dot projection
```

相位到位移、LOS/range 正负号、azimuth 投影和 acquisition side 的精确定义见
[SAR Reader 语义](../reference/sar_reader.md)与
[SAR projection 约定](../concepts/sar_projection_conventions.md)。

## 进入反演前的检查

- 同一反演的所有对象使用相同 `lon0/lat0`。
- 每个数据集的输入单位、`factor` 和输出单位已记录。
- SAR 标量与 projection 行数一致，正负号与采集几何一致。
- 第 4 列究竟是标准差还是权重已经确认。
- `.cov` 维度等于观测数；没有完整协方差时明确使用对角误差。
- GNSS 的 ENU 分量和误差列顺序正确，零误差的 `minerr` 规则明确。
- Python 中 `geodata` 的顺序与配置中的 `polys`、`sigmas`、`verticals` 和数据—断层覆盖关系一致。

## 下一步

- 原始 SAR/offset：进入 [InSAR 降采样](02_insar_downsampling.md)。
- 已有反演数据且需要估计紧凑几何：进入
  [Bayesian 非线性几何反演](03_nonlinear_geometry_bayesian.md)。
- 已有固定 fault：进入 [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md)。
