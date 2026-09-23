# 反演前读取 InSAR 与 GNSS 数据

这个短例把最常见的反演数据入口放在一页：ECAT 降采样结果、外部 ASCII SAR
点数据和 CSI ENU 格式 GNSS。非线性几何反演与 BLSE/VCE 可以复用同一组数据对象。

本页是按输入选择的顺序示例。仅使用三角 varres、外部 SAR 或 GNSS 分支时，也要先设置
共同的 `lon0/lat0`，并执行该分支需要的 import；不要为了获得一个变量而读取无关数据。
第 4 节的 `geodata` 仅展示装配方式，应换成本次实际建立的对象。

## 输入

按手头数据选择一种入口；不要把同一观测同时用两种方式读入并重复加入 `geodata`。

| 手头数据 | CSI 入口 | 需要同时准备 |
| --- | --- | --- |
| ECAT `std/data`、`trirb` 或 `from_rsp` | `insar.read_from_varres(...)` 自动识别单元 | 同前缀 `.txt/.rsp`，使用协方差时再准备 `.cov` |
| 外部整理的 SAR 点数据 | `insar.read_from_ascii(...)` | 七列 `lon lat data err Elos Nlos Ulos` |
| GNSS ENU 点数据 | `gps.read_from_enu(...)` | 九列 `station lon lat E N U sE sN sU` |

所有数据对象必须使用同一个 `lon0/lat0`。`factor` 同时缩放观测值和误差，应按输入
文件单位设置。

<a id="varres-loading"></a>

## 1. 读取 ECAT 降采样结果

`read_from_varres()` 接收共同前缀，不带 `.txt/.rsp/.cov` 后缀，并按 `.rsp` 列数自动识别
三角形、旧式矩形或完整四边形：

```python
from csi.insar import insar

lon0, lat0 = 100.0, 30.0

sar_qtree = insar("track_qtree", lon0=lon0, lat0=lat0, verbose=False)
sar_qtree.read_from_varres(
    "InSAR/downsample/track_qtree_ifg",
    cov=True,
)
```

已知输入必须是三角单元时，可以把旧参数作为校验提示保留：

```python
sar_trirb = insar("track_trirb", lon0=lon0, lat0=lat0, verbose=False)
sar_trirb.read_from_varres(
    "InSAR/downsample/track_trirb_ifg",
    triangular=True,
    cov=True,
)
```

默认 `triangular=None` 才是自动模式。显式 `True` 或 `False` 不再控制解析分支，只校验
自动识别结果是否符合调用方预期；冲突时立即报错。读入后可打印
`sar_trirb.corner_mode` 核对实际类型。

<a id="detect-varres-geometry"></a>

只想检查文件而不构建 CSI 对象时，可使用只读接口：

```python
from eqtools.csiExtend.downsample import read_csi_varres_result

prefix = "InSAR/downsample/track_ifg"
checked = read_csi_varres_result(prefix, data_type="sar", geometry="auto")

print(checked.geometry, checked.cell_count)
```

`read_csi_varres_result()` 只检查 `.txt/.rsp` 并保留单元顶点，不读取 `.cov`；真正用于
反演时仍直接调用 `read_from_varres(..., cov=True)`。

如果没有完整协方差文件，可以改为：

```python
sar_qtree.read_from_varres(
    "InSAR/downsample/track_qtree_ifg",
    cov=False,
)
sar_qtree.buildDiagCd()
```

`cov=True` 已经读取完整矩阵时，不要再调用 `buildDiagCd()` 覆盖它。

## 2. 读取外部 ASCII SAR 点数据

外部点数据应先整理为：

```text
lon lat data err Elos Nlos Ulos
100.10 30.10 0.012 0.004 -0.42 -0.10 0.90
```

其中 `data` 是一个 SAR 标量观测，`err` 是标准差或不确定度，后三列是 ENU
projection，不是 ENU 位移：

```text
scalar_observation = ENU_displacement dot projection
```

```python
sar_ascii = insar("track_ascii", lon0=lon0, lat0=lat0, verbose=False)
sar_ascii.read_from_ascii(
    "InSAR/track_ascii.txt",
    factor=1.0,
    header=1,
)
sar_ascii.buildDiagCd()
```

第 4 列必须是 `err`。如果原文件保存的是权重，应先按数据定义转换为不确定度，不能
只改列名后直接读取。LOS/range 进入 CSI 时推荐统一为朝向卫星为正，并保证标量与
projection 成对使用。

## 3. 读取 GNSS ENU 点数据

CSI ENU 格式是：

```text
station lon lat east north up sigma_e sigma_n sigma_u
STA001 100.10 30.10 0.012 -0.004 0.001 0.002 0.002 0.005
```

```python
from csi.gps import gps

gnss = gps("gnss", lon0=lon0, lat0=lat0, verbose=False)
gnss.read_from_enu(
    "GNSS/gnss_enu.txt",
    factor=1.0,
    minerr=0.001,
    header=1,
    checkNaNs=True,
)
gnss.buildCd(direction="enu")
```

`minerr` 只替换文件中等于零的误差，单位是乘以 `factor` 之前的输入单位。例如输入
位移和误差均为 mm、目标单位为 m 时使用 `factor=1e-3`，同时把 `minerr` 写成 mm。

<a id="assemble-geodata"></a>

## 4. 组成 `geodata`

只把本次反演真正使用的数据按固定顺序放入列表：

```python
geodata = [sar_qtree, sar_ascii, gnss]
```

如果只读取了上面的三角结果，应使用 `geodata = [sar_trirb]`；不要照抄包含未创建对象的列表，
也不要把同一轨道的矩形与三角降采样版本同时加入反演。

这个顺序必须与非线性配置或线性配置中的数据顺序、`polys`、`sigmas`、
`verticals` 和数据—断层覆盖关系一致。不要依赖对象名自动重排。

## 检查

```python
for data in geodata:
    print(data.name, data.lon.size)
```

进入反演前至少确认：

- 所有对象使用同一 `lon0/lat0`；
- 单位和 `factor` 已记录；
- SAR `vel` 与 `los/projection` 行数一致；
- `.cov` 维度或对角误差与观测数一致；
- GNSS 的 ENU 分量和误差列顺序正确；
- `geodata` 与配置顺序一致。

## 数据准备完成后交给哪里

完成上述检查后，保留数据对象、输入前缀、误差处理方式和共同投影原点。
需要估计紧凑几何时，进入[非线性几何工作流](../workflows/03_nonlinear_geometry_bayesian.md#入口)；
已有固定几何时，将本页的 `geodata` 接到 [BLSE 输入装配](../workflows/04_linear_slip_blse_vce.md#input-handoff)。
切换到另一个示例时，继续使用本项目的坐标和数据名称，不复制其中的占位坐标。

## 何时不用这个例子

- 手里还是 GAMMA、GMTSAR、GeoTIFF、HyP3 或 offset 栅格：先走
  [InSAR 降采样](../workflows/02_insar_downsampling.md)。
- 需要自定义原始产品 reader：看
  [自定义读入 Adapter 降采样](../workflows/02b_adapter_downsampling.md)。
- 只想检查或导出 `.txt/.rsp`，不需要 CSI 反演对象：直接使用
  `read_csi_varres_result()`。

## 相关参考

- [InSAR 与 GNSS 数据读取 workflow](../workflows/01_data_reading_insar_gps.md)
- [观测数据读入参考](../reference/observation_data_readers.md)
- [SAR projection 与正负约定](../concepts/sar_projection_conventions.md)
- [BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md)
