# Ridgecrest：GPS+InSAR 非线性几何反演

这是建议的第一个多数据非线性几何反演案例。

## GitHub 位置

[ECAT-Cases / Cases / Ridgecrest_20190706Mw7_1](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Ridgecrest_20190706Mw7_1)

关键目录：

```text
GPS/
InSAR/
Nonlinear/
```

## 文件来源与生成方式

`Nonlinear/` 中的脚本和 `default_config.yml` 来自 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 的 Ridgecrest 案例材料。它们已经按多断层、多数据集顺序整理过，适合用来学习 `geodata` 与配置顺序如何对应。

新建自己的非线性几何反演目录时，可以先生成主配置模板：

```bash
ecat-generate-nonlinear -o default_config.yml
```

随后按案例修改断层别名、每个断层的几何边界、数据集顺序和 sigma 策略。CLI 说明见 [CLI 命令参考](../reference/cli.md)，配置字段见 [非线性几何反演配置](../reference/config_nonlinear_geometry.md)。

## 为什么选这个案例

- 同时包含 GPS 和 InSAR。
- 展示多断层 alias 配置。
- 展示 Mw6.4 前震和 Mw7.1 主震这种多事件破裂场景。
- 展示 `geodata` 顺序与 `geodata.faults` 的对应关系。

## 数据入口

GPS：

```python
from csi.gps import gps

cogps = gps("co7_1", lon0=lon0, lat0=lat0, verbose=False)
cogps.read_from_enu("../GPS/GPS_ENU7_1NoEW_CSI.dat", factor=1.0, minerr=1.0, header=1, checkNaNs=True)
cogps.buildCd(direction="enu")
```

InSAR：

```python
from csi.insar import insar

sar = insar("coAscsar", lon0=lon0, lat0=lat0, verbose=False)
sar.read_from_ascii("../InSAR/coAscending_los_CSI.dat", factor=1.0, header=1)
sar.buildDiagCd()
```

Ridgecrest 的 InSAR 不是 `.txt/.rsp/.cov` varres 前缀，也不是原始 GeoTIFF/GAMMA 产品，而是外部整理好的 ASCII 点位文件。文件头类似：

```text
Lon Lat Los(m) wt E N U
```

`read_from_ascii` 的列语义是 `lon lat data err Elos Nlos Ulos`。这里 `Los(m)` 是 LOS 观测，`E/N/U` 是 LOS 投影向量；第 4 列会被读成 `sar.err` 并用于 `buildDiagCd()`。因此，Ridgecrest 文件中的 `wt` 在本案例脚本里按误差列处理；如果自己的外部 ASCII 文件第 4 列是真正的权重，应先转换为误差。更完整的数据格式说明见 [InSAR 与 GPS 数据读取](../workflows/01_data_reading_insar_gps.md)。

<a id="multi-event-coverage"></a>

## 多事件覆盖关系

Ridgecrest 案例不是单一事件单一断层。它同时包含 Mw6.4 前震和 Mw7.1 主震，数据覆盖关系也不完全相同：

| 数据对象 | 输入文件 | 覆盖事件 | 配置中约束的断层 |
| --- | --- | --- | --- |
| `coAscsar` | `InSAR/coAscending_los_CSI.dat` | 前震 + 主震的累计 LOS 形变 | `null`，表示使用全部断层别名 |
| `coDscsar` | `InSAR/coDescending_los_CSI.dat` | 前震 + 主震的累计 LOS 形变 | `null`，表示使用全部断层别名 |
| `cogps7_1` | `GPS/GPS_ENU7_1NoEW_CSI.dat` | Mw7.1 主震 | `[RCM]` |
| `cogps6_4` | `GPS/GPS_ENU6_4NoEW_CSI.dat` | Mw6.4 前震 | `[RCP]` |

脚本中的数据顺序是：

```python
geodata = [coAscsar, coDscsar, cogps7_1, cogps6_4]
```

`default_config.yml` 中的断层别名和数据-断层对应关系必须按同一顺序写：

```yaml
nfaults: 2
fault_aliasnames: ["RCM", "RCP"]

geodata:
  faults: [null, null, [RCM], [RCP]]
```

这里的 `null` 表示该数据集对所有断层源都敏感，适合覆盖两个事件累计形变的 InSAR。`[RCM]` 和 `[RCP]` 表示该数据集只参与对应断层的预测，适合只覆盖某一个地震事件的 GPS 数据。新案例只要存在“有的数据覆盖多个事件、有的数据只覆盖一个事件”的情况，就应按这个模式显式写清楚。

## 脚本对照

### 1. 读取两个事件对应的 GPS 数据

```python
gpsfile_6_4 = os.path.join("..", "GPS", "GPS_ENU6_4NoEW_CSI.dat")
gpsfile_7_1 = os.path.join("..", "GPS", "GPS_ENU7_1NoEW_CSI.dat")

cogps6_4 = gps(name="co6_4", lon0=lon0, lat0=lat0, verbose=verbose)
cogps6_4.read_from_enu(gpsfile_6_4, factor=1.0, minerr=1.0, header=1, checkNaNs=True)
cogps6_4.buildCd(direction="enu")

cogps7_1 = gps(name="co7_1", lon0=lon0, lat0=lat0, verbose=verbose)
cogps7_1.read_from_enu(gpsfile_7_1, factor=1.0, minerr=1.0, header=1, checkNaNs=True)
cogps7_1.buildCd(direction="enu")
```

教学时要说明两个 GPS 文件并不代表同一观测场：`co7_1` 对应 Mw7.1 主震，`co6_4` 对应 Mw6.4 前震，因此它们不能在配置里都写成 `null`。

### 2. 读取覆盖累计形变的 InSAR ASCII 数据

```python
coDscsar = insar(name="coDscsar", lon0=lon0, lat0=lat0, verbose=verbose)
coDscsar.read_from_ascii("../InSAR/coDescending_los_CSI.dat", factor=1.0, header=1)
coDscsar.buildDiagCd()

coAscsar = insar(name="coAscsar", lon0=lon0, lat0=lat0, verbose=False)
coAscsar.read_from_ascii("../InSAR/coAscending_los_CSI.dat", factor=1.0, header=1)
coAscsar.buildDiagCd()
```

外部 ASCII InSAR 点位读入时，第 4 列按 `sar.err` 进入误差字段。InSAR/GPS 数据入口的完整约定见 [InSAR 与 GPS 数据读取](../workflows/01_data_reading_insar_gps.md#insar-data-entry)。

### 3. 保持 `geodata` 和配置顺序一致

```python
geodata = [coAscsar, coDscsar, cogps7_1, cogps6_4]

expfault = explorefault(
    "invrc",
    lat0=lat0,
    lon0=lon0,
    config_file="default_config.yml",
    geodata=geodata,
    verbose=True,
)
```

`geodata.faults` 的每一项都按这个列表索引匹配。顺序写错时，程序不一定能替你识别科学含义错误，但反演会把某个数据集错误地分配给不对应的断层源。

### 4. 用配置区分 RCM 和 RCP 的几何先验

```yaml
bounds:
  defaults:
    lon: [Uniform, -118.0, 2.0]
    lat: [Uniform, 34.0, 2.0]
    strike: [Uniform, 90.0, 270.0]
    rake: [Uniform, 150, 60.0]
  RCP:
    rake: [Uniform, -30, 60.0]
    strike: [Uniform, 0.0, 270.0]

fixed_params:
  RCP:
    lon: -117.541
    lat: 35.6431
    depth: 0.0
    strike: 227.0
```

`defaults` 给 RCM 和 RCP 的共同先验，`RCP` 块覆盖前震断层的专属先验或固定参数。多事件案例应把断层别名、数据覆盖事件和先验差异放在同一页说明。

### 5. 运行采样并检查每个数据集拟合

```python
expfault.setPriors(bounds=None, initialSample=None, datas=None)
expfault.setLikelihood(datas=None, verticals=None)
expfault.walk(
    nchains=nchains,
    chain_length=chain_length,
    comm=comm,
    filename="samples_mag_rake_multifaults.h5",
)
expfault.extract_and_plot_bayesian_results(
    rank=rank,
    filename="samples_mag_rake_multifaults.h5",
    plot_faults=True,
    plot_sigmas=False,
    plot_data=True,
)
```

GPS 拟合图是几何验证的一部分，不只是附属图件。这个案例尤其要检查 `co7_1` 和 `co6_4` 是否分别被对应事件的断层解释。

## 教学范围

本页重点解释数据读取、配置顺序、多事件数据覆盖关系和非线性搜索运行。多断层参数、数据集顺序和 sigma 策略的配置细节可继续参考 [非线性几何反演配置](../reference/config_nonlinear_geometry.md)。

## 跑通判据

一次完整运行后，至少应能看到：

- `samples_mag_rake_multifaults.h5` 或 `samples_final.h5`
- `model_results_median.txt`
- `kde_matrix_RCM.png`
- `kde_matrix_RCP.png`
- `Modeling/coAscsar_data.txt`
- `Modeling/coAscsar_synth.txt`
- `Modeling/coAscsar_resid.txt`
- `Modeling/coDscsar_data.txt`
- `Modeling/coDscsar_synth.txt`
- `Modeling/coDscsar_resid.txt`
- `Modeling/co7_1_data.txt`
- `Modeling/co7_1_synth.txt`
- `Modeling/co7_1_res.txt`
- `Modeling/co6_4_data.txt`
- `Modeling/co6_4_synth.txt`
- `Modeling/co6_4_res.txt`

检查时不要只看总误差。Ridgecrest 的关键是确认 `co7_1` 主要由 `RCM` 解释、`co6_4` 主要由 `RCP` 解释，而累计 InSAR 数据使用两个断层源共同解释。
