# 自定义读入 Adapter 降采样

本页用于读入阶段不适合标准 reader 的情况。标准 GAMMA、GMTSAR、HyP3、GeoTIFF 或 optical
产品优先走 [InSAR 降采样](02_insar_downsampling.md)，不要使用 adapter。

adapter 只解决一个问题：**用户自己把数据读成 CSI 对象，后续降采样仍使用 ECAT 标准 runtime**。

```text
用户文件/数组/时序产品 -> input_adapter.py -> csi.insar 或 csi.opticorr
csi.insar 或 csi.opticorr -> ECAT 标准 covar/std/data/trirb/from_rsp/plot/report
```

## 适用场景

| 场景 | 是否使用 adapter |
| --- | --- |
| 标准 GAMMA/GMTSAR/HyP3/GeoTIFF | 不需要，直接用 `ecat-downsample` |
| 外部文本、非标准栅格、用户自定义预处理数组 | 使用 adapter |
| 已经在脚本中构造好了 `csi.insar` 或 `csi.opticorr` | 使用 adapter |
| 时序 InSAR：用某一时刻生成 `.rsp`，其他时刻复用同一网格 | 使用 adapter 的 `run_timeseries_downsampling.py` |
| 想改 `std/data/trirb/from_rsp` 内部算法 | 不建议通过 adapter 改；应先评估是否需要扩展 ECAT runtime |

## 生成模板

在案例处理目录生成配置和 adapter 模板：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --downsample-method std -o downsample.yml --copy-adapter-template
```

这会在当前目录生成：

```text
downsample.yml
input_adapter.py
run_adapter_downsampling.py
run_timeseries_downsampling.py
README.md
README_cn.md
```

通常只修改 `input_adapter.py` 和 `downsample.yml`。不要复制或改写 ECAT 的标准降采样大脚本。

## 配置要点

模板会写入：

```yaml
input_adapter:
  enabled: true
```

如果完全绕过标准 reader，必须手动指定投影原点，因为程序会在调用 `input_adapter.py` 前解析投影原点：

```yaml
general:
  origin: manual
  lon0: <project_lon0>
  lat0: <project_lat0>
```

如果 `input_adapter.py` 只是继续调用标准 reader，可以保留 `origin: auto`。一旦读入文件不再是标准
reader 能理解的产品，就应改为 `manual`。

其他配置仍按标准降采样设置：

```yaml
covar:
  mask_out: [lon_min, lon_max, lat_min, lat_max]

downsample:
  method: std        # std | data | trirb | from_rsp
```

字段含义见 [降采样超级入口参考](../reference/downsampling_app.md)。

## 单景数据

如果当前数据仍能用标准 reader，默认 `input_adapter.py` 不需要改，它会委托标准 ECAT reader。
这适合先确认 adapter 入口和标准 `ecat-downsample` 行为一致：

```bash
python run_adapter_downsampling.py -f downsample.yml -s
python run_adapter_downsampling.py -f downsample.yml -c
python run_adapter_downsampling.py -f downsample.yml -d
```

如果是自定义读入，只替换 `input_adapter.py` 中的：

```python
def load_data(config, context):
    # 1. 读取用户自己的文件或数组
    # 2. 构造并填充 csi.insar 或 csi.opticorr 对象
    # 3. 返回数据对象和输出前缀
    return data, out_name
```

SAR adapter 返回的对象必须包含：

```text
lon, lat, x, y, vel, los
```

Optical adapter 返回的对象必须包含：

```text
lon, lat, x, y, east, north, err_east, err_north
```

进入这些字段之后，`processing_region`、`data_filters` 之后的处理、协方差、`std/data/trirb/from_rsp`、
`guide_grid`、`extraction`、检查图和报告都由 ECAT 标准 runtime 维护。

## 时序 InSAR

时序数据先在 YAML 中增加：

```yaml
timeseries:
  mode: independent        # independent | reference_grid
  reference_epoch: 2022-01-17
  epochs: [2022-01-05, 2022-01-17, 2022-01-29]
```

然后实现：

```python
def load_epoch_data(config, context, epoch):
    # 读取一个时刻的数据，返回该时刻的 CSI 对象和输出前缀
    return data, out_name
```

运行：

```bash
python run_timeseries_downsampling.py -f downsample.yml -s
python run_timeseries_downsampling.py -f downsample.yml -c
python run_timeseries_downsampling.py -f downsample.yml -d
```

`independent` 表示每个时刻按当前 `downsample.method` 独立执行。

`reference_grid` 只影响 `-d`：参考时刻先按当前 `downsample.method` 生成 `.rsp`，其他时刻自动切换为
`from_rsp` 复用参考网格。使用 `-s` 或 `-c` 时没有可复用网格，各时刻会独立检查或估计协方差。

## 输出检查

单景和时序每个时刻都应保留同一组输出：

```text
<outName>_ifg.txt
<outName>_ifg.rsp
<outName>_ifg.cov
<outName>_run_metadata.yml
<outName>_downsample_report.yml
```

检查顺序仍与标准降采样一致：

1. `-s`：读入、单位、正负号、投影和 quick-look。
2. `-c`：`mask_out` 是否避开主形变区，协方差估计是否稳定。
3. `-d`：采样网格、输出点数、`.cov` 维度和反演输入前缀是否正确。

## 与标准流程的关系

- 跑通标准数据：读 [InSAR 降采样](02_insar_downsampling.md)。
- 按 Step1/Step2 教学调参：读 [InSAR 降采样 Step1/Step2 调参](02a_insar_downsampling_two_step.md)。
- 查全部字段：读 [降采样超级入口参考](../reference/downsampling_app.md)。
- 查 reader/mode/符号约定：读 [SAR Reader 参考](../reference/sar_reader.md)。
