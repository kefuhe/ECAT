# adapter_downsampling 模板

这个模板用于“读入阶段需要用户自定义，但降采样流程仍使用 ECAT 标准实现”的场景。

```text
用户文件或数组 -> input_adapter.py -> CSI 数据对象
CSI 数据对象 -> ECAT 标准降采样 runtime
```

通常只改 `input_adapter.py`。`run_adapter_downsampling.py` 和
`run_timeseries_downsampling.py` 只是薄入口，不应复制或改写 `std/data/trirb/from_rsp`
内部逻辑。

## 生成配置并复制模板

```powershell
ecat-generate-downsample -m sar --sar-reader gamma --sar-mode unwrapped_phase `
  -o downsample.yml --copy-adapter-template
```

如果完全绕过标准 reader，必须在 YAML 中显式设置：

```yaml
input_adapter:
  enabled: true
general:
  origin: manual
  lon0: 101.0
  lat0: 37.5
```

`origin: manual` 是必须的，因为模板会在调用 `input_adapter.py` 前解析投影原点，非标准文件没有统一的信息源可供自动推断。

## 单景数据

```powershell
python run_adapter_downsampling.py -f downsample.yml -s
python run_adapter_downsampling.py -f downsample.yml -c
python run_adapter_downsampling.py -f downsample.yml -d
```

这些开关与 `ecat-downsample` 一致：`-s` 检查读入和 quick-look，`-c` 估计协方差，
`-d` 正式降采样。

## 时序数据

增加：

```yaml
timeseries:
  mode: independent        # independent | reference_grid
  reference_epoch: 2022-01-17
  epochs: [2022-01-05, 2022-01-17, 2022-01-29]
```

`independent` 表示每个时刻按当前 `downsample.method` 独立执行，可使用
`std/data/trirb/from_rsp` 任一方法。`reference_grid` 只影响正式降采样步骤：使用 `-d` 时，
参考时刻先用当前方法生成 `.rsp`，其他时刻自动切换为 `from_rsp` 复用参考网格；使用 `-s`
或 `-c` 时没有可复用网格，各时刻会独立执行检查或协方差步骤。

使用时需要先实现 `input_adapter.load_epoch_data()`：

```powershell
python run_timeseries_downsampling.py -f downsample.yml -d
```
