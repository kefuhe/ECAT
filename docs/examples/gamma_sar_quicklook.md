# GAMMA SAR quick-look 与配置生成

这个例子用于快速查看 GAMMA LOS/phase/offset 数据是否能被 ECAT 正确读取，并生成后续降采样模板。

## 快速预览

如果文件组遵循 GAMMA prefix 约定，先用 prefix 直接预览：

```bash
ecat-downsample --sar-prefix geo_20250319_20250331 -s
```

常用覆盖项：

```bash
ecat-downsample --sar-prefix geo_20250319_20250331 --sar-dir InSAR/raw --sar-mode los_displacement -s
```

`-s` 只读入数据并画 quick-look，不估计协方差，不执行降采样。它适合先检查：

- 经纬度范围是否正确。
- 形变正负号是否符合预期。
- LOS 投影或几何文件是否被正确解析。
- `vmin/vmax` 是否需要为后续图件固定。

## 生成降采样配置

确认 quick-look 正常后，在工作目录生成 YAML 模板：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode los_displacement -o downsample.yml
```

打开 `downsample.yml` 后，通常先改：

```yaml
sar_config:
  directory: InSAR/raw
  outName: S1_T012A_ifg
  mode: los_displacement
  files:
    prefix: geo_20250319_20250331

check_plots:
  raw:
    vmin: -0.2
    vmax: 0.2
```

## 运行顺序

```bash
ecat-downsample -f downsample.yml -s
ecat-downsample -f downsample.yml -c
ecat-downsample -f downsample.yml -d
```

含义：

- `-s`：只看原始数据 quick-look。
- `-c`：估计协方差，需要配置 `covar.mask_out`。
- `-d`：正式降采样，写出 CSI 输入前缀。

## 输出检查

正式降采样后，检查工作目录中是否有：

- `<outName>_ifg.txt` 或对应 mode 的数据文件。
- `<outName>_ifg.rsp`。
- `<outName>_ifg.cov`。
- `<outName>_run_metadata.yml`。
- quick-look 和 downsample check 图件。

相关参考：
[InSAR 降采样](../workflows/02_insar_downsampling.md),
[InSAR 降采样 Step1/Step2 调参](../workflows/02a_insar_downsampling_two_step.md),
[SAR Reader](../reference/sar_reader.md),
[Downsampling App](../reference/downsampling_app.md),
[CLI Reference](../reference/cli.md)。
