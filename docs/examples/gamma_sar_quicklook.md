# GAMMA SAR quick-look 与配置生成

## 无 YAML 快速预览

默认右视 LOS displacement：

```bash
ecat-downsample -s --sar-prefix geo_20250101_20250113
```

默认右视解缠相位：

```bash
ecat-downsample -s --sar-prefix geo_20250101_20250113 --sar-mode unwrapped_phase
```

左视、由 GAMMA 导出的 NISAR 解缠相位：

```bash
ecat-downsample -s --sar-prefix nisar_pair --sar-mode unwrapped_phase --sar-look-side left
```

已知为大端 float32：

```bash
ecat-downsample -s --sar-prefix gamma_pair --sar-mode unwrapped_phase --sar-byte-order big
```

还可用 `--sar-dir InSAR/raw` 指定目录。快捷入口固定为 GAMMA reader，仅用于
`-s`；协方差估计和正式降采样应保存 YAML。

## 生成短配置

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase -o downsample.yml
```

左视：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --sar-look-side left -o downsample_left.yml
```

模板中通常只需改：

```yaml
sar_config:
  directory: InSAR/raw
  outName: S1_T012A
  mode: unwrapped_phase
  acquisition_look_side: right
  files:
    prefix: geo_20250101_20250113
  read:
    byte_order: native
```

## 三步执行

```bash
ecat-downsample -f downsample.yml -s
ecat-downsample -f downsample.yml -c
ecat-downsample -f downsample.yml -d
```

相关页面：

- [SAR 投影与方向约定](../concepts/sar_projection_conventions.md)
- [SAR Reader 参考](../reference/sar_reader.md)
- [InSAR 降采样](../workflows/02_insar_downsampling.md)
