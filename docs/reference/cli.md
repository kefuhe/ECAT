# CLI 命令参考

ECAT CLI 主要用于生成配置模板、执行降采样、查看可用选项和辅助准备断层/GF 文件。CLI 生成的是模板，不是最终科学配置；生成后仍要按数据、断层、几何和约束修改。

## 命令地图

| 任务 | 推荐命令 | 下一步 |
| --- | --- | --- |
| 生成 InSAR/optical 降采样配置 | `ecat-generate-downsample` | 修改 `downsample.yml` 后运行 `ecat-downsample -s/-c/-d` |
| 运行降采样三阶段 | `ecat-downsample` | 输出 `<effective_outName>_ifg.txt/.rsp/.cov` 供反演读入 |
| 生成非线性几何配置 | `ecat-generate-nonlinear` | 修改 `bounds/fixed_params/geodata/sigmas` |
| 生成 BLSE/VCE 主配置 | `ecat-generate-config` | 修改数据、GF、平滑和约束开关 |
| 生成 BLSE/VCE 边界配置 | `ecat-generate-boundary` | 修改滑动边界、rake、sigma/alpha 边界 |
| 查看 GF 方法选项 | `ecat-generate-config --show-gf-options` | 把选项写入 `faults.defaults.method_parameters.update_GFs.options` |
| 查看几何扰动方法 | `ecat-list-fault-perturb-methods` | 只在需要理解几何扰动族时使用 |

如果命令行入口不可用，可以使用 `python -m eqtools.cli_tools.<module>` 的模块形式。

## 降采样配置

最小 SAR 配置：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gamma \
  --sar-mode unwrapped_phase \
  --downsample-method std \
  -o downsample.yml
```

常用参数：

| 参数 | 可选值 | 说明 |
| --- | --- | --- |
| `--mode` | `sar`, `optical`, `full` | 模板数据类型；`full` 同时写 SAR 和 optical 配置段 |
| `--sar-reader` | `gamma`, `gamma_tiff`, `gmtsar`, `hyp3` | SAR reader 家族 |
| `--sar-mode` | `unwrapped_phase`, `phase_los`, `los_displacement`, `range_offset`, `azimuth_offset` | SAR 观测模式 |
| `--downsample-method` | `std`, `data`, `trirb`, `from_rsp` | 写入 `downsample.method` |
| `--template` | `minimal`, `full` | `full` 展开所有降采样方法配置块 |
| `--copy-adapter-template` | 开关 | 复制自定义读入 adapter 模板；用户只改 `input_adapter.py` |
| `-o, --output` | 文件路径 | 输出 YAML 路径 |

常见场景：

```bash
# GAMMA range offset
ecat-generate-downsample \
  --mode sar --sar-reader gamma --sar-mode range_offset \
  -o downsample_range.yml

# GAMMA GeoTIFF 解缠相位
ecat-generate-downsample \
  --mode sar --sar-reader gamma_tiff --sar-mode unwrapped_phase \
  -o downsample_gamma_tiff_phase.yml

# GMTSAR-style range offset direct-projection GRD/NetCDF + ENU projection
ecat-generate-downsample \
  --mode sar --sar-reader gmtsar --sar-mode range_offset \
  -o downsample_gmtsar_range.yml

# 按数据梯度/曲率细化
ecat-generate-downsample \
  --mode sar --sar-reader gamma --sar-mode range_offset \
  --downsample-method data \
  -o downsample_data.yml

# 复用已有 CSI .rsp 格网
ecat-generate-downsample \
  --mode sar --sar-reader gamma --sar-mode range_offset \
  --downsample-method from_rsp \
  -o downsample_from_rsp.yml

# 光学 offset
ecat-generate-downsample --mode optical -o downsample_optical.yml
```

### 高级：自定义读入 adapter

标准 GAMMA/GMTSAR/HyP3/optical reader 覆盖的场景不需要 adapter。只有当读入阶段本身不标准，
例如外部文本、用户自建 CSI 对象或时序 InSAR，需要先自行构造 `csi.insar` / `csi.opticorr`，
再复用标准降采样 runtime 时，才复制 adapter 模板：

```bash
# 非标准数据或时序 InSAR：复制 adapter 模板
ecat-generate-downsample \
  --mode sar --sar-reader gamma --sar-mode unwrapped_phase \
  -o downsample.yml --copy-adapter-template
```

复制后通常只改 `input_adapter.py`。完全绕过标准 reader 时，`general.origin` 应使用 `manual`，
并填写 `lon0/lat0`。完整操作见 [自定义读入 Adapter 降采样](../workflows/02b_adapter_downsampling.md)，字段字典见 [降采样超级入口参考](downsampling_app.md#input_adapter)。

生成后重点检查 `sar_config.reader`、`sar_config.mode`、`sar_config.files`、`sar_config.output_suffix`、`general.origin`、`covar.mask_out`、`downsample.method` 和 `downsample.compute.cutde_backend`。`cutde_backend` 默认 `cpp`，主要用于避免 `trirb` 在普通 Windows/CUDA 环境中触发 `nvcc` 编译问题。字段字典见 [降采样超级入口参考](downsampling_app.md)，SAR reader 选择见 [SAR Reader 参考](sar_reader.md)。

## 执行降采样

推荐分三步运行：

```bash
# 只读入原始数据并画 quick-look
ecat-downsample -f downsample.yml -s

# 调用 CSI imagecovariance 估计协方差；需要先设置 covar.mask_out
ecat-downsample -f downsample.yml -c

# 正式降采样，写出 <effective_outName>_ifg.txt/.rsp/.cov
ecat-downsample -f downsample.yml -d
```

运行期常用覆盖项：

| 参数 | 含义 |
| --- | --- |
| `-f, --config` | YAML 配置文件路径 |
| `-s, --show_raw_data` | 只做原始数据读入和 quick-look；不做协方差估计，也不做降采样 |
| `-c, --do_covar` | 临时启用 CSI 协方差估计器生成；执行前必须设置 `covar.mask_out` |
| `-d, --do_downsample` | 临时启用正式降采样 |
| `--vmin`, `--vmax` | 覆盖 quick-look 或降采样检查图色标 |
| `--workers` | 覆盖降采样 worker 数；Windows 建议先串行 |

如果需要剔除粗差点，在 YAML 的 `sar_config.data_filters` 或 `optical_config.data_filters` 中设置规则，而不是用命令行临时阈值。`data_filters` 会真实改变进入 quick-look、协方差估计、降采样和后续反演的数据点集合。若只希望协方差和降采样处理一个关注区域，用顶层 `processing_region`；若只是放大 `-s` 图，用 `sar_config.qc.plot.coordrange` 或 `optical_config.qc.plot.coordrange`。`covar.mask_out` 只在协方差估计阶段排除震源形变区，不删除最终降采样数据。规则字段见 [降采样超级入口参考](downsampling_app.md#data_filters-顶层字段)。

SAR 的 `<effective_outName>` 是 `sar_config.outName` 经过 `sar_config.output_suffix` 解析后的前缀。`output_suffix: auto` 会给 range/azimuth offset 分别追加 `_RngOff` 和 `_AziOff`，若 `outName` 已经带同名后缀则不会重复追加。

如果配置中已经写了 `covar.do_covar: true` 或 `downsample.enabled: true`，可以不传 `-c/-d`；命令行参数优先级更高。每次运行会写出 `<effective_outName>_run_metadata.yml`，记录有效配置、投影原点、执行步骤和预期输出文件。完整解释见 [InSAR 降采样](../workflows/02_insar_downsampling.md)；按案例脚本组织的手动调参路线见 [InSAR 降采样两步走](../workflows/02a_insar_downsampling_two_step.md)。

模块形式：

```bash
python -m eqtools.cli_tools.process_data_downsampling -f downsample.yml -s
python -m eqtools.cli_tools.process_data_downsampling -f downsample.yml -c
python -m eqtools.cli_tools.process_data_downsampling -f downsample.yml -d
```

## 非线性几何配置

生成当前目录下的非线性 Bayesian 几何反演配置模板：

```bash
ecat-generate-nonlinear -o default_config.yml
```

模块形式：

```bash
python -m eqtools.cli_tools.generate_nonlinear_config -o default_config.yml
```

不传 `-o` 时默认输出 `default_config.yml`。模板生成后，应参照案例修改：

- `bounds`：几何先验，`[Uniform, start, range]` 的上界为 `start + range`。
- `fixed_params`：固定几何参数。
- `geodata.verticals/faults/sigmas`：与脚本中的 `geodata` 顺序一致。
- `nchains/chain_length`：采样规模。

字段说明见 [非线性几何反演配置](config_nonlinear_geometry.md)。

## 线性 BLSE/VCE 配置

生成主配置：

```bash
ecat-generate-config -o default_config.yml --gf-method cutde
```

生成边界配置：

```bash
ecat-generate-boundary -o bounds_config.yml -f MyFault
```

模块形式：

```bash
python -m eqtools.cli_tools.generate_config -o default_config.yml
python -m eqtools.cli_tools.generate_bounds_config -o bounds_config.yml -f MyFault
```

常用参数：

| 命令 | 参数 | 说明 |
| --- | --- | --- |
| `ecat-generate-config` | `--gf-method cutde|okada|pscmp|edcmp` | 设置 GF 计算方法模板 |
| `ecat-generate-config` | `--include-euler-constraints` | 写入 Euler 约束示例段 |
| `ecat-generate-config` | `--des` | 写入 DES 深度均衡平滑段 |
| `ecat-generate-config` | `--show-gf-options [METHOD]` | 查看 GF 方法可用选项 |
| `ecat-generate-boundary` | `-f, --faults` | Fault 源名称列表 |

查看 GF 方法选项：

```bash
ecat-generate-config --show-gf-options edcmp
ecat-generate-config --show-gf-options pscmp --format text
```

线性配置字段见 [线性滑动反演配置](config_linear_slip.md)，约束逻辑见 [ECAT 约束管理器](constraint_manager.md)。

## 几何与辅助工具

列出几何扰动方法：

```bash
ecat-list-fault-perturb-methods
```

处理或简化断层迹线：

```bash
ecat-fault-trace-tool input_trace.txt --algo vw --output trace_simplified
```

## Green's Function 模板工具

包中还包含 PSGRN/PSCMP 和 EDGRN/EDCMP 的模板生成与运行辅助：

```bash
ecat-generate-psgrn-template --help
ecat-generate-pscmp-template --help
ecat-generate-edgrn-template --help
ecat-generate-edcmp-template --help
```

这些属于进阶内容，建议在 BLSE/VCE 主流程稳定后再引入教程。

## 典型工具链

```text
准备数据和断层几何
  -> CLI 生成配置模板
  -> 修改 YAML 数据路径、边界、约束和权重
  -> Python 脚本构造 geodata/faults
  -> 运行非线性几何或 BLSE/VCE
  -> 保存图件、模型文件和诊断表
```
