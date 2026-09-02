# CLI 参考

本页列出 ECAT 常用命令行入口。命令行负责生成模板、查看选项和执行轻量工具；正式反演仍通常由 Python 脚本组织数据对象、断层对象和输出。

## 常用入口

| 任务 | 命令 | 下一步 |
| --- | --- | --- |
| 生成旧版非线性几何配置 | `ecat-generate-nonlinear` | 修改 legacy `bounds/fixed_params/geodata/sigmas` |
| 生成新版 nonlinear geometry SMC 配置 | `ecat-generate-nonlinear-geometry` | 修改 `nonlinear_geometry.yml` |
| 生成 BLSE/VCE 主配置 | `ecat-generate-config` | 修改数据、GF、平滑、sigma/alpha 和配置文件指针 |
| 生成 BLSE/VCE 边界配置 | `ecat-generate-boundary` | 修改滑动边界、rake、sigma/alpha 和普通线性约束 |
| 生成震间配置 | `ecat-generate-interseismic` | 修改 `blocks`、`fault_loading`、可选 cap/backslip 约束 |
| 生成 SAR/optical 降采样配置 | `ecat-generate-downsample` | 核对 reader、mode、文件与处理参数 |
| 观测改正/导出、预览、协方差或降采样 | `ecat-downsample` | 无阶段选项可只执行已启用的改正/导出；`-s/-c/-d` 执行对应阶段 |
| 查看 GF 方法选项 | `ecat-generate-config --show-gf-options` | 把选项写入 `faults.defaults.method_parameters.update_GFs.options` |
| 运行已安装的 PSGRN/PSCMP 或 EDGRN/EDCMP 二进制 | `ecat-psgrn`, `ecat-pscmp`, `ecat-edgrn`, `ecat-edcmp` | 先生成并检查对应程序的输入文件，再按平台能力运行 |
| 简化或处理断层迹线 | `ecat-fault-trace-tool` | 在构建断层几何前处理 trace |
| 导出 Google Earth KMZ | `ecat-export-google-earth` | 选择标准网格、CSI varres、地震 CSV 或多图层 project |
| 打开本地科研地图 | `ecat-map` | 读取短 project YAML，按需显示地震、断层、GNSS、标准观测或 varres |

<a id="downsampling-config"></a>

## SAR 降采样配置

常用 GAMMA 模板：

```bash
# 默认右视解缠相位
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase -o downsample_phase.yml

# 默认右视 LOS、range offset、azimuth offset
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode los_displacement -o downsample_los.yml
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode range_offset -o downsample_range.yml
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode azimuth_offset -o downsample_azimuth.yml

# 左视、由 GAMMA 导出的 NISAR 相位
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase --sar-look-side left -o downsample_nisar_left.yml
```

生成器选项：

| 选项 | 值 | 默认 | 作用 |
| --- | --- | --- | --- |
| `--sar-reader` | `gamma`, `gamma_tiff`, `gmtsar`, `hyp3` | `gamma` | 输入处理平台/格式 |
| `--sar-mode` | `unwrapped_phase`, `los_displacement`, `range_offset`, `azimuth_offset` | `unwrapped_phase` | 标量观测类型 |
| `--sar-look-side` | `right`, `left` | `right` | 地面条带位于平台航向哪一侧 |
| `--sar-byte-order` | `native`, `little`, `big` | `native` | GAMMA float32 字节序 |
| `--downsample-method` | `std`, `data`, `trirb`, `from_rsp` | `std` | 选择降采样方法；短模板只保留对应的 `*_config` |
| `--template` | `minimal`, `full` | `minimal` | 短模板或带高级提示的完整模板 |

选择 `--downsample-method trirb` 时，模板中的三角断层示例预设
`use_for: [trirb]`，但仍保持 `enabled: false`；填写真实模型后启用所需条目即可。其他方法
不会预选断层计算角色。字段和多断层联合规则见
[Downsampling App 参考](downsampling_app.md#fault_traces-与-fault_models)。

无 YAML 的 GAMMA quick-look：

```bash
ecat-downsample -s --sar-prefix geo_pair --sar-mode unwrapped_phase
ecat-downsample -s --sar-prefix nisar_pair --sar-mode unwrapped_phase --sar-look-side left
```

正式运行：

```bash
# 默认模板中 covar/downsample 均关闭：只执行已启用的改正/导出
ecat-downsample -f downsample_phase.yml

# quick-look；本次强制关闭协方差和降采样
ecat-downsample -f downsample_phase.yml -s

# 临时启用经验协方差估计
ecat-downsample -f downsample_phase.yml -c

# 临时启用正式降采样
ecat-downsample -f downsample_phase.yml -d
```

`-c/-d` 与 YAML 开关是相加关系，也可组合为 `-c -d`；`-s` 会覆盖并关闭两个计算
阶段。标准 NC/H5 导出可随任一模式写出，Google Earth 联动只在有效计算/预览阶段均
关闭时写出。

字段含义和左右视约定见 [SAR Reader 参考](sar_reader.md) 与
[InSAR 降采样](../workflows/02_insar_downsampling.md)。参考改正与标准网格导出见
[观测参考改正与无重采样网格导出](observation_correction_export.md)。同一 reader
配置直接导出全分辨率 KMZ 时，启用 `export.google_earth` 并使用上面的无阶段选项
命令；详见 [Google Earth Export](google_earth_export.md#downsample-integration)。

## Google Earth 科研导出

常用单图层入口：

```bash
ecat-export-google-earth observation-grid observation.nc --variable observation -o observation.kmz
ecat-export-google-earth varres downsampled/track_a --data-type sar -o track_a_cells.kmz
ecat-export-google-earth catalog events.csv -o events.kmz
```

多图层由一个短 YAML 组织：

```bash
ecat-export-google-earth project google_earth.yml
```

导出器不直接解析原始 GAMMA/GMTSAR/HyP3，不修改观测和反演对象，也不把 KMZ
作为科学存储。完整工作顺序见
[Google Earth 科研导出](../workflows/06_google_earth_export.md)，所有 CLI、YAML 和
Python API 字段见 [Google Earth Export Reference](google_earth_export.md)。

## 本地科研地图

不写 YAML，直接查看随包发布的断层、块体和 GNSS：

```bash
ecat-map
ecat-map --region -72 -62 8 12
ecat-map --catalog events.csv
```

加入自己的地震、观测网格、GeoTIFF 或 varres 时再使用项目：

```bash
ecat-map research_map.yml
```

等价模块入口：

```bash
python -m eqtools.map_viewer
python -m eqtools.map_viewer research_map.yml
```

quick mode 只默认显示轻量 PB2002 boundaries，其余内置背景隐藏；项目模式中的内置
背景全部隐藏。项目路径相对 YAML 解析，隐藏图层首次勾选时才读取。最短项目见
[科研地图项目短例](../examples/research_map_viewer.md)，任务步骤见
[本地科研地图查看](../workflows/07_research_map_viewer.md)，完整字段见
[Research Map Viewer Reference](research_map_viewer.md)。

## 交互调整断层迹线

标准观测文件可直接打开：

```bash
ecat-trace-edit scene_observation.nc --trace published_trace.txt --output adjusted_trace.txt
```

原始 GAMMA/GMTSAR/HyP3/光学文件由现有降采样 reader 解释后交接：

```bash
ecat-downsample -f downsample.yml --edit-trace
```

该可选入口使用 Save As，不改写观测、参考迹线或 YAML。短例见
[交互迹线调整短例](../examples/interactive_trace_editing.md)，完整参数见
[交互迹线编辑器参考](interactive_trace_editor.md)。

## 非线性几何配置

旧版 legacy 模板：

```bash
ecat-generate-nonlinear -o default_config.yml
```

模块形式：

```bash
python -m eqtools.cli_tools.generate_nonlinear_config -o default_config.yml
```

新版 nonlinear geometry SMC 模板：

```bash
ecat-generate-nonlinear-geometry -o nonlinear_geometry.yml
```

模块形式：

```bash
python -m eqtools.cli_tools.generate_nonlinear_geometry_config -o nonlinear_geometry.yml
```

新版不传 `-o` 时默认输出 `nonlinear_geometry.yml`。其中 `prior_bounds_format: lower_upper` 表示 `Uniform` 使用 `[Uniform, lower, upper]` 语义。字段说明见 [非线性几何反演配置](config_nonlinear_geometry.md)。

<a id="linear-blse-vce-config"></a>

## Linear BLSE/VCE Config

生成主配置和边界配置：

```bash
ecat-generate-config -o default_config.yml --gf-method cutde
ecat-generate-boundary -o bounds_config.yml -f MyFault
```

如果是震间模型，同时生成震间配置，并让主配置记录指针：

```bash
ecat-generate-config -o default_config.yml --gf-method cutde --interseismic-config interseismic_config.yml
ecat-generate-interseismic -o interseismic_config.yml -f MyFault
```

模块形式：

```bash
python -m eqtools.cli_tools.generate_config -o default_config.yml --gf-method cutde --interseismic-config interseismic_config.yml
python -m eqtools.cli_tools.generate_bounds_config -o bounds_config.yml -f MyFault
python -m eqtools.cli_tools.generate_interseismic_config -o interseismic_config.yml -f MyFault
```

常用选项：

| 命令 | 选项 | 用途 |
| --- | --- | --- |
| `ecat-generate-config` | `--gf-method cutde|okada|pscmp|edcmp` | 设置 GF 计算方法模板 |
| `ecat-generate-config` | `--interseismic-config FILE` | 在主配置中写入 `interseismic_config_file` |
| `ecat-generate-config` | `--include-des-config` | 写入 DES 深度均衡平滑段 |
| `ecat-generate-config` | `--show-gf-options [METHOD]` | 查看 GF 方法可用选项 |
| `ecat-generate-boundary` | `-f, --faultnames` | Fault 源名称列表 |
| `ecat-generate-interseismic` | `-f, --fault` | Fault 源名称列表 |

生成的 `update_GFs` 只保留用户负责的 `method/options` 等字段。`cutde/okada`
使用 `options: {}`；观测对象和 vertical/data-fault 对应关系由脚本及顶层
`geodata` 配置在运行时注入，不需要复制到 `update_GFs`。

查看 GF 选项：

```bash
ecat-generate-config --show-gf-options edcmp
ecat-generate-config --show-gf-options pscmp --format text
```

线性配置字段见 [线性滑动反演配置](config_linear_slip.md)，约束逻辑见 [ECAT 约束管理器](constraint_manager.md)，震间公式见 [震间加载、Backslip 与 Coupling](interseismic_kinematics.md)。

`ecat-generate-config` 当前生成可直接对应标准线性子问题的
`bayesian_sampling_mode: SMC_FJ` 与 `slip_sampling_mode: ss_ds`。若改成
`FULLSMC + rake_fixed`，再取消模板中 `rake_angle` 的注释并填写固定角度。
`ecat-generate-boundary` 给出各模式可编辑的 bounds 骨架，并只保留短的
`source_bounds` / `source_constraints` 注释示例。未启用时模板明确写成
`source_bounds: {}` 和 `source_constraints: {}`；未取消注释的规则不会参与
计算。启用示例时，先移除同一行的 `{}`，再取消所需条目的注释。

## 几何与辅助工具

列出几何扰动方法：

```bash
ecat-list-fault-perturb-methods
```

该命令用于查看 `BayesianAdaptiveTriangularPatches` 当前可发现的 `perturb_*` 方法，主要服务于 [Bayesian 联合反演中的可扰动断层几何](geometry_perturbation.md) 和 `update_fault_geometry` 配置。

检查和处理断层迹线：

```bash
ecat-fault-trace-tool inspect input_trace.txt
ecat-fault-trace-tool locate input_trace.txt --lon 101.8
ecat-fault-trace-tool trim input_trace.txt --end-lon 101.8 -o trace_trimmed.txt
ecat-fault-trace-tool simplify input_trace.txt --method vw --tolerance 0.5 -o trace_simplified.txt
```

新子命令支持检查、经纬度/最近点定位、方向统一、裁剪、延长、重采样、简化、平滑和 YAML
多步处理，写文件时默认不覆盖。完整命令、marker 与 Python API 见
[断层迹线处理参考](fault_trace_processing.md)。旧的
`input_trace.txt --algo vw --param 0.5 --output PREFIX` 调用仍保留兼容，继续生成原有三类输出。

## Green's Function 模板与运行入口

包中包含 PSGRN/PSCMP 和 EDGRN/EDCMP 的输入模板生成器：

```bash
ecat-generate-psgrn-template --help
ecat-generate-pscmp-template --help
ecat-generate-edgrn-template --help
ecat-generate-edcmp-template --help
```

生成器只负责创建可编辑输入，不会自动运行外部程序。检查输入后，对应运行入口是：

```bash
ecat-psgrn [PSGRN 原生参数]
ecat-pscmp [PSCMP 原生参数]
ecat-edgrn [EDGRN 原生参数]
ecat-edcmp [EDCMP 原生参数]
```

这四个入口是已安装 CSI 二进制的轻量转发器：它们不解析 ECAT YAML，也不改写上游程序
参数；当前终端的输入输出和程序退出码会原样传递。因此输入文件名、工作目录和参数格式应
遵循对应 PSGRN/PSCMP 或 EDGRN/EDCMP 程序的约定。

运行前先检查：

- 当前平台是 Windows 或 Linux，且安装的 CSI 包包含对应平台二进制；
- 模板中的单位、层状模型、观测点和输出目录已经按上游程序要求修改；
- Windows 安装中当前随包提供的 PSGRN/PSCMP 可执行文件不可用，需改用经过验证的 Linux
  环境或自行编译的兼容二进制；EDGRN/EDCMP 仍应先用小输入检查运行时依赖和输出。

这些属于进阶正演工具。普通 `cutde`/Okada 线性反演不需要直接调用它们；建议在标准
BLSE/VCE 主流程稳定后，再根据层状介质 GF 需求引入。

## 典型工具链

```text
准备数据和断层几何
  -> CLI 生成配置模板
  -> 修改 YAML 数据顺序、边界、约束和权重
  -> Python 脚本构造 geodata/faults
  -> 运行非线性几何或 BLSE/VCE
  -> 保存图件、模型文件和诊断表
```

## 相关页面

- [非线性几何反演配置](config_nonlinear_geometry.md)
- [线性滑动反演配置](config_linear_slip.md)
- [ECAT 约束管理器](constraint_manager.md)
- [震间加载、Backslip 与 Coupling](interseismic_kinematics.md)
- [Fault Geometry Construction](fault_geometry_construction.md)
- [Google Earth Export](google_earth_export.md)
- [Research Map Viewer](research_map_viewer.md)
