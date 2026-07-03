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
| 查看 GF 方法选项 | `ecat-generate-config --show-gf-options` | 把选项写入 `faults.defaults.method_parameters.update_GFs.options` |
| 简化或处理断层迹线 | `ecat-fault-trace-tool` | 在构建断层几何前处理 trace |

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
| `ecat-generate-boundary` | `-f, --fault` | Fault 源名称列表 |
| `ecat-generate-interseismic` | `-f, --fault` | Fault 源名称列表 |

查看 GF 选项：

```bash
ecat-generate-config --show-gf-options edcmp
ecat-generate-config --show-gf-options pscmp --format text
```

线性配置字段见 [线性滑动反演配置](config_linear_slip.md)，约束逻辑见 [ECAT 约束管理器](constraint_manager.md)，震间公式见 [震间加载、Backslip 与 Coupling](interseismic_kinematics.md)。

## 几何与辅助工具

列出几何扰动方法：

```bash
ecat-list-fault-perturb-methods
```

该命令用于查看 `BayesianAdaptiveTriangularPatches` 当前可发现的 `perturb_*` 方法，主要服务于 [Bayesian 联合反演中的可扰动断层几何](geometry_perturbation.md) 和 `update_fault_geometry` 配置。

处理或简化断层迹线：

```bash
ecat-fault-trace-tool input_trace.txt --algo vw --output trace_simplified
```

它常用于断层几何构建前的 trace 减点或平滑。`--param` 分别控制 VW 面积阈值、RDP 距离阈值或 B-Spline 平滑因子。使用场景见 [Fault Geometry Construction](fault_geometry_construction.md)。

## Green's Function 模板工具

包中还包含 PSGRN/PSCMP 和 EDGRN/EDCMP 的模板生成与运行辅助：

```bash
ecat-generate-psgrn-template --help
ecat-generate-pscmp-template --help
ecat-generate-edgrn-template --help
ecat-generate-edcmp-template --help
```

这些属于进阶内容，建议在 BLSE/VCE 主流程稳定后再引入。

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
