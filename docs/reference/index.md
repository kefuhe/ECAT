# Reference Map

Reference 是查阅层，不替代工作流教程。第一次跑案例时先读 `workflows/` 和 `casebook/`；需要确认命令、配置字段、reader 语义、约束或输出细节时，再回到本目录查阅。

## 推荐阅读顺序

```text
CLI
  -> SAR Reader
  -> Data Corrections
  -> Downsampling App
  -> Nonlinear Geometry Config
  -> Linear Slip Config
  -> Fault Geometry Construction
  -> Fault Summary
  -> Fault Edges
  -> Fault Patch Indices
  -> Constraint Manager
  -> Rake Constraints
  -> Interseismic Kinematics
  -> Deep Slip Loading Proxy
  -> BLSE/VCE
```

高级联合 Bayesian 和可扰动几何放在标准两步走之后阅读。

## 基础入口

| 页面 | 什么时候读 |
| --- | --- |
| [CLI Reference](cli.md) | 不确定该用哪个命令、如何生成模板、如何用模块形式运行 CLI 时 |

## 数据准备

| 页面 | 什么时候读 |
| --- | --- |
| [SAR Reader](sar_reader.md) | 需要确认 SAR/offset 产品的 reader、mode、projection、正负号和单位转换时 |
| [Data Corrections](data_corrections.md) | 需要确认 `geodata.polys`、InSAR ramp、GPS frame transform、`poly_bounds` 或 `data_corrections` 语义时 |
| [Downsampling App](downsampling_app.md) | 正在查 `downsample.yml` 字段、执行顺序、输出文件或兼容字段时 |

## 标准两步反演

| 页面 | 什么时候读 |
| --- | --- |
| [Nonlinear Config](config_nonlinear_geometry.md) | 设置非线性几何搜索边界、固定参数、数据顺序和 sigma 策略时 |
| [Linear Slip Config](config_linear_slip.md) | 设置 BLSE/VCE 的主配置、边界配置、GF、Laplacian、poly 和线性约束入口时 |
| [Fault Geometry Construction](fault_geometry_construction.md) | 从非线性几何结果、地表迹线、等深线、slab 网格、外部 mesh 或 GMT 文件构建断层时 |
| [Fault Summary](fault_summary.md) | 建完断层或完成反演后，快速检查 trace 长度、mesh、面积、走向倾角、slip、Mw 或矩率时 |
| [Fault Edges](fault_edges.md) | 需要确认 `top/bottom/left/right`、`edge_vertices` 或边界零滑前置条件时 |
| [Fault Patch Indices](fault_patch_indices.md) | 需要按边界、深度、空间范围或 trace 段生成 patch id，并传给约束或统计接口时 |
| [Fault Contours](fault_contours.md) | 提取断层等深线或 slip/coupling 等值线时 |
| [Sigmas and Alpha](sigmas_alpha.md) | 不确定 `single / individual / grouped`、log scale 或 alpha/sigma 边界含义时 |
| [Constraint Manager](constraint_manager.md) | 需要确认 bounds、rake、zero-slip、Euler cap、震间 backslip 或自定义线性约束时 |
| [Rake Constraints](rake_constraints.md) | 需要检查 `rake_angle` 的线性公式、角度范围限制、未知参数排列和多断层约束矩阵结构时 |
| [Interseismic Kinematics](interseismic_kinematics.md) | 已有块体运动配置和线性滑动结果，需要计算 loading、backslip、coupling、creep 或导出 patch GMT 时 |
| [Deep Slip Loading Proxy](deep_slip_loading_proxy.md) | 用深部自由滑动 patch 作为浅部长期加载代理，需要建立浅深映射、底边连续约束或导出 `coupling_to_deep` 时 |
| [BLSE/VCE](blse_vce.md) | 需要理解固定权重 BLSE、smoothing loop、VCE、结果导出和报告内容时 |
| [Surface Displacement Forward](surface_displacement_forward.md) | 已有断层几何和滑动量，想生成规则网格或自定义点上的 ENU 位移场时 |

## 高级 Bayesian

| 页面 | 什么时候读 |
| --- | --- |
| [Bayesian Joint Inversion](bayesian_joint_inversion.md) | 完成标准两步走后，需要把几何不确定性、滑动、sigma/alpha 和约束放入联合后验框架时 |
| [Perturbable Fault Geometry](geometry_perturbation.md) | 联合 Bayesian 中需要让断层几何、网格和 GF 随 SMC 样本一致更新时 |

## 通用工具

| 页面 | 什么时候读 |
| --- | --- |
| [Viztools](viztools.md) | 需要统一出版尺寸、字体、dpi、经纬度格式化和项目绘图规范时 |

## 使用原则

- 想跑通流程：先读 `workflows/`，再查本目录。
- 想理解概念关系：先读 `concepts/`，再回到 workflow 或 reference。
- 想复制一个小任务的代码：先读 `examples/`。
- 想对照真实脚本：先读 `casebook/`。
- 想确认字段定义：直接查对应 reference 页面。
