# Reference Map

Reference 是查阅层，不替代工作流教程。第一次跑案例时先读 `workflows/` 和 `casebook/`；需要确认命令、配置字段、reader 语义、约束或输出细节时，再回到本目录查阅。

## 阅读方式

```text
workflow 或 example
  -> 打开其中链接的一篇 reference
  -> 按页面开头的“阅读路径”定位当前问题
  -> 需要时再进入“相关页面”
```

Reference 不要求按文件顺序通读。长页面保留完整字段和公式，同时在开头给出按任务划分的“阅读路径”；高级联合 Bayesian 和可扰动几何放在标准两步走之后查阅。

## 基础入口

| 页面 | 什么时候读 |
| --- | --- |
| [CLI Reference](cli.md) | 不确定该用哪个命令、如何生成模板、如何用模块形式运行 CLI 时 |

## 数据准备

| 页面 | 什么时候读 |
| --- | --- |
| [观测数据读入](observation_data_readers.md) | 需要复制 GAMMA/HyP3/GMTSAR/optical reader，读取 CSI varres、外部 ASCII SAR 或 GNSS ENU 时 |
| [SAR Reader](sar_reader.md) | 需要确认 SAR/offset 产品的 reader、mode、projection、正负号和单位转换时 |
| [Data Corrections](data_corrections.md) | 需要确认 `geodata.polys`、InSAR ramp、GPS frame transform、`poly_bounds` 或 `data_corrections` 语义时 |
| [Downsampling App](downsampling_app.md) | 正在查 `downsample.yml` 字段、执行顺序、输出文件或兼容字段时 |
| [交互迹线编辑器](interactive_trace_editor.md) | 需要查 `ecat-trace-edit`、降采样交接、节点操作、坐标或 Save As 规则时 |
| [经度约定与区域配置](longitude_regions.md) | 西半球或跨日界线数据需要设置处理区域、过滤、协方差掩膜或检查图范围时 |
| [Observation Correction and Grid Export](observation_correction_export.md) | 需要在降采样前归零/去一阶平面、给已确认的不连通相位分量修正整周，或无重采样导出全分辨率 SAR/optical 网格时 |

## 标准两步反演

| 页面 | 什么时候读 |
| --- | --- |
| [Nonlinear Config](config_nonlinear_geometry.md) | 设置非线性几何搜索边界、固定参数、数据顺序和 sigma 策略时 |
| [Fault Angle Conventions](../concepts/fault_angle_conventions.md) | 设置 `strike/dip/rake`、让 dip 跨过直立位置或解释 raw/canonical 几何时 |
| [Linear Slip Config](config_linear_slip.md) | 设置 BLSE/VCE 的主配置、边界配置、GF、Laplacian、poly 和线性约束入口时 |
| [Fault Geometry Construction](fault_geometry_construction.md) | 从非线性结果或地表迹线构建固定/变化倾角断层，或处理等深线、slab、外部 mesh 和 GMT 时 |
| [Fault Summary](fault_summary.md) | 建完断层或完成反演后，快速检查 trace 长度、mesh、面积、走向倾角、slip、Mw 或矩率时 |
| [Fault Edges](fault_edges.md) | 需要确认 `top/bottom/left/right`、`edge_vertices` 或边界零滑前置条件时 |
| [Fault Patch Indices](fault_patch_indices.md) | 需要按边界、深度、空间范围或 trace 段生成 patch id，并传给约束或统计接口时 |
| [Fault Contours](fault_contours.md) | 提取断层等深线或 slip/coupling 等值线时 |
| [Sigmas and Alpha](sigmas_alpha.md) | 不确定 `single / individual / grouped`、log scale 或 alpha/sigma 边界含义时 |
| [Fit Statistics](fit_statistics.md) | 需要确认 RMS/VR 的逐数据集公式、全局求解器向量公式、poly include 语义或输出表格接口时 |
| [Constraint Manager](constraint_manager.md) | 需要确认公开约束接口、配置基线/会话修改/patch 层级、bounds、领域约束或自定义线性矩阵时；只想复制常用搭配先看 [约束短例](../examples/constraint_config_runtime.md) |
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
| [Google Earth Export](google_earth_export.md) | 把标准观测、CSI 降采样单元、断层/滑动或地震目录导出成 Google Earth 显示副本时 |
| [Research Map Viewer](research_map_viewer.md) | 直接查看内置断层/GNSS，或用项目 YAML 只读叠加自己的地震、标准观测或 varres，并查格式和运行时层级时 |
| [Viztools](viztools.md) | 需要统一出版尺寸、字体、dpi、经纬度格式化和项目绘图规范时 |
| [Figure Products](figure_products.md) | 希望用一个上层入口批量生成 data/synth/residual、slip、震间或 deep proxy 常用图件，同时保持底层绘图 API 不变时 |

## 使用原则

- 想跑通流程：先读 `workflows/`，再查本目录。
- 想理解概念关系：先读 `concepts/`，再回到 workflow 或 reference。
- 想复制一个小任务的代码：先读 `examples/`。
- 想对照真实脚本：先读 `casebook/`。
- 想确认字段定义：直接查对应 reference 页面。
