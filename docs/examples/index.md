# Examples / 任务短例

本目录放短小、可复制、按任务组织的示例。它回答“这件小事怎么写”，不替代完整工作流和真实案例：

- 想跑完整科研流程：先读 [标准两步走路线](../getting_started/quickstart_two_step.md)。
- 想从完整可编辑脚本开始：看 [可运行脚本模板导航](script_templates.md)。
- 想对照真实事件脚本和数据：读 [案例选择表](../casebook/index.md)。
- 想查完整字段、参数和接口：回到 [参考手册入口地图](../reference/index.md)。

## 示例地图

### 反演主线与完整脚本入口

| 任务 | 示例 | 先准备 |
| --- | --- | --- |
| 把 ECAT 降采样、外部 SAR 点或 GNSS ENU 读入反演 | [反演前读取 InSAR 与 GNSS](inversion_data_loading.md) | 数据前缀或规范文本、统一的投影原点 |
| 从地表迹线构建单倾角、多参考点倾角或固定拓扑断层 | [地表迹线和倾角构建](fault_trace_preprocessing.md) | `lon/lat` trace、投影原点和倾角信息 |
| 从非线性几何结果构建矩形元或三角元断层 | [非线性几何结果到 fault object](fault_from_nonlinear_geometry.md) | `clon/clat/cdepth/strike/dip/length` |
| 在 BLSE、非线性几何和联合 Bayesian 脚本间选择 | [可运行脚本模板导航](script_templates.md) | 已明确当前要复现、调平滑、检查几何还是传播联合不确定性 |
| 运行一个最小 BLSE/VCE 线性滑动脚本 | [BLSE/VCE 最小脚本骨架](blse_minimal_run.md) | 已建好的 fault、geodata 和配置 |
| 用同一套三角 patch 比较一组 BLSE 倾角 | [固定拓扑倾角搜索 workflow](../workflows/04b_blse_dip_search.md) 与 [标准脚本](https://github.com/kefuhe/eqtools/blob/main/scripts/test_dip_search_BLSE.py) | 地表迹线、候选倾角、geodata 和 BLSE 配置 |
| 在 YAML 基线之上试验 coarse、patch 或 rake 约束 | [约束配置与运行时调整短例](constraint_config_runtime.md) | 已初始化的 BLSE/VCE 或 `SMC_FJ + ss_ds` inversion |
| 为联合 Bayesian 建立曲线断层参考并对齐 YAML/bounds | [联合 Bayesian 几何参考与配置](joint_bayesian_geometry_setup.md) | 已检查的迹线、投影原点和 geodata |
| 从滑动模型生成密集地表 ENU 位移 | [地表形变正演最小例子](surface_forward_grid.md) | CSI patch GMT 或已有 fault object |

### 数据、迹线与查看工具

| 任务 | 示例 | 先准备 |
| --- | --- | --- |
| 检查、定位、裁剪、延长、重采样或简化地表迹线 | [断层迹线预处理](fault_trace_processing.md) | 两列 `lon lat`、GMT 或 GeoJSON 迹线 |
| 用 GAMMA prefix 快速预览 SAR/LOS 数据 | [GAMMA SAR quick-look 与配置生成](gamma_sar_quicklook.md) | GAMMA prefix 文件组 |
| 把标准网格、CSI varres、fault/slip 或小震导出为 KMZ | [Google Earth 导出短例](google_earth_export.md) | 标准观测文件、`.txt/.rsp` 或已回填科研对象 |
| 直接查看内置断层/GNSS，或用短 YAML 叠加自己的数据 | [科研地图快速查看与项目短例](research_map_viewer.md) | 快速模式无需输入；项目模式准备规范 CSV、GeoJSON 或标准观测文件 |
| 在原始或改正后观测上复制、移动并另存断层迹线 | [交互迹线调整短例](interactive_trace_editing.md) | 标准观测文件，或已有 `downsample.yml` 和参考迹线 |
| 统一论文图字体、尺寸、经纬度刻度和栅格 quick-look | [科研绘图短例](viztools_scientific_figures.md) | 可直接使用数组，也可从已有分析脚本接入 |

## 使用原则

- 示例中的文件名都是占位符，需要替换成自己的目录和数据。
- 示例只保留当前任务需要的参数；高级选项通过相关 reference 链接查。
- 距离、长度、面积类操作默认使用投影后的 `x/y` km，不直接在经纬度上计算。
- CLI 生成的 YAML 是模板，必须按案例修改数据路径、断层名、几何边界和权重设置。
