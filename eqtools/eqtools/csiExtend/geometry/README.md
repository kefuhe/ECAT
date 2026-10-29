# 几何支撑源码导航

本目录负责共享迹线、边界、角度、倾角剖面、patch 查询和网格后端。
主要断层对象、反演入口和候选状态流程仍在 `csiExtend/` 与 `perturbations/`。
直接导入实际实现模块；目录入口不批量加载数值、绘图或网格库。

## 文件作用

| 部分 | 文件 | 作用 |
| --- | --- | --- |
| 迹线数值核 | [trace_ops.py](trace_ops.py) | 在投影平面中清理、测距、裁剪、采样和调整迹线 |
| 位置标记 | [trace_markers.py](trace_markers.py) | 将距离、比例、经纬度截线和最近点解析为统一标记 |
| 处理对象 | [trace_processing.py](trace_processing.py) | `TracePath`、坐标转换、操作组合与历史；不修改 fault |
| 文件读写 | [trace_io.py](trace_io.py) | 多段迹线的文本/GeoJSON 读写和显式覆盖策略 |
| 边界几何 | [geom_ops.py](geom_ops.py) | 顶底边界、保留节点的平面弧长、交点和单元校验 |
| patch 查询 | [patch_indices.py](patch_indices.py) | 查询、校验和选择 patch 编号；不生成约束 |
| 角度合同 | [fault_angle_conventions.py](fault_angle_conventions.py) | 分别处理有方向参考倾角与 compact solver 几何角度 |
| 角度工具 | [angle_utils.py](angle_utils.py) | 角度单位、圆周平均和参数长度展开 |
| 分层倾角 | [DipInterpolation.py](DipInterpolation.py) | 随深度变化的倾角插值与分层几何 |
| 非分层剖面 | [dip_profile.py](dip_profile.py) | 沿走向 controls、位置解析与 sampled/fixed 参数映射 |
| 转换带分析 | [dip_transition_analysis.py](dip_transition_analysis.py) | 在冻结参考顶边上只读分析曲率和倾角转换带 |
| 底边生成 | [dip_geometry.py](dip_geometry.py) | 将已解析的顶边倾角转换为走向和底边坐标 |
| 几何等深线 | [fault_contours.py](fault_contours.py) | 求断层面与深度平面的交线、拼接及诊断 |
| 结构化网格 | [MeshGenerator.py](MeshGenerator.py) | 结构化表面生成与 Gmsh 映射后端 |
| Dutta 网格 | [make_mesh_dutta.py](make_mesh_dutta.py) | 倾角曲线离散、底边修正和该方法的网格生成 |
| 参数化网格 | [parametric_mesh.py](parametric_mesh.py) | 使用三维弦长参数域生成或变形网格 |
| 方法元数据 | [mesh_registry.py](mesh_registry.py) | mesh 参数、重放与禁止项的单一静态表 |

## 阅读顺序与责任边界

- 迹线：`trace_processing` -> `trace_markers` -> `trace_ops`；格式查 `trace_io`。
- 倾角：fault 方法 -> `dip_profile` 或 `DipInterpolation` -> `dip_geometry`。
  分层与非分层分别阅读，不能互换 resolver。
- 网格：fault 方法 / `MeshPolicy` -> `mesh_registry` -> 实际网格后端。
  reference 冻结、候选物化、CSI mesh 发布与缓存失效由消费者负责。
- 等深线：`FaultGeometryEngine.extract_contours_from_fault` -> `fault_contours`。
  slip/coupling 标量等值线仍在 [statUtils/contour3D_extraction.py](../statUtils/contour3D_extraction.py)。

平面迹线距离、保留节点对应关系的边界弧长和三维网格弦长具有不同合同。
每个模块的开头说明职责，具体函数文档说明坐标、单位、数组布局和返回值。

完整依赖、计算差异、新增文件归属规则与 16 个旧模块的迁移表见
[公开手册：几何支撑源码与依赖](https://github.com/kefuhe/ECAT/blob/main/docs/developer/geometry_support.md)。
稳定的包级公开入口保留；本批旧模块仅 `statUtils.fault_contours` 保留兼容转出。
