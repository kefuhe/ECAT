# Examples / 任务短例

本目录放短小、可复制、按任务组织的示例。它回答“这件小事怎么写”，不替代完整工作流和真实案例：

- 想跑完整科研流程：先读 [标准两步走路线](../getting_started/quickstart_two_step.md)。
- 想对照真实事件脚本和数据：读 [案例选择表](../casebook/index.md)。
- 想查完整字段、参数和接口：回到 [参考手册入口地图](../reference/index.md)。

## 示例地图

| 任务 | 示例 | 先准备 |
| --- | --- | --- |
| 清理、裁剪、延伸和重采样地表迹线 | [Trace 预处理与断层顶部边界](fault_trace_preprocessing.md) | `lon/lat` trace 和投影原点 |
| 从非线性几何结果构建矩形元或三角元断层 | [非线性几何结果到 fault object](fault_from_nonlinear_geometry.md) | `clon/clat/cdepth/strike/dip/length` |
| 用 GAMMA prefix 快速预览 SAR/LOS 数据 | [GAMMA SAR quick-look 与配置生成](gamma_sar_quicklook.md) | GAMMA prefix 文件组 |
| 运行一个最小 BLSE/VCE 线性滑动脚本 | [BLSE/VCE 最小脚本骨架](blse_minimal_run.md) | 已建好的 fault、geodata 和配置 |
| 从滑动模型生成密集地表 ENU 位移 | [地表形变正演最小例子](surface_forward_grid.md) | CSI patch GMT 或已有 fault object |

## 使用原则

- 示例中的文件名都是占位符，需要替换成自己的目录和数据。
- 示例只保留当前任务需要的参数；高级选项通过相关 reference 链接查。
- 距离、长度、面积类操作默认使用投影后的 `x/y` km，不直接在经纬度上计算。
- CLI 生成的 YAML 是模板，必须按案例修改数据路径、断层名、几何边界和权重设置。
