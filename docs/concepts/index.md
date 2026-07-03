# Concepts / 核心概念

本目录解释跨多个工作流反复出现的概念。它回答“为什么这样组织”和“这些对象之间是什么关系”，不替代具体命令和字段参考。

## 概念地图

| 问题 | 先读 |
| --- | --- |
| 为什么推荐先非线性几何、再线性滑动 | [标准两步走反演逻辑](two_step_inversion.md) |
| trace、top/bottom、layers、mesh 和 patch 有什么区别 | [断层几何状态](fault_geometry_states.md) |
| SAR/offset 数据的正负号和 LOS projection 怎么理解 | [SAR 投影和观测约定](sar_projection_conventions.md) |

## 和其他文档层的关系

- `concepts/` 解释概念和判断逻辑。
- `workflows/` 给可执行步骤。
- `examples/` 给短小可复制代码。
- `reference/` 给完整字段、参数、API 和误区。
- `casebook/` 对应真实事件脚本和数据。
