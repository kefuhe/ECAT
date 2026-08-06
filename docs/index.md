# ECAT 用户手册

[ECAT](https://github.com/kefuhe/ECAT) 是面向地震大地测量建模与反演的科研工具集。公开代码包含 `eqtools` 和 `csi` 相关扩展；本手册按实际科研流程组织，而不是按源码目录组织。

标准入门路线是两步走：

1. **Bayesian 非线性几何反演**：估计断层顶边中点经纬度和深度、走向、倾角、长度、宽度等几何参数。
2. **BLSE/VCE 线性滑动分布反演**：固定优选几何后，反演分布式滑动。

数据读取和 SAR/InSAR 降采样是前期准备；固定优选几何并构建 mesh 用于衔接两个反演阶段。高级用户可在完成标准两步走检查后继续阅读 Bayesian 联合几何-滑动分布反演。

## 从这里开始

| 当前目标 | 入口 | 用法 |
| --- | --- | --- |
| 第一次使用 | [安装与环境检查](getting_started/installation.md) → [标准两步走路线](getting_started/quickstart_two_step.md) | 先跑通环境和标准反演主线 |
| 安装、导入或运行速度异常 | [安装与运行故障排查](getting_started/troubleshooting.md) | 按症状检查版本、wheel、MPI、BLAS 和线程数 |
| 不清楚 `-n`、rank、进程、线程或 affinity | [并行运行基础](concepts/parallel_process_rank_thread.md) | 先理解运行单位，再决定是否需要调参 |
| 不清楚 oneAPI、MKL、OpenBLAS 或 MPI 实现的关系 | [计算运行栈](concepts/compute_runtime_stack.md) | 先分清数值线程、MPI 进程、Python 绑定和启动器 |
| 已经知道要完成的科研任务 | [Workflows / 科研工作流](workflows/index.md) | 按输入、命令或脚本、输出和检查项执行 |
| 需要一个短小可复制片段 | [任务短例](examples/index.md) | 复制最小代码或命令，再回到 workflow |
| 需要一份完整可编辑的科研脚本 | [可运行脚本模板导航](examples/script_templates.md) | 选择普通 BLSE、平滑、倾角或联合敏感性模板 |
| 需要确认字段、接口或误区 | [Reference Map](reference/index.md) | 按需查阅，不必从头顺序阅读全部 reference |
| 需要设置跨直立倾角或核对 `strike/dip/rake` | [断层角度约定](concepts/fault_angle_conventions.md) | 先确认输入、solver geometry 和滑动基底的区别 |
| 西半球或跨日界线区域配置 | [经度约定与区域配置](reference/longitude_regions.md) | 核对处理区域、过滤、协方差掩膜和检查图范围的等价经度匹配 |
| 需要统一科研图件字体、尺寸和保存 | [科研绘图短例](examples/viztools_scientific_figures.md) | 先复制常用场景，再到 Viztools reference 查高级参数 |
| 需要在观测图上调整断层迹线 | [交互调整断层迹线](workflows/02c_interactive_trace_editing.md) | 复用降采样 reader 或直接打开标准观测，另存两列经纬度迹线 |

## 工作流主线

```text
数据读取 / 可选降采样
          ↓
Bayesian 非线性几何反演
          ↓
固定几何并构建 mesh
          ↓
BLSE/VCE 线性滑动分布反演
```

完整任务分流、降采样调参分支和高级联合反演入口见
[Workflows / 科研工作流](workflows/index.md)。

完成数据准备或反演后，如需在 Google Earth Pro 中叠加观测、降采样单元、断层、
滑动和地震目录，进入
[Google Earth 科研导出](workflows/06_google_earth_export.md)。该输出是显示副本，
不替代权威科学文件。

如需在本机浏览器中随手切换地震、断层、GNSS、标准观测和降采样图层，进入
[本地科研地图查看](workflows/07_research_map_viewer.md)。查看器同样只读科学源，
不执行数据改正或反演。

## 按需深入

| 文档层 | 什么时候使用 |
| --- | --- |
| [Concepts / 核心概念](concepts/index.md) | 理解两步走、并行运行、SAR 投影约定和断层几何状态 |
| [Examples / 任务短例](examples/index.md) | 复制一个小任务的最小代码或命令 |
| [Casebook / 公开案例](casebook/index.md) | 对照 ECAT-Cases 中的真实事件脚本、数据和输出 |
| [Reference / 完整参考](reference/index.md) | 查 CLI、配置、reader、约束、几何、结果解释和 API 细节 |

真实案例材料维护在
[ECAT-Cases](https://github.com/kefuhe/ECAT-Cases)；本仓库维护代码、用户手册与接口参考。
文档和发布维护者从 [Developer Notes](developer/index.md) 进入，参阅文档架构、
贡献规范以及独立 eqtools/CSI 与统一 ECAT 的同步边界。
