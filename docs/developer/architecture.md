# 源码与文档架构

用户手册按科研工作流组织，源码按可复用组件组织。两者不要混在一起。

## 源码层级与阅读路线

三个独立分发包的依赖方向是 `eqtools -> CSI / ecat_viz` 和 `CSI -> ecat_viz`。
ECAT 组合发布这些包；通用绘图库不依赖 CSI 或 eqtools。

| 层级 | 主要入口 | 实现归属 |
| --- | --- | --- |
| 通用绘图 | `ecat_viz.PlotStyle`、`finish_fig`、`get_cmap` | ecat-viz 的 `src/ecat_viz/`；样式、字体、颜色、尺寸、坐标格式和通用栅格显示 |
| CSI 基础对象 | `csi.gps.gps`、`csi.insar.insar`、断层类 | CSI 的 `csi/`；观测、坐标框架、source、mesh、GF 与基础对象方法 |
| CSI 内部显示 | 数据对象的 `plot_fit_comparison()` | `csi/_plotting/gps.py`、`csi/_plotting/spatial.py`；只消费准备好的字段 |
| eqtools 反演 | BLSE/VCE、Bayesian、`NonlinearGeometrySMCInversion` | `eqtools/csiExtend/` 的反演入口和现有计算、配置、约束模块 |
| eqtools 几何支撑 | `TracePath`、patch 查询、剖面解析与网格后端 | `eqtools/csiExtend/geometry/`；各模块分别保持自己的坐标、距离、角度与索引合同 |
| eqtools 观测支撑 | 行向量、预测块、改正解释与精确白化 | `eqtools/csiExtend/observations/`；不接管数据对象和物理 source |
| eqtools 约束支撑 | 公开 inversion 的约束方法 | `eqtools/csiExtend/constraints/`；声明、当前状态和目标参数空间编译 |
| eqtools 结果组织 | `plot_data_fits()`、拟合统计与参数报告 | `eqtools/csiExtend/reporting/`；标准图件、结果统计、报告格式与非线性结果显示 |
| eqtools 领域图件 | 断层边界、倾角、滑动诊断 | `eqtools/viztools/`；直接使用 ecat_viz 通用工具 |

先从用户实际调用的方法或类开始，再进入其实现模块；解释结果时再阅读相应的科学合同。
以下是常用阅读顺序：

- GPS 比较图：`gps.plot_fit_comparison()` -> `csi/_plotting/gps.py` -> ecat_viz；
  上层是否使用 U、如何发布 synthetic 则查 `observations/data_prediction.py` 和 `data_vector_layout.py`。
- 标准反演图件：反演类的 `plot_data_fits()` -> `reporting/plot_product_mixin.py` ->
  `reporting/figure_products.py` -> 数据对象绘图方法。预测准备仍委托预测拥有者。
- 非线性结果与统计：`nonlinear_geometry_smc.py` -> `reporting/nonlinear_geometry_plots.py`
  或 `reporting/nonlinear_fit_statistics_mixin.py` -> 对应的报告模块。
- 观测和约束：inversion 入口 -> `observations` 的明确合同或 `constraints` 的对应 manager；
  文件职责、参数空间差异和导入迁移见[观测与约束支撑源码](observation_constraint_support.md)。
- 迹线处理：`TracePath` / trace CLI -> `geometry/trace_processing.py` ->
  `geometry/trace_markers.py` -> `geometry/trace_ops.py`；文件格式查 `geometry/trace_io.py`。
- 倾角与网格：fault 对象 / `perturbations/pipeline.py` -> `geometry/dip_profile.py`、
  `geometry/dip_geometry.py` 和具体 mesh 后端 -> 原有 fault/CSI 发布入口。
  科学差异和完整模块导航见[几何支撑源码与依赖](geometry_support.md)。

`reporting` 的入口不批量导入绘图或求解器；按具体模块导入即可。
八个原根层结果兼容模块已移除。包内代码和功能测试直接导入 `reporting` 实现；
稳定对象方法与既有包级公开导出保持不变。旧导入与序列化边界见[结果模块迁移](#reporting-import-migration)。
原 CSI 私有 `_gps_plotting`、`_spatial_plotting` 实现已归入 `_plotting`，用户继续调用对象方法。
`csi._slip_direction` 的跨包协议探测和 `decimation_geometry` 的观测几何合同保持原位。
`geometry` 入口同样不批量加载数值或后端库。几何支撑直接从具名实现模块导入，原根层
14 个几何模块及 `perturbations.angle_utils/dip_ops` 的旧路径已移除；仅
`statUtils.fault_contours` 保留等深线兼容入口。稳定的包级公开类/函数入口保持不变。
迁移路径及旧对象序列化边界见[几何导入迁移](geometry_support.md#geometry-import-migration)。
距离定义、重复节点处理、倾角转换、参数映射和网格后端不因归组而合并。

## 结果模块导入迁移 {#reporting-import-migration}

八个原根层结果模块已移除兼容壳，直接导入 `eqtools.csiExtend.reporting` 中的具名模块。
既有 inversion 对象方法和 `eqtools.csiExtend` 包级公开类/函数导出保持不变；本次只收紧
旧实现模块路径，不改变统计公式、图件参数、返回值、模型激活或预测发布。

| 原 `eqtools.csiExtend` 下的模块 | 当前相对模块 |
| --- | --- |
| `figure_products` | `reporting.figure_products` |
| `plot_product_mixin` | `reporting.plot_product_mixin` |
| `data_plot_utils` | `reporting.data_plot_utils` |
| `nonlinear_geometry_plots` | `reporting.nonlinear_geometry_plots` |
| `fit_statistics` | `reporting.fit_statistics` |
| `hyperparameter_reporting` | `reporting.hyperparameter_reporting` |
| `parameter_layout_reporting` | `reporting.parameter_layout_reporting` |
| `nonlinear_fit_statistics_mixin` | `reporting.nonlinear_fit_statistics_mixin` |

```python
from eqtools.csiExtend import FigureProductMixin, fit_metrics_from_vectors
from eqtools.csiExtend.reporting.parameter_layout_reporting import make_parameter_layout_report
from eqtools.csiExtend.reporting.nonlinear_geometry_plots import SMCPlotData
```

显式使用 `_plot_leveling_fit`、`_plot_crossfaultoffset_fit` 等内部 helper 的脚本也应改为
从 `reporting.data_plot_utils` 导入；长期用户脚本优先使用对象方法。功能测试导入真实实现；
结构测试检查旧路径明确失败、稳定导出身份和当前 pickle 往返，不以旧测试依赖保留壳。

旧脚本可能收到 `ModuleNotFoundError: eqtools.csiExtend.<旧模块>`，或在访问旧包属性时
收到 `AttributeError`；先确认实际安装来源，再按上表修改 import。若报错缺少的是其他
依赖，应处理 traceback 中真正缺失的模块，不能把所有导入错误都归为路径迁移。

旧 `SMCPlotData` pickle 若记录了 `eqtools.csiExtend.nonlinear_geometry_plots`，已不再
保证直接恢复；其他按旧模块路径保存的类或函数也需单独检查。在可信且能读取旧对象的
旧环境中导出样本数组、参数名和必要元数据，用当前类重建并另存。修改脚本 import
不能改写存档；不提供隐式模块 alias、自定义通用 unpickler 或同名壳补丁。
当前 `reporting` 类路径的 pickle 往返保持有效；纯数组/文本文件不因模块删除自动改变。

`statUtils.fault_contours` 保持已明确的等深线兼容例外，其实现仍由 `geometry.fault_contours`
唯一拥有。运行来源、生成缓存和升级排查见[升级与依赖迁移](../getting_started/upgrading.md)。

## 新增文件归属规则

1. **先确定状态和科学语义的拥有者。** 绘图模块不能因为需要一个字段，就接管模型激活、
   预测发布、观测行布局、GF 或缓存失效；通用显示工具进入 ecat_viz，领域语义留在 CSI/eqtools。
2. **优先放入已有职责子包。** CSI 内部数据图放 `_plotting`；eqtools 的图件产品和纯结果报告
   放 `csiExtend/reporting`；断层领域诊断放 `viztools`；reader、降采样、配置分别放已有对应子包。
   共享迹线、边界、倾角、patch 查询和网格后端放 `csiExtend/geometry`；候选状态管理仍放
   `perturbations` 和原 owning fault 类。
   观测行布局、预测块、改正系数解释、精确协方差度量放 `csiExtend/observations`；约束声明、
   编译与有效状态放 `csiExtend/constraints`；纯改正结果报告放 `reporting`。
   readers、确定性预处理、协方差估计、时间序列、物理加载模型和求解器保留各自拥有者。
   不为每一种图新建子包，也不建立统一的 `helpers` 或 `internal` 杂物目录。
3. **根目录保留稳定对象和主要入口。** 新增支撑代码不自动与反演入口并列。单次使用、很短且
   紧贴一个实现的函数可以留在同一模块；形成独立责任或被多处复用后，再拆出命名明确的文件。
4. **依赖直接指向拥有者。** 包内实现不能经过旧兼容模块绕回实现，也不能让基础对象反向依赖
   结果产品。新增文件不默认生成旧路径壳层；明确保留的兼容入口只转出对象，不复制函数、
   状态、注册表或默认值。
5. **模块开头说明输入、职责和状态边界。** 用户入口提供参数与返回值说明；新文件在本页的
   实现位置或阅读路线中可发现。职责子包的 README 提供短文件索引和阅读顺序，完整合同
   链接到公开手册，不复制两套参考页。不要以文件数量或行数阈值机械拆分模块。
6. **迁移同时检查使用和发布。** 检查公开调用、序列化路径和相关测试，更新文档影响映射，
   验证 wheel 包含新子包；ECAT 同步运行源码和公开文档，保留现有发布排除规则。

反演类、CSI 数据对象、主要 fault 对象和加载模型保留现有位置。支撑职责使用浅层子包，
不改变参考冻结、候选物化、mesh 发布和缓存生命周期；同名前缀不是合并算法的充分依据。

## 主要实现位置

| 功能 | 代码位置 |
| --- | --- |
| SAR 读取 | [eqtools/csiExtend/sarUtils](https://github.com/kefuhe/eqtools/tree/main/eqtools/csiExtend/sarUtils) |
| 降采样 CLI | [eqtools/cli_tools/process_data_downsampling.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/cli_tools/process_data_downsampling.py) |
| 降采样配置校验 | [eqtools/csiExtend/downsample/config.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/downsample/config.py) |
| 断层迹线数值、marker、处理对象与文件 I/O | [geometry/trace_ops.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/geometry/trace_ops.py)、[trace_markers.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/geometry/trace_markers.py)、[trace_processing.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/geometry/trace_processing.py)、[trace_io.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/geometry/trace_io.py) |
| 倾角、边界、patch 查询与网格支撑 | [geometry](https://github.com/kefuhe/eqtools/tree/main/eqtools/csiExtend/geometry)，[模块与依赖导航](geometry_support.md) |
| 非线性几何反演（新项目入口） | [nonlinear_geometry_smc.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/nonlinear_geometry_smc.py)，`NonlinearGeometrySMCInversion` |
| 非线性几何反演（旧案例入口） | [exploremultifaults_smc.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/exploremultifaults_smc.py) |
| Bayesian 联合几何-滑动反演 | [eqtools/csiExtend/bayesian_multifaults_inversion.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/bayesian_multifaults_inversion.py) |
| 可扰动三角断层几何 | [eqtools/csiExtend/BayesianAdaptiveTriangularPatches.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/BayesianAdaptiveTriangularPatches.py) |
| BLSE/VCE | [eqtools/csiExtend/blse_multifaults_inversion.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/blse_multifaults_inversion.py) |
| 线性求解器 | [eqtools/csiExtend/multifaultsolve_boundLSE.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/multifaultsolve_boundLSE.py) |
| 观测行布局、预测、改正与精确白化 | `eqtools/csiExtend/observations`；[模块与依赖导航](observation_constraint_support.md) |
| 约束共享状态和 registry | [constraints/constraint_manager_base.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/constraints/constraint_manager_base.py) |
| BLSE/VCE 与 SMC 约束编译器 | [constraint_manager_blse.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/constraints/constraint_manager_blse.py), [constraint_manager_smc.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/constraints/constraint_manager_smc.py) |
| VCE 算法 | [simple_vce.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/simple_vce.py), [rigorous_vce.py](https://github.com/kefuhe/eqtools/blob/main/eqtools/csiExtend/rigorous_vce.py) |
| 通用科研绘图 | `ecat-viz/src/ecat_viz`；独立分发，CSI/eqtools 直接依赖 |
| 图件产品、结果统计与报告 | [eqtools/csiExtend/reporting](https://github.com/kefuhe/eqtools/tree/main/eqtools/csiExtend/reporting)；结果模块直接导入 reporting 拥有者；八个原根层壳已移除，见本页迁移表 |
| 领域图件与兼容入口 | [eqtools/viztools](https://github.com/kefuhe/eqtools/tree/main/eqtools/viztools) |

## 绘图依赖与兼容边界

通用样式、字体、formatter、figure、raster 和 CPT 的实现及状态只属于 `ecat_viz`。
CSI 与 eqtools 的普通运行模块直接导入这一公开入口；领域图件模块也直接消费这些
通用工具。断层边界、倾角和滑动诊断继续由 `eqtools.viztools` 持有。

旧 `eqtools.plottools`、`eqtools.getcpt.get_cpt`、`eqtools.viztools` 通用导出及
`eqtools.viztools.raster` 服务于已有用户脚本。包内普通模块不能把这些兼容入口当作
基础依赖；新通用功能只在 `ecat_viz` 发布。实现模块不再通过下划线 alias 转包。
导入方向与兼容范围由依赖契约测试检查，公共入口合同见
[绘图参考](../reference/viztools.md#兼容入口)。

## 文档边界

[ECAT 仓库 `docs/`](https://github.com/kefuhe/ECAT/tree/main/docs) 是用户手册，目标读者是科研用户和新学生。它应该稳定、有顺序、可学习。

公开手册应尽量自洽，不依赖未整理的本地笔记或未随手册发布的材料。需要引用实现细节时，优先链接到公开源码、案例脚本或本手册 reference 页面。

## 手册目录职责

| 目录 | 用户看到的入口 | 职责 |
| --- | --- | --- |
| `getting_started/` | 入门 / Getting Started | 安装、环境检查和最小可运行路线 |
| `concepts/` | 概念 / Concepts | 解释跨页面复用的核心概念和对象关系 |
| `workflows/` | 工作流 / Workflows | 按科研处理步骤说明输入、输出、命令和下一步 |
| `examples/` | 短例 / Examples | 给短小可复制的任务代码或命令，不承载完整案例 |
| `casebook/` | 案例 / Casebook | 将工作流映射到 ECAT-Cases 中的真实脚本、数据和输出 |
| `reference/` | 参考 / Reference | 解释 CLI、配置字段、reader、降采样、约束和反演细节 |
| `developer/` | 公开贡献说明 / Developer Notes | 说明公开文档组织、贡献规则和验证要求 |

导航标题应保留中文解释，同时显式写出英文目录名或英文功能名，便于用户在 GitHub 目录、文档页和案例仓库之间对应。

`examples/` 是短任务层，面向用户复制局部代码。它应使用通用占位文件名，不引用本地绝对路径或未公开脚本；真实事件、完整目录和输出对照放在 `casebook/`。

`concepts/` 是解释层，只放跨页面反复出现、会影响用户判断的概念。不要把 API 字段字典或完整配置块放进 concepts。

`reference/` 是查阅层，应按用户科研路线组织，而不是按源码模块或字母顺序组织。推荐分组顺序为：

1. 基础入口：CLI。
2. 数据准备：SAR Reader、Downsampling App。
3. 标准两阶段反演：非线性几何配置、线性滑动配置、Sigmas/Alpha、约束管理器、BLSE/VCE。
4. 高级 Bayesian：联合反演、可扰动断层几何。
5. 通用工具：Viztools。

长 reference 页不需要压缩完整字段，但开头必须给出“阅读路径”，帮助用户判断先读哪一段。

## 编排原则与外部参考

本手册采用“学习、执行、理解、查阅”分层，和
[Diátaxis](https://diataxis.fr/) 对 tutorial、how-to、explanation、reference 的职责分离一致，但
按地震大地测量科研使用习惯映射到本项目目录：

| 用户意图 | ECAT 文档层 |
| --- | --- |
| 第一次按顺序学习 | `getting_started/` |
| 完成一项科研任务 | `workflows/`，短代码放 `examples/` |
| 理解对象、物理量和运行关系 | `concepts/` |
| 精确查字段、API 和边界条件 | `reference/` |

`casebook/` 是科研软件额外需要的“真实事件验证层”，不替代通用 workflow；`developer/` 只服务
维护与同步。成熟科学 Python 文档也常把 Getting Started/User Guide 与 API Reference 分开，
例如 [Xarray 文档](https://docs.xarray.dev/en/stable/index.html)；API 页应像
[PyGMT API Reference](https://www.pygmt.org/dev/api/index.html) 一样保持完整、可检索，而不是承担
入门叙事。

新增内容时先判断用户意图，再决定落点。一个功能通常只需要“一篇 workflow + 一个短例 + 一篇
reference 的相关小节”，不要为每个函数建立独立学习路线，也不要把完整参数表重复到 workflow。

同一短代码块可以在 workflow 与 example 之间少量重复，前提是它就是任务交接所需的最小
片段；完整配置、长 Python 流程或运行脚本应有单一规范入口，其他页面只解释本层判断并链接。
只有稳定、可复用且同时服务多条工作流的产品契约才值得建立跨工作流页面；单个项目的任务
编排、目录习惯或运行队列不进入公共手册。

“完整脚本模板”和“公开验证案例”必须分开命名：模板提供可修改结构，可以要求用户替换
数据和占位值；casebook 页面只有在 ECAT-Cases 已公开脚本、数据入口、运行命令、预期输出
和已知限制时才称为公开案例。`--check-only` 等运行边界也应按真实代码说明它是否仍读取数据、
构建对象或写出文件。

高级联合 Bayesian 几何-滑动反演应放在 Advanced Workflows 中，不应归入入门两步走的非线性几何页面。`geometry_perturbation` 只解释联合 Bayesian 中的可扰动断层几何，不作为普通断层几何预处理教程。

## 约束内容怎样分层

约束文档既要帮助用户完成配置，也要保持公开接口边界清楚：

| 层级 | 规范入口 | 应包含 | 不应包含 |
| --- | --- | --- | --- |
| 短例 | [Constraint Runtime Example](../examples/constraint_config_runtime.md) | 可复制的配置基线、会话修改、patch/rake 调整和求解前检查 | 完整字段字典、求解器实现 |
| 工作流 | [BLSE/VCE](../workflows/04_linear_slip_blse_vce.md) 与 [联合 Bayesian](../workflows/05_joint_bayesian_geometry_slip.md) | 何时设置约束、运行前检查、失败后下一步 | 内部状态和编译过程 |
| 参考 | [Constraint Manager](../reference/constraint_manager.md), [Rake Constraints](../reference/rake_constraints.md) | 配置字段、公开方法、覆盖顺序、公式、快照和用户可见诊断 | 非公开存储结构、内部重建方法 |

推荐阅读顺序是“workflow 选择任务 → example 复制最小搭配 → reference 查完整语义”。
公开源码链接可以帮助贡献者核对实现，但源码中的辅助对象不会因此成为用户 API。

## 反演输入装配的文档所有权

fault 和 geodata 是非线性与线性反演的共同交接对象，应按同一三层规则维护：

| 层级 | 负责回答 | 当前规范入口 |
| --- | --- | --- |
| workflow | 我手里是哪类输入，下一步走哪条路线 | 数据读取、降采样、迹线预处理、非线性几何、BLSE/VCE 页面 |
| example | 最短可复制代码怎么写 | `inversion_data_loading.md`、`fault_trace_processing.md`、`fault_from_nonlinear_geometry.md`、`fault_trace_preprocessing.md` |
| reference | 列格式、坐标、marker、自动识别边界和参数契约是什么 | `observation_data_readers.md`、`fault_trace_processing.md`、`fault_geometry_construction.md` |

同一段完整代码不要同时在多篇 workflow 中复制维护。workflow 保留路线表和关键骨架，example
保存可复制代码，reference 保存完整模式和边界条件。降采样 workflow 的末尾可以保留一次最短
“读回反演”代码，因为它属于该任务的直接输出交接点。

降采样空间支撑的所有权属于 CSI 数据对象。`insar` 与 `opticorr` 通过只读
`corner_mode` 发布 `triangle`、`legacy_rectangle`、`quadrilateral` 或 `None`；reader、
writer 和 eqtools 模板都消费这一个分类结果。eqtools 不再根据 `corner.shape` 建立平行规则，
也不把非法 corner 静默当作点数据。`triangular=True/False` 只保留为旧调用的类型断言。

案例证据只有在脚本、数据入口和必要说明已经公开时才能写入用户手册。公开短例一律使用
通用文件名和占位数值；只有稳定、可访问且具有明确教学目标的完整案例才进入 `casebook/`。

公开接口或层级语义变化时，应同步更新 workflow、example 或 reference 中真正受影响的页面，
并用实现、模板和测试核对。只有实现细节变化且用户行为不变时，不需要为了展示源码改动而
扩写用户手册。

Viztools 也遵守同一阅读顺序：普通用户从
[科研绘图短例](../examples/viztools_scientific_figures.md) 进入，再到
[Viztools reference](../reference/viztools.md) 和
[Figure Products](../reference/figure_products.md) 查完整语义；内部参数传递和兼容实现不进入
用户必读路径。

## 手册当前覆盖

- InSAR/GPS 数据读取
- InSAR 降采样
- 核心概念和任务短例
- Bayesian 非线性几何反演
- BLSE/VCE 线性滑动分布反演
- Bayesian 联合几何-滑动分布反演
- 图件样式与出版尺寸


## Observation subsets and plotting ownership

CSI owns station selection, covariance row indexing and cached predictions on each data object.
Source GF and solver state are rebuilt by their owners after observation changes. ecat_viz owns
one-time style/config initialization, validated inch-based widths and affine raster drawing.
eqtools compatibility entries share these objects; they do not own a second registry.
CSI alone declares the installed ecat-psgrn entry point; eqtools keeps a delegating module.

## 网格细分与事件后分析

`TriangularPatches.subdivided_copy()` -> `csi/_mesh/subdivision.py`：CSI 生成共享中点
和输入面片映射，在新普通断层上一次发布完整网格；不复制原反演状态。
`statUtils/earthquakesProj2Fault.py` 消费这一接口，把来源映射保存为既有
`receiver_dense.original_indices`；事件归属及计数留在 eqtools，计数不写入滑移。
目录内 README 是文件功能导航。新数组级 mesh 核归 CSI `_mesh`；事件分析不能接管
底层连接关系或边界缓存。接口和迁移见[地震投影参考](../reference/seismic_projection.md)。
