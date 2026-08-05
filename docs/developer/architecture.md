# 文档架构说明

用户手册按科研工作流组织，源码按可复用组件组织。两者不要混在一起。

## 主要实现位置

| 功能 | 代码位置 |
| --- | --- |
| SAR 读取 | [eqtools/csiExtend/sarUtils](https://github.com/kefuhe/ECAT/tree/main/eqtools/csiExtend/sarUtils) |
| 降采样 CLI | [eqtools/cli_tools/process_data_downsampling.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/cli_tools/process_data_downsampling.py) |
| 降采样配置校验 | [eqtools/csiExtend/downsample/config.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/downsample/config.py) |
| 非线性几何反演 | [eqtools/csiExtend/exploremultifaults_smc.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/exploremultifaults_smc.py) |
| Bayesian 联合几何-滑动反演 | [eqtools/csiExtend/bayesian_multifaults_inversion.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/bayesian_multifaults_inversion.py) |
| 可扰动三角断层几何 | [eqtools/csiExtend/BayesianAdaptiveTriangularPatches.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/BayesianAdaptiveTriangularPatches.py) |
| BLSE/VCE | [eqtools/csiExtend/blse_multifaults_inversion.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/blse_multifaults_inversion.py) |
| 线性求解器 | [eqtools/csiExtend/multifaultsolve_boundLSE.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/multifaultsolve_boundLSE.py) |
| 约束共享状态和 registry | [eqtools/csiExtend/constraint_manager_base.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/constraint_manager_base.py) |
| BLSE/VCE 与 SMC 约束编译器 | [constraint_manager_blse.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/constraint_manager_blse.py), [constraint_manager_smc.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/constraint_manager_smc.py) |
| VCE 算法 | [simple_vce.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/simple_vce.py), [rigorous_vce.py](https://github.com/kefuhe/ECAT/blob/main/eqtools/csiExtend/rigorous_vce.py) |
| 图件样式工具 | [eqtools/viztools](https://github.com/kefuhe/ECAT/tree/main/eqtools/viztools) |

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
| `developer/` | 维护说明 / Developer Notes | 说明文档组织、维护规则和面向维护者的边界 |

导航标题应保留中文解释，同时显式写出英文目录名或英文功能名，便于用户在 GitHub 目录、文档页和案例仓库之间对应。

`examples/` 是短任务层，面向用户复制局部代码。它应使用通用占位文件名，不引用本地绝对路径或未公开脚本；真实事件、完整目录和输出对照放在 `casebook/`。

`concepts/` 是解释层，只放跨页面反复出现、会影响用户判断的概念。不要把 API 字段字典或完整配置块放进 concepts。

`reference/` 是查阅层，应按用户科研路线组织，而不是按源码模块或字母顺序组织。推荐分组顺序为：

1. 基础入口：CLI。
2. 数据准备：SAR Reader、Downsampling App。
3. 标准两步反演：非线性几何配置、线性滑动配置、Sigmas/Alpha、约束管理器、BLSE/VCE。
4. 高级 Bayesian：联合反演、可扰动断层几何。
5. 通用工具：Viztools。

长 reference 页不需要压缩完整字段，但开头必须给出“阅读路径”，帮助用户判断先读哪一段。

高级联合 Bayesian 几何-滑动反演应放在 Advanced Workflows 中，不应归入入门两步走的非线性几何页面。`geometry_perturbation` 只解释联合 Bayesian 中的可扰动断层几何，不作为普通断层几何预处理教程。

## 约束文档的双层边界

约束文档同时服务科研用户和代码维护者，但两类内容不能混写：

| 层级 | 规范入口 | 应包含 | 不应包含 |
| --- | --- | --- | --- |
| 公开短例层 | [Constraint Runtime Example](../examples/constraint_config_runtime.md) | 可复制的配置基线、会话修改、patch/rake 调整和求解前检查 | 完整字段字典、私有实现 |
| 公开用户层 | [Constraint Manager](../reference/constraint_manager.md), [Rake Constraints](../reference/rake_constraints.md) | 配置字段、稳定 facade、`update/add/replace/set/clear` 语义、config/runtime 生命周期、公式和诊断 | 私有 registry、`owner/family` 写入、内部重建方法 |
| 内部维护层 | [Constraint Manager Developer Guide](../../eqtools/csiExtend/docs/CONSTRAINT_MANAGER_DEVELOPER_GUIDE.md) | 声明态与解析态、owner/family、CRUD 和 reconciliation、事务、参数布局、solver handoff、扩展清单 | 重复维护面向用户的完整配置教程 |
| 追溯层 | `eqtools/csiExtend/docs/` 中的 plan、audit、changelog | 设计原因、迁移记录、验证证据 | 作为当前公开接口的唯一依据 |

推荐阅读顺序是“workflow 选择任务 → example 复制最小搭配 → reference 查完整语义”。
内部维护指南不进入普通用户的必读路径。

公开接口或层级语义变化时，应同时更新公开 reference 和内部 developer guide。
只有内部实现细节变化且用户行为不变时，只更新 developer guide 和相应测试/变更记录。

Viztools 同样遵守双层边界：普通用户从 [科研绘图短例](../examples/viztools_scientific_figures.md) 进入，再到 [Viztools reference](../reference/viztools.md) 和 [Figure Products](../reference/figure_products.md) 查完整语义；代码层级、兼容入口、参数所有权和扩展模板只维护在 [Viztools Developer Guide](../../eqtools/viztools/docs/VIZTOOLS_DEVELOPER_GUIDE.md)。

## 手册当前覆盖

- InSAR/GPS 数据读取
- InSAR 降采样
- 核心概念和任务短例
- Bayesian 非线性几何反演
- BLSE/VCE 线性滑动分布反演
- Bayesian 联合几何-滑动分布反演
- 图件样式与出版尺寸
