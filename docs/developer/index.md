# 开发者说明

本部分说明源码阅读路线、文件归属、公开文档和发布维护规则。

- [源码与文档架构](architecture.md)：三个包的层级、源码阅读路线、新增文件归属和用户手册职责；含[结果模块旧路径迁移](architecture.md#reporting-import-migration)。
- [几何支撑源码与依赖](geometry_support.md)：按任务选择 trace、倾角、边界、patch 查询与网格后端，查阅职责、计算差异和旧导入迁移。
- [观测与约束支撑源码](observation_constraint_support.md)：数据合同、约束状态、改正报告的归属、科学差异与导入迁移。
- [文档贡献规范](contributing_docs.md)：公开页面的内容、链接、案例和验证要求。
- [代码—文档影响评估](documentation_impact.md)：代码修改后怎样路由、判断和验证公开文档影响。
- [依赖与运行环境策略](dependency_environment_policy.md)：兼容范围、默认 Python、
  pip/Conda 边界、BLAS/MPI 策略和发布验证要求。
