# Viztools 内部文档导航

本目录只面向 ECAT/eqtools 维护者。普通用户请阅读：

- `docs/examples/viztools_scientific_figures.md`：最短可复制场景；
- `docs/reference/viztools.md`：样式、字体、尺寸、栅格和兼容入口；
- `docs/reference/figure_products.md`：反演结果的批量图件入口。

内部文档只保留两份长期资料：

| 文档 | 用途 |
| --- | --- |
| [VIZTOOLS_DEVELOPER_GUIDE.md](VIZTOOLS_DEVELOPER_GUIDE.md) | 代码层级、契约、参数职责和扩展模板 |
| [VIZTOOLS_VALIDATION_REPORT.md](VIZTOOLS_VALIDATION_REPORT.md) | 本轮兼容迁移、缺陷修复和验证证据 |

不在本目录复制完整用户 API。公开行为改变时，应同时更新 public reference、契约测试和本指南；纯内部重构只更新本指南与测试。
