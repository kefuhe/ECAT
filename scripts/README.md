# ECAT 可运行脚本模板

本目录放公开、可编辑的起始脚本。把所需模板复制到案例目录，与相应 YAML 配置放在一起，修改占位路径和几何参数后再运行。第一次选择脚本先看 [可运行脚本模板导航](../docs/examples/script_templates.md)。

## 常用完整模板

| 任务 | 模板 | 用户文档 |
| --- | --- | --- |
| 新版 Bayesian 非线性几何反演 | [`test_nonlinear_geometry_smc.py`](test_nonlinear_geometry_smc.py) | [非线性几何工作流](../docs/workflows/03_nonlinear_geometry_bayesian.md) |
| 复现 legacy `explorefault` 案例 | [`test_nonlinear_bayesian.py`](test_nonlinear_bayesian.py) | [非线性几何配置](../docs/reference/config_nonlinear_geometry.md) |
| 单次 BLSE/VCE 线性滑动反演 | [`test_slip_inv_BLSE.py`](test_slip_inv_BLSE.py) 的 `--mode single` | [BLSE/VCE 工作流](../docs/workflows/04_linear_slip_blse_vce.md) |
| 既有传统 smoothing loop | [`test_slip_inv_BLSE.py`](test_slip_inv_BLSE.py) 的 `--mode loop` | [BLSE/VCE 参考](../docs/reference/blse_vce.md#smoothing-loop) |
| 固定几何平滑搜索 | [`test_smoothing_search_BLSE.py`](test_smoothing_search_BLSE.py) | [平滑搜索工作流](../docs/workflows/04a_blse_smoothing_search.md) |
| 固定拓扑倾角搜索 | [`test_dip_search_BLSE.py`](test_dip_search_BLSE.py) | [倾角搜索工作流](../docs/workflows/04b_blse_dip_search.md) |
| 倾角 × 平滑敏感性 | [`test_dip_smoothing_search_BLSE.py`](test_dip_smoothing_search_BLSE.py) | [联合敏感性工作流](../docs/workflows/04c_blse_dip_smoothing_search.md) |
| 地表 ENU 位移正演 | [`test_surface_displacement_forward.py`](test_surface_displacement_forward.py) | [正演短例](../docs/examples/surface_forward_grid.md) |
| SAR LOS 正演与 GeoTIFF 输出 | [`test_sar_los_surface_forward.py`](test_sar_los_surface_forward.py) | [地表位移正演参考](../docs/reference/surface_displacement_forward.md) |
| BLSE 棋盘格分辨率检查 | [`test_BLSE_Inv_Checkboard.py`](test_BLSE_Inv_Checkboard.py) | [BLSE/VCE 工作流](../docs/workflows/04_linear_slip_blse_vce.md) |

普通 BLSE、平滑搜索、倾角搜索和联合敏感性各自保留独立模板，便于按科研场景演进。共同的配置、约束和拟合统计语义统一放在 BLSE/reference，不在每个脚本中重复定义。

## 兼容入口和历史变体

| 文件 | 定位 |
| --- | --- |
| [`process_data_downsampling.py`](process_data_downsampling.py) | 从源码树直接调用降采样 CLI 的薄入口；已安装 ECAT 时优先使用 `ecat-downsample`。 |
| [`test_BLSE_Inv_CovDiag_Checkboard_simple.py`](test_BLSE_Inv_CovDiag_Checkboard_simple.py) | 既有 CovDiag checkerboard 变体，用于复现对应旧脚本组织。 |
| [`test_BLSE_Inv_CovDiag_Checkboard_general.py`](test_BLSE_Inv_CovDiag_Checkboard_general.py) | 既有 general CovDiag checkerboard 变体，不作为新用户默认入口。 |

兼容脚本不会被隐藏，但也不与当前推荐模板混在同一学习路径中。复制前先确认它使用的配置类、数据布局和输出接口是否与当前项目一致。

## 发布维护入口

| 文件 | 定位 |
| --- | --- |
| [`generate_requirements.py`](generate_requirements.py) | 从 CSI 与 eqtools 的直接依赖元数据生成或核查 ECAT 统一环境清单；普通科研用户不需要运行。 |
