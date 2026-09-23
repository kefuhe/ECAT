# ECAT 可运行脚本模板

本目录放公开、可编辑的起始脚本。把所需模板复制到案例目录，与相应 YAML 配置放在一起，修改占位路径和几何参数后再运行。第一次选择脚本先看 [可运行脚本模板导航](../docs/examples/script_templates.md)。

推荐模板的 `Editing guide` 给出首次修改项和路径基准。沿分段标题依次修改数据、几何、
求解与结果；启用可选数据块时，还需更新数据列表和对应 YAML 设置。具体关联修改见
[复制后从哪里修改](../docs/examples/script_templates.md#editing-template)。

## MPI 启动模板

| 平台 | 启动器 | 使用说明 |
| --- | --- | --- |
| Windows PowerShell | [`run_ecat_mpi_windows.ps1`](run_ecat_mpi_windows.ps1) | [Windows 与 WSL 的 MPI 启动脚本](../docs/examples/mpi_launcher_scripts.md) |
| WSL/Linux Bash | [`run_ecat_mpi_wsl.sh`](run_ecat_mpi_wsl.sh) | [Windows 与 WSL 的 MPI 启动脚本](../docs/examples/mpi_launcher_scripts.md) |

启动器只管理 MPI rank、每 rank 数值线程、案例工作目录和无窗口绘图环境，不拥有
反演配置或科学参数。复制到案例目录后修改顶部默认值，或在运行命令中覆盖；先激活
Conda 环境，不能用启动器代替 MPI/mpi4py 配套检查。

Windows 下 Python 脚本自己的 `-r` 等选项必须写成
`-ScriptArguments "-r"`；裸写 `-r` 会被 PowerShell 识别为启动器参数 `-Ranks` 的缩写。
完整的单行、多行和续算示例见上表链接的启动说明。

## 常用完整模板

| 任务 | 模板 | 用户文档 |
| --- | --- | --- |
| 新版 Bayesian 非线性几何反演 | [`test_nonlinear_geometry_smc.py`](test_nonlinear_geometry_smc.py) | [非线性几何工作流](../docs/workflows/03_nonlinear_geometry_bayesian.md) |
| 复现 legacy `explorefault` 案例 | [`test_nonlinear_bayesian.py`](test_nonlinear_bayesian.py) | [非线性几何配置](../docs/reference/config_nonlinear_geometry.md) |
| 联合 Bayesian：单一底边位移 | [`test_joint_bayesian_bottom_offset.py`](test_joint_bayesian_bottom_offset.py) | [联合 Bayesian 工作流](../docs/workflows/05_joint_bayesian_geometry_slip.md) |
| 联合 Bayesian：多个倾角控制点示例（模板使用 3 点） | [`test_joint_bayesian_three_dip_controls.py`](test_joint_bayesian_three_dip_controls.py) | [联合几何设置短例](../docs/examples/joint_bayesian_geometry_setup.md) |
| 联合 Bayesian：组合扰动示例（当前方法使用 4 个参数） | [`test_joint_bayesian_custom_perturbation.py`](test_joint_bayesian_custom_perturbation.py) | [可扰动断层几何参考](../docs/reference/geometry_perturbation.md) |
| 单次固定权重 BLSE 线性滑动反演 | [`test_slip_inv_BLSE.py`](test_slip_inv_BLSE.py) 的 `--mode single` | [BLSE/VCE 工作流](../docs/workflows/04_linear_slip_blse_vce.md) |
| 单次 VCE 线性滑动反演 | [`test_slip_inv_VCE.py`](test_slip_inv_VCE.py) | [BLSE/VCE 工作流](../docs/workflows/04_linear_slip_blse_vce.md) |
| 紧凑 smoothing scan | [`test_slip_inv_BLSE.py`](test_slip_inv_BLSE.py) 的 `--mode loop` | [BLSE/VCE 参考](../docs/reference/blse_vce.md#smoothing-scan) |
| 固定几何 BLSE L-curve | [`test_BLSE_L_Curve.py`](test_BLSE_L_Curve.py) | [L-curve 工作流](../docs/workflows/04a_blse_l_curve.md) |
| 固定拓扑倾角搜索 | [`test_dip_search_BLSE.py`](test_dip_search_BLSE.py) | [倾角搜索工作流](../docs/workflows/04b_blse_dip_search.md) |
| 倾角 × 平滑敏感性 | [`test_dip_smoothing_search_BLSE.py`](test_dip_smoothing_search_BLSE.py) | [联合敏感性工作流](../docs/workflows/04c_blse_dip_smoothing_search.md) |
| 地表 ENU 位移正演 | [`test_surface_displacement_forward.py`](test_surface_displacement_forward.py) | [正演短例](../docs/examples/surface_forward_grid.md) |
| SAR LOS 正演与 GeoTIFF 输出 | [`test_sar_los_surface_forward.py`](test_sar_los_surface_forward.py) | [地表位移正演参考](../docs/reference/surface_displacement_forward.md) |
| BLSE 棋盘格分辨率检查 | [`test_BLSE_Inv_Checkboard.py`](test_BLSE_Inv_Checkboard.py) | [BLSE/VCE 工作流](../docs/workflows/04_linear_slip_blse_vce.md) |

普通 BLSE、平滑搜索、倾角搜索和联合敏感性各自保留独立模板，便于按科研场景演进。共同的配置、约束和拟合统计语义统一放在 BLSE/reference，不在每个脚本中重复定义。

BLSE、VCE、L-curve、倾角搜索和 checkerboard 都是单 Python 进程模板，直接用
`python <script>.py` 运行，不需要 `mpiexec`、rank 判断或 MPI 输出门控。底层数值库仍可按
环境设置使用线程；MPI 启动器只用于 SMC/Bayesian 模板，因此项目仍保留 `mpi4py` 依赖。

`test_slip_inv_BLSE.py --mode single` 与 `test_slip_inv_VCE.py` 按顺序生成标准滑动图、fault/slip 文件和
GPS/InSAR/opticorr 的 data/synth/resid 文本。有 corner 的 raster 默认写多边形；没有
corner 的点输入直接写点表。`--export-point-values` 只为有 corner 的数据在
`Modeling/points/` 增加中心点表。`--no-plot` 只关闭图件，不跳过最终模型
分发和文本导出。VCE 模板只执行一次 `run_simple_vce()`，不提供 loop mode。BLSE/VCE
都是单一最终解，没有联合 Bayesian 的 posterior 滑动标准差图。

单次 BLSE/VCE 模板依赖 `verbose=True` 时的默认 compact 尺度报告，不再额外调用第二个
打印接口。如需在求解后或交互检查时单独重打当前 sigma/alpha，调用
`inversion.print_scale_parameters()`；该调用只读取已冻结的活动尺度，不会重新求解。
VCE 模板使用独立的 `default_config_VCE.yml`，并在 `run_simple_vce()` 中保留一行可选
`qp_acceleration='certified_kkt'` 注释；不取消注释时仍使用默认可信 QP 路径。

`test_BLSE_L_Curve.py` 的固定几何可以由一个或多个已经准备好的断层组成；模板通过
`scan_penalty_weights()` 扫描
一个共享的 scalar penalty weight，并在主流程中逐轮收集全局和逐数据集统计。绘图单独由
纯展示函数完成，同时输出三联诊断图和单幅 roughness–RMS L-curve；CSV 不写只适用于单断层
或单倾角的说明字段。GPS 可以与 InSAR 一起放入
`geodata`，但观测、协方差和断层几何必须在整个扫描中保持不变。
通用 BLSE 模板的 `--mode loop` 也直接调用同一规范扫描入口，并输出摘要、逐数据集长表和
三联图；旧 `simple_run_loop()` 的迁移示例集中在 [BLSE/VCE 参考](../docs/reference/blse_vce.md#smoothing-scan)。更完整的绘图选项和说明仍
集中在独立 L-curve 模板中。

`test_BLSE_Inv_Checkboard.py` 是串行 InSAR/GPS/optical 棋盘格模板：先生成 truth，再正演、
加噪、替换观测并运行 BLSE。GPS 按配置和输入有效分量自动使用 EN 或 ENU；optical 使用
east/north，且对应 `verticals` 必须为 `false`。两者都可用标量表示活动分量的共同标准差，
也可分别使用 `east/north/up` 或 `east/north` 映射，并在同一入口重建对应对角协方差。
脚本不用 MPI 外壳，也不在模板层复制 synthetic 注入逻辑。兼容区中的两个旧 checkerboard
变体也已去掉无效的 MPI/rank 外壳，但仍保留旧脚本组织，仅用于迁移参考。

当前推荐模板直接用 `faults_list = [fault]` 声明一个或多个 source；列表顺序就是 source
与参数块顺序，source 对象的 `name` 与配置名称对应。无需为此先建立 `OrderedDict` 再转回
列表；兼容区中的历史 checkerboard 变体保留原有写法，不作为新项目示例。

## 联合 Bayesian 模板

三份联合模板分别展示标量底边位移、三个倾角控制点和一个四参数组合扰动实例。控制点数量
与采样参数个数由所选扰动方法决定，不是联合框架的固定要求。`lon0/lat0` 由数据和断层共同
使用，应保持同一来源；geodata 顺序、reference、mesh 和配置 source 名也必须彼此一致。
配套配置位于
[`configs/joint_bayesian/`](configs/joint_bayesian/)，复制时让 Python、主配置和 bounds
保持成套：

| 场景 | 主配置 | Bounds |
| --- | --- | --- |
| 标量底边位移示例 | [`bottom_offset.yml`](configs/joint_bayesian/bottom_offset.yml) | [`bottom_offset_bounds.yml`](configs/joint_bayesian/bottom_offset_bounds.yml) |
| 多倾角控制点示例（模板使用 3 点） | [`three_dip_controls.yml`](configs/joint_bayesian/three_dip_controls.yml) | [`three_dip_controls_bounds.yml`](configs/joint_bayesian/three_dip_controls_bounds.yml) |
| 组合扰动示例（当前方法使用 4 个参数） | [`custom_perturbation.yml`](configs/joint_bayesian/custom_perturbation.yml) | [`custom_perturbation_bounds.yml`](configs/joint_bayesian/custom_perturbation_bounds.yml) |

三份模板默认绘制 median 代表滑动；只有显式传入 `--plot-std` 才额外求解并绘制 posterior
滑动分量离散度。该统计在 SMC-FJ 中需要对已接受样本重求条件线性解，适合结果确定后按需
生成。完成后会恢复 median 代表模型，再生成几何改正、geometry/sigma/alpha KDE、拟合统计
和 fault/slip 文件。`output/` 保存断层、统计与后验产品，`Modeling/` 保存数据拟合图以及
raster 的 data/synth/resid 文本：有 corner 时默认写多边形，没有 corner 时默认写点表。
`--export-point-values` 只为有 corner 的数据在 `Modeling/points/` 额外写中心点表。
`--no-plot` 只跳过图件，不跳过代表模型回填和文本导出。
STD patch/center GMT 属于高级、按需产品，公共模板不再复制临时状态切换；个人案例需要时
按[联合 Bayesian 参考](../docs/reference/bayesian_joint_inversion.md#标准结果入口与脚本层导出)
中的 `try/finally` 示例增加独立开关。
脚本末尾另给可选的定制滑动图调用，用户可直接修改视角、范围和色标。

路径处理统一使用 Python `pathlib`，不绑定 Windows 盘符或 POSIX 绝对路径。当前公开支持的
平台和环境要求以[安装说明](../docs/getting_started/installation.md)为准；若环境只提供
`python3` 或 `mpirun`，替换相应可执行文件名即可。

## 兼容入口和历史变体

| 文件 | 定位 |
| --- | --- |
| [`process_data_downsampling.py`](process_data_downsampling.py) | 从源码树直接调用降采样 CLI 的薄入口；已安装 ECAT 时优先使用 `ecat-downsample`。 |
| [`test_BLSE_Inv_Checkboard_simple.py`](test_BLSE_Inv_Checkboard_simple.py) | 既有 checkerboard 简化变体，用于复现对应旧脚本组织。 |
| [`test_BLSE_Inv_Checkboard_general.py`](test_BLSE_Inv_Checkboard_general.py) | 既有 checkerboard general 变体，不作为新用户默认入口。 |

兼容脚本不会被隐藏，但也不与当前推荐模板混在同一学习路径中。复制前先确认它使用的配置类、数据布局和输出接口是否与当前项目一致。
