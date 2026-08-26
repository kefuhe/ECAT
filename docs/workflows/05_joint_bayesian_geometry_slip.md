# Bayesian 联合几何-滑动分布反演 / Joint Bayesian Geometry-Slip Inversion

本页说明 ECAT 中的高级 Bayesian 联合反演路线。它不是入门两步走的替代品，而是在几何不确定性会显著影响滑动分布、数据覆盖足够、计算预算允许时使用。

入门路线通常是：

```text
Bayesian 非线性几何反演 -> BLSE/VCE 线性滑动分布反演
```

联合路线则把可扰动断层几何、滑动分布、数据权重和多源约束放进同一个 Bayesian 采样框架：

```text
数据对象 + 可扰动断层对象 + Bayesian 配置
  -> BayesianMultiFaultsInversion
  -> SMC_FJ 或 FULLSMC
  -> 几何后验、滑动后验、sigma/alpha 后验和拟合诊断
```

## 适用场景

优先使用两步走路线，除非至少满足以下条件之一：

- 几何扰动会明显改变滑动分布，固定单一优选几何会低估不确定性。
- 需要比较多源、多断层或多事件场景下的几何-滑动耦合。
- 希望在同一后验中报告几何、滑动、sigma/alpha 和约束影响。
- 已经有稳定的两步走结果，可作为联合采样的初值、边界和质量检查基准。

不建议把联合 Bayesian 当作第一个案例。它对几何参数化、网格更新、GF 重建、约束模式和计算成本都更敏感。

## 模式选择

| 模式 | 采样内容 | 滑动处理 | 适合场景 |
| --- | --- | --- | --- |
| `SMC_FJ` | 几何、sigma、alpha 等超参数 | 每个样本内用约束线性求解滑动 | 推荐的高级主线；维度较低，可复用 BLSE 约束体系 |
| `FULLSMC` | 几何、sigma、alpha 和滑动参数 | 滑动也作为采样参数 | 研究滑动先验或小规模模型；维度高，计算成本大 |

`SMC_FJ` 是规范配置值，读入后也作为唯一内部名称保存。旧配置中的
`SMC_F_J` 仍会在配置入口自动归一化为 `SMC_FJ`，但新配置、导出结果和
运行状态不再使用旧拼写。`SMC_FJ` 会使用 `ss_ds` 滑动参数化；线性 rake、
Euler、零滑和自定义矩阵约束只在 `SMC_FJ + ss_ds` 或 BLSE 中形成线性约束
矩阵。完整约束差异见
[ECAT 约束管理器 / Constraint Manager](../reference/constraint_manager.md)。

两种模式复用同一份 bounds 和震间配置。切换到 `FULLSMC` 时，不必维护第二份
YAML；线性子问题专属约束会被标为 inactive，bounds 和适用先验仍按当前参数化
生效。纯断层小模型可使用 `magnitude_rake` 或 `rake_fixed`；包含 Pressure、
Sbarbot 等非断层源时，`FULLSMC` 使用 `ss_ds`。

两种模式也共用同一套候选几何刷新顺序。每个有效候选都更新 GF；刚体平移/旋转可复用
面积与 Laplacian，非刚性变形按真实消费者重建。`alpha` 未参与时不为候选额外计算
Laplacian，magnitude prior 未参与时不额外物化面积。该判断在每个 MPI rank 本地完成，
不会在候选循环中增加 collective communication。`none/rigid/deform/remesh` 的含义和各派生量
刷新规则见[可扰动断层几何：几何变化与派生量刷新](../reference/geometry_perturbation.md#几何变化与派生量刷新)。

`alpha.enabled: false` 只表示不构造和评分 Laplacian 平滑块。`SMC_FJ` 仍会从数据法矩阵
计算消去线性滑动参数所需的 \(-\tfrac12\log|H_d|\) 曲率项；因此无平滑模型必须由数据
充分约束全部线性参数。实现先在无量纲对角平衡坐标中检查数值秩，再把尺度行列式精确
加回，因此参数单位差异不会被当成秩亏。单个无效候选会获得低似然而不复用旧线性解；
只有初始粒子全部无效时，MPI 任务才会一致报错退出。程序不会静默退回另一种 profile
目标。完整推导、平衡公式和约束近似边界见
[Bayesian 联合反演参考：为什么消去线性参数会产生 log-determinant](../reference/bayesian_joint_inversion.md#为什么消去线性参数会产生-log-determinant)。

## 输入和完成标准

联合反演仍使用 CSI/ECAT 标准数据对象。开始前应有：

- 已完成单位、正负号、projection、协方差和降采样检查的 InSAR/GNSS/光学数据。
- 已构建并检查 top/bottom、mesh、patch 点序和下倾侧的断层对象。
- 一份冻结的 `GeometryReference`，以及和它配套的扰动方法、样本位置与 bounds。
- 能在固定参考几何上稳定运行的 GF、Laplacian、滑动约束和拟合诊断。

数据准备见 [InSAR 与 GNSS 数据读取](01_data_reading_insar_gps.md) 和
[InSAR 降采样](02_insar_downsampling.md)。参考几何的状态关系见
[Bayesian 联合反演中的几何参考](../concepts/bayesian_geometry_reference.md)。

## 执行步骤

### 1. 从稳定的基线问题开始

先在固定几何上完成 BLSE/VCE 检查。联合反演不应同时承担“几何是否合理”“mesh 是否
有效”“线性约束是否可行”三个尚未解决的问题。两步走结果还可用于设置扰动中心、边界
和网格尺度。

### 2. 选择参考来源

| 常见场景 | 参考中应固化什么 | 建立方式 |
| --- | --- | --- |
| 迹线+倾角、解析构造或外部 top/bottom 是权威几何 | top/bottom；通常不需要 vertices/layers | 写入或构造坐标后 `snapshot(capture_vertices=False, capture_layers=False)` |
| 导入、裁切或人工修整后的 mesh 是权威几何 | 从当前 mesh 提取并排序的 top/bottom | `set_edges_for_bayesian_optimization()` |
| top 必须严格来自迹线、bottom 来自 mesh | trace top + mesh bottom | 上述入口加 `use_trace=True` |
| 直接变换整个已有 mesh | top/bottom + `Vertices/Faces` | 最终参考 mesh 后 `snapshot(capture_vertices=True, capture_layers=False)` |
| 多层或变化倾角 | layers 或 dip controls + top/bottom | 设置对应字段后按方法需求显式 `snapshot(...)` |

曲线形状本身不要求先生成 mesh。若曲线 top 已由迹线确定、bottom 已由显式倾角或控制点
确定，它属于第一行：直接冻结 top/bottom，然后只生成一次参数化 mesh。只有实际 mesh
边界才是零扰动权威输入时，才需要临时/外部 mesh 和 edge extraction。

完整场景表和 `ref*` 接口见
[可扰动断层几何参考](../reference/geometry_perturbation.md)。曲线断层的可复制片段见
[联合 Bayesian 几何参考与配置短例](../examples/joint_bayesian_geometry_setup.md)。

### 3. 冻结一次参考

采样前明确建立一次 reference，并检查：

```python
assert fault.geometry_ref is not None
fault.geometry_summary()
```

参考定义零扰动几何。同一次 SMC run 中不要再次 `snapshot()`；每个样本应满足
`candidate = transform(reference, delta)`，而不是从上一样本累计变形。

### 4. 从模板生成主配置和 bounds

希望直接修改完整科研脚本时，可先从三个参数化实例中选择。它们不是框架支持范围的固定
清单，控制点数量和采样参数个数由所选扰动方法决定：

| 场景 | 模板 |
| --- | --- |
| 标量底边位移示例 | [`test_joint_bayesian_bottom_offset.py`](../../scripts/test_joint_bayesian_bottom_offset.py) |
| 多个沿走向倾角控制点示例（模板使用 3 点） | [`test_joint_bayesian_three_dip_controls.py`](../../scripts/test_joint_bayesian_three_dip_controls.py) |
| 组合扰动示例（当前方法使用 4 个参数） | [`test_joint_bayesian_custom_perturbation.py`](../../scripts/test_joint_bayesian_custom_perturbation.py) |

它们各有一套位于 [`scripts/configs/joint_bayesian/`](../../scripts/configs/joint_bayesian/)
的主配置和 bounds。新用户可以成套复制；需要从当前版本全部默认字段开始时，使用 CLI：

模板中的 `lon0/lat0` 是数据与断层共享的坐标参考。修改案例时，还要一起核对 geodata 顺序、
迹线与断层物理参数、reference 建立时机和首次构网参数。

```bash
ecat-generate-config -o joint_config.yml --gf-method cutde
ecat-generate-boundary -o joint_bounds.yml -f MainFault
```

保留生成模板中的 geodata、sigma/alpha、GF、Laplacian 和约束配置，再修改联合几何相关
字段。生成器中的 `ExampleFault` 必须删除或改成与 fault object 完全一致的名称。一个
“底边沿固定方向移动、随后单独变形 mesh”的完整激活块是：

```yaml
nonlinear_inversion: true
bayesian_sampling_mode: SMC_FJ
slip_sampling_mode: ss_ds
nchains: 100
chain_length: 50

faults:
  defaults:
    geometry:
      update: false
      sample_positions: [0, 0]
    method_parameters:
      update_GFs:
        method: cutde
        options: {}
      update_Laplacian:
        method: Mudpy
        bounds: [free, locked, free, free]
  MainFault:
    geometry:
      update: true
      sample_positions: [0, 1]
    method_parameters:
      update_fault_geometry:
        method: perturb_bottom_coords_along_fixed_direction
        average_direction: 10.0
        angle_unit: degrees
        perturbation_direction: horizontal
        use_average_strike: false
      update_mesh:
        method: generate_and_deform_mesh
        top_size: 3.0
        bottom_size: 6.0
        num_segments: 25
        disct_z: 10
```

三项缺一不可：顶层 `nonlinear_inversion: true`、断层级 `geometry.update: true`，以及
与方法参数个数匹配的 `sample_positions`。`[0, 1]` 是全局几何采样向量的半开区间，
不是断层节点编号。

这里的“匹配”由方法契约决定，不等于所有方法都有固定长度。固定组合要求精确个数；
`fixed_nodes`、可变数量 dip controls 等动态方法接受一个广播标量，或每个可移动项一个值。
可先运行 `fault.help("方法名")` 查看 reference 和 sample cardinality，再同步修改
`sample_positions` 与 bounds。配置初始化会在启动采样前检查这些关系。

对应 bounds：

```yaml
geometry:
  MainFault: [-15.0, 15.0]
```

本例扰动量是水平距离，单位 km。其他方法可能混合 km 和 degrees，不能仅凭
`geometry` 字段名推断单位。

多个几何参数具有不同边界时，使用显式上下界数组。例如参数顺序为
`[bottom_offset_km, rotation_deg, dx_km, dy_km]`：

```yaml
geometry:
  MainFault:
    lb: [-10.0, -15.0, -5.0, -5.0]
    ub: [10.0, 15.0, 5.0, 5.0]
```

不要写成四行 `[lb, ub]`；当前解析器不把 `4 x 2` 数组解释为逐参数边界。

### 5. 对齐扰动方法和 mesh 职责

方法必须来自当前版本注册表：

```bash
ecat-list-fault-perturb-methods
```

```python
fault.help("perturb_bottom_coords_along_fixed_direction")
```

`perturbations` 由采样器从当前样本自动传入，不写进 YAML。无 mesh 后缀的方法需要独立
`update_mesh`；`_simpleMesh`、`_DeformMesh` 和 `_multiLayerMesh` 方法内部已负责 mesh，
不要再配置第二次重建。

采样前通常用 `generate_and_deform_mesh(remap=True)` 从 frozen reference 对应的同一组
top/bottom 建立参数映射；`bottom_norm_offset` 保持默认 `None`。只要该参数不是 `None`，
当前实现就会在 mesh 方法内部调用一次底边扰动：`0.0` 也会执行该调用，非零值还会让
current bottom/初始 mesh 与 frozen reference 分离。它不是 sampler 初值，也不会作为固定项
自动叠加到后续样本。

如果非零 offset 是正式物理基线的一部分，应先用与采样方法一致的方向、固定节点和单位
显式修改 current bottom，再 `snapshot(...)`，最后以
`bottom_norm_offset=None` 生成参数化 mesh。若只想改变 mesh mapping anchor，则属于需要单独
验证映射范围和离散误差的高级数值策略，不放入标准流程。样本内 `update_mesh` 不能配置
`remap`、`use_current_mesh`、`bottom_norm_offset` 或平滑坐标类选项，因为 mesh 阶段不应
重新定义几何或破坏 patch 对应关系。

创建采样 target 时，`generate_and_deform_mesh` 路径会在每个 MPI rank 本地做一次只读
就绪检查：准备映射必须存在，逐顶点参数坐标、Gmsh mesh 与当前 face rows 必须匹配，且
`num_segments/disct_z` 等实际映射参数必须与 YAML 重放设置一致。这个检查不 remap、不变形、
不重算 GF/GL/area，也不会进入候选循环。简单 mesh、multi-layer mesh 和 whole-mesh 刚性
变换没有该参数映射依赖，因此不会被要求先建立这组状态。

### 6. 创建 inversion 并做轻量校验

```python
inversion.print_parameter_positions()

constraint_state = inversion.get_constraint_snapshot(validate=True)
print(constraint_state["sampling_mode"])
print(constraint_state["inactive_constraints"])
print(constraint_state["validation"])
```

这里的 `constraint_state` 是约束诊断副本，不是断层 `geometry_ref`。还应先对 bounds 的
极值或少量代表样本做 mesh、GF、Laplacian、面积和残差检查，再启动正式 SMC。

初始化阶段的无效 active constraint 会直接报错，并回滚整次约束更新。`SMC_FJ`
运行中若某个样本的受约束线性求解失败，该样本只获得低似然；程序不会移除等式约束后
重算。

### 7. 理解每个样本的执行顺序

状态与职责的图示说明见
[一个样本从配置到似然的完整流转](../concepts/bayesian_geometry_reference.md#一个样本从配置到似然的完整流转)；
下面保留便于运行时排查的顺序清单。

```text
从全局样本向量取得 geometry slice
  -> target 构造时对方法、kwargs、reference、参数个数和所需 mesh replay 状态完成无副作用预检
  -> 从 GeometryReference 创建样本状态
  -> update_fault_geometry
  -> update_mesh（仅在扰动方法未自行更新 mesh 时）
  -> update_GFs
  -> alpha 启用且当前 GL 已失效时更新 Laplacian
  -> magnitude prior 启用时按需取得当前候选面积
  -> SMC_FJ 受约束线性滑动求解，或 FULLSMC 滑动似然
  -> likelihood 和 SMC 权重
```

联合路线的关键不是单独“画一个扰动几何”，而是保证每个样本内的参考解释、候选几何、
mesh、GF、Laplacian、约束矩阵和协方差保持一致。
GF 随非线性候选几何更新；Laplacian 和面积由实际消费者触发，不会仅因诊断绘图或未启用的
先验在每个候选中额外重算。

整网格平移/旋转使用同一次 snapshot 的 `Vertices/Faces` pair；首次验证当前 Faces 与该
reference 的 row、编号和连接关系一致后，固定拓扑候选复用验证结果，不会逐样本重找边界
或重建邻接。若采样前又 remesh，应先建立新的 reference，而不是把旧 vertices 与新 Faces
混用。

## MPI 进程与进程内线程

联合反演与非线性几何 SMC 一样，以 MPI rank 分配样本。`nchains` 是粒子数，
`chain_length` 是每阶段的链长度；二者都不等于 MPI 进程数。进程数应不大于
`nchains`，并优先比较能整除 `nchains` 的候选值，使各 rank 工作量更均衡。
进程、rank、线程和 affinity 的基础关系见
[并行运行基础](../concepts/parallel_process_rank_thread.md)。

两种模式的资源瓶颈不同：

| 模式 | 每个样本的主要工作 | 进程数选择重点 |
| --- | --- | --- |
| `FULLSMC` | 直接使用样本中的滑动计算似然；几何可变时还会重建 GF | 先看 GF 重建成本，再比较 rank 并行与进程内线程 |
| `SMC_FJ` | 每个样本都进行一次受约束线性滑动求解；固定几何复用 Gram/cross 块，变化几何只保留当前候选的瞬态块 | GF、协方差度量和求解工作区会随 rank 复制，先看峰值内存 |

第一次运行先保留 MPI 和数值库默认设置，从 4 个进程开始：

```bash
python test_joint_bayesian_bottom_offset.py --check-only
mpiexec -n 4 python test_joint_bayesian_bottom_offset.py --run
```

`--check-only` 仍会读取真实数据、构建 reference/mesh 和 inversion；它用于在采样前校验完整
装配，不是只检查 YAML 语法。模板不附带可直接完成科研反演的观测数据，运行前必须替换
reader、迹线、投影中心和配置占位值。

三份模板都提供 `--smooth-prior-weight`，默认值 `1.0`。它只在 `SMC_FJ` 已完成当前样本的
受约束线性求解后，缩放该模型的平滑先验对后验分数的贡献；它不改变该样本使用的 `alpha`，
也不改变线性解本身。大于 `1.0` 可作为“对粗糙模型给出更低后验分数”的受控敏感性试验；
不要把它与扩大 `alpha` 边界或固定 `alpha` 混为同一操作。若配置禁用了 `alpha` 平滑项，
这个倍率没有作用。正式比较时应保持数据、bounds 和随机设置相同，并报告倍率值。

确认结果和 MPI rank 正常后，再用同一配置和随机设置比较 8、16 或 `nchains` 的
其他因数。`SMC_FJ` 不应只按逻辑 CPU 数扩展：先观察小进程数下的峰值内存，确保
增加 rank 后仍有足够内存余量。MPI 已自动 pin rank 并限制 BLAS 时，直接保留默认
命令；只有发现过量线程或 affinity 不合理时，才把每进程 1 个 BLAS/OpenMP 线程
作为受控对照。通用判断流程见
[MPI 进程与 BLAS 线程相互叠加](../getting_started/troubleshooting.md#6-mpi-进程与-blas-线程相互叠加)。

模板路径使用 Python `pathlib`，不绑定 Windows 盘符或 POSIX 绝对路径。当前公开支持的
平台和环境要求以[安装说明](../getting_started/installation.md)为准；若环境使用 `python3`
或 `mpirun`，相应替换命令中的可执行文件名。

希望采样结束后照常保存几何改正、KDE 和拟合图，但不弹出窗口时，在启动 MPI 前设置
`MPLBACKEND=Agg`，不要传 `--no-plot`。三份模板已经对保存图使用 `show=False`；`Agg` 会由
各 rank 继承，也不改变 likelihood、线性求解或采样结果。只有明确不需要任何图件时才使用
`--no-plot`，统计、GMT 和 `Modeling/` 文本仍会输出。

正式重跑前还应确认样本文件不会覆盖旧结果；详见
[样本文件和覆盖语义](../reference/bayesian_joint_inversion.md#sample-file-overwrite)。

## 与两步走的关系

两步走仍然是推荐公开入门路线：

- 非线性几何反演负责寻找合理的几何参数范围和优选模型。
- BLSE/VCE 负责在固定几何上做分布式滑动和权重诊断。
- 联合 Bayesian 用于进一步传播几何不确定性到滑动分布，而不是替代前两步的基础检查。

实际研究中，建议先用两步走获得稳定结果，再用这些结果设置联合 Bayesian 的几何先验、扰动尺度、网格尺度和计算预算。

## 输出和检查

三份公共模板默认把断层、统计和后验图放在 `output/`，把观测、合成值、残差和数据拟合图
放在 `Modeling/`：

```text
output/
  geometry_correction.pdf
  geometry_correction/
  kde_geometry_sigmas_alpha.png
  fit_statistics_median.txt / .tsv
  slip_<fault>.gmt / slip_<fault>_center.gmt / slipdir_<fault>.txt
  stat_infos/
Modeling/
  <dataset>_fit_comparison.pdf
  <dataset>_data.txt / _synth.txt / _resid.txt
```

结果阶段先调用标准高层入口。它在 rank 0 加载 HDF5、激活代表模型、更新后验几何与滑动，
并按配置中的 `verticals` 和 `polys` 重建 synthetic。模板将 `plot_sigmas=False`，因为随后会
单独绘制同时包含 geometry、sigma 和 alpha 的联合 KDE，避免重复图片：

```python
inversion.extract_and_plot_bayesian_results(
    rank=rank, filename=str(samples_file), model="median",
    plot_faults=not args.no_plot, plot_data=not args.no_plot,
    plot_sigmas=False,   # 后面单独生成 geometry/sigma/alpha 联合 KDE。
    data_poly="config",  # 使用配置解析后的多项式改正设置。
    file_type="pdf",
    fault_outdir=str(output_dir), data_outdir=str(modeling_dir), show=False,
)
```

随后才可绘制几何改正、联合 KDE，并导出 fault/slip 文件。模板使用
`collect_fit_statistics(..., rebuild_synth=False)` 复用高层入口已经生成的 synthetic；
`writeDecim2file(..., triangular=None)` 根据 corner 的 4/6/8 列自动识别输出多边形，不把
降采样格式硬编码为三角形。`--no-plot` 只关闭图件，仍会激活代表模型并输出统计、GMT 和
`Modeling/` 文本。比较 MAP、mean 或 median 时，每次都要重新激活对应代表模型后立即导出。
详细状态规则见 [联合反演参考](../reference/bayesian_joint_inversion.md)，统计公式见
[Fit Statistics](../reference/fit_statistics.md)。

脚本末尾保留可选的 `plot_multifaults_slip(...)` 调用，供用户修改视角、深度范围、色标和
图幅；这些发表图参数不进入 YAML。

联合反演结果至少应检查：

- 几何参数后验是否收敛，是否贴边。
- 滑动后验均值、中位数和可信区间是否受几何扰动主导。
- 每条数据的残差、sigma 后验和权重是否合理。
- 超参数摘要中 geometry 的角色/单位是否与配置一致，sigma/alpha 的 `Scale (s)`、
  `Sampling`、`State` 和 `Row mult. (1/s)` 是否能逐组对上；不要把 `log10(s)` 当成物理尺度。
- `SMC_FJ` 中线性约束是否按预期生效。
- `inactive_constraints` 是否只包含当前模式预期不消费的配置项。
- 是否出现重复的 constrained linear solve 失败告警；若有，应先检查约束可行性。
- 网格更新和 GF 重建是否与扰动方法一致。

报告联合 Bayesian 结果时，应明确写出 `bayesian_sampling_mode`、`slip_sampling_mode`、几何扰动方法、扰动参数边界、网格策略、GF 后端、数据协方差处理和约束类型。

## 相关页面

- [Bayesian 联合反演中的几何参考](../concepts/bayesian_geometry_reference.md)
- [联合 Bayesian 几何参考与配置短例](../examples/joint_bayesian_geometry_setup.md)
- [Bayesian 联合反演参考 / Bayesian Joint Inversion](../reference/bayesian_joint_inversion.md)
- [Bayesian 联合反演中的可扰动断层几何 / Perturbable Fault Geometry](../reference/geometry_perturbation.md)
- [ECAT 约束管理器 / Constraint Manager](../reference/constraint_manager.md)
- [Sigmas 与 Alpha 配置模式 / Sigmas and Alpha](../reference/sigmas_alpha.md)
- [BLSE/VCE 参考 / BLSE/VCE](../reference/blse_vce.md)
