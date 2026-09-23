# 可运行脚本模板导航

`scripts/` 中的文件是可复制、可逐段修改的完整起始脚本；`docs/examples/` 的其他页面主要
提供短代码片段。第一次使用先按科研任务选择模板，再进入对应 workflow 理解输入、输出和
检查项，不需要从头阅读全部 reference。

<a id="editing-template"></a>

## 复制后从哪里修改

推荐模板开头的 `Editing guide` 提示首次修改项和相对路径基准。按阶段阅读分隔标题，
可选数据、诊断和定制绘图使用段内小标题；参数仍就近放在所属步骤。

| 修改需求 | 脚本中的位置 | 需要一起核对 |
| --- | --- | --- |
| 换一组观测 | `Data` 中的路径和 reader 调用 | 输入格式、单位、投影原点和协方差 |
| 加 GPS 或 optical | 启用完整可选块，再加入对应数据列表 | `geodata` 顺序及 YAML 对应设置；checkerboard 还需配置 `noise_config` |
| 改断层几何 | `Fault geometry and mesh` 或 `Reference fault geometry` | source 名与 YAML/bounds 对应，几何单位、方向及网格设置 |
| 改搜索范围或求解设置 | `Search settings` 或相应 inversion 段 | 平滑参数表示方式、配置分组和参数顺序 |
| 改图件 | 结果段的绘图调用或本地绘图设置 | 视角、范围、色标与保存位置；正演模板的常用选项集中在顶部 |

BLSE/VCE、参数搜索和 checkerboard 模板中的相对路径基于**当前工作目录**；新版非线性
SMC 模板也使用这一基准。联合 Bayesian 和两个地表正演模板的主要路径基于**脚本目录**。
复制脚本后先确认相应目录布局，再运行；YAML 内部路径仍按对应配置接口的规则解析。
BLSE/VCE 中的 `verbose` 还控制可选边界诊断图，打开日志时应一并留意该段。

大型 SMC 已有独立的
[Windows PowerShell 与 WSL/Linux MPI 启动模板](mpi_launcher_scripts.md)。启动器和科研
脚本保持分离：前者只设置 rank、线程与工作目录，后者继续按顺序组织数据、断层、配置、
反演和输出，便于直接阅读修改。

## BLSE 模板怎么选

| 当前任务 | 模板 | 先读 |
| --- | --- | --- |
| 用已经选定的几何和平滑强度运行一次 BLSE | [`test_slip_inv_BLSE.py`](../../scripts/test_slip_inv_BLSE.py) 的 `--mode single` | [BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md) |
| 固定几何，由 VCE 估计 sigma/alpha 后运行一次线性反演 | [`test_slip_inv_VCE.py`](../../scripts/test_slip_inv_VCE.py) | [BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md) |
| 固定几何，构建 BLSE L-curve | [`test_BLSE_L_Curve.py`](../../scripts/test_BLSE_L_Curve.py) | [固定几何 L-curve](../workflows/04a_blse_l_curve.md) |
| 固定平滑强度，只搜索倾角 | [`test_dip_search_BLSE.py`](../../scripts/test_dip_search_BLSE.py) | [固定拓扑倾角搜索](../workflows/04b_blse_dip_search.md) |
| 已完成前两项，需要检查倾角和平滑耦合 | [`test_dip_smoothing_search_BLSE.py`](../../scripts/test_dip_smoothing_search_BLSE.py) | [倾角 × 平滑参数敏感性分析](../workflows/04c_blse_dip_smoothing_search.md) |

既有 `simple_run_loop()` 作为兼容入口继续保留。新实验优先复制独立的
`test_BLSE_L_Curve.py`：它直接调用 `scan_penalty_weights()`，输出候选摘要、逐数据集长表和
三联诊断图，并同时保存单幅 roughness–RMS L-curve；不会让单次 BLSE 模板同时承担参数
搜索职责。规范扫描用未加权 \(L_0\) 报告
roughness，并恢复进入前的活动模型和必要的数据合成状态；选中候选后仍需用
`run(penalty_weight=...)` 正式求解。旧接口的迁移写法见
[BLSE/VCE 参考](../reference/blse_vce.md#smoothing-scan)。
通用 `test_slip_inv_BLSE.py --mode loop` 同样直接调用规范扫描入口；它保留较紧凑的默认
输出，而独立 L-curve 模板集中提供科研绘图选项和更完整的工作流说明。

单次固定权重结果使用 `python test_slip_inv_BLSE.py --mode single`；单次 VCE 使用
`python test_slip_inv_VCE.py`。两者默认输出滑动图、fault/slip 文件和
data/synth/resid 文本；只有需要逐点 InSAR/opticorr 表格时才增加
`--export-point-values`。BLSE/VCE 只给一个最终线性解，不能把联合 Bayesian 的 posterior
滑动标准差图机械套到该模板。

这些 BLSE/VCE 模板、L-curve/倾角搜索模板和 checkerboard 模板都按单 Python 进程运行，
不要用 `mpiexec` 或 SMC 启动脚本包裹它们。需要调节 CPU 使用量时，应测试 BLAS/CUTDE 的
线程设置；MPI rank 只属于 SMC/Bayesian 采样模板。

两个单次模板都使用默认 compact 尺度报告，不再紧接着重复打印同一张表。需要稍后单独
核对当前活动 sigma/alpha 时，可调用 `inversion.print_scale_parameters()`；它不会重新
运行 BLSE 或 VCE。VCE 模板读取独立的 `default_config_VCE.yml`，可选的 certified KKT
加速以一行注释保留，默认仍关闭。

## 非线性、正演和分辨率检查模板

| 当前任务 | 模板 | 先读 |
| --- | --- | --- |
| 新项目做 Bayesian 非线性几何搜索 | [`test_nonlinear_geometry_smc.py`](../../scripts/test_nonlinear_geometry_smc.py) | [Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md) |
| 原样复现旧多断层 `exploremultifaults_smc` 案例 | [`test_nonlinear_bayesian.py`](../../scripts/test_nonlinear_bayesian.py) | [非线性几何配置](../reference/config_nonlinear_geometry.md) |
| 在规则点或自定义点计算 ENU 正演 | [`test_surface_displacement_forward.py`](../../scripts/test_surface_displacement_forward.py) | [地表位移正演](surface_forward_grid.md) |
| 在 SAR 有效像元计算并输出 LOS | [`test_sar_los_surface_forward.py`](../../scripts/test_sar_los_surface_forward.py) | [地表位移正演参考](../reference/surface_displacement_forward.md) |
| 检查 BLSE 空间分辨率 | [`test_BLSE_Inv_Checkboard.py`](../../scripts/test_BLSE_Inv_Checkboard.py) | [BLSE/VCE 工作流](../workflows/04_linear_slip_blse_vce.md) |

当前 checkerboard 模板是**串行 InSAR/GPS/optical 分辨率检查**。它读取观测几何与活动分量，生成
truth 滑动、正演并加入可复现噪声，再用同一固定网格进行 BLSE 恢复。GPS 会根据 YAML 的
`verticals` 与输入 U 分量是否有效选择 EN 或 ENU；启用脚本中的 GPS 块时，还需同步调整
`gpsdata`、`noise_config` 和 YAML 数据顺序。optical 使用 CSI `opticorr` 的 east/north
双分量接口，对应 YAML 的 `verticals` 必须为 `false`。不要用 MPI 启动该模板。

GPS 的标量噪声不是整条 ENU 向量共用一次随机扰动。对每个站点、每个活动分量都会独立
抽取高斯噪声，只是共用同一个标准差。例如 `{"GNSS": 0.002}` 表示 E/N（以及启用时的
U）各自使用 0.002。若垂向误差更大，直接按分量给出标准差：

```python
np.random.seed(2026)
noise_config = {
    "GNSS": {
        "east": 0.002,
        "north": 0.002,
        "up": 0.006,
    },
    "Optical": {"east": 0.05, "north": 0.08},
    "T012A": 0.003,
    "T121D": 0.005,
}
inversion.apply_synthetics(
    noise_sigma=noise_config,
    update_weight=True,
    save_dir="Modeling",
)
```

`noise_sigma=0.003` 表示所有数据集使用同一个标量；列表按 `geodata` 顺序；按数据名的字典
最适合混合数据。字典中的 InSAR 值为标量，GPS 可写标量或 `east/north/up`，opticorr 可写
标量或 `east/north`。`update_weight=True` 会把这些标准差同步写入活动分量并重建对角
`Cd`，因此噪声模型和反演权重一致，此时每个活动分量的标准差必须为正。GPS EN 模式只
消费 `east/north`；U 不进入观测向量、噪声或协方差，原 U 列保持不变。该入口表示各站或
像元、各分量相互独立的对角噪声；
若研究问题需要站间或分量间相关噪声，应显式构造完整协方差，而不能用这个短例代替。

当前线性与 checkerboard 模板直接以列表声明断层：

```python
faults_list = [fault_em1]
# 多断层时按希望的 source/参数块顺序排列：
# faults_list = [west_fault, east_fault]
```

中间不需要先建立 `OrderedDict` 再立即转回列表。断层对象自身的 `name` 负责与 YAML 中的
source 名匹配，列表顺序负责 source 和参数块顺序；两者应分别明确，不应靠字典绕转表达。

## 联合 Bayesian 模板怎么选

联合 Bayesian 是已经跑通两步走之后的高级路线。三份模板分别展示三类参数化实例；它们
不是框架支持范围的固定清单，控制点数量和采样参数个数由所选扰动方法决定：

| 要搜索的几何 | 脚本 | 配套配置 |
| --- | --- | --- |
| 标量底边位移示例 | [`test_joint_bayesian_bottom_offset.py`](../../scripts/test_joint_bayesian_bottom_offset.py) | [`bottom_offset.yml`](../../scripts/configs/joint_bayesian/bottom_offset.yml) + [`bottom_offset_bounds.yml`](../../scripts/configs/joint_bayesian/bottom_offset_bounds.yml) |
| 多个倾角参考点示例（模板使用 3 点） | [`test_joint_bayesian_three_dip_controls.py`](../../scripts/test_joint_bayesian_three_dip_controls.py) | [`three_dip_controls.yml`](../../scripts/configs/joint_bayesian/three_dip_controls.yml) + [`three_dip_controls_bounds.yml`](../../scripts/configs/joint_bayesian/three_dip_controls_bounds.yml) |
| 组合扰动示例（当前方法使用 4 个参数） | [`test_joint_bayesian_custom_perturbation.py`](../../scripts/test_joint_bayesian_custom_perturbation.py) | [`custom_perturbation.yml`](../../scripts/configs/joint_bayesian/custom_perturbation.yml) + [`custom_perturbation_bounds.yml`](../../scripts/configs/joint_bayesian/custom_perturbation_bounds.yml) |

初学者可以复制成套文件；熟练用户也可先运行 `ecat-generate-config` 和
`ecat-generate-boundary`，再对照模板修改生成文件。CLI 生成的是当前版本的完整配置，
配套文件则把一个具体场景的 Python、参数顺序和 bounds 对齐。

### 复制模板

Linux 或 WSL 的 Bash：

```bash
cp scripts/test_joint_bayesian_bottom_offset.py my_case/
cp scripts/configs/joint_bayesian/bottom_offset.yml my_case/default_config.yml
cp scripts/configs/joint_bayesian/bottom_offset_bounds.yml my_case/bounds_config.yml
```

Windows PowerShell：

```powershell
Copy-Item scripts/test_joint_bayesian_bottom_offset.py my_case/
Copy-Item scripts/configs/joint_bayesian/bottom_offset.yml my_case/default_config.yml
Copy-Item scripts/configs/joint_bayesian/bottom_offset_bounds.yml my_case/bounds_config.yml
```

进入案例目录后，各系统使用同样的运行命令：

```bash
python test_joint_bayesian_bottom_offset.py --check-only
mpiexec -n 4 python test_joint_bayesian_bottom_offset.py --run
python test_joint_bayesian_bottom_offset.py
# 需要额外逐点表时：
python test_joint_bayesian_bottom_offset.py --export-point-values
```

这些是完整的可编辑起点，不是附带真实观测数据的一键演示。`--check-only` 仍会
读取数据、构建 fault/reference/mesh 和 inversion，只跳过采样与绘图；因此必须先替换数据
路径、迹线、投影中心和配置占位值。

如果环境只提供 `python3` 或 MPI 发行版只提供 `mpirun`，分别替换命令中的 `python`
或 `mpiexec` 即可；这不是脚本或配置格式的差异。

模板使用 `pathlib` 从脚本位置解析相对路径，不要求 Windows 盘符或 POSIX 绝对路径。当前
公开支持的平台和环境要求以[安装说明](../getting_started/installation.md)为准。

联合模板的默认结果分为 `output/` 和 `Modeling/`：前者保存几何改正、联合 KDE、拟合统计、
fault/slip GMT 与代表滑动图，后者保存数据拟合图以及 raster data/synth/resid 文本。有
corner 的 raster 写多边形，没有 corner 的输入直接写点表。需要 posterior 滑动分量离散度
图时显式增加 `--plot-std`；有 corner 的数据需要额外中心点表时增加
`--export-point-values`，输出进入 `Modeling/points/`。两者都不改变反演或默认代表模型。
模板末尾另给可选的 `plot_multifaults_slip(...)` 调用，便于修改发表图的视角、范围和色标。

降采样通常不需要复制 Python：先用 `ecat-generate-downsample` 生成 YAML，再运行 `ecat-downsample`。完整命令见 [InSAR 降采样](../workflows/02_insar_downsampling.md)。`scripts/process_data_downsampling.py` 只是在源码树中调用同一 CLI 的薄入口。

旧 checkerboard 变体仍列在 [`scripts/README.md`](../../scripts/README.md)，用于复现已有项目；新用户先从上表的单一 checkerboard 入口开始。

## 推荐学习顺序

```text
普通 BLSE 单次运行
  -> 确认数据、固定几何、bounds、rake、poly 和输出链条

固定几何平滑搜索
  -> 选择合理 penalty 范围

固定平滑倾角搜索
  -> 比较几何候选

倾角 × 平滑联合敏感性
  -> 仅在前两者显示明显耦合时运行
```

## 复制后先改哪些位置

模板采用一致的注释分块，优先修改：

1. `lon0/lat0` 和数据文件路径；
2. `geodata` 顺序；
3. `fault_name`、trace、top/depth、dip direction 和 mesh size；
4. `default_config_BLSE.yml`、`bounds_config.yml` 路径；
5. 当前任务对应的候选列表；
6. 输出目录。

`fault_name` 必须匹配配置 source 名，`geodata` 必须匹配配置数据顺序。模板中的文件路径
只是占位符，不能不检查就用于正式案例。

联合 Bayesian 模板中的 `lon0/lat0` 同时服务数据和断层，应保持为同一共享定义。修改案例时，
还要一起核对 geodata 顺序、迹线与断层物理参数、reference 建立时机和 initial mesh 参数，
避免坐标参考、配置 source 名或采样基线彼此错位。

<a id="loop-statistics"></a>

## 循环中怎样获取统计信息

无论循环变量是倾角、平滑权重还是约束方案，都应在每次 `run()` 完成后、进入下一轮前
立即收集该轮统计。下面是可直接放进循环体的公共骨架：

```python
inversion.run(
    penalty_weight=penalty_weight,
    alpha=None,
    verbose=False,
)

roughness, solver_rms, solver_vr = inversion.returnModel(
    print_fit_statistics=False
)

fit_rows = inversion.collect_fit_statistics(
    model=f"penalty_{penalty_weight:g}",
    data_poly="config",
    include_dataset=True,
    include_global=True,
)
global_fit = next(
    row for row in fit_rows if row["scope"] == "global_solver_vector"
)
dataset_rows = [row for row in fit_rows if row["scope"] == "dataset"]
fit_df = inversion.fit_statistics_to_dataframe(fit_rows)
```

上面的 `run()` 参数展示直接传入平滑权重的场景；倾角、mesh 或约束循环仍按各自 workflow
更新对象和配置，后续三行统计接口保持不变。

三个接口各司其职：

- `returnModel()` 返回当前模型的 roughness 以及组装后 solver vector 的 RMS/VR；
- `collect_fit_statistics()` 按 `data_poly="config"` 重建配置对应的 synthetic，并返回逐数据集
  rows 和独立计算的 `global_solver_vector` row；
- `fit_statistics_to_dataframe()` 只把已经获得的 rows 转成表格，不重新求解或重建模型。

当前搜索模板把各数据集统计展开为宽表。当实验维度或数据集会变化时，可把循环变量附加到
每个 statistics row，改用更便于扩展的长表：

```python
all_rows.extend(
    {
        "dip_deg": dip_deg,
        "penalty_weight": penalty_weight,
        "constraint_case": constraint_case,
        **row,
    }
    for row in fit_rows
)
```

没有参与当前实验的字段可以删掉，也可以加入 mesh size、数据组合或其他方案标识；不要改动
求解后立即采集统计这一顺序。下一轮 `run()` 会覆盖当前 `mpost` 和 penalty 状态，因此不要
等循环结束后再补取前面各轮统计，也不要把逐数据集 RMS/VR 的算术平均当作全局拟合。
字段、公式和 scope 的完整定义见 [Fit Statistics](../reference/fit_statistics.md)，penalty 的
解析语义见 [BLSE/VCE Reference](../reference/blse_vce.md#结构化拟合统计)。

## 文档层级怎么配合

| 层级 | 负责回答 |
| --- | --- |
| 本页 | 当前任务应该复制哪个脚本 |
| `scripts/README.md` | 仓库中有哪些公开脚本入口 |
| workflow | 输入、执行顺序、输出、科学检查和下一步 |
| example | 某个小任务的短代码怎么写 |
| reference | 类、方法、配置字段和完整接口语义 |
| casebook | 公开真实案例如何组织脚本、数据和结果 |

这样模板可以按场景独立演进，而公共计算和配置语义仍由 BLSE/reference 统一说明，避免在
每个脚本页面复制同一套长参数表。
