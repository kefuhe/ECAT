# Figure Products

Figure Products 是已完成计算后的批量出图层。它把常见科研图件组织成少量入口，但仍调用现有 CSI/ECAT 绘图方法；`ecat_viz` 负责样式、字体、尺寸和通用栅格显示；`eqtools.viztools` 保留兼容入口与断层领域图件。

```text
已求解的 inversion / fault / geodata
  -> Figure Product（选择数据和图件组）
  -> 现有 CSI/ECAT plot 方法（实际绘图）
  -> ecat_viz / Matplotlib（样式与保存）
```

只想画一张自定义图，直接使用底层 `fault.plot()`、`data.plot()`、`plot_multifaults_slip()` 或 [Viztools](viztools.md)；需要重复生成一组标准图时再使用本页接口。

## 科学边界

Figure Products：

- 不构建 Green's functions；
- 不改变解算器矩阵、约束或后验样本；
- 不把临时绘图数组写回持久 `fault.slip`；
- BLSE/VCE 正式结果直接发布已组装的 `G @ mpost`，诊断或 Bayesian 路线按当前 source
  声明调用 `buildsynth()`；
- 保留底层绘图方法原本的返回值和文件组织。

`plot_data_fits()` 使用与拟合统计相同的预测契约。BLSE/VCE 的完整配置模型从
`G @ mpost` 按数据行切分，在全部目标字段布局预检通过后批量发布，图件不会再次正演；显式选择部分 fault、强制
`data_poly` 或 Bayesian 路线才调用 CSI `buildsynth(direction="source")`。GPS 使用配置中
的 `vertical`；其他数据类型也消费同一份规范化配置，若其 CSI 接口不使用该参数则不会改变
其物理预测。水平 GPS 未参与反演的 U synthetic 明确为 0，不沿用观测值或前一模型的状态。

`faults` 使用一套统一选择规则：`None` 或 `"all"` 表示完整配置模型；单个 fault 名称、
fault 对象以及名称/对象组成的 list 或 tuple 表示显式贡献诊断。显式选择会先解析为同一组
fault 对象，再同时用于预测和图上的 fault 迹线；未知名称立即报错，不会回退到全部 fault。
因此多断层图件不会出现“迹线只画子集、synthetic 却包含全部 fault”的错配。

共享入口支持上述全部类型。BLSE/VCE 与联合 Bayesian 结果入口都可生成
GPS、InSAR、opticorr、leveling 和 cross-fault offset 产品；独立非线性几何 SMC
当前只接受其似然支持的数据类型，不会因为绘图接口增加 opticorr 反演支持。

## 图像格式与路径契约

所有 Figure Product 使用同一个 `file_type` 规范：忽略前导点和大小写，例如
`".PNG"` 会规范为 `"png"`；支持 `png/jpg/jpeg/tif/tiff/pdf/svg/eps`，不支持的
格式会在发布预测或创建图件前报错。

推荐把断层场图和数据拟合图分开放置：

```text
output/     # 断层滑动、标准差和后验图
Modeling/   # GPS、InSAR、opticorr、leveling、cross-fault offset 拟合图
```

默认 legacy GPS 使用 CSI `geodeticplot.savefig()` 的 prefix 接口，因此实际文件名带 `_map`：

```text
Modeling/gps_<dataset>_map.<file_type>
```

`plot_data_fits()` 返回的 GPS 路径与这个真实文件名一致。

## Data / Synth 图组

完成 `run()` 和 `returnModel()` 后：

```python
written = inv.plot_data_fits(
    datasets="all",
    faults="all",
    data_poly="config",
    outdir="Modeling",
    file_type="pdf",
    plot_data=True,
    show=False,
)
```

默认的 `data_poly="config"` 按数据集跟随已经解析并对齐的 `config.geodata["polys"]`：未配置改正项的数据集使用 source/slip-only 预测，配置了 offset、ramp 或 frame transform 的数据集使用包含已估计改正项的总预测。单值 `polys` 会在配置解析阶段先展开为与数据集等长的列表，Figure Product 不会再次猜测或展开。

这个高层选择会在统一预测适配层转换为各 CSI 数据类型的真实调用参数。InSAR、GPS、
opticorr 和 leveling 使用 `poly="include"`；cross-fault offset 需要改正项具体类型，因而
使用配置中已经解析的单个整数或列表 estimator 规格。图件层不会自行推断 estimator
类型，也不会再次维护一份合法值清单。

| `data_poly` | 用途 |
| --- | --- |
| `"config"`（默认） | 每个数据集跟随自己的 `geodata.polys`；正式结果图推荐使用 |
| `"include"` | 对所有选中数据集强制请求包含已求解改正项的总预测 |
| `None` | 明确只画 source/slip-only 预测，用于诊断改正项贡献 |

不同数据类型仍由其既有方法处理：GPS 默认使用 `data.plot()`，comparison 模式使用 `data.plot_fit_comparison()`，InSAR 与 opticorr 使用各自
数据对象的 `plot_fit_comparison()`，leveling 和 cross-fault offset 使用 ECAT 的专用比较图。

InSAR 和 opticorr 的空间显示由一个顶层选项控制，不改变数据内容或文本输出格式：

```python
inv.plot_data_fits(
    antisymmetric=True,  # 默认使用以 0 为中心的对称色标
    raster_render_mode="cells",
    raster_cell_edge_width=0.25,
    show=False,
)
```

| `raster_render_mode` | 图件语义 |
| --- | --- |
| `"points"`（默认） | 绘制观测中心点；保持既有图件、速度和内存特征 |
| `"cells"` | 按存储的 4/6/8 列 corner 绘制矩形、三角形或四边形；无 corner 时明确报错 |
| `"auto"` | 有合法 corner 时绘制单元，否则回退到点 |

`raster_cell_edge_width` 的单位是 point，只在单元模式下生效。它与 reader 的
`triangular` 校验以及结果文件的 `sar_corner` 互不替代：前者决定输入几何契约，
`sar_corner` 决定文本输出，`raster_render_mode` 只决定拟合图怎样显示。

`antisymmetric=True` 是 InSAR/opticorr Data 与 Model 的默认自动色标策略：每个
raster 场使用 `[-M, M]`。改为 `False` 后，自动范围使用该场有限值的实际最小值和
最大值。`sar_kwargs` 或 `opticorr_kwargs` 中显式给出的 `vmin`/`vmax` 优先级最高，
不会被再次对称化；只给一侧时，仅自动补齐缺失的一侧。`res_use_data_norm=True` 让
Residual 共用 Data/Model 范围，否则 Residual 独立使用零中心对称范围。以上选择只
影响色标，不修改 observation、synthetic、residual、corner 或反演状态。

opticorr 默认输出 2×3 图：East/North 两行和 Data/Model/Residual 三列。只查看一个
分量时，把组件选择放入专用显示参数：

```python
inv.plot_data_fits(
    opticorr_kwargs={"components": ("east",)},
    show=False,
)
```

可用 `gps_kwargs`、`sar_kwargs` 和 `opticorr_kwargs` 调整显示参数：

```python
inv.plot_data_fits(
    show=False,
    gps_kwargs={"scale": 0.03, "figsize": (7, 5)},
    sar_kwargs={"cmap": "cmc.roma_r", "vmin": -0.1, "vmax": 0.2},
)
```

`faults`、GPS 的 `data=["data", "synth"]`、raster 的 `save_path`、`show`、
`render_mode` 和 `cell_edge_width` 由产品层拥有，不能在自由 kwargs 中重复指定。

## GPS 单图比较

`gps_plot_mode="comparison"` 显式选择新版 GPS 比较图；默认 `"legacy"` 继续生成
原 CSI 地图。观测和模拟 EN 箭头使用相同线宽、箭头形状和纸面长度比例，默认分别
为红色和蓝色。参与反演的 U 同时显示：观测是底层大圆，模拟是上层小圆，填色表示
带符号的 Up 位移；大小只区分角色，不表示位移幅值。两个 U 场共用一个色标。

```python
written = inv.plot_data_fits(
    data_types=("gps",),
    gps_plot_mode="comparison",
    gps_figsize=(3.5, 2.7),
    gps_kwargs={
        "coordinates": "lonlat",  # 默认区域经纬度，保留度数刻度，不显示轴标题
        "value_scale": 1000.0,  # 当前数组为 m 时，仅显示换算为 mm
        "value_unit": "mm",
        "arrow_scale": 500.0,  # 显示值单位/inch；100 mm 箭头长 0.2 inch
        "legend_value": 100.0,
        "vertical_sizes": (64, 25),  # 观测/模拟面积，单位 point²
        "colorbar_size": 0.35,  # 右侧竖直色条，底端对齐主轴底边
        "error": False,
    },
    outdir="Modeling",
    file_type="pdf",
    show=False,
)
```

新版输出 `Modeling/gps_<dataset>_fit_comparison.<file_type>`，返回路径与实际文件一致。
`extract_and_plot_blse_results()`、联合 Bayesian 和独立几何 SMC 的
`extract_and_plot_bayesian_results()` 也接受 `gps_plot_mode` 和 `gps_kwargs`。
独立几何 SMC 保持其现有输出格式，新 GPS 比较图为 PNG。

| 参数 | 语义 |
| --- | --- |
| `coordinates="lonlat"` | 默认区域经纬度显示，按局地纬度修正轴比例，EN 箭头使用屏幕东/北方向；不提供全球制图或底图 |
| `coordinates="xy"` | 显式使用该 GPS 对象的投影公里坐标，等比例轴，ENU 方向转换为投影方向 |
| `xlabel`, `ylabel` | None 在 lonlat 中不显示轴标题，在 xy 中显示公里标题；字符串显式覆盖，空字符串隐藏；不隐藏刻度 |
| `remove_direction_labels` | 仅移除经纬度刻度的 E/W/N/S 后缀，保留度数及足以区分刻度的精度 |
| `extent` | 所选坐标下的 `[xmin, xmax, ymin, ymax]`；默认从有效配对站点生成带边距的范围，并包含箭头端点 |
| `figsize`, `unit` | 通用发表图幅，`unit` 为图幅 inch/cm，与位移单位分开 |
| `value_scale`, `value_unit` | 显式显示换算和标签，同时用于两组值及可选误差；不修改原数组 |
| `arrow_scale` | 显示数值单位/inch；None 从两组水平幅值生成同一自动比例 |
| `legend_value` | 红蓝标定棒共同代表的 EN 数值；每根长度为 `legend_value / arrow_scale` inch；默认对应 0.2 inch |
| `color` | 观测/模拟颜色对；默认红/蓝 |
| `width`, `headwidth`, `headlength`, `headaxislength` | 两组箭头共用的外观控制；width 为主轴宽度比例，值越大越粗 |
| `show_vertical` | 默认 True；只能隐藏已采用的 U，不能开启未参与反演的 U |
| `vertical_sizes` | 观测/模拟圆圈面积，要求观测大于模拟且均大于零；默认 `(64, 25)` |
| `vertical_cmap` | 默认 `RdBu_r`，正值表示 Up，负值表示 Down |
| `vertical_vmin`, `vertical_vmax` | 显示单位下的 U 色标边界；自动范围覆盖两组值并以零为中心，显式值不再对称化 |
| `colorbar_orientation="vertical"` | 默认 U 色条在轴外右侧，底端对齐主轴底边；horizontal 时默认放在主轴下方 |
| `colorbar_size=0.35` | 相对于最终主轴高度（竖直）或宽度（水平）的色条长度，要求 `0 < size <= 1` |
| `cbaxis` | 手动指定 `[left, bottom, width, height]`，单位为主轴比例；覆盖自动位置/长度，方向仍由 colorbar_orientation 决定 |
| `name`, `title`, `xticks`, `yticks` | 站名、标题和所选坐标下的刻度 |
| `legend_loc` | 合并图例位置，默认 best；包含两根标定棒、一次居中的共同尺度文字，以及可选 U 角色样本 |
| `key_position` | 已弃用：发出 FutureWarning，提示改用 legend_loc；不再绘制独立黑色箭头 |
| `error=False` | 默认不画误差；True 使用 `err_enu` 的 E/N 边际标准差，假定独立，不代替完整 Cd 或 VCE 结果 |
| `style="science"`, `style_kwargs` | 直接使用 ecat_viz `PlotStyle`；例如 `{"fontsize": 9, "pdf_fonttype": 42}` |
| `close`, `dpi` | 高层默认无显示时关闭批量图件；底层默认保留可编辑 Figure，保存 dpi 默认 300 |

水平身份图例同时承担尺度标定，不另外绘制黑色箭头。红蓝两根棒分别完整表示同一个
`legend_value`，共同数值居中放在标定棒列上方。例如 `legend_value=100`、
`arrow_scale=500` 时，每根棒和 100 显示单位的数据箭头均长 0.2 inch（5.08 mm）。
字体、DPI 和地图坐标单位不会改变这一长度。箭头长度随 arrow_scale 增大而缩短；
改变 legend_value 只改变标定棒，不改变数据箭头。只有 U 有效时不显示 EN 标定图例。

**分量和单位由当前科学流程决定。** 高层读取规范化反演配置中的 `vertical`，不会根据
数组有三列、U 是否有限或是否为零来猜测。EN 正式预测仍可能有零占位 U 列，那不是
拟合得到的 U。绘图不改变 `verticals`、模型、观测、协方差或预测发布合同。
`value_scale` 也不会读取或修正 reader 的 `factor`；数组曾被手动换算时，用户必须按
当前单位设置显示换算。位移与速度之间的转换不属于绘图。

E/N 与 U 各自使用观测/模拟共同的有效站点掩码。缺失值不补零，不改变源对象的站点；
跳过的数量会发出警告。没有任何有效配对或没有预先准备的 synth 时明确报错。

旧 `gps_scale`/`gps_legendscale` 只作用于 `legacy`，不是新版的 `arrow_scale`/
`legend_value`。旧地图的经纬度缩放不能按数值原样迁移为纸面长度比例。新版拒绝
`scale`、`legendscale`、`box`、`verticalsize`、`verticalnorm`、`drawCoastlines`、
`Map`、`Fault` 等旧地图选项；产品层还拥有 `vertical`、`faults`、`save_path`、
`show`、`data` 和 `ax`，不允许在 `gps_kwargs` 中覆盖。

新版默认值迁移时，原先省略 coordinates、但 extent 使用公里坐标的调用必须补上
`coordinates="xy"`；不根据数值猜测坐标单位。已有横向 cbaxis 需要显式指定
`colorbar_orientation="horizontal"`。原 key_position 应改为 legend_loc。上述变化
只作用于 comparison，legacy 的颜色、地图参数和输出保持原合同。

需要自行组合或进一步编辑图时，直接使用 CSI：

```python
# gps_data.synth 必须已由当前模型准备；vertical 由调用者明确声明。
fig, ax = gps_data.plot_fit_comparison(
    vertical=True,
    value_scale=1000.0,
    value_unit="mm",
    style="science",
    style_kwargs={"fontsize": 9},
    show=False,
)
ax.set_title("GNSS displacement comparison")
fig.savefig("gps_comparison.pdf", dpi=300, bbox_inches="tight")
```

底层可以借用 `ax`，并返回普通 `(fig, ax)`；不会覆盖旧 `gps_data.fig`。
借用的 Figure 不允许由绘图方法关闭。仅处理 EN 时直接传 `vertical=False`。

实现阅读顺序是 CSI `gps.plot_fit_comparison` 的参数合同、`csi._gps_plotting` 的
七个绘制步骤，再到 eqtools Figure Products 的分量/文件组织。坐标投影只决定箭头方向，
原始 `sqrt(E² + N²)` 决定箭头幅值；颜色 normalization 只影响 U 显示，不参与预测。
这是一张二维图表达 ENU 三个位移分量，不是空间三维箭头图。

## 断层滑动图组

```python
results = inv.plot_fault_fields(
    faults="all",
    fields=("total", "strikeslip", "dipslip"),
    outdir="Modeling/slip",
    file_type="pdf",
    show=False,
)
```

字段别名：`slip`/`total_slip` → `total`，`ss`/`strike` → `strikeslip`，`ds`/`dip` → `dipslip`。

公共显示参数放在函数的自由 kwargs 中；某个字段的覆盖放在 `field_plot_kwargs`：

```python
inv.plot_fault_fields(
    fields=("ss", "ds"),
    cmap="viridis",
    shape=(1.0, 1.0, 0.4),
    field_plot_kwargs={
        "ss": {"cmap": "cmc.roma_r", "norm": [-1.0, 1.0]},
        "ds": {"cmap": "cmc.vik"},
    },
    show=False,
)
```

解析顺序为：

```text
产品默认值 < 公共 plot kwargs < field_plot_kwargs[当前字段]
```

因此示例中 `ss` 使用 `cmc.roma_r`，`ds` 使用 `cmc.vik`，其他未覆盖字段才会使用公共 `viridis`。

产品层固定 `faults`、`slip`、`show`、`savefig`、`outdir` 和 `ftype`。这些键要用顶层显式参数表达，不能放进自由 kwargs 或 `field_plot_kwargs`；重复给出会立即报出清晰的 `ValueError`，避免 Python 的重复关键字错误或静默错画字段。

## 震间场图组

```python
results = inv.plot_interseismic_summary(
    faults=["FaultA"],
    fields=("tectonic_loading_rate", "backslip_rate", "coupling_ratio"),
    slip_component="strikeslip",
    outdir="Modeling/interseismic",
    show=False,
)
```

每个 fault 只计算一次震间结果，多个字段复用同一结果对象。不同字段的显示差异仍通过 `field_plot_kwargs` 表达：

```python
inv.plot_interseismic_summary(
    faults=["FaultA"],
    fields=("tectonic_loading_rate", "coupling_ratio"),
    field_plot_kwargs={
        "tectonic_loading_rate": {"cmap": "cmc.hawaii", "cblabel": "Loading"},
        "coupling_ratio": {"cmap": "cmc.roma_r", "cblabel": "Coupling"},
    },
    plot_on_2d=False,
    show=False,
)
```

`field`、`result`、`show` 和 `savefig` 属于产品层；计算参数如 `slip_component`、`solution` 和 `model` 通过各自的显式参数传入，不混入绘图 kwargs。

## Deep-slip loading 图组

```python
results = inv.plot_deep_slip_loading_summary(
    shallow_fault="ShallowFault",
    deep_faults=["DeepFault"],
    fields=("deep_loading_proxy_rate", "shallow_slip_rate", "coupling_to_deep"),
    component="strikeslip",
    mapping_kwargs={"max_distance": 5.0},
    outdir="Modeling/deep_loading",
    show=False,
)
```

如果已有 `result`，可直接传入避免重新计算。产品层固定 `field`、`shallow_fault`、`deep_faults`、`result`、`mapping`、`show` 和 `savefig`；公共和逐字段显示参数仍遵守相同覆盖顺序。

## 参数应放在哪一层

| 参数类别 | 放置位置 | 例子 |
| --- | --- | --- |
| 科学对象和字段选择 | 顶层显式参数 | `faults`、`fields`、`datasets`、`slip_component` |
| 计算或映射参数 | 对应显式参数/专用 kwargs | `solution`、`model`、`mapping_kwargs` |
| 全部图共享的显示参数 | 自由 kwargs 或数据类型专用 kwargs | `cmap`、`shape`、`figsize` |
| 单一字段显示覆盖 | `field_plot_kwargs[field]` | `norm`、`cblabel`、字段专用 `cmap` |
| 输出生命周期 | 顶层显式参数 | `outdir`、`file_type`、`show`、`savefig` |

同一含义的参数只应在一层设置；单一字段需要不同显示参数时，使用
`field_plot_kwargs[field]` 覆盖公共设置。

## 返回值和诊断

- `plot_data_fits()` 返回按数据类型组织的已写出路径字典，并记录跳过的数据集名。
- `plot_fault_fields()` 返回以规范化滑动字段为键的底层返回值字典。
- 震间和 deep-slip 图组返回按 fault/field 组织的底层结果。
- 不认识的 fault、字段或产品层保留键会立即报错；不会为了“尽量画图”而猜测科学字段。

## 相关页面

- [科研绘图短例](../examples/viztools_scientific_figures.md)
- [Viztools](viztools.md)
- [震间运动学](interseismic_kinematics.md)
- [Deep-slip loading proxy](deep_slip_loading_proxy.md)
