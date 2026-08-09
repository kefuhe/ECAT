# Figure Products

Figure Products 是已完成计算后的批量出图层。它把常见科研图件组织成少量入口，但仍调用现有 CSI/ECAT 绘图方法；`eqtools.viztools` 继续负责样式、字体、尺寸和通用栅格显示。

```text
已求解的 inversion / fault / geodata
  -> Figure Product（选择数据和图件组）
  -> 现有 CSI/ECAT plot 方法（实际绘图）
  -> viztools / Matplotlib（样式与保存）
```

只想画一张自定义图，直接使用底层 `fault.plot()`、`data.plot()`、`plot_multifaults_slip()` 或 [Viztools](viztools.md)；需要重复生成一组标准图时再使用本页接口。

## 科学边界

Figure Products：

- 不构建 Green's functions；
- 不改变解算器矩阵、约束或后验样本；
- 不把临时绘图数组写回持久 `fault.slip`；
- 可按现有流程调用 `buildsynth()`，生成 data/synth 对比所需的合成观测；
- 保留底层绘图方法原本的返回值和文件组织。

`plot_data_fits()` 使用与 BLSE/Bayesian 结果入口相同的 `buildsynth()` 契约：GPS
使用配置中的 `vertical` 和逐数据集 `poly`，InSAR
固定 `vertical=True`，opticorr 固定 `vertical=False`，leveling 固定
`vertical=True`。因此统一的是绘图、格式和路径，不是反演或预测公式。

共享入口支持上述全部类型，但各高层工作流仍保留自己的数据类型参与范围：BLSE
结果入口选择 GPS、InSAR、leveling 与 cross-fault offset；Bayesian 结果入口还会执行
opticorr 合成计算。共享绘图产品不会让某个结果入口自动处理额外的数据类型。

## 图像格式与路径契约

所有 Figure Product 使用同一个 `file_type` 规范：忽略前导点和大小写，例如
`".PNG"` 会规范为 `"png"`；支持 `png/jpg/jpeg/tif/tiff/pdf/svg/eps`，不支持的
格式会在调用 `buildsynth()` 或创建图件前报错。

推荐把断层场图和数据拟合图分开放置：

```text
output/     # 断层滑动、标准差和后验图
Modeling/   # GPS、InSAR、leveling、cross-fault offset 拟合图
```

GPS 使用 CSI `geodeticplot.savefig()` 的 prefix 接口，因此实际文件名带 `_map`：

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

| `data_poly` | 用途 |
| --- | --- |
| `"config"`（默认） | 每个数据集跟随自己的 `geodata.polys`；正式结果图推荐使用 |
| `"include"` | 对所有选中数据集强制请求包含已求解改正项的总预测 |
| `None` | 明确只画 source/slip-only 预测，用于诊断改正项贡献 |

不同数据类型仍由其既有方法处理：GPS 使用 `data.plot()`，InSAR 使用 `plot_fit_comparison()`，leveling 和 cross-fault offset 使用 ECAT 的专用比较图。

可用 `gps_kwargs` 和 `sar_kwargs` 调整显示参数：

```python
inv.plot_data_fits(
    show=False,
    gps_kwargs={"scale": 0.03, "figsize": (7, 5)},
    sar_kwargs={"cmap": "cmc.roma_r", "vmin": -0.2, "vmax": 0.2},
)
```

`faults`、GPS 的 `data=["data", "synth"]`、InSAR 的 `save_path` 和 `show` 由产品层拥有，不能在自由 kwargs 中重复指定。

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
| 全部图共享的显示参数 | 自由 kwargs 或 `gps_kwargs`/`sar_kwargs` | `cmap`、`shape`、`figsize` |
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
