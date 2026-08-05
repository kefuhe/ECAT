# Viztools 兼容优化验证报告

日期：2026-07-31  
状态：实现完成，验证通过

## 1. 验证结论

本轮修改保持了 Viztools、Figure Products 和旧 `eqtools.plottools` 的公开调用边界，同时修复了会造成状态污染、参数歧义或参数失效的确定性问题。仓库全量测试通过，未发现 BLSE、VCE、SMC、约束索引、断层几何或降采样数值路径回退。

没有修改以下内容：

- 求解器矩阵、参数索引和约束编译；
- Green's functions、后验样本或模型数值；
- 既有绘图函数签名、默认 `show/savefig`、返回值和文件命名；
- 外部旧脚本的 `eqtools.plottools` 入口。

## 2. 已验证的实现变化

### 2.1 兼容导入

ECAT 包内 Python 代码已从弃用的 `eqtools.plottools` 改为直接从 `eqtools.viztools` 导入。`eqtools.plottools` 本身仍是 re-export shim，契约测试确认关键对象在新旧入口中具有相同对象身份。

这次迁移只改变 import 路径，没有把 `sci_plot_style()` 批量替换成 `PlotStyle`，因此保留了旧 wrapper 对 raw style、serif、mathtext 和历史参数的兼容映射。

### 2.2 样式状态和 formatter

验证范围：

- `PlotStyle` 正常退出恢复 rcParams；
- 异常退出恢复 rcParams；
- 嵌套 context 逐层恢复；
- `science`/`science-serif` 字体与 mathtext 契约；
- 颜色 overlay 不改变基础字体；
- 显式 `rcparams` 具有最终优先级；
- x/y 使用独立 `DegreeFormatter`，不会在 Axis 之间重新绑定同一实例。

`PlotStyle.subplots()` 保留历史的 figure-lifetime 行为，没有进行会改变后续 artist 样式的语义修订。它依赖 `close_event`，无界面 backend 不保证触发；公开文档和模块 quick start 已改为推荐 `with PlotStyle(...)`。

### 2.3 栅格 quick-look

验证范围：

- 数组、1D/2D 坐标、DataArray、NetCDF 和 GeoTIFF；
- 自动 percentile、对称色阶和 Matplotlib `norm`；
- `norm` 与 `vmin/vmax/symmetric/non-zero center` 的冲突被明确拒绝；
- projected CRS 和不完整地理元数据在 `axis="geo"` 下告警；
- 返回 `(fig, ax, artist)`、保存和显示契约不变。

`plot_geotiff()` 仍不做重投影。告警只阻止用户把投影坐标误读为经纬度，不改变数据数组或坐标值。

### 2.4 3D slip 状态

临时 custom slip 现在采用事务式 snapshot/restore，覆盖：

- `fault.slip` 原本不存在；
- `fault.slip is None`；
- 原本已有数组；
- 多 fault 中后一个输入失败；
- 已进入实际绘图后发生异常。

测试确认所有路径都恢复进入函数前的原始状态。CPT 参数改为按 `method=`、`N=` 关键字传递，与 `get_cmap(cpt_path, name=None, method='cdict', N=None)` 的真实签名对齐。

### 2.5 Figure Products

统一的显示参数优先级为：

```text
产品默认值 < 公共 plot kwargs < 当前字段/数据集 kwargs
```

科学身份和生命周期键由产品层锁定。测试覆盖 GPS/SAR 显示覆盖、slip 字段覆盖、逐字段 `cmap/norm`、3D 视角参数和 locked-key 清晰报错。产品函数仍调用已有 CSI/ECAT 绘图方法，没有复制第二套科学计算。

### 2.6 leveling / cross-fault 绘图参数

`figsize` 和 `dpi` 以前虽被 `**style_kwargs` 接受，却被函数内硬编码值覆盖。现在通过单一私有 helper 同时传给 PlotStyle、具体 figure 和保存函数；默认尺寸与 300 dpi 不变，只有用户显式设置时才改变输出。

## 3. 测试证据

使用项目指定环境：

```text
D:\Anaconda\envs\cutde\python.exe
```

### 3.1 聚焦契约与文档测试

命令覆盖：

- `tests/test_viztools_lifecycle.py`
- `tests/test_viztools_raster.py`
- `tests/test_viztools_contracts.py`
- `eqtools/csiExtend/tests/test_figure_products.py`
- `eqtools/csiExtend/tests/test_nonlinear_geometry_plots.py`
- `tests/test_docs_navigation.py`

最终结果：`76 passed, 2 warnings in 20.45s`。两条 warning 均来自有意构造的无 georeferencing GeoTIFF 测试：rasterio 的 `NotGeoreferencedWarning` 和 Matplotlib 对退化刻度范围的 RuntimeWarning。

### 3.2 全仓库回归

命令：

```text
python -m pytest -q
```

结果：

```text
1463 passed, 20090 warnings in 482.24s
```

大量 warning 来自既有 covariance benchmark 对 `numpy.matrix` 的 PendingDeprecationWarning；另有既有 rasterio georeferencing、测试函数 return-not-none、稀疏 mesh 和 Agg show 提示。全量运行发现一处遗漏的内部 `plottools` import，随后已迁移为同对象的 `viztools` import，并通过模块导入检查。

### 3.3 内部模块导入

smoke test 共导入 18 个本轮迁移涉及的 csiExtend、InvTools、optiUtils、sartsUtils 和 statUtils 唯一模块，全部成功；使用 `-W error::DeprecationWarning` 再次导入也通过。最终源码检索确认 ECAT 包内 `.py` 文件不再导入 `eqtools.plottools`；该名称只保留在兼容 shim、测试和说明文档中。

## 4. 文档检查

已建立三层阅读路径：

```text
docs/examples/viztools_scientific_figures.md
  -> docs/reference/viztools.md / figure_products.md
  -> eqtools/viztools/docs/VIZTOOLS_DEVELOPER_GUIDE.md
```

公开文档检查内容：

- 相对 Markdown 链接；
- 本地绝对路径；
- Python 代码块语法；
- 不存在的 `fontfamily` 和错误的实例 `apply()/restore()`；
- `title_fontsize`、percentile、`norm`、GeoTIFF CRS 和 kwargs 所有权语义。

旧的重复使用指南、内部说明、版本更新页和已执行计划已删除；长期规则并入 Developer Guide，验证证据保留在本报告。

## 5. 剩余边界

- 本轮不改变 `PlotStyle.subplots()` 的历史 lifetime 语义，只停止推荐；未来若删除或重设计，应走单独的弃用周期和图像回归。
- 本轮不新增 mega `PlotConfig`、绘图 YAML 或 backend 抽象。
- `plot_slip_distribution()` 的长签名为兼容契约；内部按职责维护，不强迫现有用户改成新对象。
- GeoTIFF quick-look 不代替 GIS 重投影，SAR/LOS 物理约定仍由 reader/workflow 层负责。

## 6. 最终检查

提交前应再次执行：

```text
python -m pytest tests/test_viztools_lifecycle.py tests/test_viztools_raster.py tests/test_viztools_contracts.py eqtools/csiExtend/tests/test_figure_products.py eqtools/csiExtend/tests/test_nonlinear_geometry_plots.py tests/test_docs_navigation.py -q
git diff --check
```

最终实际结果：

```text
76 passed, 2 warnings in 20.45s
```

另外，3 个新增/重写的公开绘图页面共 27 个 Python 代码块均通过 `ast.parse`；相关 Python 文件通过 `compileall`；`git diff --check` 通过，仅报告工作树既有的 LF/CRLF 转换提示，没有空白错误。
