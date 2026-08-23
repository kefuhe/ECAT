# ECAT 科研绘图参考 / Viztools

`eqtools.viztools` 是 ECAT 的 Matplotlib 绘图公共入口。它负责科研图件的样式、字体、出版尺寸、保存收尾、经纬度刻度和轻量栅格 quick-look；它不解释 SAR 正负号、LOS 投影或反演结果的物理意义。

只想复制常用画法，先看 [科研绘图短例](../examples/viztools_scientific_figures.md)。本页用于查完整语义、参数优先级和兼容边界。

## 阅读路径

| 想完成的事 | 从这里开始 |
| --- | --- |
| 直接复制一个论文图样式 | [科研绘图短例](../examples/viztools_scientific_figures.md) |
| 选择 preset、理解叠加关系 | [Preset 的职责](#preset-的职责) |
| 覆盖字体、线宽、DPI 等参数 | [参数覆盖顺序](#参数覆盖顺序) 与 [PlotStyle 常用参数](#plotstyle-常用参数) |
| 统一字体、公式和出版尺寸 | [字体与数学公式](#字体与数学公式) 与 [出版尺寸](#出版尺寸) |
| 保存图件或绘制栅格 quick-look | [保存与显示](#保存与显示) 与 [二维科学栅格 quick-look](#二维科学栅格-quick-look) |
| 检查三角断层四边、角点和命名 | [断层边界诊断](#断层边界诊断) |

## 最短推荐用法

在函数或脚本局部使用上下文管理器：

```python
import matplotlib.pyplot as plt
from eqtools.viztools import PlotStyle, Presets

with PlotStyle(Presets.SCIENCE, figsize="single", fontsize=8, dpi=600):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Displacement (mm)")
    fig.savefig("figure.pdf")
```

离开 `with` 后，Matplotlib 的全局 `rcParams` 会恢复。库函数和可复用模块应优先采用这一模式。

## 三个使用层级

| 层级 | 推荐入口 | 适用情况 |
| --- | --- | --- |
| 普通用户 | `with PlotStyle(...):` | 单图、论文图、报告图；最清晰且不会污染后续绘图 |
| 项目脚本 | `PlotStyle.apply(...)` / `PlotStyle.reset()` | 同一脚本连续生成许多同风格图；必须成对恢复 |
| 高级扩展 | `register_preset()`、`rcparams`、自定义 handler | 项目统一规范或新增可复用 preset；不建议为单张图使用 |

`PlotStyle(...).subplots()` 是保留的兼容接口。它依赖 Matplotlib `close_event` 恢复样式，
而无界面后端不保证触发该事件；新脚本应优先使用 `with PlotStyle(...):`。

## Preset 的职责

完整 preset 已包含基础字体、线宽和版式，可单独使用：

| Preset | 用途 |
| --- | --- |
| `science` | 默认无衬线科研图 |
| `science-serif` | 衬线科研图 |
| `chinese` / `chinese-serif` | 在对应基础 preset 上加入系统 CJK 字体回退 |
| `minimal` | 继承 `science` 的极简坐标轴 |
| `scatter` | 继承 `science` 的散点循环 |
| `ieee` | 继承 `science-serif` 的 IEEE 线型循环 |
| `notebook` | Notebook 快速检查 |
| `presentation` | 幻灯片和海报 |

颜色 preset 只负责颜色循环，用来叠加到一个完整 preset 上：

```python
with PlotStyle([Presets.SCIENCE, Presets.COLORS_BRIGHT], figsize="double"):
    fig, axes = plt.subplots(1, 2)
```

可选颜色层为 `colors-bright`、`colors-vibrant` 和 `colors-contrast`。不要重复叠加已经包含基础 preset 的 `minimal`、`scatter`、`ieee` 或 `chinese`。

查看当前可用项：

```python
from eqtools.viztools import list_presets

print(list_presets())
```

## 参数覆盖顺序

同一个 rcParam 被多处设置时，后者优先：

```text
基础 preset / mplstyle
  < 后续叠加的 preset
  < PlotStyle 显式参数
  < 自定义 handler
  < rcparams
```

因此 `rcparams` 是最后的高级逃生口，不适合作为普通图件的主要配置方式。

## PlotStyle 常用参数

```python
PlotStyle(
    preset="science",
    figsize="single",
    fontsize=8,
    tick_fontsize=7,
    legend_fontsize=7,
    title_fontsize=9,
    legend_frame=False,
    dpi=600,
    figure_dpi=None,
    pdf_fonttype=42,
    usetex=False,
    mathfont=None,
    rcparams=None,
)
```

| 参数 | 作用 |
| --- | --- |
| `preset` | 一个 preset 名或从左到右覆盖的名称列表 |
| `figsize` | 列宽名、数值宽度或 `(width, height)` |
| `fontsize` | 基础字号和轴标签字号；未单设时派生 tick、legend 和 figure title 字号 |
| `tick_fontsize` | `xtick.labelsize` 与 `ytick.labelsize` |
| `legend_fontsize` | legend 字号 |
| `title_fontsize` | `figure.titlesize`，即 `fig.suptitle()` 的默认字号；轴标题用 `rcparams={"axes.titlesize": ...}` 或 `ax.set_title(..., fontsize=...)` |
| `dpi` | 默认 `savefig.dpi`，不改变交互窗口的 figure dpi |
| `figure_dpi` | 显式设置交互 figure dpi；普通用户通常不需要 |
| `pdf_fonttype` | PDF/PS 字体类型；可编辑文本常用 42 |
| `usetex` | 调用外部 LaTeX 渲染；默认不建议开启 |
| `mathfont` | Matplotlib mathtext 字体族 |
| `rcparams` | 最终覆盖的原生 Matplotlib rcParams 字典 |

`PlotStyle` 没有 `fontfamily` 参数。字体族由 preset 决定；如确需覆盖，使用 `rcparams={"font.family": ...}`。

## 字体与数学公式

- `science` 使用无衬线文本并匹配 sans 数学字体。
- `science-serif` 使用衬线文本并匹配 serif 数学字体。
- `chinese` 和 `chinese-serif` 在上述基础上探测可用 CJK 字体。
- `usetex=True` 依赖本机 LaTeX；CJK preset 会禁用不兼容的 pdfLaTeX 路径并给出提示。
- 默认 PDF/PS 字体为可编辑的 Type 42；最终投稿前仍应在目标机器检查字体嵌入。

查询本机中文字体：

```python
from eqtools.viztools import list_chinese_fonts

print(list_chinese_fonts())
```

## 出版尺寸

`publication_figsize()` 返回英寸单位的 `(width, height)`：

```python
from eqtools.viztools import publication_figsize

publication_figsize("single")
publication_figsize("double", fraction=0.8)
publication_figsize((10, 8), unit="cm")
```

常用名字包括 `single`、`double`、`full`、`nature`、`nature_double`、`science`、`science_double`、`ieee_column`、`ieee_page`、`pnas`、`pnas_double`、`a4` 和 `a4_margin`。

## 保存与显示

单一格式直接使用 Matplotlib：

```python
fig.savefig("result.pdf", dpi=600, bbox_inches="tight")
```

批量格式使用 `save_fig()`：

```python
from eqtools.viztools import save_fig

save_fig(fig, "result", fmts=["pdf", "png"], dpi=600)
```

ECAT 内部绘图函数需要统一处理保存、显示和关闭时，可用：

```python
from eqtools.viztools import finish_fig

finish_fig(fig, "result.png", show=True, dpi=600, screen_dpi=200)
```

`screen_dpi` 只限制异常高的交互预览 dpi，不改变已保存文件的分辨率。正式质量检查应打开保存后的 PDF/SVG/PNG，而不是只看 `plt.show()` 窗口。

## 经纬度刻度

```python
from eqtools.viztools import LatFormatter, LonFormatter

ax.xaxis.set_major_formatter(LonFormatter())
ax.yaxis.set_major_formatter(LatFormatter())
```

只需给数值添加度符号时：

```python
from eqtools.viztools import set_degree_formatter

set_degree_formatter(ax, axis="both")
```

x、y 轴分别拥有独立 formatter 实例，避免 Matplotlib 在两个 Axis 之间重新绑定同一个 formatter。

## 二维科学栅格 quick-look

数组或坐标网格：

```python
from eqtools.viztools import plot_raster

fig, ax, image = plot_raster(
    data,
    x=lon,
    y=lat,
    axis="geo",
    cmap="RdBu_r",
    symmetric=True,
    percentile=99,
    colorbar_label="LOS displacement (m)",
    save="quicklook.png",
)
```

文件入口：

```python
from eqtools.viztools import plot_geotiff, plot_netcdf_grid

plot_geotiff("los.tif", axis="geo", colorbar_label="LOS displacement (m)")
plot_netcdf_grid("los.nc", variable="los", colorbar_label="LOS displacement (m)")
```

色阶规则：

- 非对称模式的 `percentile=99` 保留有限值的中央 99%，两端各裁掉 0.5%。
- `symmetric=True` 时，对 `abs(data - center)` 取指定 percentile，再围绕 `center` 对称。
- `percentile=None` 使用完整有限范围。
- 传入 Matplotlib `norm=...` 时，`norm` 独立负责色阶；不能再同时传 `vmin`、`vmax`、`symmetric=True` 或非零 `center`。

坐标规则：

- 同时给 `x`、`y` 时使用 `pcolormesh`，支持一维坐标或二维 mesh，不把二维经纬度错误压成一维插值。
- 只给 `extent` 时使用 `imshow`。
- `plot_geotiff(axis="geo")` 不做重投影。缺少 CRS、使用投影 CRS、旋转/剪切 transform 或索引式坐标时会告警；应先把数据重投影到经纬度后再使用地理标签。

这些入口只画已准备好的二维数据，不读取 GAMMA/GMTSAR/HyP3 物理约定，也不改变 LOS 正负号或单位。

## 断层边界诊断

三角断层完成四边识别后，可以用一个只读诊断图检查三维位置和四边的平面命名：

```python
from eqtools.viztools import plot_fault_boundary_diagnostics

fault.find_fault_fouredge_vertices(
    edge_method="topology",
    gap_policy="strict",
)

fig, axes = plot_fault_boundary_diagnostics(
    fault,
    coordinates="xy",
    save="fault_boundary_diagnostics.pdf",
    show=False,
)
```

默认包含：

| panel | 用途 |
| --- | --- |
| `3d` | 检查 mesh、四条边、inclusive boundary faces 和 junction vertices 的三维位置 |
| `map` | 检查平面投影、left/right 命名以及已记录的走向/投影方向 |
| `sequence`（可选） | 按 `top -> right -> bottom -> left` 展开节点顺序和深度；横轴不是距离或真实剖面 |

近直立断层的 left/right 边在平面投影中可能退化到两个端点并相互遮盖；这是投影几何，
不等同于边界提取失败，此时应以 `3d` panel 为主进行核对。

默认只包含 `3d` 和 `map`。需要核对四边节点顺序时再显式增加 `sequence`：

```python
fig, axes = plot_fault_boundary_diagnostics(
    fault,
    views=("3d", "map", "sequence"),
    coordinates="lonlat",
    show_boundary_faces=False,
)
```

`coordinates="lonlat"` 使用现有 `fault.Vertices_ll`，不重新投影或修改 fault。平面投影视图在
`xy` 模式下可以显示 `edge_extraction_info` 已记录的 strike/projection vectors；在 lon/lat
模式下不会把公里方向分量错误当作经纬度增量。经纬度刻度精度根据当前跨度自动选择；
地图比例使用中心纬度的 `cos(latitude)` 修正，并通过调整 axes box 保留紧贴数据的坐标范围，
不会为了填满方形 panel 而人为扩大纬度范围。

这个函数要求边界已经成功识别。它不会：

- 调用 `find_fault_fouredge_vertices()`；
- 自动选择 `topology`、`geometry` 或 fallback；
- 使用 `refind=True` 重建边界；
- 修改 mesh、边界字段、MudPy stencil、Laplacian、面积或 Bayesian 更新标记。

因此 topology 提取本身失败时，应先根据异常和 `edge_extraction_info` 检查网格；当前诊断入口
不复制一套 topology 算法去猜测失败边界。MPI 脚本中应在科学边界准备由各 rank 一致完成后，
只在 rank 0 保存或显示图件。完整边界字段、方法和 gap policy 说明见
[断层边界识别](fault_edges.md)。

## 兼容入口

`eqtools.plottools`、`sci_plot_style()` 和 `set_plot_style()` 仍保留给旧脚本；新建用户
脚本统一从下面入口导入：

```python
from eqtools.viztools import PlotStyle, Presets, save_fig
```

## 相关页面

- [科研绘图短例](../examples/viztools_scientific_figures.md)
- [Figure Products](figure_products.md)
- [SAR Reader](sar_reader.md)
- [降采样应用](downsampling_app.md)
