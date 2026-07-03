# ECAT 图件样式参考 / Viztools

`eqtools.viztools` 是 ECAT 的 Matplotlib 样式、出版尺寸、字体和常用格式化工具入口。它适合帮助用户建立稳定的科研绘图规范：统一字号、列宽、字体、保存格式和经纬度标注，而不是在每个脚本里临时调整 Matplotlib 全局参数。

## 阅读路径

- 只想快速画一张统一风格的图：看 [快速使用](#快速使用)。
- 正在做论文图或报告图：看 [推荐科研图件配方](#推荐科研图件配方)、[出版尺寸](#出版尺寸) 和 [保存图件](#保存图件)。
- 需要统一整个项目的绘图规范：看 [建立项目绘图规范](#建立项目绘图规范)。
- 需要中文、数学公式或 LaTeX：看 [字体和数学公式](#字体和数学公式)。
- 地理图件经纬度坐标不统一：看 [经纬度格式化](#经纬度格式化)。
- 想快速重绘 GeoTIFF、NetCDF/GRD 或二维数组：看 [科学栅格 Quick-Look](#科学栅格-quick-look)。

## 快速使用

推荐在函数或脚本局部使用上下文管理器，避免污染后续绘图：

```python
import matplotlib.pyplot as plt
from eqtools.viztools import PlotStyle, Presets

with PlotStyle(Presets.SCIENCE, figsize="single", fontsize=8, dpi=600):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Displacement (mm)")
    fig.savefig("figure.pdf")
    plt.show()
```

推荐先理解为三件事：

- `PlotStyle` 管样式、字体、字号、出版尺寸和默认保存 dpi。
- `fig.savefig(..., dpi=...)` 或 `PlotStyle(..., dpi=...)` 管保存质量。
- `plt.show()` 是正常屏显方式；`show_fig()` / `screen_dpi` 只是异常高 dpi figure 的保护工具。

因此，普通用户可以像使用 Matplotlib 或 scienceplots 一样工作：`with PlotStyle(...)` 激活风格，`fig.savefig()` 保存，`plt.show()` 显示。`finish_fig()` 不是必需步骤，它只是 ECAT 内部和批量绘图常用的保存/屏显收尾工具。

论文图保存 600 dpi 时，可以在 `fig.savefig("figure.png", dpi=600)` 中指定，也可以用 `PlotStyle(..., dpi=600)` 设置当前样式上下文的默认 `savefig.dpi`。`PlotStyle(dpi=...)` 不改变交互 figure dpi。

屏显图应理解为保存图的自然预览：布局、字号、线宽和图面比例应一致，保存 dpi 只提高输出分辨率。需要显式控制交互 figure dpi 时，使用 `PlotStyle(..., figure_dpi=...)` 或 `rcparams={"figure.dpi": ...}`。`screen_dpi=200` 只是异常保护：当 figure dpi 已经过高时限制窗口大小。检查最终发表质量时，应打开保存出的 `pdf/svg/png` 文件。

快速创建 figure：

```python
fig, ax = PlotStyle(Presets.SCIENCE, figsize="single", fontsize=8).subplots()
```

脚本顶部全局应用：

```python
from eqtools.viztools import PlotStyle, Presets

PlotStyle.apply(Presets.SCIENCE, figsize="double", fontsize=9)
# ... many plots ...
PlotStyle.reset()
```

只保存不屏显：

```python
with PlotStyle(Presets.SCIENCE, figsize="single", fontsize=8, dpi=600):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fig.savefig("figure.pdf")
```

批量保存多格式时再使用 `save_fig()`：

```python
from eqtools.viztools import save_fig

save_fig(fig, "figure", fmts=["pdf", "png"], dpi=600)
```

只屏显检查时，普通情况直接 `plt.show()`；如果外部样式把 figure dpi 设得很高，可以使用 `show_fig()` 做屏显保护：

```python
from eqtools.viztools import show_fig

with PlotStyle(Presets.SCIENCE, figsize="single", fontsize=8):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    show_fig(fig)
```

## 推荐科研图件配方

论文单栏图：

```python
with PlotStyle(Presets.SCIENCE, figsize="single", fontsize=8):
    fig, ax = plt.subplots()
```

论文双栏或两列子图：

```python
with PlotStyle([Presets.SCIENCE, Presets.COLORS_BRIGHT], figsize="double", fontsize=8):
    fig, axes = plt.subplots(1, 2)
```

中文报告或中文论文图：

```python
with PlotStyle(Presets.CHINESE, figsize="single", fontsize=8):
    fig, ax = plt.subplots()
```

Notebook 快速检查：

```python
PlotStyle.apply(Presets.NOTEBOOK, fontsize=10, figure_dpi=150)
```

演示文稿：

```python
with PlotStyle(Presets.PRESENTATION, figsize="double", fontsize=12):
    fig, ax = plt.subplots()
```

## 常用 Presets

| 常量 | 字符串 | 用途 |
| --- | --- | --- |
| `Presets.SCIENCE` | `science` | 默认科学图件，无衬线 |
| `Presets.SCIENCE_SERIF` | `science-serif` | 衬线科学图件 |
| `Presets.MINIMAL` | `minimal` | 极简坐标轴和图件 |
| `Presets.PRESENTATION` | `presentation` | 演示文稿 |
| `Presets.NOTEBOOK` | `notebook` | Notebook 交互检查 |
| `Presets.IEEE` | `ieee` | IEEE 风格 |
| `Presets.SCATTER` | `scatter` | 散点图优化 |
| `Presets.CHINESE` | `chinese` | 中文无衬线 |
| `Presets.CHINESE_SERIF` | `chinese-serif` | 中文衬线 |
| `Presets.COLORS_BRIGHT` | `colors-bright` | 明亮色板，可叠加 |
| `Presets.COLORS_VIBRANT` | `colors-vibrant` | 高饱和色板，可叠加 |
| `Presets.COLORS_CONTRAST` | `colors-contrast` | 高对比色板，可叠加 |

多个 preset 可以叠加：

```python
with PlotStyle([Presets.SCIENCE, Presets.COLORS_BRIGHT], figsize="double"):
    fig, axes = plt.subplots(1, 2)
```

查看可用 preset：

```python
from eqtools.viztools import list_presets

print(list_presets())
```

## 出版尺寸

`figsize` 支持字符串、数值宽度或显式 `(width, height)`。字符串会由 `publication_figsize()` 转换为 inch。

| 名称 | 宽度含义 |
| --- | --- |
| `single` | 通用单栏，约 3.5 inch |
| `double` | 通用双栏，约 7.0 inch |
| `full` | 全页宽度，约 7.16 inch |
| `nature` / `nature_double` | Nature 单栏/双栏 |
| `science` / `science_double` | Science 单栏/双栏 |
| `ieee_column` / `ieee_page` | IEEE 单栏/全页 |
| `pnas` / `pnas_double` | PNAS 单栏/双栏 |
| `a4` / `a4_margin` | A4 纸宽或带边距宽度 |

示例：

```python
from eqtools.viztools import publication_figsize

publication_figsize("single")                 # (width, height), inch
publication_figsize("double", fraction=0.8)
publication_figsize(10, unit="cm")
publication_figsize((10, 8), unit="cm")
```

注册自己的期刊列宽：

```python
from eqtools.viztools import register_column_width

register_column_width("agu_single", 3.37)
```

## 建立项目绘图规范

一个项目或案例目录建议固定一套出图约定，避免每张图独立调参：

```python
from eqtools.viztools import PlotStyle, Presets, save_fig

PLOT_STYLE = Presets.SCIENCE
PLOT_FIGSIZE = "single"
PLOT_FONTSIZE = 8
PLOT_DPI = 300

def save_paper_figure(fig, name):
    save_fig(fig, name, fmts=["pdf", "png"], dpi=PLOT_DPI)
```

在绘图函数中使用：

```python
def plot_result(x, y, output):
    with PlotStyle(PLOT_STYLE, figsize=PLOT_FIGSIZE, fontsize=PLOT_FONTSIZE):
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Displacement (mm)")
        save_paper_figure(fig, output)
        return fig
```

如果一个项目要遵循特定期刊列宽，可先注册列宽，再在所有脚本中使用同一个字符串：

```python
register_column_width("my_journal_single", 3.35)

with PlotStyle(Presets.SCIENCE, figsize="my_journal_single", fontsize=8):
    fig, ax = plt.subplots()
```

## 字体和数学公式

常用参数：

| 参数 | 含义 |
| --- | --- |
| `fontsize` | 基础字号 |
| `fontfamily` | 文本字体族 |
| `mathfont` | Matplotlib mathtext 字体 |
| `usetex` | 是否使用 LaTeX 渲染 |
| `chinese` preset | 自动选择可用 CJK 字体 |

建议：

- 普通论文图优先用 `usetex=False`，减少环境依赖。
- 需要中文标签时使用 `Presets.CHINESE` 或 `Presets.CHINESE_SERIF`。
- 封装绘图函数返回 figure 前，如果担心后续样式重置影响文字，可调用 `bake_text_fonts(fig)`。

```python
from eqtools.viztools import bake_text_fonts

bake_text_fonts(fig)
```

## 保存图件

普通单图优先使用 Matplotlib 标准保存：

```python
with PlotStyle(Presets.SCIENCE, figsize="single", fontsize=8, dpi=600):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fig.savefig("result.pdf")
```

`dpi=600` 也可以直接写在 `fig.savefig("result.png", dpi=600)` 中。`PlotStyle(dpi=...)` 设置的是默认 `savefig.dpi`，不会改变屏幕 figure dpi。

需要多格式保存或自动创建输出目录时，使用 `save_fig`：

```python
from eqtools.viztools import save_fig

save_fig(fig, "result", fmts=["pdf", "png"])
save_fig(fig, "result.pdf", dpi=600, transparent=True)
```

论文图通常保存为 `pdf` 或 `svg`，检查图可保存为 `png` 或 `jpg`。高 DPI 只影响栅格格式；矢量格式主要由线宽、字体和图元复杂度决定。

如果同一个内部绘图函数既要保存又要可选屏显，使用 `finish_fig` 统一收尾：

```python
from eqtools.viztools import finish_fig

finish_fig(fig, "result.png", show=True, dpi=600, screen_dpi=200)
```

这里 `dpi=600` 只控制保存图，`screen_dpi=200` 只在交互 figure dpi 已经过高时限制窗口。`finish_fig()` 适合 ECAT 内部批量绘图和脚本批处理，不是普通单图必须步骤。若只想显示，不保存：

```python
from eqtools.viztools import show_fig

show_fig(fig, max_dpi=200)
```

注意：`show_fig()` 和 `screen_dpi` 面向异常高 dpi 屏显保护，不是最终质量检查。保存图仍由 `fig.savefig()` / `save_fig()` / `finish_fig(..., dpi=...)` 和矢量/栅格格式决定。若确实需要高 dpi 交互预览，应显式设置 `figure_dpi`，并接受窗口可能变大的代价。

## 经纬度格式化

地理图件常用：

```python
from eqtools.viztools import LatFormatter, LonFormatter

ax.xaxis.set_major_formatter(LonFormatter())
ax.yaxis.set_major_formatter(LatFormatter())
```

也可以使用：

```python
from eqtools.viztools import set_degree_formatter

set_degree_formatter(ax)
```

## 检查清单

正式保存图件前建议检查：

- 图件尺寸是否对应目标版式：单栏用 `single`，双栏或多子图用 `double`。
- 字号是否统一；论文图通常从 8 pt 左右开始，特别窄的图不要低于可读范围。
- 坐标轴和 colorbar label 是否写清单位，例如 `(m)`、`(cm)`、`(mm/yr)`。
- 经纬度图是否使用统一格式化器。
- 线宽、marker 大小和透明度是否在最终保存尺寸下仍然清晰。
- 论文图优先保存 `pdf` 或 `svg`，同时保留 `png` 便于快速查看。
- 高 DPI 只提高栅格图分辨率，不会改善矢量图线条或字体质量。
- 不要只按 `plt.show()` 窗口判断最终质量；屏显是 preview，正式检查应打开保存文件。

## 常见误区

- 不要在库函数中长期 `PlotStyle.apply()` 后不恢复；函数内部优先用 `with PlotStyle(...)`。
- 不要用放大 `figsize` 代替合理字号；版面尺寸和字号应一起设计。
- 不要把 `usetex=True` 作为默认选项；它依赖本机 LaTeX 环境，适合最终论文图而不是所有脚本。
- 不要只保存高 DPI 位图作为论文最终图；线图、散点图和多数模型图更适合矢量格式。
- 不要在一个项目中混用过多 preset；通常固定 1 个基础 preset，再按需要叠加一个颜色 preset。

## 兼容入口

旧代码中的 `eqtools.plottools` 和 `sci_plot_style` 仍可用作兼容入口，但新代码建议直接使用：

```python
from eqtools.viztools import PlotStyle, Presets, save_fig
```

这样能获得统一 preset、出版尺寸和字体系统，也更容易和新文档对应。

## 科学栅格 Quick-Look

`eqtools.viztools` 提供轻量的二维栅格绘图入口，适合已经准备好的科学栅格数据，例如正演位移 GeoTIFF、NetCDF/GRD 网格、`xarray.DataArray` 或普通 `numpy.ndarray`。

推荐入口：

```python
from eqtools.viztools import plot_raster, plot_geotiff, plot_netcdf_grid
```

普通二维数组：

```python
fig, ax, im = plot_raster(
    data,
    cmap="RdBu_r",
    symmetric=True,
    percentile=99,
    colorbar_label="LOS displacement (m)",
    save="quicklook.png",
    show=True,
)
```

已经保存的 GeoTIFF：

```python
fig, ax, im = plot_geotiff(
    "forward_los.tif",
    symmetric=True,
    percentile=99,
    axis="geo",
    axis_max_major_ticks=5,
    colorbar_label="LOS displacement (m)",
    colorbar_max_major_ticks=4,
    save="forward_los.png",
)
```

NetCDF/GRD 文件：

```python
fig, ax, im = plot_netcdf_grid(
    "forward_disp.nc",
    variable="los",
    symmetric=True,
    colorbar_label="LOS displacement (m)",
)
```

如果 NetCDF/GRD 中只有一个数据变量，`variable` 可以省略；如果有多个变量，必须显式指定，避免误画错误分量。

这些函数只做通用二维栅格显示：

- 自动处理 `NaN`、mask 和 GeoTIFF `nodata`。
- 可用 `percentile` 设置稳健色标，避免少量极端值撑大色标。
- 可用 `symmetric=True` 让形变、残差等正负场围绕 0 对称显示。
- 可用 `axis="geo"` 显示经纬度轴、Longitude/Latitude 标签和经纬度 tick 格式；用 `axis="off"` 可生成无边框 quick-look。
- `plot_geotiff(axis="geo")` 只使用 GeoTIFF 文件自身的 bounds；如果文件缺少 CRS、transform 接近行列号，或 bounds 像像素 index，会提示检查 georeferencing metadata。
- 可用 `axis_max_major_ticks`、`colorbar_max_major_ticks`、`tickfontsize` 和 `labelfontsize` 做基础刻度控制。
- 返回 `fig, ax, im`，用户可以继续叠加断层线、台站、标注或自定义 colorbar。
- 使用 `PlotStyle` 和 `finish_fig` 的保存/屏显逻辑，保存 dpi 与屏显预览仍按本页前述规则处理。

不要把这些函数理解为 SAR reader 或 GIS 绘图框架。它们不解释 GAMMA、HyP3、GMTSAR 的正负号，不计算 LOS 投影，也不依赖 PyGMT、Cartopy 或底图服务。SAR/LOS 的物理语义应先由 reader 或脚本处理清楚，再把可画的二维数据交给这些 quick-look 函数。

如果需要论文或 PPT 细调，保留返回对象后继续使用 Matplotlib：

```python
fig, ax, im = plot_geotiff("forward_los.tif", axis="geo", show=False)
ax.plot(fault_lon, fault_lat, "k-", linewidth=0.8)
fig.savefig("forward_los_paper.png", dpi=600)
```

## 相关页面

- [降采样超级入口参考：`check_plots`](downsampling_app.md#check_plots)
- [SAR Reader 参考：诊断和 Quick-Look](sar_reader.md#诊断和-quick-look)
