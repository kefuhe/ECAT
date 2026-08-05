# 科研绘图短例

本页只给最常复制的 `eqtools.viztools` 画法。完整参数、优先级和兼容说明见 [Viztools 参考](../reference/viztools.md)；反演结果批量图见 [Figure Products](../reference/figure_products.md)。

## 1. 论文单栏曲线图

```python
import matplotlib.pyplot as plt
from eqtools.viztools import PlotStyle, Presets

with PlotStyle(Presets.SCIENCE, figsize="single", fontsize=8, dpi=600):
    fig, ax = plt.subplots()
    ax.plot(distance_km, displacement_mm, linewidth=1.0)
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Displacement (mm)")
    fig.savefig("profile.pdf", bbox_inches="tight")
```

## 2. 双栏多子图与色盲友好颜色

```python
import matplotlib.pyplot as plt
from eqtools.viztools import PlotStyle, Presets

with PlotStyle(
    [Presets.SCIENCE, Presets.COLORS_BRIGHT],
    figsize="double",
    fontsize=8,
):
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    axes[0].plot(x, observed, label="Observed")
    axes[0].plot(x, synthetic, label="Synthetic")
    axes[0].legend()
    axes[1].plot(x, observed - synthetic)
    axes[1].set_ylabel("Residual")
    fig.savefig("fit.pdf", dpi=600)
```

## 3. 中文图件

```python
import matplotlib.pyplot as plt
from eqtools.viztools import PlotStyle, Presets

with PlotStyle(Presets.CHINESE, figsize="single", fontsize=9):
    fig, ax = plt.subplots()
    ax.plot(time, displacement)
    ax.set_xlabel("时间")
    ax.set_ylabel("位移（毫米）")
    ax.set_title("形变时间序列")
```

如果中文仍显示为方框，先运行 `list_chinese_fonts()` 检查当前系统是否有可用 CJK 字体。

## 4. 经纬度散点图

```python
import matplotlib.pyplot as plt
from eqtools.viztools import LonFormatter, LatFormatter, PlotStyle

with PlotStyle("science", figsize="single", fontsize=8):
    fig, ax = plt.subplots()
    points = ax.scatter(lon, lat, c=value, cmap="RdBu_r", s=10)
    ax.xaxis.set_major_formatter(LonFormatter())
    ax.yaxis.set_major_formatter(LatFormatter())
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(points, ax=ax, label="LOS displacement (m)")
```

## 5. 二维 SAR/光学网格 quick-look

```python
from eqtools.viztools import plot_raster

fig, ax, image = plot_raster(
    displacement,
    x=lon_mesh,
    y=lat_mesh,
    axis="geo",
    symmetric=True,
    percentile=99,
    colorbar_label="Displacement (m)",
    save="displacement_quicklook.png",
    show=False,
)
```

`lon_mesh` 和 `lat_mesh` 可以是与数据同形状的二维坐标；函数会逐点保持坐标对应，不会把二维经纬度压成一维插值。

## 6. 一次保存 PDF 和 PNG

```python
from eqtools.viztools import save_fig

save_fig(fig, "result", fmts=["pdf", "png"], dpi=600)
```

论文线图优先检查 PDF/SVG；PNG 适合快速预览和汇报材料。
