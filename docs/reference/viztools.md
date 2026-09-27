# ECAT 科研绘图参考 / Viztools

`ecat_viz` 是独立的通用 Matplotlib 科研绘图库，可单独安装，不需要 CSI 或 eqtools。
它拥有样式、字体、出版尺寸、保存/显示、刻度、通用 3D 轴、轻量栅格和 CPT 色表；
`eqtools.viztools` 保留兼容入口及断层边界、dip、滑动分布等领域图件。
它们不解释 SAR 正负号、LOS 投影或反演结果的物理意义。

依赖方向是 CSI → ecat_viz，以及 eqtools → CSI / ecat_viz。ECAT 协调三个独立分发包。
通用 API 在新旧入口具有相同对象身份和同一注册表；仅导入通用工具不会加载领域图件。
独立绘图安装是在 `ecat-viz` 项目目录运行 `python -m pip install .`；文件读取功能可安装
`python -m pip install ".[raster]"`。完整 ECAT 安装见[安装说明](../getting_started/installation.md)。

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
from ecat_viz import PlotStyle, Presets

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
from ecat_viz import list_presets

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
from ecat_viz import list_chinese_fonts

print(list_chinese_fonts())
```

## 字体固定与显式设置

`finish_fig()` 默认调用 `bake_text_fonts(fig)`。应在 `PlotStyle` 上下文内收尾：
它把每个已有文本对象的通用字体族展开为当前样式的具体字体候选列表，保留候选顺序，
不统一覆盖用户指定的字体名称或字体文件，也不改变字号、粗细和斜体。
TeX 文本不参与此处理，应在所需样式上下文内渲染；之后新增的文字仍使用创建时的设置。重复固定不会不断扩展列表。
处理失败会发出警告，不能仅凭文件生成成功判断字体正确。

## 配置错误、恢复与并发

显式 `rcparams` 中的未知键或无效值会抛出包含键和值的 `ValueError`，不再静默忽略。
完整配置先验证再应用，验证或应用失败不留下部分全局样式；预设循环继承也会报错并列出路径。
未知预设仍沿用既有警告后回退行为，自定义 handler 异常仍告警。

同一 `PlotStyle` 实例可以嵌套使用和顺序复用。每层退出（包括异常退出）仅恢复本层实际修改的
rcParams，不回滚用户对无关键的主动修改。`apply/reset` 仍须按后进先出顺序配对，
不要将持久化样式跨越不匹配的上下文边界。

注册表的锁只保护内部状态操作，不提供 Matplotlib 或 rcParams 的跨线程隔离。
支持单线程顺序/嵌套绘图；并行批处理使用独立进程，GUI 使用遵守后端主线程要求。

## 初始化与用户配置

预设、样式目录和列宽的公开注册/查询入口先完成一次初始化，再处理用户操作。
无需先调用 `PlotStyle` 或 `list_presets()`。用户 preset 可以注销；内置名称仍受保护，
对内置名称的显式覆盖不会被后续初始化重置。旧兼容入口共享相同状态。

列宽配置读取优先级保持为 `~/.config/eqtools/viztools.json`、
`~/.config/eqtools/plottools.json`、`~/.config/statutils/plottools.json`、
`~/.plottools.json`，只读取第一个存在的文件。配置必须是 JSON 对象，
`column_widths` 必须是对象，全部宽度必须是有限正数。无效配置告警且不应用其中任何宽度。

`save_column_width(name, width_inch, config_path=None)` 校验原文件并保留其他字段，
在同目录写临时文件后原子替换。解析、校验和写入失败会抛出异常，保留原文件，
不注册未保存的新宽度。损坏的配置需要用户显式修复；多个进程同时修改配置应由调用者协调。
CSI 的 leveling/crossfaultoffset 绘图同样报告样式参数错误；`style=None` 是显式关闭样式的入口。

## 出版尺寸

`publication_figsize()` 返回英寸单位的 `(width, height)`：

```python
from ecat_viz import publication_figsize

publication_figsize("single")
publication_figsize("double", fraction=0.8)
publication_figsize((10, 8), unit="cm")
```

常用名字包括 `single`、`double`、`full`、`nature`、`nature_double`、`science`、`science_double`、`ieee_column`、`ieee_page`、`pnas`、`pnas_double`、`a4` 和 `a4_margin`。

命名列宽及 `register_column_width()`/`save_column_width()` 的宽度始终以英寸保存。
`unit="cm"` 只换算数字/二元组输入以及显式 `height`，不会再次换算命名列宽。
尺寸、fraction 和实际使用的 aspect 必须是有限正数；fraction 可大于 1。
二元组已经指定完整尺寸，因此忽略 fraction、height 和 aspect。

```python
publication_figsize("single", unit="cm")  # 仍是英寸命名列宽 (3.5, 2.625)
publication_figsize("single", height=5.08, unit="cm")  # 高度 2 inch
```

## GeoTIFF 坐标

`plot_geotiff()` 在文件原生 CRS 中绘制。普通北向上栅格保留原 `imshow` 路径；
旋转、剪切或轴翻转使用仿射变换后的二维像元边界和 `pcolormesh`，避免把包围盒误作像元位置。
此接口不重投影、不换算 CRS、不解释 SAR 正号。文件拥有坐标和行方向，不能通过
`x/y/extent` 或 `origin="lower"` 覆盖；需要自定义坐标时使用 `plot_raster()`。

## 保存与显示

单一格式直接使用 Matplotlib：

```python
fig.savefig("result.pdf", dpi=600, bbox_inches="tight")
```

批量格式使用 `save_fig()`：

```python
from ecat_viz import save_fig

save_fig(fig, "result", fmts=["pdf", "png"], dpi=600)
```

ECAT 内部绘图函数需要统一处理保存、显示和关闭时，可用：

```python
from ecat_viz import finish_fig

finish_fig(fig, "result.png", show=True, dpi=600, screen_dpi=200)
```

`screen_dpi` 只限制异常高的交互预览 dpi，不改变已保存文件的分辨率。正式质量检查应打开保存后的 PDF/SVG/PNG，而不是只看 `plt.show()` 窗口。

`show_fig(fig)` 中的 `fig` 只指定预览 DPI 的调整对象；显示仍调用 `plt.show()`，
可能显示所有打开的 Figure。`finish_fig` 的保存和 `close=True` 只针对传入的 Figure，
`show=False` 不主动显示。`block=False` 与 `close=True` 合用可能立即关闭窗口。
Notebook 后端的自动显示不属于此函数的显式显示调用。

## 经纬度刻度

```python
from ecat_viz import LatFormatter, LonFormatter

ax.xaxis.set_major_formatter(LonFormatter())
ax.yaxis.set_major_formatter(LatFormatter())
```

只需给数值添加度符号时：

```python
from ecat_viz import set_degree_formatter

set_degree_formatter(ax, axis="both")
```

x、y 轴分别拥有独立 formatter 实例，避免 Matplotlib 在两个 Axis 之间重新绑定同一个 formatter。

## 二维科学栅格 quick-look

数组或坐标网格：

```python
from ecat_viz import plot_raster

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
from ecat_viz import plot_geotiff, plot_netcdf_grid

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

## 倾角 profile 诊断

非分层 Bayesian 倾角 profile 使用独立诊断入口，不把参数控制信息混入 mesh 四边拓扑图：

```python
from eqtools.viztools import plot_dip_profile_diagnostics

fig, axes = plot_dip_profile_diagnostics(
    fault,
    perturbations=None,  # None 表示零扰动参考 profile
    coordinates="lonlat",
    save="dip_profile_diagnostics.pdf",
    show=False,
)
```

`map` panel 同时画输入位置、投影到 top 后的位置、sampled/fixed 角色和 transition；
`profile` panel 画统一坐标 \(u\) 上实际用于底边生成的倾角。该函数直接消费 fault 的只读
`resolve_dip_profile()`，不会复制投影/插值规则，也不会更新 bottom、mesh、GF、Laplacian
或缓存状态。完整设置协议与可复制的过渡区写法见
[Bayesian 倾角剖面组合](dip_profile/bayesian_mixed.md#只读诊断)。

transition 尚未确定时，可先显示 reference-top 曲率分析：

```python
from eqtools.viztools import plot_dip_transition_analysis

fig, axes = plot_dip_transition_analysis(
    fault,
    analysis,
    suggestion,
    coordinates="lonlat",
    save="dip_transition_preflight.pdf",
    show=False,
)
```

该函数只消费已经计算好的 `TopCurvatureAnalysis`，不会在绘图层重算曲率。它会先核对
reference fingerprint，避免把旧 top 的建议画到新 top。计算公式、报告字段和标准 setup
顺序见[曲率与转换带预分析](dip_profile/transition_preflight.md)。

## 色表与资源迁移

```python
from pathlib import Path
from ecat_viz import get_cmap, load_cpt, list_cmaps

cmap = get_cmap("viridis")  # Matplotlib 名称
cmap = get_cmap("cpt:precip3_16lev_change", samples=15)
cmap = load_cpt(Path("custom.cpt"))  # 显式本地文件
print(list_cmaps())  # 内置 CPT 名称，不含扩展名
```

`load_cpt(source, *, name=None, kind="continuous", samples=None)` 返回标准 Matplotlib
Colormap；`kind="listed"` 使用 CPT 的原始颜色，samples 只截取原始列表，不做插值。
连续色表未指定 samples 时沿用历史插值，指定后按该数量采样。裸名称用于 Matplotlib；
CPT 用 `cpt:` 前缀或显式 Path，URL 只能显式请求并有 30 秒超时。不会隐式注册全局色表、
搜索当前目录、建立 norm 或显示图件。历史解析保留忽略 B/F/N 和归一化色表位置的行为。

旧 CPT 调用需要保留 `method`、`N` 和返回值合同，而只更新导入时，可以使用：

```python
from ecat_viz import cpt as get_cpt

cmap = get_cpt.get_cmap("precip3_16lev_change.cpt", method="list", N=15)
positions, listed = get_cpt.get_listed_cmap("precip3_16lev_change.cpt", N=8)
```

这里的 `get_cpt` 是本地变量名，直接指向 `ecat_viz.cpt`。它与上面的通用
`get_cmap("cpt:...", samples=...)` 是两个签名合同；不要只换函数名而保留旧参数，
也不要把 `.cpt` 裸名称传给通用入口后期待搜索内置资源。

旧 `from eqtools.getcpt import get_cpt` 仍指向同一实现，函数签名保留；
`get_cpt.get_listed_cmap(...)` 仍返回 `(positions, cmap)`。
旧物理目录 `eqtools/cpt`、`eqtools/viztools/styles` 已移除，不能继续依赖
`resource_filename("eqtools", "cpt")`、`files("eqtools").joinpath("cpt")`、
硬编码目录或修改 `get_cpt.basedir`。外部色表改用 `load_cpt(Path(...))`；自定义样式
改用 `register_style_directory(Path(...))`。需要资源内容时可使用
`importlib.resources.files("ecat_viz").joinpath("cpt", "NAME.cpt").read_bytes()`。
已有样式名、用户配置路径及读取优先级在本次迁移中保持。

参数错误现在直接报告；`method="list"` 必须显式给出正整数 N，不再靠错误参数顺序
或无效输入继续执行。旧 CPT 的有效 RGB 采样保持；gray 命名颜色、缺省 RGB 和 HSV
补齐原解析器遗漏。色表文件保留原始来源及许可证，不能统一视为 MIT 资源。

`update_style_library()` 在 Matplotlib 的 `plt.style.reload_library()` 后重载
包内、SciencePlots 和已注册目录的样式，保留用户 preset/列宽/rcParams，不重新初始化
这些状态。此操作只刷新样式库，不应用样式；目录内容变化后也可显式调用。

## 兼容入口

本版本继续保留 `eqtools.viztools` 根入口的 32 个通用导出、
`eqtools.viztools.raster`、`eqtools.plottools` 和 `eqtools.getcpt.get_cpt`。
这些入口转出同一对象，不复制样式注册器、色表解析器或绘图状态；兼容入口不再扩展
新的通用 API。断层边界、倾角和滑动图件继续从 `eqtools.viztools` 导入。

下划线实现模块不是公开 API。本版本移除了 `eqtools.viztools` 下的 `_color_utils`、
`_compat`、`_constants`、`_core`、`_font_utils`、`_formatters`、`_registry` 和
`_style_utils` 转包。使用 `PlotStyle`、formatter 或 `finish_fig` 的脚本应从
`ecat_viz` 公开入口导入；用户脚本不应依赖另一份私有实现模块路径。

`eqtools.plottools`、`sci_plot_style()` 和 `set_plot_style()` 仍保留给旧脚本；新建用户
脚本统一从下面入口导入：

```python
from ecat_viz import PlotStyle, Presets, save_fig
```

## 相关页面

- [科研绘图短例](../examples/viztools_scientific_figures.md)
- [Figure Products](figure_products.md)
- [SAR Reader](sar_reader.md)
- [降采样应用](downsampling_app.md)
