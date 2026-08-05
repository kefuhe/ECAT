# Viztools 维护与扩展指南

## 1. 设计目标

Viztools 是 Matplotlib 上的薄科研绘图层，不是独立 backend、绘图 DSL 或大型配置系统。它应同时满足：

1. ECAT 内部绘图具有一致字体、尺寸、保存和坐标格式；
2. 用户可直接组合 Matplotlib，而不必理解内部 registry；
3. 旧 `eqtools.plottools` 脚本继续工作；
4. 新增数据类型和科研图件时，不把科学计算、样式和输出生命周期混在一个对象中；
5. 默认参数、返回值、文件名和模型状态均可通过测试锁定。

## 2. 层级与依赖方向

```text
用户/科研脚本
  ├─ eqtools.viztools 公共基础 API
  └─ inversion FigureProductMixin 公共产品 API

Figure Products（eqtools/csiExtend/figure_products.py）
  └─ 现有 CSI/ECAT fault/data plot 方法
       └─ eqtools.viztools / Matplotlib

兼容层 eqtools.plottools
  └─ 仅 re-export eqtools.viztools
```

依赖只能朝下。`viztools` 不得导入 inversion、约束管理器或求解器；Figure Products 可以调用既有绘图方法，但不得反向成为 `fault.plot()` 的必需依赖。

## 3. 模块职责

| 模块 | 唯一职责 |
| --- | --- |
| `_core.py` | `PlotStyle`、preset 注册、rcParams 解析和恢复 |
| `_registry.py` | preset、样式目录、列宽和持久 apply 栈的状态 |
| `_compat.py` | `sci_plot_style`、`set_plot_style` 等旧接口转发 |
| `_style_utils.py` | 出版尺寸、保存、屏幕 dpi 防护和统一收尾 |
| `_font_utils.py` | CJK 字体探测和 figure 文本字体固化 |
| `_formatters.py` | 经纬度、DMS 和度符号 formatter |
| `_color_utils.py` | 色板读取，不改变科学数据 |
| `raster.py` | 已准备好的二维数组/DataArray/GeoTIFF/NetCDF quick-look |
| `viz_3d.py` | CSI 断层 3D 视角整理和滑动分布薄封装 |
| `csiExtend/figure_products.py` | 已求解结果的批量图件编排 |

不要在 `plottools.py` 新增实现；它必须一直是同一对象的兼容 re-export。

## 4. 稳定公共契约

重构前先锁定以下行为：

- 函数签名、默认值和关键字名称；
- 返回 `Figure`、`Axes`、tuple、字典或 `None` 的既有形式；
- `show`、`savefig`、`outdir`、后缀和扩展名；
- `fault.slipfig`、`data.fig` 等 CSI 既有副作用；
- 输入 fault、slip、observations 和 solver 状态不能因绘图而持久改变；
- `eqtools.plottools.X is eqtools.viztools.X` 的兼容身份。

因此不能仅为了“统一”而强迫所有接口返回 `(fig, ax)`，也不能一次性把长函数签名替换成新的配置对象。

## 5. PlotStyle 解析与生命周期

### 5.1 优先级

`PlotStyle._build_final_rcparams()` 的覆盖顺序为：

```text
基础 preset / mplstyle
  < 后续 preset
  < 显式 PlotStyle 字段
  < 自定义 handler
  < rcparams
```

颜色 preset 是 overlay；`minimal`、`scatter`、`ieee`、`chinese` 和 `chinese-serif` 已有 base，不应再要求用户重复写基础 preset。

### 5.2 状态恢复

库函数应使用：

```python
with PlotStyle("science", figsize="single", fontsize=8):
    fig, ax = plt.subplots()
    # create every artist that depends on rcParams here
```

正常退出、异常退出和嵌套上下文都必须恢复进入前的 rcParams。

`PlotStyle.apply()` 是脚本级持久状态，必须由 `reset()` 或 `reset_all()` 回收。`PlotStyle.subplots()` 是旧的 figure-lifetime 便利方法，依赖 GUI `close_event`；无界面 backend 可能不触发，因此不作为新代码模板。

### 5.3 字体字段

- `fontsize` 设置基础字号和轴标签，并派生 tick、legend、figure title；
- `title_fontsize` 当前只对应 `figure.titlesize`；
- 轴标题属于 `axes.titlesize`，需要 `rcparams` 或 `ax.set_title()`；
- 公共 API 没有 `fontfamily` 参数；字体族由 preset 或原生 rcParams 控制；
- `dpi` 是 `savefig.dpi`，`figure_dpi` 才是交互 figure dpi。

## 6. 参数多时如何组织

长签名可以保留以维护兼容性，但实现和文档按职责分组：

| 参数组 | 示例 | 归属 |
| --- | --- | --- |
| 科学对象 | `fault`、`slip`、`fields`、`datasets` | 调用者显式选择，不由样式层猜测 |
| 显示映射 | `cmap`、`norm`、`cblabel`、`cbticks` | 可按公共默认和逐字段覆盖合并 |
| 几何/视角 | `plot_on_2d`、`elevation`、`azimuth`、`shape` | 绘图实现 |
| 版式 | `figsize`、字号、label/tick padding | PlotStyle 或底层 plot |
| 输出生命周期 | `show`、`savefig`、`outdir`、`file_type` | 产品/最外层调用拥有 |

可以新增私有 normalizer、snapshot 或 kwargs merge helper；不要为了减少参数数量立即新增公共 `PlotConfig`、YAML schema 或 backend 接口。

## 7. Figure Product kwargs 契约

`_merge_product_plot_kwargs()` 使用统一规则：

```text
产品默认显示值 < 公共 plot kwargs < 当前 field/dataset kwargs
```

产品拥有科学身份和生命周期键，例如 `faults`、`slip`、`field`、`result`、`show`、`savefig`、`outdir`。自由 kwargs 中出现这些键时应清晰报错，不能让 Python 以重复 keyword 的偶然顺序决定行为。

新增产品时的模板：

```python
display = _merge_product_plot_kwargs(
    {"cmap": default_cmap},
    common_plot_kwargs,
    field_plot_kwargs.get(field, {}),
    locked=("field", "result", "show", "savefig"),
    context=f"my_product({field!r})",
)
return inversion.existing_plot_method(
    field=field,
    result=result,
    show=show,
    savefig=False,
    **display,
)
```

优先复用现有 `fault.plot()` 或 `data.plot()`。只有多个调用者重复相同“计算无关”的组图逻辑时，才增加 Figure Product。

## 8. 栅格契约

`raster.py` 只接受已经准备好的二维值和坐标：

- `x`、`y` 必须同时提供；一维中心坐标会转边界，二维 mesh 原样交给 `pcolormesh`；
- 仅有 `extent` 时用 `imshow`；
- 不重投影、不解释 LOS 符号、不转换相位或单位；
- percentile 自动范围与显式 Matplotlib `norm` 二选一；
- `axis="geo"` 只适用于真实经纬度坐标。投影 CRS、旋转 transform 或不完整元数据必须告警。

新增 reader 不应写进 `raster.py`；先在 reader/workflow 层形成明确的值和坐标，再交给 quick-look。

## 9. 模型状态保护

绘图函数临时替换模型属性时必须采用 snapshot/restore，并覆盖四种路径：

1. 属性原本不存在；
2. 属性原本为 `None`；
3. 属性是数组；
4. 多对象处理到中途发生异常。

`plot_slip_distribution()` 的 custom slip 只写入 strike-slip 列用于绘图，`finally` 后必须恢复完全相同的原始状态。准备多个 fault 时需要事务式回滚，不能只恢复已成功进入绘图阶段的对象。

## 10. 扩展模板

### 10.1 新增 preset

1. 判断它是完整 preset 还是 overlay；
2. 在 `styles/` 增加最小 `.mplstyle`，不要复制无关 rcParams；
3. 在 `_register_builtin_presets()` 注册 base 和描述；
4. 增加字体、线型或颜色契约测试；
5. 更新 public reference 的 preset 表。

### 10.2 新增通用绘图函数

```python
def plot_quantity(data, *, ax=None, style="science", save=None, show=False, **artist_kwargs):
    context = PlotStyle(style) if ax is None and style is not None else nullcontext()
    with context:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        artist = ax.plot(data, **artist_kwargs)
        finish_fig(fig, save, save=save is not None, show=show)
    return fig, ax, artist
```

如果接受外部 `ax`，不要改变用户整个进程的样式；如果保留 figure 给调用者继续画，必须清楚说明哪些 artist 已在 style context 内创建。

### 10.3 迁移旧内部调用

只改变 import：

```python
# old
from ..plottools import sci_plot_style

# new
from ..viztools import sci_plot_style
```

不要顺手把 `sci_plot_style` 全部替换为 `PlotStyle`。旧 wrapper 对 raw Matplotlib style、`serif`、`use_mathtext` 和历史参数有兼容映射，只有图像回归和 rcParams 对照测试证明等价时才能逐个迁移。

## 11. 验证清单

每次相关修改至少检查：

- PlotStyle 正常、异常、嵌套恢复；
- public signature 和默认值；
- legacy re-export 身份；
- `show/save/close` 生命周期；
- raster 数组、坐标、NetCDF、GeoTIFF、norm 和告警；
- custom slip 的 missing/None/array/多 fault 回滚；
- Figure Product 默认值、逐字段覆盖和 locked keys；
- csiExtend 相关绘图与 nonlinear geometry 回归；
- public Markdown 链接、Python 代码块、绝对路径和 `git diff --check`。

数值求解器测试仍需在跨模块改动时运行，但 Viztools 不应写入 BLSE/VCE/SMC 的参数索引、矩阵或约束状态。
