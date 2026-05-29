# eqtools.viztools — Matplotlib 样式管理

`eqtools.viztools` 提供统一的 matplotlib 样式管理，支持**上下文管理器、持久应用、装饰器**三种使用模式，以及灵活的字体控制、尺寸注册、格式化工具和函数封装支持。

---

## 目录

1. [快速上手（手头用）](#快速上手手头用)
2. [使用 Presets 常量（v2.2.0+）](#使用-presets-常量v220推荐)
3. [PlotStyle 三种使用模式](#plotstyle-三种使用模式)
4. [subplots() 一步创建图窗](#subplots-一步创建图窗)
5. [在封装函数中正确使用](#在封装函数中正确使用)
6. [内置 Preset 列表](#内置-preset-列表)
7. [自定义与扩展（custom 用）](#自定义与扩展custom-用)
8. [高级扩展功能](#高级扩展功能)
9. [参数完整参考](#参数完整参考)
10. [图尺寸系统](#图尺寸系统)
11. [字体系统](#字体系统)
12. [其他工具函数](#其他工具函数)
13. [常用场景示例](#常用场景示例)
14. [在可视化模块中的集成](#在可视化模块中的集成)
15. [常见问题排查](#常见问题排查)
16. [性能优化（v2.2.0+）](#性能优化v220)
17. [API 速查](#api-速查)

---

## 快速上手（手头用）

### 三行开画

```python
from eqtools.viztools import PlotStyle, Presets
import matplotlib.pyplot as plt

# 推荐：使用 Presets 常量（IDE 自动补全）
PlotStyle.apply(Presets.SCIENCE, figsize='single', fontsize=8)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel(r'Distance $\Delta x$ (km)')
ax.set_ylabel(r'Displacement $u_z$ (mm)')
fig.savefig('figure.pdf')
PlotStyle.reset()

# 或：字符串方式（仍然支持）
PlotStyle.apply('science', figsize='single', fontsize=8)
```

### 一步出图（更简洁）

```python
fig, ax = PlotStyle('science', figsize='single', fontsize=8).subplots()
ax.plot(x, y)
fig.savefig('figure.pdf')
# 关闭图窗时自动 reset rcParams，无需手动调用 PlotStyle.reset()
```

### 保存多格式

```python
from eqtools.viztools import save_fig

save_fig(fig, 'output', fmts=['pdf', 'png'])    # → output.pdf + output.png
save_fig(fig, 'output.pdf')                      # 有扩展名直接用
save_fig(fig, 'output.pdf', dpi=600, transparent=True)
```

### 查看 preset 内容

```python
# describe() 打印并返回 dict
rc = PlotStyle.describe('minimal')
# Preset: minimal  (base: science)
# mplstyles : ['eqtools-minimal']
# Resolved rcparams (N keys):
#   axes.spines.right = False
#   ...

# 只看字体相关参数
rc_fonts = PlotStyle.describe('science', filter_prefix='font.')
```

---

## 使用 Presets 常量（v2.2.0+，推荐）

为了减少拼写错误并获得 IDE 自动补全支持，现在可以使用 `Presets` 常量类：

```python
from eqtools.viztools import PlotStyle, Presets

# 新方式：使用常量（推荐）
with PlotStyle(Presets.SCIENCE, figsize='single', fontsize=8):
    fig, ax = plt.subplots()
    ax.plot(x, y)

# 旧方式：使用字符串（仍然支持）
with PlotStyle('science', figsize='single', fontsize=8):
    fig, ax = plt.subplots()
```

**可用的 Presets 常量**:

```python
# 基础样式
Presets.SCIENCE          # 'science' - 无衬线，默认推荐
Presets.SCIENCE_SERIF    # 'science-serif' - 衬线
Presets.MINIMAL          # 'minimal' - 极简风格
Presets.PRESENTATION     # 'presentation' - 演示/幻灯片
Presets.NOTEBOOK         # 'notebook' - Jupyter Notebook
Presets.IEEE             # 'ieee' - IEEE 期刊
Presets.SCATTER          # 'scatter' - 散点图模式

# 中文支持
Presets.CHINESE          # 'chinese' - 无衬线+中文
Presets.CHINESE_SERIF    # 'chinese-serif' - 衬线+中文

# 颜色预设（色盲友好）
Presets.COLORS_VIBRANT   # 'colors-vibrant' - 高饱和度
Presets.COLORS_BRIGHT    # 'colors-bright' - 明亮
Presets.COLORS_CONTRAST  # 'colors-contrast' - 高对比度（黑白打印友好）
```

**优势**:
- ✅ IDE 自动补全（输入 `Presets.` 即可看到所有选项）
- ✅ 类型检查支持（mypy, pylance 等）
- ✅ 避免拼写错误（常量在导入时就会验证）
- ✅ 代码更易读（`Presets.SCIENCE` vs `'science'`）

**使用示例**:

```python
from eqtools.viztools import PlotStyle, Presets
import matplotlib.pyplot as plt

# 单个预设
with PlotStyle(Presets.SCIENCE, figsize='single'):
    fig, ax = plt.subplots()

# 多个预设叠加（可以混用常量和字符串）
with PlotStyle([Presets.SCIENCE, Presets.COLORS_BRIGHT], figsize='double'):
    fig, axes = plt.subplots(1, 2)

# 持久应用
PlotStyle.apply(Presets.MINIMAL, fontsize=9)
# ...
PlotStyle.reset()

# 装饰器
@PlotStyle.decorator(Presets.PRESENTATION, figsize='single')
def my_plot():
    fig, ax = plt.subplots()
    return fig
```

---

## PlotStyle 三种使用模式

### 模式 1：上下文管理器（局部作用域，推荐）

```python
from eqtools.viztools import PlotStyle, Presets

# 使用 Presets 常量（推荐）
with PlotStyle(Presets.SCIENCE, figsize='single', fontsize=8):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fig.savefig('out.pdf')

# 或使用字符串（仍然支持）
with PlotStyle('science', figsize='single', fontsize=8):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    fig.savefig('out.pdf')
# 退出 with 块后精确恢复被改动的 rcParams（不影响未改动的键）
# 即使 with 块内抛出异常也能正确恢复
```

适合：函数内、单图绘制、需要嵌套多种样式时。

### 模式 2：持久应用（脚本 / Notebook）

```python
from eqtools.viztools import PlotStyle, Presets

# 使用 Presets 常量
PlotStyle.apply(Presets.SCIENCE, fontsize=9, dpi=150)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
# ...

PlotStyle.reset()       # 恢复上一次 apply 前的状态
# PlotStyle.reset_all() # 若多次 apply，一次恢复到最初始状态
```

适合：Jupyter Notebook 全局设置、批量绘图。

### 模式 3：装饰器（函数级）

```python
from eqtools.viztools import PlotStyle, Presets

# 使用 Presets 常量
@PlotStyle.decorator(Presets.SCIENCE, figsize='single', fontsize=8)
def plot_profile(data):
    fig, ax = plt.subplots()
    ax.plot(data.distances, data.values)
    return fig

fig = plot_profile(my_data)   # 每次调用自动应用、自动恢复
```

适合：封装绘图函数，保证每次调用都有一致样式。

---

## subplots() 一步创建图窗

`PlotStyle.subplots()` 将**样式激活**和**创建图窗**合并为一步，
并在图窗**关闭时自动 reset** rcParams：

```python
# 基本用法
fig, ax = PlotStyle('science', figsize='single', fontsize=8).subplots()

# 多子图
fig, axes = PlotStyle('science', figsize='double', fontsize=8).subplots(1, 3, sharey=True)

# 叠加 preset
fig, ax = PlotStyle(['science', 'minimal'], figsize='single').subplots()
```

> **关闭时自动 reset**：图窗关闭（`plt.close(fig)` 或点击关闭按钮）时，
> rcParams 自动恢复，不需要手动 `PlotStyle.reset()`。
> 适合交互式绘图和 Notebook。

---

## 在封装函数中正确使用

### 问题背景

当你将绘图逻辑封装在函数里，然后在外部调用 `plt.show()`，
字体可能与封装内不一致：

```python
def plot_result(data):
    PlotStyle.apply('science', fontsize=8)
    fig, ax = plt.subplots()
    ax.set_xlabel(r'Distance $\Delta x$ (km)')
    PlotStyle.reset()      # ← reset 后 rcParams 变回默认
    return fig

fig = plot_result(data)
plt.show()                 # ← 此时 font.sans-serif 已是默认值，字体变了！
```

**根本原因**：matplotlib `Text` 对象存储 `fontfamily='sans-serif'` 字符串，
在渲染时才从当前 `font.sans-serif` rcParam 列表查找实际字体。
`PlotStyle.reset()` 后列表变回系统默认，`plt.show()` 触发渲染时字体已不是预设字体。

### 解决方案：`bake_text_fonts(fig)`

在 `PlotStyle` 仍激活时调用 `bake_text_fonts(fig)`，
将所有 `Text` artist 的字体从字符串解析为具体字体名，使其独立于后续 rcParam 变化：

```python
from eqtools.viztools import PlotStyle, bake_text_fonts

def plot_result(data):
    PlotStyle.apply('science', fontsize=8)
    fig, ax = plt.subplots()
    ax.set_xlabel(r'Distance $\Delta x$ (km)')
    ax.plot(data.x, data.y)
    bake_text_fonts(fig)   # ← 在 reset 前调用，此时 rcParams 仍是预设值
    PlotStyle.reset()
    return fig

fig = plot_result(data)
plt.show()                 # ✓ 字体与封装内一致
```

### 封装函数最佳实践

#### 方法 A：`with` + `bake_text_fonts`（推荐，异常安全）

```python
from eqtools.viztools import PlotStyle, bake_text_fonts

def plot_offset_along_fault(results, save_path=None, fontsize=8):
    """沿断层位移图，可在外部调用 plt.show()"""
    with PlotStyle('science', figsize='double', fontsize=fontsize):
        fig, ax = plt.subplots()
        ax.errorbar(results.distances, results.offsets, yerr=results.errors,
                    fmt='o-', markersize=3, capsize=2, linewidth=0.8)
        ax.set_xlabel('Distance along fault (km)')
        ax.set_ylabel('|Offset| (m)')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        bake_text_fonts(fig)   # reset 前固化字体
    # with 退出：rcParams 自动恢复（即使 savefig 出错也会恢复���
    return fig
```

#### 方法 B：`apply` / `reset` + `bake_text_fonts`

```python
def plot_result(data, config=None):
    preset = config.style if config else 'science'
    fontsize = config.fontsize if config else 8

    PlotStyle.apply(preset, fontsize=fontsize)
    try:
        fig, ax = plt.subplots()
        # ... 绘图代码 ...
        bake_text_fonts(fig)
    finally:
        PlotStyle.reset()   # 无论是否出错都会 reset
    return fig
```

#### 方法 C：装饰器（最简洁，但 bake 需手动加）

```python
@PlotStyle.decorator('science', figsize='single', fontsize=8)
def plot_profile(data):
    fig, ax = plt.subplots()
    ax.plot(data.x, data.y)
    bake_text_fonts(fig)   # 在 return 前固化
    return fig
```

### 何时不需要 `bake_text_fonts`

- 在 `with PlotStyle(...)` 块内直接调用 `plt.show()` — 渲染时 rcParams 仍是预设值
- 只保存 PDF/PNG 不在外部 show — 渲染在 `savefig` 时发生，此时 rcParams 仍激活
- 使用 `PlotStyle.subplots()` — 图窗关闭时才 reset，show 时 rcParams 仍激活

---

## 内置 Preset 列表

### 基础样式

| Preset | 字体 | 数学字体 | 说明 |
| --- | --- | --- | --- |
| `science` | 无衬线（Arial/Helvetica） | STIXSans | **默认推荐**，适合地球科学期刊 |
| `science-serif` | 衬线（Times New Roman） | STIX | 传统学术风格 |
| `chinese` | 无衬线 + 中文（SimHei 等） | STIXSans | 含中文图件 |
| `chinese-serif` | 衬线 + 中文（SimSun 等） | STIX | 衬线中文 |
| `presentation` | 无衬线，字体更大 | STIXSans | 幻灯片 / 海报 |
| `notebook` | 无衬线，中等大小 | STIXSans | Jupyter Notebook |
| `minimal` | 继承 science，去掉顶/右边框 | STIXSans | 极简风格 |

### 色彩调色板（叠加用）

只修改 `axes.prop_cycle`，叠加在基础样式之上：

| Preset | 色板 | 颜色数 | 说明 |
| --- | --- | --- | --- |
| `colors-bright` | Paul Tol Bright | 7 | 色盲友好，明亮，通用首选 |
| `colors-vibrant` | Paul Tol Vibrant | 7 | 色盲友好，高饱和度 |
| `colors-contrast` | High-contrast | 3 | 色盲友好 + 黑白打印安全，3 条线首选 |

```python
with PlotStyle(['science', 'colors-bright'], figsize='single'):
    fig, ax = plt.subplots()
    for s in dataset:
        ax.plot(s.x, s.y)   # 自动循环色盲友好色板
```

### 绘图模式（叠加用）

| Preset | 说明 | 适用场景 |
| --- | --- | --- |
| `scatter` | 仅标记点，无连线（7 种标记 × std-colors） | 散点图、观测数据 |
| `ieee` | 4 色 × 4 线型，黑白打印安全（继承 science-serif） | IEEE 双栏投稿 |

```python
# 散点模式：ax.plot() 自动变为仅标记
with PlotStyle(['science', 'scatter'], figsize='single'):
    fig, ax = plt.subplots()
    ax.plot(lon, lat)    # → 圆形标记，无连线
    ax.plot(lon2, lat2)  # → 方形标记，下一个颜色
```

```python
# 列出所有 preset
from eqtools.viztools import list_presets
for name, desc in list_presets().items():
    print(f'  {name:<20s}  {desc}')
```

---

## 自定义与扩展（custom 用）

### 注册自定义 Preset

```python
from eqtools.viztools import register_preset, PlotStyle

register_preset(
    'my_lab',
    base='science',               # 继承 science 的所有设置
    rcparams={
        'axes.grid':       True,
        'grid.alpha':      0.25,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
    },
    description='Lab house style with grid',
)

with PlotStyle('my_lab', figsize='single', fontsize=8):
    fig, ax = plt.subplots()
    ax.plot(x, y)
```

### 注销自定义 Preset

```python
from eqtools.viztools import register_preset, unregister_preset

register_preset('test', base='science', rcparams={'axes.grid': True})
unregister_preset('test')   # 删除用户注册的 preset

# 内置 preset（science、minimal 等）不可注销：
try:
    unregister_preset('science')
except ValueError as e:
    print(e)   # Cannot unregister built-in preset 'science'
```

### 多重继承（`base` 为列表）

```python
register_preset(
    'my_bright',
    base=['science', 'colors-bright'],  # 从左到右叠加，右边优先级高
    rcparams={'axes.grid': True},
    description='Science + bright colors + grid',
)

with PlotStyle('my_bright', figsize='double'):
    ...
```

### 中文支持

```python
register_preset(
    'my_lab_cn',
    base='my_lab',
    chinese=True,
    chinese_prefer_serif=False,    # True 则优先 SimSun 等宋体
    description='Lab style with Chinese support',
)

PlotStyle.apply('my_lab_cn', fontsize=9)
ax.set_title('断层地表形变剖面')
PlotStyle.reset()
```

### 注册自定义期刊列宽

```python
from eqtools.viztools import register_column_width, publication_figsize

register_column_width('agu_single',   3.37)
register_column_width('agu_double',   6.83)
register_column_width('copernicus',   3.15)

# 之后直接用名称
fig, ax = PlotStyle('science', figsize='agu_single').subplots()
w, h = publication_figsize('agu_single', aspect=0.8)
```

**用户配置文件（持久化）**：创建 `~/.config/eqtools/viztools.json`，import 时自动加载：

```json
{
  "column_widths": {
    "agu_single":  3.37,
    "agu_double":  6.83,
    "copernicus":  3.15,
    "my_journal":  4.00
  }
}
```

> 旧路径 `~/.config/eqtools/plottools.json`、`~/.config/statutils/plottools.json`、
> `~/.plottools.json` 仍可使用，但会触发 `DeprecationWarning`。

### 从文件系统加载自定义样式（v2.2.0+）

除了使用 `register_preset()` 在代码中注册样式，现在可以直接从文件系统加载 `.mplstyle` 文件：

```python
from eqtools.viztools import register_style_directory, PlotStyle

# 注册包含 .mplstyle 文件的目录
register_style_directory('~/my_matplotlib_styles')
register_style_directory('/path/to/project/styles')

# 目录中的所有 .mplstyle 文件自动注册为用户预设
# 例如: my_custom.mplstyle → 可以使用 'my_custom'
with PlotStyle('my_custom', figsize='single'):
    fig, ax = plt.subplots()
```

#### 文件结构示例

```
~/my_matplotlib_styles/
├── lab_default.mplstyle          # → 注册为 'lab_default'
├── paper_nature.mplstyle         # → 注册为 'paper_nature'
├── presentation_16x9.mplstyle    # → 注册为 'presentation_16x9'
├── thesis_chapter.mplstyle       # → 注册为 'thesis_chapter'
└── _experimental.mplstyle        # 下划线开头，被忽略
```

#### .mplstyle 文件内容示例

```ini
# lab_default.mplstyle
# Lab house style with grid

# 基础设置
axes.linewidth: 1.2
axes.grid: True
grid.alpha: 0.25
grid.linestyle: --
grid.color: gray

# 线条样式
lines.linewidth: 1.5
lines.markersize: 4

# 字体（将被 PlotStyle 的 fontsize 参数覆盖）
font.size: 10
```

#### 使用场景

1. **团队协作** - 共享实验室/团队的统一样式
   ```bash
   # 克隆团队样式库
   git clone https://github.com/mylab/mpl-styles ~/lab_styles
   ```
   ```python
   # Python 代码中使用
   register_style_directory('~/lab_styles')
   with PlotStyle('lab_2024', figsize='single'):
       ...
   ```

2. **项目级配置** - 每个项目维护自己的样式
   ```
   my_project/
   ├── styles/
   │   ├── figure_main.mplstyle
   │   └── figure_supplement.mplstyle
   ├── scripts/
   │   └── plot.py
   └── data/
   ```
   ```python
   # plot.py
   from pathlib import Path
   from eqtools.viztools import register_style_directory

   PROJECT_ROOT = Path(__file__).parent.parent
   register_style_directory(PROJECT_ROOT / 'styles')

   with PlotStyle('figure_main'):
       ...
   ```

3. **期刊特定样式** - 为不同期刊准备专用样式
   ```python
   register_style_directory('~/journal_styles')

   # Nature 投稿
   with PlotStyle('nature_single_column'):
       fig, ax = plt.subplots()

   # Science 投稿
   with PlotStyle('science_double_column'):
       fig, axes = plt.subplots(1, 2)
   ```

#### 与 register_preset() 的对比

| 特性 | `register_style_directory()` | `register_preset()` |
|------|------------------------------|---------------------|
| 样式来源 | 文件系统（.mplstyle 文件） | 代码中定义 |
| 团队共享 | 容易（Git/共享文件夹） | 需要共享代码 |
| 版本控制 | 直接（样式文件） | 需要提交代码 |
| 继承支持 | 通过 mplstyle 语法 | `base=` 参数 |
| 中文支持 | 需要在 mplstyle 中配置 | `chinese=True` 参数 |
| 灵活性 | 中等（mplstyle 语法限制） | 高（Python 字典） |
| 适用场景 | 固定样式库、团队协作 | 动态生成、复杂逻辑 |

#### 组合使用

两种方法可以组合使用：

```python
from eqtools.viztools import register_style_directory, register_preset, PlotStyle

# 1. 从文件系统加载基础样式
register_style_directory('~/lab_styles')

# 2. 在代码中基于文件样式创建变体
register_preset(
    'lab_with_grid',
    base='lab_default',  # 继承文件定义的样式
    rcparams={
        'axes.grid': True,
        'grid.alpha': 0.3,
    },
    description='Lab default + grid',
)

# 3. 使用
with PlotStyle('lab_with_grid', figsize='single'):
    ...
```

### 检查 Preset 内容

```python
# 打印并返回 dict
rc = PlotStyle.describe('minimal')
# Preset: minimal  (base: science)
# mplstyles : ['eqtools-minimal']
# Resolved rcparams (N keys):
#   axes.spines.right = False
#   axes.spines.top   = False
#   axes.unicode_minus = False
#   mathtext.fontset  = stixsans
#   ...

# 只返回指定前缀的键（同时打印）
rc_fonts = PlotStyle.describe('science', filter_prefix='font.')
assert all(k.startswith('font.') for k in rc_fonts)

# 静默模式（只返回 dict，不打印）
rc = PlotStyle.describe('science', print_result=False)
```

---

## 高级扩展功能

### 自定义 Handler 注册

从 v2.1 开始，PlotStyle 支持注册自定义 handler 来扩展样式处理链，无需修改源代码。

#### 基本用法

```python
from eqtools.viztools import PlotStyle

# 定义自定义 handler
def grid_handler(ps, acc):
    """自定义网格样式 handler"""
    if hasattr(ps, '_enable_grid') and ps._enable_grid:
        acc['axes.grid'] = True
        acc['grid.alpha'] = 0.3
        acc['grid.linestyle'] = '--'
        acc['grid.color'] = 'gray'

# 注册 handler
PlotStyle.register_handler(grid_handler, 'grid', priority=55)

# 使用
ps = PlotStyle('science', fontsize=10)
ps._enable_grid = True

with ps:
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    # 网格自动启用

# 清理
PlotStyle.unregister_handler('grid')
```

#### Handler 签名

```python
def my_handler(plotstyle_instance, acc_dict):
    """
    Parameters
    ----------
    plotstyle_instance : PlotStyle
        PlotStyle 实例，可访问其属性（如 _fontsize, _presets）
    acc_dict : dict
        累积的 rcParams 字典，原地修改以添加/覆盖设置
    """
    if some_condition:
        acc_dict['some.rcparam'] = value
```

#### 优先级系统

- **0-100**: 保留给内置 handler
- **50 (默认)**: 推荐用于大多数自定义 handler
- **更高值**: 更晚执行（更高优先级）
- 自定义 handler 在内置 handler 之后、`_apply_extra` 之前执行

#### Handler 管理 API

```python
# 注册
PlotStyle.register_handler(handler_fn, name='my_handler', priority=50)

# 注销
PlotStyle.unregister_handler('my_handler')
PlotStyle.unregister_handler('my_handler', priority=50)  # 只从指定优先级移除

# 列出所有自定义 handler
handlers = PlotStyle.list_handlers()
# {50: ['handler1'], 60: ['handler2']}
```

#### 使用场景

- **条件样式**: 基于自定义属性应用样式
- **第三方集成**: 从外部样式系统注入设置
- **领域特定默认值**: 地震学、气候科学等领域的专用样式
- **动态主题**: 程序化切换明暗模式

### Axes 级别样式应用

对单个 Axes 应用样式，而不影响全局 rcParams。适用于混合样式图（如主图 + 插图）。

```python
from eqtools.viztools import PlotStyle
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 对不同 axes 应用不同样式
PlotStyle('science', fontsize=10).apply_to_axes(ax1)
PlotStyle('minimal', fontsize=8).apply_to_axes(ax2)

x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x))
ax1.set_title('Science 样式 (10pt)')

ax2.plot(x, np.cos(x))
ax2.set_title('Minimal 样式 (8pt)')

plt.tight_layout()
plt.show()
```

#### 支持的设置

- **字体大小**: `axes.labelsize`, `xtick.labelsize`, `ytick.labelsize`, `axes.titlesize`
- **网格**: `axes.grid`, `grid.alpha`, `grid.linewidth`
- **脊柱**: `axes.spines.{top,right,left,bottom}`, `axes.linewidth`
- **刻度**: `{x,y}tick.{major,minor}.{width,size}`

全局设置（如 `figure.figsize`, `font.family`）会被忽略。

#### 使用场景

- 主图 + 插图（不同字体大小）
- 多面板图（每个面板类型一致的样式）
- 叠加图（每层不同的网格/脊柱设置）
- 交互式仪表板（每个小部件的样式）

### 增强的地理格式化器

`LatFormatter` 和 `LonFormatter` 现在支持度分秒（DMS）格式和精度控制。

#### 小数度格式（带精度）

```python
from eqtools.viztools import LatFormatter, LonFormatter

fig, ax = plt.subplots()
ax.contourf(lons, lats, data)

# 小数度，2 位小数
ax.xaxis.set_major_formatter(LonFormatter(decimal_places=2))  # 96.50°E
ax.yaxis.set_major_formatter(LatFormatter(decimal_places=2))  # 21.25°N
```

#### DMS 格式（度分秒）

```python
# 基本 DMS 格式
ax.xaxis.set_major_formatter(LonFormatter(format='dms'))  # 96°30'15"E
ax.yaxis.set_major_formatter(LatFormatter(format='dms'))  # 21°15'30"N

# DMS 格式，秒带小数
ax.xaxis.set_major_formatter(LonFormatter(format='dms', dms_precision=2))  # 96°30'15.50"E
ax.yaxis.set_major_formatter(LatFormatter(format='dms', dms_precision=2))  # 21°15'30.25"N
```

#### DMSFormatter（通用 DMS，无半球后缀）

```python
from eqtools.viztools import DMSFormatter

fig, ax = plt.subplots()
x = np.linspace(-180, 180, 100)
y = np.sin(x * np.pi / 180)
ax.plot(x, y)

# 不带符号
ax.xaxis.set_major_formatter(DMSFormatter())  # 45°30'15", -12°15'30"

# 带符号
ax.xaxis.set_major_formatter(DMSFormatter(show_sign=True))  # +45°30'15", -12°15'30"

# 秒带小数
ax.xaxis.set_major_formatter(DMSFormatter(precision=1))  # 45°30'15.5"
```

#### 格式对比

| 格式 | 示例输出 | 使用场景 |
| --- | --- | --- |
| `decimal_places=0` | `96°E`, `21°N` | 低精度地图 |
| `decimal_places=2` | `96.50°E`, `21.25°N` | 高精度地图 |
| `format='dms'` | `96°30'15"E`, `21°15'30"N` | 导航、测量 |
| `format='dms', dms_precision=2` | `96°30'15.50"E`, `21°15'30.25"N` | 高精度大地测量 |

#### API 参考

```python
LatFormatter(decimal_places=0, format='decimal', dms_precision=0)
LonFormatter(decimal_places=0, format='decimal', dms_precision=0)
DMSFormatter(precision=0, show_sign=False)
```

**参数说明**:
- `decimal_places`: 小数度格式的小数位数
- `format`: `'decimal'`（小数度）或 `'dms'`（度分秒）
- `dms_precision`: DMS 格式中秒的小数位数
- `show_sign`: 是否显示正负号（仅 DMSFormatter）

---

## 参数完整参考

### `PlotStyle.__init__` 参数

```python
PlotStyle(
    preset           = 'science',  # str 或 list[str]，preset 名称，左→右优先级递增

    # ── 图尺寸 ──────────────────────────────────────────────────────────────
    figsize          = None,       # 'single'|'double'|'nature'|'ieee'|'a4'
                                   # | 自定义名 | 数值 | (w, h) 元组
    figsize_unit     = 'inch',     # 'inch' 或 'cm'
    figsize_fraction = 1.0,        # 列宽比例（0~1）
    figsize_aspect   = 0.75,       # 高宽比（figsize_height 未指定时生效）
    figsize_height   = None,       # 显式高度（覆盖 figsize_aspect）

    # ── 字体大小（细粒度控制）──────────────────────────────────────────────
    fontsize         = None,       # float，基准字号：设置 font.size + axes.labelsize
    tick_fontsize    = None,       # float，刻度字号（默认 max(fontsize-1, 6)）
    legend_fontsize  = None,       # float，图例字号（默认 max(fontsize-1, 6)）
    title_fontsize   = None,       # float，标题字号（默认 fontsize+1）

    # ── 其他样式 ────────────────────────────────────────────────────────────
    legend_frame     = False,      # bool，是否显示图例边框（含 alpha=0.7）
    dpi              = None,       # float，同时设置 figure.dpi 和 savefig.dpi
    pdf_fonttype     = None,       # 3 或 42（42 = 文字可选中，推荐）

    # ── 字体引擎 ────────────────────────────────────────────────────────────
    usetex           = None,       # True=LaTeX（自动 preamble）/False=关/None=不变
    mathfont         = None,       # 'stixsans'|'stix'|'cm'|'dejavusans'|... / None=自动

    # ── 最高优先级覆盖 ───────────────────────────────────────────────────────
    rcparams         = None,       # dict，原始 rcParam 键值对，覆盖所有上述参数
)
```

### 字号细粒度示例

```python
# 全部由 fontsize 自动计算（向后兼容默认行为）
with PlotStyle('science', fontsize=8):
    # font.size = axes.labelsize = 8
    # xtick.labelsize = ytick.labelsize = legend.fontsize = max(7, 6) = 7
    # figure.titlesize = 9
    ...

# 自定义各层字号
with PlotStyle('science', fontsize=8,
               tick_fontsize=6,      # 刻度比正文小
               legend_fontsize=7,    # 图例与正文一样大
               title_fontsize=10):   # 标题显著更大
    ...
```

### 优先级链（从低到高）

```text
mplstyle 文件（.mplstyle）
  → preset 的 rcparams 字典
  → chinese 字体注入（若 chinese=True）
  → figsize 参数
  → fontsize / tick_fontsize / legend_fontsize / title_fontsize 参数
  → legend_frame 参数
  → dpi 参数
  → pdf_fonttype 参数
  → mathfont 参数（含自动检测）
  → usetex 参数（含自动 preamble 注入）
  → rcparams= 字典（最高优先级，覆盖一切）
```

---

## 图尺寸系统

### 内置列宽

| 名称 | 宽度（英寸） | 说明 |
| --- | --- | --- |
| `'single'` | 3.50 | AGU/GRL 单栏（89 mm） |
| `'double'` | 7.20 | AGU/GRL 双栏（183 mm） |
| `'nature'` | 3.42 | Nature 单栏（87 mm） |
| `'ieee'` | 3.50 | IEEE 单栏 |
| `'ieee_double'` | 7.16 | IEEE 双栏 |
| `'a4'` | 8.27 | A4 文本区宽 |

### `publication_figsize()` 使用

```python
from eqtools.viztools import publication_figsize

w, h = publication_figsize('single')                    # (3.5, 2.625)
w, h = publication_figsize('double', fraction=0.8)      # (5.76, 4.32)
w, h = publication_figsize('single', aspect=0.618)      # 黄金比例
w, h = publication_figsize(8.8, unit='cm')              # cm 输入
w, h = publication_figsize((8.8, 6.5), unit='cm')       # 精确元组
w, h = publication_figsize('single', height=2.0)        # 固定高度
```

### 常用场景尺寸

```python
# AGU 单栏
PlotStyle.apply('science', figsize='single', fontsize=8)

# AGU 双栏（两子图）
PlotStyle.apply('science', figsize='double', fontsize=9)
fig, axes = plt.subplots(1, 2, sharey=True)

# 宽幅（沿断层剖面）
PlotStyle.apply('science', figsize='double', figsize_aspect=0.4, fontsize=8)

# cm 单位（Nature 精确列宽 87 mm）
PlotStyle.apply('science', figsize=8.7, figsize_unit='cm', fontsize=7)

# 自定义期刊（需先注册）
register_column_width('agu_single', 3.37)
PlotStyle.apply('science', figsize='agu_single', fontsize=8)
```

---

## 字体系统

### 文字字体

由 preset 决定，可通过 `rcparams` 覆盖字体列表：

```python
# science → Arial/Helvetica（无衬线）
PlotStyle.apply('science')

# science-serif → Times New Roman（衬线）
PlotStyle.apply('science-serif')

# 手动指定字体回退列表
PlotStyle.apply('science', rcparams={
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
})
```

### 数学字体

`mathfont` 参数控制，**默认跟随文字字体自动选择**：

| 文字 preset | 自动数学字体 | 效果 |
| --- | --- | --- |
| `science`（无衬线） | `stixsans` | 无衬线数学，与 Arial 匹配 |
| `science-serif`（衬线） | `stix` | 衬线数学，与 Times 匹配 |

```python
PlotStyle.apply('science')                    # → stixsans（自动）
PlotStyle.apply('science', mathfont='cm')     # Computer Modern（传统 LaTeX 风格）
PlotStyle.apply('science', mathfont='stix')   # 衬线 STIX（显式）
```

可用值：`'stixsans'` · `'stix'` · `'cm'` · `'computer-modern'` ·
`'dejavusans'` · `'dejavuserif'`

### LaTeX 渲染（usetex）

> **前提**：系统已安装 MiKTeX 或 TeX Live

```python
# 无衬线 LaTeX（science + helvet + sfmath，自动注入 preamble）
PlotStyle.apply('science', usetex=True)
# → 文字：Helvetica；数学：sfmath（无衬线）

# 衬线 LaTeX（science-serif + lmodern）
PlotStyle.apply('science-serif', usetex=True)
# → 文字：Latin Modern Roman；数学：Computer Modern
```

**preamble 自动注入规则**：

```text
font.family = sans-serif  →  \usepackage{helvet}
                               \renewcommand{\familydefault}{\sfdefault}
                               \usepackage{sfmath}
                               \usepackage{amsmath}
                               \usepackage{amssymb}

font.family = serif       →  \usepackage[T1]{fontenc}
                               \usepackage{lmodern}
                               \usepackage{amsmath}
                               \usepackage{amssymb}
```

---

## 其他工具函数

### `save_fig()` — 多格式保存

```python
from eqtools.viztools import save_fig

save_fig(fig, 'result.pdf')                          # 有扩展名直接用
save_fig(fig, 'result', fmts=['pdf', 'png'])         # 多格式
save_fig(fig, 'result', fmts=['pdf', 'svg', 'png'])  # 三格式
save_fig(fig, 'result.pdf', dpi=600, transparent=True)  # 额外 kwargs
# 每个文件保存后打印: Saved: result.pdf
```

### `LatFormatter` / `LonFormatter` — 地理坐标格式化

```python
from eqtools.viztools import LatFormatter, LonFormatter

fig, ax = plt.subplots()

# 小数度格式（整数）
ax.yaxis.set_major_formatter(LatFormatter())          # 21°N, 0°, 5°S
ax.xaxis.set_major_formatter(LonFormatter())          # 96°E, 180°, 5°W

# 小数度格式（带精度）
ax.yaxis.set_major_formatter(LatFormatter(decimal_places=1))  # 21.5°N
ax.xaxis.set_major_formatter(LonFormatter(decimal_places=2))  # 96.50°E

# DMS 格式（度分秒）
ax.yaxis.set_major_formatter(LatFormatter(format='dms'))  # 21°15'30"N
ax.xaxis.set_major_formatter(LonFormatter(format='dms'))  # 96°30'15"E

# DMS 格式（秒带小数）
ax.yaxis.set_major_formatter(LatFormatter(format='dms', dms_precision=2))  # 21°15'30.25"N
ax.xaxis.set_major_formatter(LonFormatter(format='dms', dms_precision=2))  # 96°30'15.50"E
```

### `DMSFormatter` — 通用度分秒格式化

```python
from eqtools.viztools import DMSFormatter

fig, ax = plt.subplots()
x = np.linspace(-180, 180, 100)
ax.plot(x, np.sin(x * np.pi / 180))

# 基本 DMS 格式（无半球后缀）
ax.xaxis.set_major_formatter(DMSFormatter())  # 45°30'15", -12°15'30"

# 带正负号
ax.xaxis.set_major_formatter(DMSFormatter(show_sign=True))  # +45°30'15", -12°15'30"

# 秒带小数
ax.xaxis.set_major_formatter(DMSFormatter(precision=1))  # 45°30'15.5"
```

### `set_degree_formatter()` / `DegreeFormatter` — 通用度符号

```python
from eqtools.viztools import set_degree_formatter

set_degree_formatter(ax, axis='both')   # 两轴加 °（不区分 N/S/E/W）
set_degree_formatter(ax, axis='x')     # 只对 x 轴
set_degree_formatter(ax, axis='y')     # 只对 y 轴
```

### `get_color_cycle()` — 获取当前或 preset 颜色列表

```python
from eqtools.viztools import get_color_cycle

colors = get_color_cycle()                  # 当前 rcParams 中的颜色列表
colors = get_color_cycle('colors-bright')   # 指定 preset（不修改全局状态）
colors = get_color_cycle('science')         # science preset 的默认色板

# 用途示例
for i, data in enumerate(dataset):
    ax.plot(data.x, data.y, color=colors[i % len(colors)])
```

### `bake_text_fonts()` — 固化字体（封装函数必备）

```python
from eqtools.viztools import bake_text_fonts

# 在 PlotStyle 仍激活时调用，将 Text artist 字体固化为具体字体名
# 使 plt.show() 的渲染结果与封装内一致
# → 详见"在封装函数中正确使用"章节
bake_text_fonts(fig)
```

### `PlotStyle.describe()` — 检查 preset 详情

```python
# 打印并返回完整 dict
rc = PlotStyle.describe('science')

# 只返回 font. 开头的键（也会打印）
rc_fonts = PlotStyle.describe('science', filter_prefix='font.')

# 静默模式（只返回 dict，不打印）
rc = PlotStyle.describe('minimal', print_result=False)
assert isinstance(rc, dict)
```

### `PlotStyle.reset_all()` — 清空所有 apply 层

```python
PlotStyle.apply('science')
PlotStyle.apply('colors-bright')   # 叠加第二层

PlotStyle.reset()      # 撤销最近一次 apply（仅 colors-bright）
PlotStyle.reset()      # 再撤销上一次（仅 science）

# 或一次性全部撤销：
PlotStyle.apply('science')
PlotStyle.apply('colors-bright')
PlotStyle.reset_all()              # 两层一次恢复
```

### `publication_figsize()` — 计算出版图尺寸

```python
from eqtools.viztools import publication_figsize

w, h = publication_figsize('single')                   # (3.5, 2.625)
w, h = publication_figsize('double', fraction=0.5)     # 半双栏宽
w, h = publication_figsize(8.7, unit='cm')             # cm 输入
w, h = publication_figsize('single', height=2.0)       # 固定高度（英寸）
```

### `list_chinese_fonts()` — 查询系统 CJK 字体

```python
from eqtools.viztools import list_chinese_fonts

fonts = list_chinese_fonts()
print(fonts)
# {'sans': 'SimHei', 'serif': 'SimSun'}
# 值为 None 表示未找到
```

---

## 常用场景示例

### AGU/GRL 投稿图

```python
from eqtools.viztools import PlotStyle, Presets, save_fig

# 使用 Presets 常量（推荐）
PlotStyle.apply(Presets.SCIENCE, figsize='single', fontsize=8)

fig, ax = plt.subplots()
ax.plot(distances, values, 'b-', lw=1.2, label=r'$u_z$')
ax.set_xlabel(r'Fault-normal distance $\Delta x$ (km)')
ax.set_ylabel(r'Displacement $u_z$ (mm)')
ax.legend()
save_fig(fig, 'figure', fmts=['pdf', 'png'])

PlotStyle.reset()
```

### 含公式的高质量图（LaTeX）

```python
from eqtools.viztools import PlotStyle, Presets, save_fig

PlotStyle.apply(Presets.SCIENCE, usetex=True, figsize='single', fontsize=8)

fig, ax = plt.subplots()
ax.plot(x, y, lw=1.2,
        label=r'$u_z = \frac{S}{2}\left[1-\frac{x}{\sqrt{x^2+d^2}}\right]$')
ax.set_xlabel(r'Distance $\Delta x$ (km)')
ax.legend(fontsize=7)
save_fig(fig, 'figure.pdf')

PlotStyle.reset()
```

### 地理坐标轴（经纬度标注）

```python
from eqtools.viztools import PlotStyle, Presets, LatFormatter, LonFormatter, save_fig

with PlotStyle(Presets.SCIENCE, figsize='single', fontsize=8):
    fig, ax = plt.subplots()
    ax.scatter(lon, lat, c=values, cmap='RdBu_r', s=5)
    ax.xaxis.set_major_formatter(LonFormatter())
    ax.yaxis.set_major_formatter(LatFormatter())
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    save_fig(fig, 'map.pdf')
```

### 多子图共享颜色

```python
from eqtools.viztools import PlotStyle, Presets, get_color_cycle, save_fig

colors = get_color_cycle(Presets.COLORS_BRIGHT)   # 色盲友好色板

with PlotStyle([Presets.SCIENCE, Presets.COLORS_BRIGHT], figsize='double', fontsize=8):
    fig, axes = plt.subplots(1, 3, sharey=True)
    for i, (ax, data) in enumerate(zip(axes, dataset)):
        ax.plot(data.x, data.y, color=colors[i], lw=1.2)
    save_fig(fig, 'comparison.pdf')
```

### Notebook 全局设置

```python
from eqtools.viztools import PlotStyle, Presets

PlotStyle.apply(Presets.NOTEBOOK, fontsize=10)

# 后续所有 plt 命令自动使用该样式
for i, data in enumerate(dataset):
    fig, ax = plt.subplots()
    ax.plot(data.x, data.y)
    ax.set_title(f'Profile {i+1}')
    plt.show()
    plt.close()

PlotStyle.reset()
```

---

## 在可视化模块中的集成

### `fault_profile/visualization.py`

模块级绘图函数内部自动管理 `PlotStyle`，并在 return 前调用 `bake_text_fonts(fig)`：

```python
from statUtils.fault_profile.visualization import (
    plot_single_profile,
    plot_all_profiles,
    plot_offset_along_fault,
    plot_fit_quality,
)
from statUtils.fault_profile.visualization import PlotConfig

# 内部自动 apply/bake/reset，用户只传配置
fig = plot_offset_along_fault(batch_results, save_path='offset.pdf')
plt.show()   # ✓ 字体与封装内一致

# 自定义样式
cfg = PlotConfig(style='science', fontsize=9, dpi=300)
fig = plot_offset_along_fault(batch_results, config=cfg)
```

### `profile_extraction/visualization.py`

`ProfileVisualizer` 在每个公开绘图方法开头自动调用 `self._setup_style()`：

```python
from statUtils.profile_extraction import ProfileVisualizer

vis = ProfileVisualizer()                        # 使用 'science' 默认样式
fig = vis.plot_profile_summary(profile_data)     # 内部自动管理 PlotStyle
```

---

## 常见问题排查

### Q1：`usetex=True` 后 tick label 变成衬线字体

```python
# ❌ 直接开 usetex，LaTeX 默认用 Computer Modern（衬线）
PlotStyle.apply('science', rcparams={'text.usetex': True})

# ✓ 用 usetex=True 参数，自动注入 helvet+sfmath
PlotStyle.apply('science', usetex=True)
```

### Q2：`plt.show()` 字体与 savefig 不一致

原因分两种，处理方式不同：

**情形 A：`usetex=False`（matplotlib 内置渲染）**

`Text` 对象存储 `fontfamily='sans-serif'` 字符串，渲染时才查 `font.sans-serif` 列表。
`PlotStyle.reset()` 后列表变回默认，`plt.show()` 触发渲染时字体已不是预设字体。

修复：在 `PlotStyle` 仍激活时调用 `bake_text_fonts(fig)`，固化字体名：

```python
with PlotStyle('science', fontsize=8):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    bake_text_fonts(fig)   # ← reset 前固化
# with 退出：rcParams 恢复
return fig
# 外部 plt.show() → 字体已固化，一致 ✓
```

**情形 B：`usetex=True`（pdflatex 渲染管道）**

`bake_text_fonts` 对 usetex 模式**无效**。`Text.fontfamily` 属性根本不参与 pdflatex 渲染，
关键状态是 `text.usetex` 这个 rcParam 本身。
`PlotStyle.reset()` 后 `text.usetex` 变回 `False`，`plt.show()` 退回到 matplotlib 内置渲染：

- 字体：DejaVu（不是 Helvetica）
- `\textbf{A}` 之类的 LaTeX 命令显示为字面量字符串

修复：`plt.show()` 必须在 `PlotStyle` 激活期间调用（`usetex` 仍为 `True`）：

```python
def plot_func(data, show=False):
    with PlotStyle('science', usetex=True, fontsize=8):
        fig, ax = plt.subplots()
        ax.set_title(r'\textbf{A} --- Result')   # LaTeX 命令
        ax.plot(data.x, data.y)
        fig.savefig('out.pdf')        # savefig 在 with 内 → pdflatex → 正确 ✓

        if show:
            plt.show()                # ← with 内调用，usetex 仍激活 → 正确 ✓

        bake_text_fonts(fig)          # 对 usetex 无效，但对后续非 show 用途无害
    return fig

fig = plot_func(data, show=True)      # ✓ 字体一致
# plt.show()  ← 不要在这里调用，usetex 已恢复 False
```

| 模式 | 渲染引擎 | 修复方式 |
| --- | --- | --- |
| `usetex=False` | matplotlib 内置 | `bake_text_fonts(fig)` 在 reset 前 |
| `usetex=True` | pdflatex | `plt.show()` 在 PlotStyle 激活期间调用 |

### Q3：PDF 文字不可选中

确认 `pdf_fonttype=42`（science preset 默认已设置）：

```python
PlotStyle.apply('my_preset', pdf_fonttype=42)
# 或在注册时加入：
register_preset('my_preset', rcparams={'pdf.fonttype': 42, 'ps.fonttype': 42})
```

### Q4：中文字符显示为方块

```python
from eqtools.viztools import list_chinese_fonts
print(list_chinese_fonts())   # 若 sans/serif 为 None，需安装字体

# Linux: sudo apt install fonts-noto-cjk
# macOS: 安装 Noto Sans CJK
# Windows: 黑体(SimHei)/微软雅黑(Microsoft YaHei) 通常已内置

# 安装后清除字体缓存：
import matplotlib.font_manager
matplotlib.font_manager._load_fontmanager(try_read_cache=False)
```

### Q5：`chinese` + `usetex=True` 报错

PlotStyle 自动检测并禁用不兼容的组合，发出 `UserWarning`：

```python
PlotStyle.apply(['science', 'chinese'], usetex=True)
# UserWarning: usetex has been automatically disabled (CJK incompatible)
```

如需 CJK + 高质量数学，改用 XeLaTeX + PGF backend：

```python
import matplotlib
matplotlib.use('pgf')
matplotlib.rcParams.update({
    'pgf.texsystem': 'xelatex',
    'pgf.preamble': r'\usepackage{xeCJK}\setCJKmainfont{SimSun}',
})
```

### Q6：scienceplots 未安装

可选依赖，未安装时自动跳过，功能完整（使用 eqtools.viztools 自带 `.mplstyle`）。

### Q7：自定义列宽不生效

确认在 `PlotStyle(figsize=...)` 之前已调用 `register_column_width()`，
或在 `~/.config/eqtools/viztools.json` 中注册（import 时自动加载）。

### Q8：旧代码 `from statUtils.plottools import ...` 报 DeprecationWarning

这是预期行为。`statUtils.plottools` 和 `eqtools.plottools` 均已成为向后兼容 shim，
所有功能已迁移到 `eqtools.viztools`。更新导入即可消除警告：

```python
# 旧（仍然可用，但有 DeprecationWarning）
from statUtils.plottools import PlotStyle

# 新（推荐）
from eqtools.viztools import PlotStyle
```

### Q9: matplotlib 版本过低警告

**问题**:
```
UserWarning: eqtools.viztools requires matplotlib 3.3.0 or later,
but version 3.2.0 is installed. Some features may not work correctly.
Please upgrade: pip install --upgrade matplotlib
```

**原因**: viztools 使用了 matplotlib 3.3.0 引入的某些 API

**解决**:
```bash
# 升级 matplotlib
pip install --upgrade matplotlib

# 或指定版本
pip install matplotlib>=3.3.0

# conda 用户
conda update matplotlib
```

**最低版本要求**: matplotlib 3.3.0

---

## 性能优化（v2.2.0+）

### 内存优化

v2.2.0 对样式栈进行了内存优化，现在只保存实际改变的 rcParams 值：

**优化前** (v2.1):
```python
PlotStyle.apply('science', fontsize=8)
# 保存所有 ~300 个 rcParams → 每次 apply 约 150 KB
```

**优化后** (v2.2):
```python
PlotStyle.apply('science', fontsize=8)
# 只保存改变的 ~30 个 rcParams → 每次 apply 约 15 KB
# 内存减少 90%
```

**实测效果**:
- 100 次 apply/reset 循环
- 优化前: ~15 MB
- 优化后: ~0.16 MB
- **减少 99%**

对于频繁调用 apply/reset 的场景（如批量生成图表），内存占用显著降低。

### 字体探测缓存

CJK 字体探测现在使用持久化缓存，极大提升启动速度：

**首次运行**:
```python
from eqtools.viztools import list_chinese_fonts
fonts = list_chinese_fonts()
# 扫描所有系统字体（~100-500ms）
# 结果保存到 ~/.cache/eqtools/font_cache.pkl
```

**后续运行**:
```python
fonts = list_chinese_fonts()
# 从缓存加载（<1ms）
# 提速 100-500 倍
```

**缓存特性**:
- **位置**: `~/.cache/eqtools/font_cache.pkl` (Linux/macOS)
  `C:\Users\<user>\.cache\eqtools\font_cache.pkl` (Windows)
- **有效期**: 7 天自动过期
- **验证**: 检查缓存的字体是否仍然存在
- **刷新**: 手动删除缓存文件或调用 `list_chinese_fonts(refresh=True)`

**清除缓存**:
```python
# 方法 1: 强制刷新
from eqtools.viztools import list_chinese_fonts
fonts = list_chinese_fonts(refresh=True)

# 方法 2: 手动删除缓存文件
import os
from pathlib import Path
cache_file = Path.home() / '.cache' / 'eqtools' / 'font_cache.pkl'
if cache_file.exists():
    cache_file.unlink()
```

---

## API 速查

```python
from eqtools.viztools import (
    # 主类
    PlotStyle,             # 上下文管理器 / apply / reset / decorator / subplots

    # 预设常量（v2.2.0+）
    Presets,               # 预设名称常量类（IDE 友好）

    # Preset 管理
    register_preset,       # 注册自定义 preset
    unregister_preset,     # 删除用户注册的 preset（内置 preset 不可删）
    list_presets,          # 列出所有已注册 preset
    register_style_directory,  # 从文件系统加载样式（v2.2.0+）

    # 图尺寸
    publication_figsize,   # 计算出版图尺寸（英寸）
    register_column_width, # 注册期刊列宽

    # 字体工具
    bake_text_fonts,       # 固化 Text artist 字体（封装函数必备）
    list_chinese_fonts,    # 探测系统 CJK 字体
    get_color_cycle,       # 获取当前或 preset 的颜色列表

    # 刻度格式化
    LatFormatter,          # 纬度格式（21°N, 5°S, DMS 支持）
    LonFormatter,          # 经度格式（96°E, 5°W, DMS 支持）
    DMSFormatter,          # 通用度分秒格式（无半球后缀）
    DegreeFormatter,       # 通用度符号
    set_degree_formatter,  # 快捷设置度符号

    # 保存
    save_fig,              # 多格式保存封装

    # 向后兼容
    sci_plot_style,        # 旧版上下文管理器（wraps PlotStyle）
)

# ─── PlotStyle 类方法 ────────────────────────────────────────────────────────
PlotStyle.apply('science', **kwargs)          # 持久应用
PlotStyle.reset()                             # 恢复上一次 apply
PlotStyle.reset_all()                         # 恢复所有 apply 层
PlotStyle.decorator('science', **kwargs)      # 装饰器工厂
rc = PlotStyle.describe('science')            # 打印并返回 preset 详情 dict
rc = PlotStyle.describe('science', filter_prefix='font.')   # 只返回 font. 开头
rc = PlotStyle.describe('science', print_result=False)      # 静默模式

# ─── 扩展性 API (v2.1+) ──────────────────────────────────────────────────────
PlotStyle.register_handler(handler_fn, name='my_handler', priority=50)  # 注册自定义 handler
PlotStyle.unregister_handler('my_handler')                              # 注销 handler
PlotStyle.list_handlers()                                               # 列出所有自定义 handler
plotstyle.apply_to_axes(ax)                                             # 对单个 Axes 应用样式

# ─── 使用模式 ───────────────────────────────────────────────────────────────
# 上下文管理器
with PlotStyle('science', figsize='single', fontsize=8):
    fig, ax = plt.subplots()

# 一步创建图窗
fig, ax = PlotStyle('science', figsize='single', fontsize=8).subplots()

# 多 preset 叠加
with PlotStyle(['science', 'colors-bright', 'minimal'], figsize='single'):
    fig, axes = plt.subplots(1, 2)

# 装饰器
@PlotStyle.decorator('science', figsize='single')
def my_plot(): ...

# 封装函数（异常安全 + 字体固化）
def my_plot_func(data):
    with PlotStyle('science', fontsize=8):
        fig, ax = plt.subplots()
        ax.plot(data.x, data.y)
        bake_text_fonts(fig)   # plt.show() 字体一致
    return fig
```
