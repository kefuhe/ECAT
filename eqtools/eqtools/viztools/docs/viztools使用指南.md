# eqtools.viztools 使用指南 - 开发者推荐

## 📌 推荐使用模式

### 模式 1: 上下文管理器 (推荐用于函数内部)

**适用场景**: 在函数内部临时应用样式，函数结束后自动恢复

```python
from eqtools.viztools import PlotStyle, Presets
import matplotlib.pyplot as plt

def plot_seismic_data(time, amplitude):
    """绘制地震数据"""
    with PlotStyle(Presets.SCIENCE, figsize='single', fontsize=8):
        fig, ax = plt.subplots()
        ax.plot(time, amplitude)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        return fig

# 样式只在 with 块内生效，退出后自动恢复
```

**优点**:
- ✅ 自动清理，不影响其他代码
- ✅ 代码结构清晰
- ✅ 适合写库函数

---

### 模式 2: 一步创建图表 (推荐用于快速绘图)

**适用场景**: 快速创建带样式的图表

```python
from eqtools.viztools import PlotStyle, Presets

# 一行代码创建带样式的 figure 和 axes
fig, ax = PlotStyle(Presets.SCIENCE, figsize='single', fontsize=8).subplots()

ax.plot(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
```

**优点**:
- ✅ 代码简洁
- ✅ 适合脚本和快速原型
- ✅ 样式自动应用

---

### 模式 3: 持久化应用 (推荐用于脚本顶部)

**适用场景**: 在脚本开头设置全局样式，整个脚本都使用

```python
from eqtools.viztools import PlotStyle, Presets
import matplotlib.pyplot as plt

# 在脚本顶部应用样式
style = PlotStyle(Presets.SCIENCE, figsize='single', fontsize=8)
style.apply()  # 持久化应用

# 后续所有绘图都使用这个样式
fig1, ax1 = plt.subplots()
ax1.plot(x1, y1)

fig2, ax2 = plt.subplots()
ax2.plot(x2, y2)

# 脚本结束时恢复（可选）
style.restore()
```

**优点**:
- ✅ 适合长脚本
- ✅ 统一整个脚本的风格
- ✅ 可以手动控制恢复时机

---

### 模式 4: 混合样式 (推荐用于复杂图表)

**适用场景**: 同一个 figure 中不同子图使用不同样式

```python
from eqtools.viztools import PlotStyle, Presets
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图使用科学风格
PlotStyle(Presets.SCIENCE, fontsize=10).apply_to_axes(ax1)
ax1.plot(x, y1)
ax1.set_title('Science Style')

# 右图使用简约风格
PlotStyle(Presets.MINIMAL, fontsize=8).apply_to_axes(ax2)
ax2.plot(x, y2)
ax2.set_title('Minimal Style')

plt.tight_layout()
```

**优点**:
- ✅ 灵活控制每个子图
- ✅ 适合对比展示
- ✅ 不影响全局设置

---

## 🎯 针对 eqtools 开发的具体建议

### 场景 1: 开发库函数（给其他人调用）

**推荐**: 使用上下文管理器 + 参数化样式

```python
from eqtools.viztools import PlotStyle, LatFormatter, LonFormatter
import matplotlib.pyplot as plt

def plot_fault_model(lon, lat, slip, style='science', figsize='single'):
    """
    绘制断层模型

    Parameters
    ----------
    lon, lat : array
        经纬度坐标
    slip : array
        滑动量
    style : str, optional
        绘图样式，默认 'science'
    figsize : str or tuple, optional
        图表尺寸，默认 'single'
    """
    with PlotStyle(style, figsize=figsize, fontsize=9):
        fig, ax = plt.subplots()

        # 绘制数据
        im = ax.contourf(lon, lat, slip, levels=20, cmap='RdBu_r')

        # 设置经纬度格式化器
        ax.xaxis.set_major_formatter(LonFormatter())
        ax.yaxis.set_major_formatter(LatFormatter())

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.colorbar(im, ax=ax, label='Slip (m)')

        return fig, ax

# 用户可以自定义样式
fig, ax = plot_fault_model(lon, lat, slip, style='chinese', figsize='double')
```

---

### 场景 2: 写分析脚本（自己用）

**推荐**: 脚本顶部持久化应用 + 组合预设

```python
#!/usr/bin/env python
"""
地震数据分析脚本
"""
from eqtools.viztools import PlotStyle, Presets
import matplotlib.pyplot as plt
import numpy as np

# ========== 配置绘图样式 ==========
# 组合多个预设：科学风格 + 中文支持 + 明亮色盘
style = PlotStyle(
    [Presets.SCIENCE, Presets.CHINESE, Presets.COLORS_BRIGHT],
    figsize='double',
    fontsize=10
)
style.apply()

# ========== 数据分析和绘图 ==========
# 图1: 时间序列
fig1, ax1 = plt.subplots()
ax1.plot(time, displacement, label='位移')
ax1.set_xlabel('时间 (s)')
ax1.set_ylabel('位移 (m)')
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.savefig('output/displacement.pdf', dpi=300)

# 图2: 频谱分析
fig2, ax2 = plt.subplots()
ax2.plot(freq, amplitude, label='振幅谱')
ax2.set_xlabel('频率 (Hz)')
ax2.set_ylabel('振幅')
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.savefig('output/spectrum.pdf', dpi=300)

# 脚本结束时恢复（可选）
style.restore()
```

---

### 场景 3: Jupyter Notebook 交互式分析

**推荐**: 使用 notebook 预设 + 上下文管理器

```python
from eqtools.viztools import PlotStyle, Presets
import matplotlib.pyplot as plt
%matplotlib inline

# Notebook 预设：中等字号，适合屏幕显示
with PlotStyle(Presets.NOTEBOOK, figsize=(8, 6)):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
```

---

### 场景 4: 论文插图制作

**推荐**: 根据期刊要求选择预设

```python
from eqtools.viztools import PlotStyle, Presets, publication_figsize
import matplotlib.pyplot as plt

# ========== Nature 期刊单栏图 ==========
with PlotStyle(Presets.SCIENCE, figsize='nature', fontsize=7):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.savefig('figure1.pdf', dpi=300, bbox_inches='tight')

# ========== Science 期刊双栏图 ==========
with PlotStyle(Presets.SCIENCE_SERIF, figsize='science_double', fontsize=8):
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.5))
    axes[0].plot(x1, y1)
    axes[1].plot(x2, y2)
    plt.savefig('figure2.pdf', dpi=300, bbox_inches='tight')

# ========== IEEE 期刊（黑白打印友好）==========
with PlotStyle([Presets.SCIENCE_SERIF, Presets.IEEE], figsize='ieee_column'):
    fig, ax = plt.subplots()
    for i in range(5):
        ax.plot(x, y + i, label=f'Line {i+1}')
    ax.legend()
    plt.savefig('figure3.pdf', dpi=300, bbox_inches='tight')
```

---

## 🔧 常用预设组合推荐

### 英文论文（推荐）
```python
PlotStyle(Presets.SCIENCE, figsize='single', fontsize=8)
```

### 中文论文/报告（推荐）
```python
PlotStyle([Presets.SCIENCE, Presets.CHINESE], figsize='double', fontsize=10)
```

### 地学数据可视化（推荐）
```python
from eqtools.viztools import PlotStyle, Presets, LatFormatter, LonFormatter

with PlotStyle(Presets.SCIENCE, figsize='double', fontsize=9):
    fig, ax = plt.subplots()
    # 绘制地图数据
    ax.xaxis.set_major_formatter(LonFormatter())
    ax.yaxis.set_major_formatter(LatFormatter())
```

### 演示文稿（推荐）
```python
PlotStyle(Presets.PRESENTATION, figsize=(10, 7.5), fontsize=14)
```

### 多线条图（色盲友好）
```python
PlotStyle([Presets.SCIENCE, Presets.COLORS_BRIGHT], figsize='double')
```

### IEEE 投稿（黑白打印）
```python
PlotStyle([Presets.SCIENCE_SERIF, Presets.IEEE], figsize='ieee_column')
```

---

## 📊 完整示例：地震断层反演结果可视化

```python
"""
完整示例：绘制地震断层反演结果
"""
from eqtools.viztools import PlotStyle, Presets, LatFormatter, LonFormatter, save_fig
import matplotlib.pyplot as plt
import numpy as np

def plot_inversion_results(lon, lat, slip, residual, output_dir='figures'):
    """
    绘制反演结果：断层滑动分布 + 残差分布

    Parameters
    ----------
    lon, lat : 2D array
        经纬度网格
    slip : 2D array
        滑动量分布
    residual : 2D array
        残差分布
    output_dir : str
        输出目录
    """
    # 使用科学风格 + 中文支持 + 明亮色盘
    with PlotStyle(
        [Presets.SCIENCE, Presets.CHINESE, Presets.COLORS_BRIGHT],
        figsize='double',
        fontsize=9
    ):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 左图：滑动分布
        im1 = ax1.contourf(lon, lat, slip, levels=20, cmap='RdBu_r')
        ax1.xaxis.set_major_formatter(LonFormatter())
        ax1.yaxis.set_major_formatter(LatFormatter())
        ax1.set_xlabel('经度')
        ax1.set_ylabel('纬度')
        ax1.set_title('断层滑动分布')
        plt.colorbar(im1, ax=ax1, label='滑动量 (m)')

        # 右图：残差分布
        im2 = ax2.contourf(lon, lat, residual, levels=20, cmap='seismic')
        ax2.xaxis.set_major_formatter(LonFormatter())
        ax2.yaxis.set_major_formatter(LatFormatter())
        ax2.set_xlabel('经度')
        ax2.set_ylabel('纬度')
        ax2.set_title('反演残差')
        plt.colorbar(im2, ax=ax2, label='残差 (m)')

        plt.tight_layout()

        # 保存为多种格式
        save_fig(fig, f'{output_dir}/inversion_results.pdf', dpi=300)
        save_fig(fig, f'{output_dir}/inversion_results.png', dpi=150)

        return fig, (ax1, ax2)

# 使用示例
lon = np.linspace(100, 105, 50)
lat = np.linspace(30, 35, 50)
lon_grid, lat_grid = np.meshgrid(lon, lat)
slip = np.random.randn(50, 50) * 2
residual = np.random.randn(50, 50) * 0.5

fig, axes = plot_inversion_results(lon_grid, lat_grid, slip, residual)
plt.show()
```

---

## 💡 最佳实践建议

### 1. 库函数开发
- ✅ 使用上下文管理器
- ✅ 提供 `style` 参数让用户自定义
- ✅ 使用 `Presets` 常量提供 IDE 自动补全

### 2. 脚本开发
- ✅ 顶部持久化应用样式
- ✅ 组合多个预设（如 science + chinese + colors-bright）
- ✅ 使用 `save_fig` 统一保存格式

### 3. 交互式分析
- ✅ 使用 `notebook` 预设
- ✅ 配合 `%matplotlib inline` 或 `%matplotlib widget`
- ✅ 使用上下文管理器避免污染全局

### 4. 论文插图
- ✅ 根据期刊要求选择 figsize（nature, science, ieee 等）
- ✅ 设置合适的 fontsize（通常 7-9）
- ✅ 保存为 PDF 格式（矢量图）
- ✅ 使用 `dpi=300` 和 `bbox_inches='tight'`

### 5. 地学应用
- ✅ 使用 `LatFormatter` 和 `LonFormatter`
- ✅ 使用 `DegreeFormatter` 标注角度
- ✅ 使用 `DMSFormatter` 显示度分秒

---

## 🚀 快速参考

### 导入常用组件
```python
from eqtools.viztools import (
    PlotStyle,           # 核心样式管理器
    Presets,             # 预设常量（IDE 自动补全）
    LatFormatter,        # 纬度格式化器
    LonFormatter,        # 经度格式化器
    DegreeFormatter,     # 度数格式化器
    DMSFormatter,        # 度分秒格式化器
    save_fig,            # 保存图片
    list_presets,        # 列出所有预设
)
```

### 可用的 figsize 预设
```python
'single'           # 3.5 英寸（单栏）
'double'           # 7.0 英寸（双栏）
'nature'           # 3.42 英寸（Nature 单栏）
'nature_double'    # 7.08 英寸（Nature 双栏）
'science'          # 3.54 英寸（Science 单栏）
'science_double'   # 7.08 英寸（Science 双栏）
'ieee_column'      # 3.5 英寸（IEEE 单栏）
'ieee_page'        # 7.16 英寸（IEEE 全页）
'pnas'             # 3.42 英寸（PNAS 单栏）
'pnas_double'      # 7.0 英寸（PNAS 双栏）
```

### 可用的样式预设
```python
Presets.SCIENCE              # Sans-serif 科学风格
Presets.SCIENCE_SERIF        # Serif 学术风格
Presets.MINIMAL              # 简约风格
Presets.PRESENTATION         # 演示风格
Presets.NOTEBOOK             # 笔记本风格
Presets.CHINESE              # 中文支持（黑体）
Presets.CHINESE_SERIF        # 中文支持（宋体）
Presets.IEEE                 # IEEE 线型循环
Presets.SCATTER              # 散点图优化
Presets.COLORS_BRIGHT        # 明亮色盘
Presets.COLORS_VIBRANT       # 鲜艳色盘
Presets.COLORS_CONTRAST      # 高对比色盘（黑白打印友好）
```

---

**总结**: 对于 eqtools 开发，推荐使用**上下文管理器模式**（模式 1）作为主要方式，配合 `Presets` 常量和参数化设计，既保证代码清晰，又给用户足够的灵活性。
