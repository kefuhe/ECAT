"""
_constants.py — Centralized constants for eqtools.viztools.

This module provides a single location for all magic numbers, strings,
and configuration constants used throughout the viztools package.
"""

from typing import Dict

# ==============================================================================
# 默认列宽配置 (英寸)
# ==============================================================================

DEFAULT_COLUMN_WIDTHS: Dict[str, float] = {
    'single': 3.5,           # 单栏宽度
    'double': 7.0,           # 双栏宽度
    'full': 7.16,            # 全页宽度
    'nature': 3.42,          # Nature 期刊单栏
    'nature_double': 7.08,   # Nature 期刊双栏
    'science': 3.54,         # Science 期刊单栏
    'science_double': 7.08,  # Science 期刊双栏
    'ieee_column': 3.5,      # IEEE 单栏
    'ieee_page': 7.16,       # IEEE 全页
    'pnas': 3.42,            # PNAS 单栏
    'pnas_double': 7.0,      # PNAS 双栏
    'a4': 8.27,              # A4 纸宽度
    'a4_margin': 6.5,        # A4 纸带边距后的宽度
}

# ==============================================================================
# 样式栈配置
# ==============================================================================

MAX_STYLE_STACK_DEPTH: int = 50  # 样式应用栈的最大深度

# ==============================================================================
# 数学字体别名映射
# ==============================================================================

# 用户友好名称 -> matplotlib mathtext.fontset 值
MATHFONT_ALIASES: Dict[str, str] = {
    'stixsans': 'stixsans',
    'stix-sans': 'stixsans',
    'stix_sans': 'stixsans',
    'stix': 'stix',
    'cm': 'cm',
    'computer-modern': 'cm',
    'computer_modern': 'cm',
    'dejavusans': 'dejavusans',
    'dejavu-sans': 'dejavusans',
    'dejavuserif': 'dejavuserif',
    'dejavu-serif': 'dejavuserif',
}

# ==============================================================================
# LaTeX 前导码
# ==============================================================================

# Sans-serif LaTeX 前导码
SANS_TEX_PREAMBLE: str = '\n'.join([
    r'\usepackage{helvet}',
    r'\renewcommand{\familydefault}{\sfdefault}',
    r'\usepackage{sfmath}',
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
])

# Serif LaTeX 前导码
SERIF_TEX_PREAMBLE: str = '\n'.join([
    r'\usepackage[T1]{fontenc}',
    r'\usepackage{lmodern}',
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}',
])

# ==============================================================================
# CJK 字体候选列表
# ==============================================================================

# 中文无衬线字体候选（按优先级排序）
CHINESE_SANS_CANDIDATES: list = [
    'SimHei',
    'Microsoft YaHei',
    'PingFang SC',
    'Heiti SC',
    'WenQuanYi Micro Hei',
    'Noto Sans CJK SC',
    'Noto Sans CJK',
    'Source Han Sans SC',
    'Noto Sans SC',
    'Arial Unicode MS',
]

# 中文衬线字体候选（按优先级排序）
CHINESE_SERIF_CANDIDATES: list = [
    'SimSun',
    'STSong',
    'FangSong',
    'AR PL UMing CN',
    'Noto Serif CJK SC',
    'Source Han Serif SC',
    'SimHei',  # 如果没有衬线字体，回退到无衬线
]

# ==============================================================================
# 预设类型标识
# ==============================================================================

PRESET_TYPES: set = {'package', 'user', 'builtin'}

# ==============================================================================
# 格式化器配置
# ==============================================================================

# 支持的格式化器类型
FORMATTER_TYPES: set = {'decimal', 'dms'}

# ==============================================================================
# 文件格式配置
# ==============================================================================

# 支持的图像格式扩展名
KNOWN_IMAGE_FORMATS: set = {
    'pdf', 'png', 'svg', 'eps',
    'jpg', 'jpeg', 'tiff', 'tif'
}

# ==============================================================================
# 字体缓存配置
# ==============================================================================

FONT_CACHE_EXPIRY_DAYS: int = 7  # 字体缓存过期时间（天）
FONT_CACHE_DIR_NAME: str = '.cache/eqtools'  # 相对于用户主目录
FONT_CACHE_FILE_NAME: str = 'font_cache.pkl'

# ==============================================================================
# 预设名称常量类
# ==============================================================================

class Presets:
    """预设名称常量，提供 IDE 自动补全和类型检查支持。

    Examples
    --------
    >>> from eqtools.viztools import PlotStyle, Presets
    >>> with PlotStyle(Presets.SCIENCE, figsize='single'):
    ...     fig, ax = plt.subplots()
    """

    # 基础科学风格
    SCIENCE = 'science'
    SCIENCE_SERIF = 'science-serif'

    # 简约风格
    MINIMAL = 'minimal'

    # 演示风格
    PRESENTATION = 'presentation'

    # 笔记本风格
    NOTEBOOK = 'notebook'

    # IEEE 期刊风格
    IEEE = 'ieee'

    # 散点图优化风格
    SCATTER = 'scatter'

    # 中文支持风格
    CHINESE = 'chinese'
    CHINESE_SERIF = 'chinese-serif'

    # 颜色预设（色盲友好）
    COLORS_VIBRANT = 'colors-vibrant'
    COLORS_BRIGHT = 'colors-bright'
    COLORS_CONTRAST = 'colors-contrast'  # 高对比度（黑白打印友好）


# ==============================================================================
# 单位转换常量
# ==============================================================================

CM_TO_INCH: float = 1 / 2.54  # 厘米转英寸
INCH_TO_CM: float = 2.54      # 英寸转厘米

# ==============================================================================
# 默认值
# ==============================================================================

DEFAULT_DPI: int = 300                    # 默认 DPI
DEFAULT_BBOX_INCHES: str = 'tight'        # 默认边界框
DEFAULT_FIGSIZE_ASPECT: float = 0.75      # 默认高宽比
DEFAULT_FIGSIZE_FRACTION: float = 1.0     # 默认列宽比例
DEFAULT_FIGSIZE_UNIT: str = 'inch'        # 默认单位

# ==============================================================================
# 版本要求
# ==============================================================================

MIN_MATPLOTLIB_VERSION: str = '3.3.0'  # 最低 matplotlib 版本要求
