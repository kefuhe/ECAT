# eqtools.viztools — 设计原理

本文面向需要**理解内部机制、二次开发或调试**的读者。
记录各核心模块的设计决策、数据流和关键注意事项。

---

## 目录

1. [整体架构](#整体架构)
2. [模块初始化顺序](#模块初始化顺序)
3. [Preset 注册系统](#preset-注册系统)
4. [PlotStyle 核心流程](#plotstyle-核心流程)
5. [rcParams 精确保存与恢复](#rcparams-精确保存与恢复)
6. [字体系统设计](#字体系统设计)
7. [bake_text_fonts 的作用边界](#bake_text_fonts-的作用边界)
8. [usetex 渲染管道](#usetex-渲染管道)
9. [列宽注册与用户配置](#列宽注册与用户配置)
10. [向后兼容层](#向后兼容层)
11. [扩展性系统 (v2.1+)](#扩展性系统-v21)
12. [扩展点索引](#扩展点索引)

---

## 整体架构

```text
eqtools/viztools/                    ← 顶级子包
│
├── __init__.py                      ← 公开 API 入口，__all__
├── _core.py                         ← PlotStyle 类、preset 注册、惰性初始化
├── _font_utils.py                   ← CJK 字体探测、bake_text_fonts
├── _style_utils.py                  ← 列宽注册、publication_figsize、save_fig、用户配置
├── _formatters.py                   ← LatFormatter、LonFormatter、DegreeFormatter、DMSFormatter
├── _color_utils.py                  ← get_color_cycle
├── _compat.py                       ← sci_plot_style、set_plot_style（向后兼容层）
├── _registry.py                     ← 集中式注册表（preset、handler、样式状态）
├── viz_3d.py                        ← optimize_3d_plot、plot_slip_distribution（可选）
└── styles/                          ← 随包分发的 .mplstyle 文件
    ├── eqtools-science.mplstyle
    ├── eqtools-science-serif.mplstyle
    ├── eqtools-minimal.mplstyle
    ├── eqtools-notebook.mplstyle
    ├── eqtools-presentation.mplstyle
    ├── eqtools-scatter.mplstyle
    ├── eqtools-ieee.mplstyle
    ├── eqtools-colors-bright.mplstyle
    ├── eqtools-colors-vibrant.mplstyle
    └── eqtools-colors-contrast.mplstyle
```

```text
_core.py 内部结构
│
├── 模块级全局状态（通过 _registry 模块管理）
│   ├── _PRESET_REGISTRY      : Dict[str, Dict]  ← preset 注册表
│   ├── _BUILTIN_PRESET_NAMES : set               ← 内置 preset 名（不可 unregister）
│   ├── _STYLES_REGISTERED    : bool              ← .mplstyle 注册标志（惰性）
│   ├── _INITIALIZED          : bool              ← 全局初始化标志（惰性）
│   └── _CUSTOM_HANDLERS      : Dict[int, list]   ← 自定义 handler 注册表（v2.1+）
│
├── _ensure_initialized()                         ← 惰性初始化入口
│   ├── _register_package_styles()                ← 注册 .mplstyle 文件（一次）
│   ├── _register_builtin_presets()               ← 注册 13 个内置 preset（一次）
│   └── _load_user_config()                       ← 加载 JSON 配置（一次）
│
├── PlotStyle 类
│   ├── __init__()                存储参数，不修改任何全局状态
│   ├── _resolve_preset()         递归解析继承链
│   ├── _build_final_rcparams()   调度处理器链（含自定义 handler）
│   ├── _HANDLERS = [...]         9 个内置处理器的有序列表
│   ├── _CUSTOM_HANDLERS          自定义 handler 注册表（类变量）
│   ├── _apply_xxx(acc) × 9       各内置处理器（就地修改 acc dict）
│   ├── _apply_custom_handlers()  调度自定义 handler（v2.1+）
│   ├── _apply_to(saved)          修改 mpl.rcParams，同时把原值存入 saved
│   ├── _restore(saved)           还原 mpl.rcParams
│   ├── __enter__ / __exit__      上下文管理器接口
│   ├── apply() / reset()         持久应用接口（_apply_stack）
│   ├── decorator()               装饰器工厂
│   ├── subplots()                一步式图窗创建（close_event 自动 reset）
│   ├── describe()                调试：打印并返回解析后的 rcParams dict
│   ├── reset_all()               清空整个 _apply_stack
│   ├── register_handler()        注册自定义 handler（v2.1+）
│   ├── unregister_handler()      注销自定义 handler（v2.1+）
│   ├── list_handlers()           列出所有自定义 handler（v2.1+）
│   └── apply_to_axes()           对单个 Axes 应用样式（v2.1+）
│
└── register_preset / unregister_preset / list_presets
```

---

## 模块初始化顺序

```text
import eqtools.viztools
  │
  ├─ 1. 顶层 import（各子模块）
  │
  ├─ 2. __init__.py 执行
  │     导入 _core.PlotStyle、_font_utils.*、_style_utils.*、
  │     _formatters.*、_color_utils.*、_compat.*
  │     （尝试导入 viz_3d，失败则静默跳过）
  │
  └─ 3. 此时 _INITIALIZED = False，_STYLES_REGISTERED = False
         无任何副作用（无 .mplstyle 注册，无 preset 填充）

首次使用 PlotStyle / list_presets / register_preset / describe 时：
  │
  └─ _ensure_initialized() 被调用（之后 _INITIALIZED = True）
       ├─ _register_package_styles()
       │    ├─ (可选) scienceplots 样式注册
       │    └─ eqtools/viztools/styles/*.mplstyle → plt.style.library
       ├─ _register_builtin_presets()
       │    └─ 注册 13 个内置 preset → _PRESET_REGISTRY
       │       填充 _BUILTIN_PRESET_NAMES（用于 unregister_preset 保护）
       └─ _load_user_config()
            └─ 读 ~/.config/eqtools/viztools.json（或旧路径）
               → _COLUMN_WIDTHS 追加用户自定义列宽
```

**惰性初始化的意义**：import 时没有任何副作用（不访问磁盘、不修改 matplotlib 状态），
只有真正使用 `PlotStyle` 时才触发一次性初始化，对嵌入式和测试环境友好。

---

## Preset 注册系统

### 数据结构

每个 preset 在 `_PRESET_REGISTRY` 中存储为一个 dict：

```python
{
    'base':               None,      # str 或 list[str]，父 preset
    'mplstyles':          [...],     # plt.style.library 中的样式名列表
    'rcparams':           {...},     # 直接的 rcParam 覆盖
    'chinese':            False,     # 是否注入 CJK 字体
    'chinese_prefer_serif': False,   # 注入 CJK serif 还是 sans
    'description':        '...',
}
```

### 继承解析：`_resolve_preset(name)`

递归深度优先展开 `base` 链，用 `_visited` set 防止循环依赖：

```text
_resolve_preset('minimal')
  → base='science' → _resolve_preset('science')
       → base=None → result = {mplstyles:[], rcparams:{}}
       → 追加 science.mplstyles = ['eqtools-science']
       → 追加 science.rcparams  = {}
       → 返回
  → 追加 minimal.mplstyles = ['eqtools-minimal']
  → 追加 minimal.rcparams  = {}
  → 返回 {mplstyles: ['eqtools-science', 'eqtools-minimal'], rcparams: {}}
```

**多重继承**（`base` 为列表时）：按列表顺序依次解析，后者覆盖前者。
`base` 列表中的每一项：
- 若名称在 `_PRESET_REGISTRY` 中 → 作为 preset 递归解析
- 否则 → 视为 mplstyle 名称直接追加（支持混合 preset/mplstyle 基类）

```python
register_preset('my_bright', base=['science', 'colors-bright'], ...)
# resolve 顺序：science → colors-bright → my_bright 自身
# 后解析的 rcparams 键会覆盖前面的
```

### `unregister_preset()` 保护机制

内置 preset 名称在 `_register_builtin_presets()` 结束时写入 `_BUILTIN_PRESET_NAMES`（不可变 set）。
`unregister_preset(name)` 检查该 set，若命中则抛 `ValueError`：

```python
_BUILTIN_PRESET_NAMES: set = set()   # 填充于 _register_builtin_presets()

def unregister_preset(name: str) -> None:
    if name in _BUILTIN_PRESET_NAMES:
        raise ValueError(f"Cannot unregister built-in preset '{name}'")
    _PRESET_REGISTRY.pop(name, None)
```

### .mplstyle 文件加载时机

`_resolve_preset()` 只收集样式**名称字符串**。
实际加载发生在 `_apply_preset_layers()` 处理器中：

```python
for style_name in resolved['mplstyles']:
    if style_name in plt.style.library:
        acc.update(dict(plt.style.library[style_name]))
    else:
        warnings.warn(f"PlotStyle: mplstyle '{style_name}' not in library.")
```

这意味着：如果 `.mplstyle` 文件不存在于 `plt.style.library`，
会发出 `UserWarning` 并跳过，不会崩溃。

---

## PlotStyle 核心流程

### 处理器链（Handler Chain）

`_build_final_rcparams()` 不再是 100+ 行单体函数，而是调度 9 个有序内置处理器 + 自定义处理器：

```python
_HANDLERS = [
    '_apply_preset_layers',   # mplstyles + preset rcparams + CJK 注入
    '_apply_figsize',
    '_apply_fontsize',        # font.size + xtick/ytick/legend/title 联动
    '_apply_legend_frame',
    '_apply_dpi',
    '_apply_pdf_fonttype',
    '_apply_mathfont',        # 必须在 usetex 前（计算 self._is_sans）
    '_apply_usetex',          # CJK 检测 + preamble 注入
    '_apply_extra',           # self._extra_rcparams，永远最后
]

_CUSTOM_HANDLERS: Dict[int, list] = {}  # {priority: [(name, handler_fn), ...]}

def _build_final_rcparams(self) -> Dict:
    """Collect all rcParams to apply, lowest -> highest priority."""
    acc: Dict = {}
    for handler_name in self._HANDLERS:
        if handler_name == '_apply_extra':
            # 在 _apply_extra 之前插入自定义 handler
            self._apply_custom_handlers(acc)
        getattr(self, handler_name)(acc)
    return acc

def _apply_custom_handlers(self, acc: Dict) -> None:
    """Apply user-registered custom handlers in priority order."""
    for priority in sorted(self._CUSTOM_HANDLERS.keys()):
        for name, handler_fn in self._CUSTOM_HANDLERS[priority]:
            try:
                handler_fn(self, acc)
            except Exception as e:
                warnings.warn(f"Custom handler '{name}' failed: {e}")
```

每个处理器签名：`def _apply_xxx(self, acc: dict) -> None`（就地修改 `acc`）。

优先级链（从低到高）：

```text
1. _apply_preset_layers
   ├── mplstyle 文件内容（plt.style.library）
   ├── preset 自身 rcparams dict
   └── CJK 字体注入（若 chinese=True）
2. _apply_figsize     → figure.figsize
3. _apply_fontsize    → font.size, axes.labelsize, xtick/ytick/legend/title.fontsize
4. _apply_legend_frame → legend.frameon, legend.framealpha, legend.fancybox
5. _apply_dpi         → figure.dpi, savefig.dpi
6. _apply_pdf_fonttype → pdf.fonttype, ps.fonttype
7. _apply_mathfont    → mathtext.fontset（+ 计算 self._is_sans）
8. _apply_usetex      → text.usetex, text.latex.preamble
9. _apply_custom_handlers → 用户注册的自定义 handler（按优先级排序）
10. _apply_extra      → self._extra_rcparams（用户 rcparams= 参数，最高优先级）
```

**处理器顺序约束**：
- `_apply_mathfont` 必须在 `_apply_usetex` 之前（计算 `self._is_sans`）
- `_apply_custom_handlers` 在 `_apply_extra` 之前（用户 rcparams 最高优先级）
- 自定义 handler 内部按优先级数值排序（低优先级先执行）

### `describe()` 返回 dict

`describe()` 现在打印并**返回**解析后的 rcParams dict，支持静默模式和前缀过滤：

```python
@classmethod
def describe(cls, name: str, print_result: bool = True,
             filter_prefix: Optional[str] = None) -> Dict:
    ...
    result = tmp._build_final_rcparams()
    if filter_prefix is not None:
        result = {k: v for k, v in result.items() if k.startswith(filter_prefix)}
    if print_result:
        for k, v in sorted(result.items()):
            print(f"  {k} = {v}")
    return result
```

### `_apply_to(saved)` — 原子性修改

```python
def _apply_to(self, save_target):
    final = self._build_final_rcparams()
    for k, v in final.items():
        try:
            save_target[k] = copy.deepcopy(mpl.rcParams[k])  # 先保存
            mpl.rcParams[k] = v                               # 再修改
        except (KeyError, ValueError):
            pass  # 未知键或值类型错误时静默跳过
```

`deepcopy` 是必要的：`mpl.rcParams` 中某些值（如 `font.sans-serif`）是列表，
浅拷贝会导致保存的原值和当前值指向同一对象，恢复时失效。

---

## rcParams 精确保存与恢复

### 设计目标

与 `mpl.rcdefaults()` 不同，PlotStyle **只恢复它自己改过的键**，
不影响用户在其他地方设置的 rcParams。

### 持久 apply 的栈结构

```text
_apply_stack: List[Dict]   ← 类变量，模块级单例

PlotStyle.apply('science')           → 压入 saved_A
PlotStyle.apply('colors-bright')     → 压入 saved_B
PlotStyle.reset()                    → 弹出 saved_B，恢复 B 改过的键
PlotStyle.reset()                    → 弹出 saved_A，恢复 A 改过的键
PlotStyle.reset_all()                → while stack: pop + restore
```

### 上下文管理器的栈

上下文管理器使用实例属性 `self._saved`（不是类级栈），
因此**嵌套 `with` 块**是安全的：

```python
with PlotStyle('science'):           # self._saved = {原始值}
    with PlotStyle('chinese'):       # inner._saved = {science 激活后的值}
        ...
    # 内层退出：恢复到 science 状态
# 外层退出：恢复到 science 激活前状态
```

### `subplots()` 的 close_event 绑定

```python
def subplots(self, *args, **kwargs):
    self.__enter__()                  # 激活样式，填充 self._saved
    fig, axes = plt.subplots(...)
    saved = self._saved               # 捕获引用（闭包）
    fig.canvas.mpl_connect(
        'close_event',
        lambda _: PlotStyle._restore(saved)   # 关闭图窗时自动恢复
    )
    return fig, axes
```

注意：`saved` 通过闭包捕获，图窗存活期间 dict 不会被 GC。
`__exit__` 永远不会被调用（没有 `with` 块），
恢复完全依赖 `close_event`。

---

## 字体系统设计

### CJK 字体探测

`_probe_chinese_fonts()` 用 `@lru_cache(maxsize=1)` 修饰，全程只探测一次：

```python
@lru_cache(maxsize=1)
def _probe_chinese_fonts():
    available = {f.name for f in fontManager.ttflist}
    return {
        'sans':  next((f for f in _CHINESE_SANS_CANDIDATES  if f in available), None),
        'serif': next((f for f in _CHINESE_SERIF_CANDIDATES if f in available), None),
    }
```

探测结果是候选列表中**第一个系统存在的字体名**。
候选列表按优先级排序（`SimHei` > `Microsoft YaHei` > `PingFang SC` > ...）。

CJK 字体注入逻辑（在 `_apply_preset_layers` 处理器中）：

```python
# 把 CJK 字体名插到 font.sans-serif 列表最前面（最高优先级）
current = list(acc.get('font.sans-serif', []))
if cjk_font not in current:
    current.insert(0, cjk_font)
acc['font.sans-serif'] = current
```

这保留了后续英文字体的回退链（如 Arial → Helvetica → DejaVu Sans）。

### mathfont 自动检测

检测 `acc` 中已累积的 `font.family`，自动选择匹配的数学字体。
注意这发生在 `_apply_mathfont` 处理器中（早于 `_apply_usetex`），
并通过实例属性 `self._is_sans` 传递结果：

```python
def _apply_mathfont(self, acc: Dict) -> None:
    _effective_family = acc.get('font.family', 'sans-serif')
    self._is_sans = (_effective_family == 'sans-serif')   # ← 实例属性，供 _apply_usetex 使用
    # 自动：stixsans ↔ sans-serif,  stix ↔ serif
    acc['mathtext.fontset'] = 'stixsans' if self._is_sans else 'stix'
```

### usetex preamble 自动注入

```python
def _apply_usetex(self, acc: Dict) -> None:
    _has_cjk = any(
        _PRESET_REGISTRY.get(p, {}).get('chinese', False)
        for p in self._presets
    )
    if self._usetex is True and not _has_cjk:
        acc['text.usetex'] = True
        if 'text.latex.preamble' not in self._extra_rcparams:
            preamble = _SANS_TEX_PREAMBLE if self._is_sans else _SERIF_TEX_PREAMBLE
            acc['text.latex.preamble'] = preamble
```

用户通过 `rcparams={'text.latex.preamble': '...'}` 可完全覆盖自动 preamble，
因为 `self._extra_rcparams` 在最后一步（`_apply_extra`）以最高优先级覆盖 `acc`。

---

## bake_text_fonts 的作用边界

### matplotlib 字体延迟解析机制

matplotlib 的 `Text` 对象在 `set_xlabel()` 等调用时只存储字体**族名字符串**：

```python
ax.set_xlabel('Distance (km)')
# Text._fontproperties.family = ['sans-serif']   ← 存的是族名
```

实际字体文件查找发生在渲染时（`savefig` 或 `show` 触发 draw）：

```python
# 伪代码，matplotlib 内部
resolved_font = font_manager.findfont(
    FontProperties(family=mpl.rcParams['font.sans-serif'])
)
```

### bake_text_fonts 实现

```python
def bake_text_fonts(fig):
    family = mpl.rcParams.get('font.family', ['sans-serif'])
    font_list = mpl.rcParams.get(f'font.{family}', [])
    available = {f.name for f in fontManager.ttflist}
    resolved = next((f for f in font_list if f in available), None)

    def _walk(artist):
        if isinstance(artist, Text) and artist.get_text():
            artist.set_fontname(resolved)   # 固化为具体字体名
        for child in artist.get_children():
            _walk(child)

    _walk(fig)
```

`set_fontname(name)` 把 `Text._fontproperties` 中的 family 替换为具体名称，
渲染时不再走 `font.sans-serif` 列表查找，直接用该名称。

### 作用边界（重要）

| 渲染路径 | bake 是否有效 | 原因 |
| --- | --- | --- |
| `usetex=False` + `plt.show()` | ✓ 有效 | 走 matplotlib 内置渲染，`Text.fontname` 参与 |
| `usetex=False` + `savefig` | 本身无需 bake，savefig 在 with 内时已正确 | 同上 |
| `usetex=True` + `plt.show()` | ✗ **无效** | 走 pdflatex，`Text.fontname` 不参与渲染 |
| `usetex=True` + `savefig`（在 with 内）| 本身无需 bake | pdflatex 在此时 `text.usetex=True` |

**结论**：`usetex=True` 时，`plt.show()` 必须在 `PlotStyle` 激活期间调用，
`bake_text_fonts` 无法解决该场景。

---

## usetex 渲染管道

### 两条渲染路径对比

```text
usetex=False（matplotlib 内置 mathtext）
  plt.show() / savefig
    → matplotlib Agg/PDF/SVG backend
    → Text._fontproperties → font_manager.findfont()
    → 查 mpl.rcParams['font.sans-serif'] 列表
    → 调用 FreeType 渲染字形

usetex=True（LaTeX 外部进程）
  plt.show() / savefig
    → matplotlib 生成 .tex 临时文件
    → 调用 pdflatex（外部进程）
    → 生成 DVI/PDF → 转换为位图
    → mpl.rcParams['font.sans-serif'] 完全不参与
    → 字体由 .tex preamble 中的 \usepackage{helvet} 等决定
```

### 关键 rcParam：`text.usetex`

这是**渲染路径的开关**，不是某个 `Text` 对象的属性。
`PlotStyle.reset()` 恢复该键后，之后所有渲染都走内置路径，
即使图已经创建好也无法补救（`bake_text_fonts` 亦然）。

### CJK + usetex 自动禁用

pdflatex 不支持 Unicode CJK 字符（T1/OT1 编码限制）。
`_apply_usetex()` 检测到 CJK preset 时自动禁用 usetex：

```python
_has_cjk = any(
    _PRESET_REGISTRY.get(p, {}).get('chinese', False)
    for p in self._presets
)
if self._usetex is True and _has_cjk:
    acc['text.usetex'] = False
    warnings.warn(...)
```

---

## 列宽注册与用户配置

### `_COLUMN_WIDTHS` dict（位于 `_style_utils.py`）

模块级单例，所有 `publication_figsize()` 调用和 `PlotStyle(figsize=...)` 共享：

```python
_COLUMN_WIDTHS: Dict[str, float] = {
    'single': 3.5, 'double': 7.2, 'nature': 3.42,
    'ieee': 3.5, 'ieee_double': 7.16, 'a4': 8.27,
}
```

`register_column_width(name, width_inch)` 直接向该 dict 写入，
因此**注册后立即对所有后续调用生效**（无需重新实例化 `PlotStyle`）。

### 用户配置文件搜索路径

```python
def _load_user_config():
    candidates = [
        (Path.home() / '.config' / 'eqtools' / 'viztools.json',   None),          # 新标准路径
        (Path.home() / '.config' / 'eqtools' / 'plottools.json',  None),          # 新兼容路径
        (Path.home() / '.config' / 'statutils' / 'plottools.json', DeprecationWarning),  # 旧路径
        (Path.home() / '.plottools.json',                          DeprecationWarning),  # 旧路径
    ]
    for cfg_path, warn_cls in candidates:
        if cfg_path.exists():
            if warn_cls is not None:
                warnings.warn(
                    f"Config at {cfg_path} is deprecated. "
                    f"Move to ~/.config/eqtools/viztools.json",
                    DeprecationWarning, stacklevel=2,
                )
            data = json.loads(cfg_path.read_text())
            for name, w in data.get('column_widths', {}).items():
                register_column_width(name, float(w))
            break   # 只加载第一个找到的文件
```

**JSON 格式**（各路径通用）：

```json
{
    "column_widths": {
        "agu_single": 3.37,
        "agu_double": 6.83
    }
}
```

该函数由 `_ensure_initialized()` 调用一次。

---

## 向后兼容层

### shim 文件

两个旧入口点通过 shim 转发到 `eqtools.viztools`：

| 旧导入路径 | shim 位置 | 行为 |
| --- | --- | --- |
| `from eqtools.plottools import ...` | `eqtools/plottools.py` | `DeprecationWarning` + re-export |
| `from statUtils.plottools import ...` | `statUtils/plottools.py` | `DeprecationWarning` + re-export |

shim 文件结构（以 `eqtools/plottools.py` 为例）：

```python
import warnings
warnings.warn(
    "eqtools.plottools is deprecated; use eqtools.viztools instead.",
    DeprecationWarning, stacklevel=2,
)
from .viztools import *        # noqa: F401, F403
from .viztools import __all__  # noqa: F401
```

### `sci_plot_style()`（位于 `_compat.py`）

旧版接口，通过 `_map_legacy_style_to_preset()` 将旧参数映射到新 preset 名，
然后内部创建并返回一个 `PlotStyle` 实例：

```python
# 旧代码继续有效
with sci_plot_style(serif=False, fontsize=8, figsize='single'):
    ...

# 等价于新代码
with PlotStyle('science', fontsize=8, figsize='single'):
    ...
```

**`use_tes` 拼写错误修复**：旧接口同时接受 `use_tes=True`（拼写错误）和 `use_tex=True`（正确）。
使用 `use_tes` 时触发 `DeprecationWarning`，两者均正确注入 LaTeX preamble（通过新的 `usetex=True` 路径）：

```python
def sci_plot_style(..., use_tes: bool = False, use_tex: bool = False, ...):
    if use_tes and not use_tex:
        warnings.warn(
            "'use_tes' is a misspelling; use 'use_tex=True' instead.",
            DeprecationWarning, stacklevel=2,
        )
    _do_tex = use_tex or use_tes
    # 通过 usetex=True 调用新接口 → 正确注入 preamble
```

### `plt.style.use('eqtools-science')` 直接使用

`.mplstyle` 文件在 `_register_package_styles()` 触发后注册到 `plt.style.library`，
可直接通过 `plt.style.use()` 使用，但不经过 PlotStyle 的 CJK 注入、fontsize 联动、preamble 注入等逻辑。

旧的 `statutils-*` 样式文件仍保留在 `statUtils/styles/` 目录（未删除），
`plt.style.use('statutils-science')` 仍然有效。

---

## 扩展性系统 (v2.1+)

### 自定义 Handler 注册机制

从 v2.1 开始，PlotStyle 支持第三方注册自定义 handler，无需修改源代码。

#### 数据结构

```python
# 类变量，所有 PlotStyle 实例共享
_CUSTOM_HANDLERS: Dict[int, List[Tuple[str, Callable]]] = {}
# {priority: [(name, handler_fn), ...]}
```

#### 注册流程

```python
@classmethod
def register_handler(cls, handler_fn, name=None, priority=50):
    if name is None:
        name = getattr(handler_fn, '__name__', f'handler_{id(handler_fn)}')

    if priority not in cls._CUSTOM_HANDLERS:
        cls._CUSTOM_HANDLERS[priority] = []

    # 检查重名（同优先级）
    existing_names = [n for n, _ in cls._CUSTOM_HANDLERS[priority]]
    if name in existing_names:
        warnings.warn(f"Handler '{name}' already registered at priority {priority}. Overwriting.")
        cls._CUSTOM_HANDLERS[priority] = [
            (n, fn) for n, fn in cls._CUSTOM_HANDLERS[priority] if n != name
        ]

    cls._CUSTOM_HANDLERS[priority].append((name, handler_fn))
```

#### 执行流程

自定义 handler 在 `_build_final_rcparams()` 中被调用，位于 `_apply_extra` 之前：

```python
def _build_final_rcparams(self) -> Dict:
    acc: Dict = {}
    for handler_name in self._HANDLERS:
        if handler_name == '_apply_extra':
            self._apply_custom_handlers(acc)  # ← 插入点
        getattr(self, handler_name)(acc)
    return acc
```

这保证了：
1. 自定义 handler 可以覆盖内置 handler 的设置
2. 用户的 `rcparams=` 参数（通过 `_apply_extra`）仍然是最高优先级

#### Handler 签名

```python
def my_handler(plotstyle_instance: PlotStyle, acc: Dict[str, Any]) -> None:
    """
    Parameters
    ----------
    plotstyle_instance : PlotStyle
        当前 PlotStyle 实例，可访问其属性（_fontsize, _presets, 等）
    acc : dict
        累积的 rcParams 字典，原地修改以添加/覆盖设置
    """
    if hasattr(plotstyle_instance, '_my_custom_attr'):
        acc['some.rcparam'] = plotstyle_instance._my_custom_attr
```

#### 错误处理

自定义 handler 的异常会被捕获并转换为 `UserWarning`，不会中断整个样式应用流程：

```python
def _apply_custom_handlers(self, acc: Dict) -> None:
    for priority in sorted(self._CUSTOM_HANDLERS.keys()):
        for name, handler_fn in self._CUSTOM_HANDLERS[priority]:
            try:
                handler_fn(self, acc)
            except Exception as e:
                warnings.warn(
                    f"PlotStyle: custom handler '{name}' failed: {e}",
                    UserWarning, stacklevel=4
                )
```

#### 优先级系统

- **0-100**: 保留给内置 handler（未来扩展）
- **50 (默认)**: 推荐用于大多数自定义 handler
- **更高值**: 更晚执行（更高优先级）

同一优先级内的 handler 按注册顺序执行。

### Axes 级别样式应用

`apply_to_axes(ax)` 方法将样式应用到单个 Axes 对象，而不修改全局 rcParams。

#### 实现原理

```python
def apply_to_axes(self, ax) -> None:
    final = self._build_final_rcparams()  # 复用完整的处理器链

    # 映射 rcParams 到 Axes 方法
    axes_mappings = {
        'axes.labelsize': lambda v: (
            ax.xaxis.label.set_fontsize(v),
            ax.yaxis.label.set_fontsize(v)
        ),
        'xtick.labelsize': lambda v: ax.tick_params(axis='x', labelsize=v),
        'ytick.labelsize': lambda v: ax.tick_params(axis='y', labelsize=v),
        'axes.grid': lambda v: ax.grid(v),
        'axes.spines.top': lambda v: ax.spines['top'].set_visible(v),
        # ... 更多映射
    }

    for key, value in final.items():
        if key in axes_mappings:
            try:
                axes_mappings[key](value)
            except Exception:
                pass  # 静默跳过不兼容的设置
```

#### 支持的 rcParams

只有可以在 Axes 级别设置的参数会被应用：

| 类别 | 支持的 rcParams |
| --- | --- |
| 字体大小 | `axes.labelsize`, `xtick.labelsize`, `ytick.labelsize`, `axes.titlesize` |
| 网格 | `axes.grid`, `grid.alpha`, `grid.linewidth` |
| 脊柱 | `axes.spines.{top,right,left,bottom}`, `axes.linewidth` |
| 刻度 | `{x,y}tick.{major,minor}.{width,size}` |

全局设置（`figure.figsize`, `font.family`, `text.usetex` 等）会被忽略。

#### 设计权衡

**为什么不支持 `font.family`？**

`font.family` 是全局 rcParam，matplotlib 在渲染时从 `mpl.rcParams['font.sans-serif']` 列表查找字体。
单个 Axes 无法拥有独立的字体回退列表。

**解决方案**：使用 `set_fontname()` 直接设置具体字体名（类似 `bake_text_fonts` 的原理）：

```python
def apply_to_axes(self, ax) -> None:
    final = self._build_final_rcparams()

    # 解析字体
    family = final.get('font.family', 'sans-serif')
    font_list = final.get(f'font.{family}', [])
    resolved_font = next((f for f in font_list if f in available_fonts), None)

    # 应用到所有 Text artist
    for text in ax.get_xticklabels() + ax.get_yticklabels() + [ax.xaxis.label, ax.yaxis.label, ax.title]:
        if resolved_font:
            text.set_fontname(resolved_font)
```

当前实现未包含此逻辑（避免复杂性），用户需要手动设置字体。

### 增强的地理格式化器

#### DMS 格式实现

```python
def _decimal_to_dms(decimal_deg: float, precision: int = 0) -> tuple:
    """Convert decimal degrees to degrees, minutes, seconds."""
    abs_deg = abs(decimal_deg)
    degrees = int(abs_deg)
    minutes_float = (abs_deg - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60

    if precision == 0:
        seconds = int(round(seconds))
        # 处理舍入溢出
        if seconds == 60:
            seconds = 0
            minutes += 1
        if minutes == 60:
            minutes = 0
            degrees += 1
    else:
        seconds = round(seconds, precision)
        if seconds >= 60:
            seconds = 0
            minutes += 1
        if minutes >= 60:
            minutes = 0
            degrees += 1

    return degrees, minutes, seconds
```

**舍入溢出处理**：当秒数舍入到 60 时，需要进位到分钟，分钟进位到度数。
这是 DMS 格式化器的关键边界情况。

#### LatFormatter / LonFormatter 增强

```python
class LatFormatter(mpl.ticker.FuncFormatter):
    def __init__(
        self,
        decimal_places: int = 0,
        format: Literal['decimal', 'dms'] = 'decimal',
        dms_precision: int = 0
    ):
        self.format = format
        self.dms_precision = dms_precision

        if format == 'dms':
            def formatter(x, _):
                if x == 0:
                    d, m, s = _decimal_to_dms(abs(x), dms_precision)
                    return f'{d}°{m}\'{s}"'

                d, m, s = _decimal_to_dms(abs(x), dms_precision)
                suffix = 'N' if x > 0 else 'S'
                return f'{d}°{m}\'{s}"{suffix}'
        else:
            fmt = f'{{:.{decimal_places}f}}'
            def formatter(x, _):
                return (
                    f'{fmt.format(abs(x))}°N' if x > 0 else
                    f'{fmt.format(abs(x))}°S' if x < 0 else
                    f'{fmt.format(abs(x))}°'
                )

        super().__init__(formatter)
```

**设计决策**：
- 使用 `Literal` 类型提示（Python 3.8+）限制 `format` 参数
- 0° 不显示半球后缀（N/S/E/W）
- ±180° 不显示半球后缀（经度边界）

#### DMSFormatter（新增）

通用 DMS 格式化器，不带半球后缀，适用于任意角度值：

```python
class DMSFormatter(mpl.ticker.FuncFormatter):
    def __init__(self, precision: int = 0, show_sign: bool = False):
        self.precision = precision
        self.show_sign = show_sign

        def formatter(x, _):
            d, m, s = _decimal_to_dms(abs(x), precision)
            sign = ''
            if show_sign:
                sign = '+' if x >= 0 else '-'
            elif x < 0:
                sign = '-'

            if precision > 0:
                return f'{sign}{d}°{m}\'{s:.{precision}f}"'
            return f'{sign}{d}°{m}\'{s}"'

        super().__init__(formatter)
```

**使用场景**：
- 方位角（0-360°）
- 倾角（-90° 到 +90°）
- 任意角度测量（不限于地理坐标）

---

## 扩展点索引

| 扩展目标 | 在哪里修改 | 说明 |
| --- | --- | --- |
| 新增内置 preset | `_core._register_builtin_presets()` | 调用 `register_preset()` 即可 |
| 新增 .mplstyle 文件 | `eqtools/viztools/styles/` 目录 | 文件名 `eqtools-<name>.mplstyle` |
| 新增列宽名称 | `_style_utils._COLUMN_WIDTHS` 或 `register_column_width()` | |
| 新增 mathfont 别名 | `_core._MATHFONT_ALIASES` dict | 值为 matplotlib `mathtext.fontset` 有效值 |
| 新增 CJK 字体候选 | `_font_utils._CHINESE_SANS_CANDIDATES` / `_CHINESE_SERIF_CANDIDATES` | 插入列表前部优先级更高 |
| 修改 preamble 默认值 | `_core._SANS_TEX_PREAMBLE` / `_SERIF_TEX_PREAMBLE` 常量 | 影响所有 `usetex=True` 调用 |
| 新增 PlotStyle 参数 | `_core.PlotStyle.__init__` + 新增 `_apply_xxx` 处理器 | 加入 `_HANDLERS` 列表 |
| 用户持久配置 | `~/.config/eqtools/viztools.json` | `_style_utils._load_user_config()` 解析 |
| **注册自定义 handler** | `PlotStyle.register_handler()` | **v2.1+ 无需修改源代码** |
| **Axes 级别样式** | `plotstyle.apply_to_axes(ax)` | **v2.1+ 单个 Axes 样式** |
| **DMS 格式化器** | `LatFormatter(format='dms')` / `DMSFormatter()` | **v2.1+ 度分秒格式** |

### 添加新内置 preset 的最小步骤

1. 在 `eqtools/viztools/styles/` 创建 `eqtools-<name>.mplstyle`
2. 在 `_core._register_builtin_presets()` 中调用：
   ```python
   register_preset(
       '<name>',
       mplstyles=['eqtools-<name>'],
       description='...',
   )
   ```
3. 用户即可通过 `PlotStyle('<name>')` 使用

### 添加新 PlotStyle 参数的最小步骤

以添加 `spine_width` 参数为例：

1. `_core.PlotStyle.__init__` 加参数：`spine_width: Optional[float] = None`
2. `__init__` 存储：`self._spine_width = spine_width`
3. 新建处理器方法（就地修改 `acc`）：
   ```python
   def _apply_spine_width(self, acc: Dict) -> None:
       if self._spine_width is not None:
           acc['axes.linewidth'] = float(self._spine_width)
   ```
4. 将 `'_apply_spine_width'` 插入 `_HANDLERS` 列表的适当位置
5. 无需改其他代码，继承链、apply/reset、decorator 全部自动支持

---

## 已知局限与注意事项

1. **`usetex=True` + `plt.show()` 在封装外调用**：`bake_text_fonts` 无法解决，
   必须在 PlotStyle 激活期间调用 `plt.show()`（见[usetex 渲染管道](#usetex-渲染管道)）。

2. **`subplots()` 不支持 `with` 语法**：`subplots()` 调用 `__enter__` 但没有配对的
   `__exit__`，close_event 是唯一的恢复路径。若图窗被 `plt.close('all')` 批量关闭，
   所有绑定的 close_event 均会触发，行为正确。

3. **`_probe_chinese_fonts()` 的 lru_cache**：安装新字体后 `fontManager.ttflist`
   会更新，但缓存的探测结果不会自动刷新。需清空缓存：
   ```python
   from eqtools.viztools._font_utils import _probe_chinese_fonts
   _probe_chinese_fonts.cache_clear()
   ```

4. **`_apply_stack` 是类变量**：多线程环境下，不同线程共享同一个 `_apply_stack`，
   可能产生竞态条件。当前版本未做线程保护，多线程绘图场景下请用上下文管理器
   （实例属性 `self._saved`）而不是 `apply/reset`。

5. **`_ensure_initialized()` 非线程安全**：惰性初始化没有加锁，
   并发调用可能导致重复初始化（副作用是 preset 被注册多次，
   实际上 `_PRESET_REGISTRY` 写入是幂等的，不会产生错误，但属于技术债务）。

6. **自定义 handler 的异常处理** (v2.1+)：自定义 handler 抛出的异常会被捕获并转换为 `UserWarning`，
   不会中断样式应用流程。这是有意设计，避免第三方 handler 破坏核心功能。

7. **`apply_to_axes()` 不支持字体族** (v2.1+)：`font.family` 是全局 rcParam，
   单个 Axes 无法拥有独立的字体回退列表。需要手动使用 `set_fontname()` 设置具体字体。

8. **DMS 格式化器的舍入边界** (v2.1+)：当秒数舍入到 60 时会进位到分钟，
   分钟进位到度数。这是正确行为，但在某些边界情况下可能产生意外的度数值
   （如 89°59'59.5" 舍入为 90°0'0"）。
