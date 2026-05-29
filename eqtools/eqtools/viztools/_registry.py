"""
_registry.py — 全局状态注册表，避免循环依赖并提供线程安全的状态管理。

这个模块集中管理所有全局状态：
- 预设注册表
- 列宽注册表
- 样式应用栈
- 初始化标志

通过单例模式提供线程安全的访问接口。
"""

import threading
from typing import Dict, List, Optional, Any

# 导入常量
from ._constants import DEFAULT_COLUMN_WIDTHS, MAX_STYLE_STACK_DEPTH


class _StateRegistry:
    """线程安全的全局状态管理单例。

    集中管理 viztools 的所有全局状态，避免模块间循环依赖，
    并提供线程安全的访问接口。
    """

    def __init__(self):
        self._lock = threading.RLock()

        # 预设注册表
        self._presets: Dict[str, Dict] = {}
        self._builtin_presets: set = set()

        # 列宽注册表（使用常量初始化）
        self._column_widths: Dict[str, float] = DEFAULT_COLUMN_WIDTHS.copy()

        # 样式应用栈（用于 PlotStyle.apply/reset）
        self._apply_stack: List[Dict] = []
        self._max_stack_depth: int = MAX_STYLE_STACK_DEPTH

        # 初始化标志
        self._initialized: bool = False
        self._styles_registered: bool = False

        # 自定义样式目录列表（用于插件式样式加载）
        self._custom_style_dirs: List = []

    # ======================================================================
    # 预设管理
    # ======================================================================

    def register_preset(self, name: str, spec: Dict) -> None:
        """注册一个预设。

        Parameters
        ----------
        name : str
            预设名称
        spec : dict
            预设规格，包含 base, mplstyles, rcparams 等
        """
        with self._lock:
            self._presets[name] = spec

    def get_preset(self, name: str) -> Optional[Dict]:
        """获取预设规格。

        Parameters
        ----------
        name : str
            预设名称

        Returns
        -------
        dict or None
            预设规格，如果不存在返回 None
        """
        with self._lock:
            return self._presets.get(name)

    def list_presets(self) -> Dict[str, Dict]:
        """列出所有预设。

        Returns
        -------
        dict
            预设名称到规格的映射（副本）
        """
        with self._lock:
            return self._presets.copy()

    def preset_exists(self, name: str) -> bool:
        """检查预设是否存在。

        Parameters
        ----------
        name : str
            预设名称

        Returns
        -------
        bool
        """
        with self._lock:
            return name in self._presets

    def mark_builtin_preset(self, name: str) -> None:
        """标记预设为内置预设（不可删除）。

        Parameters
        ----------
        name : str
            预设名称
        """
        with self._lock:
            self._builtin_presets.add(name)

    def is_builtin_preset(self, name: str) -> bool:
        """检查预设是否为内置预设。

        Parameters
        ----------
        name : str
            预设名称

        Returns
        -------
        bool
        """
        with self._lock:
            return name in self._builtin_presets

    def unregister_preset(self, name: str) -> None:
        """删除用户注册的预设。

        Parameters
        ----------
        name : str
            预设名称

        Raises
        ------
        ValueError
            如果尝试删除内置预设
        """
        with self._lock:
            if name in self._builtin_presets:
                raise ValueError(
                    f"Cannot unregister built-in preset '{name}'. "
                    f"Built-in presets: {sorted(self._builtin_presets)}"
                )
            self._presets.pop(name, None)

    # ======================================================================
    # 列宽管理
    # ======================================================================

    def register_column_width(self, name: str, width_inch: float) -> None:
        """注册列宽。

        Parameters
        ----------
        name : str
            列宽名称（不区分大小写）
        width_inch : float
            宽度（英寸）
        """
        with self._lock:
            self._column_widths[str(name).lower()] = float(width_inch)

    def get_column_width(self, name: str) -> Optional[float]:
        """获取列宽。

        Parameters
        ----------
        name : str
            列宽名称（不区分大小写）

        Returns
        -------
        float or None
            宽度（英寸），如果不存在返回 None
        """
        with self._lock:
            return self._column_widths.get(str(name).lower())

    def list_column_widths(self) -> Dict[str, float]:
        """列出所有列宽。

        Returns
        -------
        dict
            列宽名称到宽度的映射（副本）
        """
        with self._lock:
            return self._column_widths.copy()

    # ======================================================================
    # 样式栈管理
    # ======================================================================

    def push_style(self, saved: Dict) -> None:
        """将样式状态压入栈。

        Parameters
        ----------
        saved : dict
            保存的 rcParams 状态

        Warns
        -----
        ResourceWarning
            如果栈深度超过限制
        """
        with self._lock:
            if len(self._apply_stack) >= self._max_stack_depth:
                import warnings
                warnings.warn(
                    f"PlotStyle.apply() stack depth reached {self._max_stack_depth}. "
                    f"You may have forgotten to call reset(). "
                    f"Current stack size: {len(self._apply_stack)}. "
                    f"Consider using context manager (with PlotStyle(...)) instead.",
                    ResourceWarning, stacklevel=4
                )
            self._apply_stack.append(saved)

    def pop_style(self) -> Dict:
        """从栈中弹出样式状态。

        Returns
        -------
        dict
            保存的 rcParams 状态

        Raises
        ------
        IndexError
            如果栈为空
        """
        with self._lock:
            if not self._apply_stack:
                raise IndexError("No style to reset")
            return self._apply_stack.pop()

    def clear_style_stack(self) -> None:
        """清空样式栈。"""
        with self._lock:
            self._apply_stack.clear()

    def get_stack_depth(self) -> int:
        """获取当前栈深度。

        Returns
        -------
        int
        """
        with self._lock:
            return len(self._apply_stack)

    @property
    def max_stack_depth(self) -> int:
        """获取最大栈深度限制。"""
        with self._lock:
            return self._max_stack_depth

    @max_stack_depth.setter
    def max_stack_depth(self, value: int) -> None:
        """设置最大栈深度限制。

        Parameters
        ----------
        value : int
            新的限制值（必须 > 0）
        """
        if value <= 0:
            raise ValueError("max_stack_depth must be positive")
        with self._lock:
            self._max_stack_depth = int(value)

    # ======================================================================
    # 初始化标志
    # ======================================================================

    def is_initialized(self) -> bool:
        """检查是否已初始化。"""
        with self._lock:
            return self._initialized

    def mark_initialized(self) -> None:
        """标记为已初始化。"""
        with self._lock:
            self._initialized = True

    def is_styles_registered(self) -> bool:
        """检查样式文件是否已注册。"""
        with self._lock:
            return self._styles_registered

    def mark_styles_registered(self) -> None:
        """标记样式文件已注册。"""
        with self._lock:
            self._styles_registered = True

    # ======================================================================
    # 自定义样式目录管理（插件式加载）
    # ======================================================================

    def register_style_directory(self, path) -> None:
        """注册自定义样式目录，扫描其中的 .mplstyle 文件。

        Parameters
        ----------
        path : str or Path
            样式目录路径

        Raises
        ------
        ValueError
            如果路径不是一个目录
        FileNotFoundError
            如果路径不存在

        Examples
        --------
        >>> from eqtools.viztools import register_style_directory
        >>> register_style_directory('~/my_matplotlib_styles')
        >>> # 现在可以使用该目录中的样式
        >>> with PlotStyle('my_custom_style'):
        ...     fig, ax = plt.subplots()
        """
        from pathlib import Path
        import warnings

        path = Path(path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(f"Style directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        with self._lock:
            if path not in self._custom_style_dirs:
                self._custom_style_dirs.append(path)
                # 扫描并注册该目录中的样式
                self._scan_directory_for_styles(path)

    def _scan_directory_for_styles(self, directory) -> None:
        """扫描目录中的 .mplstyle 文件并注册为用户预设。

        Parameters
        ----------
        directory : Path
            要扫描的目录
        """
        import warnings
        from pathlib import Path

        try:
            import matplotlib.pyplot as plt
            from matplotlib.style.core import read_style_directory, update_nested_dict

            # 首先使用 matplotlib 的官方方法加载目录中的所有样式到 plt.style.library
            try:
                styles = read_style_directory(str(directory))
                update_nested_dict(plt.style.library, styles)
                # 刷新 available 列表
                plt.style.core.available[:] = sorted(plt.style.library.keys())
            except Exception as e:
                warnings.warn(
                    f"Failed to load styles from directory {directory} into matplotlib library: {e}",
                    UserWarning
                )
                # 如果批量加载失败，继续尝试逐个文件加载
                styles = {}

            # 然后为每个样式注册预设
            for style_file in directory.glob('*.mplstyle'):
                preset_name = style_file.stem

                # 跳过以下划线开头的私有样式
                if preset_name.startswith('_'):
                    continue

                # 注册为用户预设
                try:
                    # 使用样式名称（stem）而非文件路径
                    spec = {
                        'base': None,
                        'mplstyles': [preset_name],  # 使用名称而非路径
                        'rcparams': {},
                        'chinese': False,
                        'chinese_prefer_serif': False,
                        'description': f'Custom style from {directory.name}/{style_file.name}',
                    }
                    self.register_preset(preset_name, spec)
                except Exception as e:
                    warnings.warn(
                        f"Failed to register preset '{preset_name}' from {style_file}: {e}",
                        UserWarning
                    )
        except Exception as e:
            warnings.warn(
                f"Failed to scan directory {directory}: {e}",
                UserWarning
            )

    def list_custom_style_directories(self) -> list:
        """列出所有注册的自定义样式目录。

        Returns
        -------
        list
            自定义样式目录路径列表（副本）
        """
        with self._lock:
            return self._custom_style_dirs.copy()


# 全局单例实例
_registry = _StateRegistry()


# ======================================================================
# 向后兼容的便捷函数
# ======================================================================

def get_registry() -> _StateRegistry:
    """获取全局注册表实例。

    Returns
    -------
    _StateRegistry
        全局注册表单例
    """
    return _registry
