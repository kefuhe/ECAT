"""
eqtools.stress - 区域应力场与断层动力学加载模块

模块结构:
    core.py   - 核心计算（参数定义、应力剖面、断层应力分解）
    loader.py - 断层应力加载器（继承 SourceInv，用于动力学模拟）

基本用法:
    >>> from eqtools.stress import StressParams, calc_stress_profile
    >>> params = StressParams(R=0.5, k=0.4, regime='SS')
    >>> stresses = calc_stress_profile(depths, params)

动力学模拟:
    >>> from eqtools.stress import FaultStressLoader, HypocenterParams
    >>> loader = FaultStressLoader('MyFault', fault, params)
    >>> result = loader.compute_stress(rake=0.0)
    >>> nucleation = loader.create_nucleation_stress(hypocenter)
"""

from .core import (
    # 参数定义
    RDefinition,
    KDefinition,
    StressParams,
    # 转换函数
    convert_R,
    convert_k,
    # 应力计算
    calc_stress_profile,
    calc_effective_stress,
    # 断层应力
    resolve_stress_on_fault,
    diagnose_stress_direction,
    calc_coulomb_stress,
)

from .loader import (
    FaultStressLoader,
    HypocenterParams,
    FaultStressResult,
)

__all__ = [
    # 参数
    'RDefinition',
    'KDefinition',
    'StressParams',
    'convert_R',
    'convert_k',
    # 计算
    'calc_stress_profile',
    'calc_effective_stress',
    'resolve_stress_on_fault',
    'diagnose_stress_direction',
    'calc_coulomb_stress',
    # 加载器
    'FaultStressLoader',
    'HypocenterParams',
    'FaultStressResult',
]

__version__ = '0.1.0'