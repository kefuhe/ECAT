"""
应力计算核心模块

包含:
    - 应力参数定义与转换 (StressParams, RDefinition, KDefinition)
    - 深度依赖的应力剖面计算 (calc_stress_profile)
    - 断层面应力分解 (resolve_stress_on_fault)
    - 库仑应力计算 (calc_coulomb_stress)

设计原则:
    - 所有函数都是纯函数（无副作用、无状态）
    - 支持标量和数组输入
    - 使用 NumPy 广播实现向量化计算
"""

import numpy as np
from enum import Enum
from typing import Dict, Optional, Tuple, Union


# ============================================================
# 参数定义约定
# ============================================================

class RDefinition(Enum):
    """
    应力形因子 R 的不同定义约定
    
    不同文献中 R 的定义：
        - STANDARD:  R = (S1 - S2) / (S1 - S3)  [Angelier 1979; Delvaux 1997]
        - INVERTED:  R' = (S2 - S3) / (S1 - S3) [部分欧洲文献]
    
    关系: R + R' = 1
    
    物理含义:
        - R = 0: S1 = S2 (扁平应力椭球)
        - R = 1: S2 = S3 (拉长应力椭球)
        - R = 0.5: S2 在 S1 和 S3 中间
    """
    STANDARD = "standard"
    INVERTED = "inverted"


class KDefinition(Enum):
    """
    有效主应力比 k 的不同定义约定
    
    不同文献中 k 的定义：
        - S3_OVER_S1: k = S3'/S1'  [常用, 0 < k <= 1]
        - S1_OVER_S3: k = S1'/S3'  [部分工程文献, k >= 1]
    
    关系: k (S3/S1) = 1 / k (S1/S3)
    """
    S3_OVER_S1 = "s3/s1"
    S1_OVER_S3 = "s1/s3"


def convert_R(value: float, from_def: RDefinition, to_def: RDefinition) -> float:
    """
    在不同 R 定义之间转换
    
    参数:
        value: 输入的 R 值
        from_def: 输入值的定义约定
        to_def: 目标定义约定
    
    返回:
        转换后的 R 值
    
    示例:
        >>> convert_R(0.3, RDefinition.INVERTED, RDefinition.STANDARD)
        0.7
    """
    if from_def == to_def:
        return value
    if {from_def, to_def} == {RDefinition.STANDARD, RDefinition.INVERTED}:
        return 1.0 - value
    raise ValueError(f"不支持的转换: {from_def} -> {to_def}")


def convert_k(value: float, from_def: KDefinition, to_def: KDefinition) -> float:
    """
    在不同 k 定义之间转换
    
    参数:
        value: 输入的 k 值
        from_def: 输入值的定义约定
        to_def: 目标定义约定
    
    返回:
        转换后的 k 值
    
    示例:
        >>> convert_k(2.5, KDefinition.S1_OVER_S3, KDefinition.S3_OVER_S1)
        0.4
    """
    if from_def == to_def:
        return value
    if {from_def, to_def} == {KDefinition.S3_OVER_S1, KDefinition.S1_OVER_S3}:
        if value == 0:
            raise ValueError("k 值为 0，无法取倒数")
        return 1.0 / value
    raise ValueError(f"不支持的转换: {from_def} -> {to_def}")


# ============================================================
# 应力参数容器
# ============================================================

class StressParams:
    """
    区域应力参数容器
    
    内部统一使用:
        - R: STANDARD 定义 (S1-S2)/(S1-S3)
        - k: S3_OVER_S1 定义 (S3'/S1')
    
    属性:
        R: 应力形因子 [0, 1]
        k: 有效主应力比 (0, 1]
        gamma_v: 孔隙流体因子 Pf/Sv [0, 1]
        theta_SHmax: SHmax 方位角 (度，北偏东)
        regime: 应力体系 ('NF', 'SS', 'TF')
    
    使用示例:
        # 方式1: 直接使用标准定义
        >>> params = StressParams(R=0.5, k=0.4, regime='SS')
        
        # 方式2: 从其他约定转换
        >>> params = StressParams.from_convention(
        ...     R=0.5, R_def=RDefinition.INVERTED,
        ...     k=2.5, k_def=KDefinition.S1_OVER_S3,
        ...     regime='SS'
        ... )
    """
    
    _INTERNAL_R_DEF = RDefinition.STANDARD
    _INTERNAL_K_DEF = KDefinition.S3_OVER_S1
    
    def __init__(self, R: float, k: float, gamma_v: float = 0.4,
                 theta_SHmax: float = 0.0, regime: str = 'SS'):
        """
        使用标准定义创建参数对象
        
        参数:
            R: 应力形因子 (STANDARD 定义)，范围 [0, 1]
            k: 有效主应力比 (S3'/S1')，范围 (0, 1]
            gamma_v: 孔隙流体因子，范围 [0, 1]
            theta_SHmax: SHmax 方位角 (度)
            regime: 应力体系
        """
        self._validate(R, k, gamma_v, regime)
        self.R = R
        self.k = k
        self.gamma_v = gamma_v
        self.theta_SHmax = theta_SHmax
        self.regime = regime
    
    @classmethod
    def from_convention(cls, R: float, R_def: RDefinition,
                        k: float, k_def: KDefinition,
                        gamma_v: float = 0.4,
                        theta_SHmax: float = 0.0,
                        regime: str = 'SS') -> 'StressParams':
        """从指定约定创建参数对象"""
        R_std = convert_R(R, R_def, cls._INTERNAL_R_DEF)
        k_std = convert_k(k, k_def, cls._INTERNAL_K_DEF)
        return cls(R=R_std, k=k_std, gamma_v=gamma_v,
                   theta_SHmax=theta_SHmax, regime=regime)
    
    @staticmethod
    def _validate(R: float, k: float, gamma_v: float, regime: str):
        """验证参数范围"""
        if not 0 <= R <= 1:
            raise ValueError(f"R 应在 [0, 1] 范围内，当前: {R}")
        if not 0 < k <= 1:
            raise ValueError(f"k 应在 (0, 1] 范围内，当前: {k}")
        if not 0 <= gamma_v <= 1:
            raise ValueError(f"gamma_v 应在 [0, 1] 范围内，当前: {gamma_v}")
        if regime not in ('NF', 'SS', 'TF'):
            raise ValueError(f"regime 必须是 'NF', 'SS', 'TF' 之一")
    
    def get_R(self, definition: RDefinition = RDefinition.STANDARD) -> float:
        """获取指定定义下的 R 值"""
        return convert_R(self.R, self._INTERNAL_R_DEF, definition)
    
    def get_k(self, definition: KDefinition = KDefinition.S3_OVER_S1) -> float:
        """获取指定定义下的 k 值"""
        return convert_k(self.k, self._INTERNAL_K_DEF, definition)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'R': self.R, 'k': self.k, 'gamma_v': self.gamma_v,
            'theta_SHmax': self.theta_SHmax, 'regime': self.regime
        }
    
    def __repr__(self):
        return (f"StressParams(R={self.R:.3f}, k={self.k:.3f}, "
                f"γv={self.gamma_v:.2f}, θSHmax={self.theta_SHmax}°, "
                f"regime='{self.regime}')")


# ============================================================
# 应力剖面计算
# ============================================================

def calc_stress_profile(z: Union[float, np.ndarray],
                        params: Union[StressParams, Dict],
                        rho: float = 2700,
                        g: float = 9.8,
                        z_limit: Optional[float] = None) -> Dict[str, np.ndarray]:
    """
    计算随深度变化的主应力剖面
    
    根据 Anderson 断层理论，不同应力体系下主应力与 Sv, SH, Sh 的对应关系:
        - NF (正断): S1=Sv, S2=SH, S3=Sh
        - SS (走滑): S1=SH, S2=Sv, S3=Sh
        - TF (逆冲): S1=SH, S2=Sh, S3=Sv
    
    参数:
        z: 深度 (m, 正值向下)，标量或数组
        params: StressParams 对象或字典
        rho: 岩石密度 (kg/m^3)
        g: 重力加速度 (m/s^2)
        z_limit: 深度上限 (m)，超过此深度应力保持不变
    
    返回:
        字典，包含 (单位: Pa):
            - sigma_v: 垂直应力 (总应力)
            - sigma_H: 最大水平主应力 (总应力)
            - sigma_h: 最小水平主应力 (总应力)
            - Pf: 孔隙流体压力
    
    示例:
        >>> params = StressParams(R=0.5, k=0.4, regime='SS')
        >>> stresses = calc_stress_profile(np.linspace(0, 10000, 11), params)
    """
    if isinstance(params, dict):
        params = StressParams(**params)
    
    z = np.atleast_1d(np.asarray(z, dtype=float))
    z_eff = np.minimum(z, z_limit) if z_limit else z
    
    R, k = params.R, params.k
    gamma_v, regime = params.gamma_v, params.regime
    
    # 垂直应力和孔隙压
    sigma_v = rho * g * z_eff
    Pf = gamma_v * sigma_v
    sv_eff = sigma_v - Pf  # 有效垂直应力
    
    # 中间因子: S2'/S1' = 1 - R*(1-k)
    factor = 1 - R * (1 - k)
    
    # 根据应力体系计算有效主应力
    if regime == 'NF':  # S1=Sv, S2=SH, S3=Sh
        s1_eff = sv_eff
        sH_eff = s1_eff * factor
        sh_eff = s1_eff * k
        
    elif regime == 'SS':  # S1=SH, S2=Sv, S3=Sh
        # sv_eff = s1_eff * factor => s1_eff = sv_eff / factor
        s1_eff = np.divide(sv_eff, factor, where=factor != 0,
                          out=np.full_like(sv_eff, np.inf))
        sH_eff = s1_eff
        sh_eff = s1_eff * k
        
    elif regime == 'TF':  # S1=SH, S2=Sh, S3=Sv
        # sv_eff = s1_eff * k => s1_eff = sv_eff / k
        s1_eff = np.divide(sv_eff, k, where=k != 0,
                          out=np.full_like(sv_eff, np.inf))
        sH_eff = s1_eff
        sh_eff = s1_eff * factor
    else:
        raise ValueError(f"未知应力体系: {regime}")
    
    return {
        'sigma_v': sigma_v,
        'sigma_H': sH_eff + Pf,  # 转回总应力
        'sigma_h': sh_eff + Pf,
        'Pf': Pf
    }


def calc_effective_stress(stresses: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    从总应力计算有效应力
    
    参数:
        stresses: calc_stress_profile 返回的字典
    
    返回:
        包含有效应力的字典
    """
    Pf = stresses['Pf']
    return {
        'sigma_v_eff': stresses['sigma_v'] - Pf,
        'sigma_H_eff': stresses['sigma_H'] - Pf,
        'sigma_h_eff': stresses['sigma_h'] - Pf
    }


# ============================================================
# 断层面应力分解
# ============================================================

def resolve_stress_on_fault(sigma_v: Union[float, np.ndarray],
                            sigma_H: Union[float, np.ndarray],
                            sigma_h: Union[float, np.ndarray],
                            theta_SHmax: float,
                            strike: Union[float, np.ndarray],
                            dip: Union[float, np.ndarray],
                            rake: Optional[float] = None,
                            dip_direction: str = 'right'
                            ) -> Union[Tuple[np.ndarray, np.ndarray],
                                       Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    计算断层面上的正应力和剪应力
    
    使用 ENU (东-北-上) 坐标系，通过应力张量与断层法向量计算牵引力，
    再分解为正应力和剪应力分量。
    
    坐标约定:
        - E (东): x 轴正向
        - N (北): y 轴正向  
        - U (上): z 轴正向
    
    断层几何约定 (Aki & Richards):
        - strike: 走向角，从北顺时针
        - dip: 倾角，从水平面向下
        - rake: 滑动角，在断层面内从走向方向逆时针
            - 0°: 左旋走滑
            - 90°: 逆冲
            - -90°: 正断
            - 180°: 右旋走滑
    
    参数:
        sigma_v: 垂直应力 (Pa)
        sigma_H: 最大水平主应力 (Pa)
        sigma_h: 最小水平主应力 (Pa)
        theta_SHmax: SHmax 方位角 (度，北偏东)
        strike: 断层走向 (度)
        dip: 断层倾角 (度)
        rake: 滑动角 (度)，可选
        dip_direction: 'right' (右手法则) 或 'left'
    
    返回:
        如果 rake 为 None:
            (sigma_n, tau_total)
        如果提供 rake:
            (sigma_n, tau_rake, tau_total)
        
        其中:
            sigma_n: 正应力 (Pa, 压为正)
            tau_rake: 沿滑动方向剪应力 (Pa, 正值促进滑动)
            tau_total: 剪应力总大小 (Pa)
    """
    # 转换为弧度
    th = np.radians(theta_SHmax)
    s = np.radians(np.atleast_1d(strike))
    d = np.radians(np.atleast_1d(dip))
    
    # 确保应力数组维度匹配
    sigma_v = np.atleast_1d(sigma_v)
    sigma_H = np.atleast_1d(sigma_H)
    sigma_h = np.atleast_1d(sigma_h)
    
    # 广播到相同形状
    shape = np.broadcast_shapes(sigma_v.shape, s.shape)
    sigma_v = np.broadcast_to(sigma_v, shape)
    sigma_H = np.broadcast_to(sigma_H, shape)
    sigma_h = np.broadcast_to(sigma_h, shape)
    s = np.broadcast_to(s, shape)
    d = np.broadcast_to(d, shape)
    
    # 1. ENU 坐标系下的应力张量
    sin_t, cos_t = np.sin(th), np.cos(th)
    
    T_ee = sigma_H * sin_t**2 + sigma_h * cos_t**2
    T_nn = sigma_H * cos_t**2 + sigma_h * sin_t**2
    T_uu = sigma_v
    T_en = (sigma_H - sigma_h) * sin_t * cos_t
    
    # 2. 断层法向量 (指向上盘)
    sign = 1.0 if dip_direction == 'right' else -1.0
    dip_azimuth = s + sign * np.pi / 2
    
    n_e = np.sin(dip_azimuth) * np.sin(d)
    n_n = np.cos(dip_azimuth) * np.sin(d)
    n_u = np.cos(d)
    
    # 3. 走向向量
    strike_e = np.sin(s)
    strike_n = np.cos(s)
    strike_u = np.zeros_like(s)
    
    # 4. 下倾向量 (downdip = strike × n)
    downdip_e = strike_n * n_u - strike_u * n_n
    downdip_n = strike_u * n_e - strike_e * n_u
    downdip_u = strike_e * n_n - strike_n * n_e
    
    # 归一化
    downdip_mag = np.sqrt(downdip_e**2 + downdip_n**2 + downdip_u**2)
    mask = downdip_mag > 1e-10
    downdip_e = np.where(mask, downdip_e / downdip_mag, downdip_e)
    downdip_n = np.where(mask, downdip_n / downdip_mag, downdip_n)
    downdip_u = np.where(mask, downdip_u / downdip_mag, downdip_u)
    
    # 5. 牵引力向量 t = T · n
    t_e = T_ee * n_e + T_en * n_n
    t_n = T_en * n_e + T_nn * n_n
    t_u = T_uu * n_u
    
    # 6. 正应力 σn = t · n
    sigma_n = t_e * n_e + t_n * n_n + t_u * n_u
    
    # 7. 剪应力向量
    tau_e = t_e - sigma_n * n_e
    tau_n = t_n - sigma_n * n_n
    tau_u = t_u - sigma_n * n_u
    
    # 8. 剪应力大小
    tau_total = np.sqrt(tau_e**2 + tau_n**2 + tau_u**2)
    
    # 9. 沿滑动方向的剪应力
    if rake is not None:
        r = np.radians(rake)
        
        slip_e = np.cos(r) * strike_e - np.sin(r) * downdip_e
        slip_n = np.cos(r) * strike_n - np.sin(r) * downdip_n
        slip_u = np.cos(r) * strike_u - np.sin(r) * downdip_u
        
        tau_rake = -(tau_e * slip_e + tau_n * slip_n + tau_u * slip_u)
        
        return sigma_n, tau_rake, tau_total
    
    return sigma_n, tau_total


def diagnose_stress_direction(sigma_v: float, sigma_H: float, sigma_h: float,
                               theta_SHmax: float, strike: float, dip: float,
                               dip_direction: str = 'right') -> Dict[str, float]:
    """
    诊断断层面上的剪应力方向
    
    参数:
        sigma_v, sigma_H, sigma_h: 主应力 (Pa)
        theta_SHmax: SHmax 方位角 (度)
        strike, dip: 断层走向和倾角 (度)
        dip_direction: 'right' 或 'left'
    
    返回:
        字典，包含:
            - optimal_rake: 最优滑动 rake 角 (度)
            - tau_total: 剪应力大小 (Pa)
            - tau_strike: 走向分量 (Pa, 正=左旋)
            - tau_dip: 倾向分量 (Pa, 正=逆冲)
            - sigma_n: 正应力 (Pa)
    """
    sigma_n, tau_total = resolve_stress_on_fault(
        sigma_v, sigma_H, sigma_h, theta_SHmax,
        strike, dip, rake=None, dip_direction=dip_direction
    )
    
    _, tau_strike, _ = resolve_stress_on_fault(
        sigma_v, sigma_H, sigma_h, theta_SHmax,
        strike, dip, rake=0, dip_direction=dip_direction
    )
    
    _, tau_dip, _ = resolve_stress_on_fault(
        sigma_v, sigma_H, sigma_h, theta_SHmax,
        strike, dip, rake=90, dip_direction=dip_direction
    )
    
    optimal_rake = np.degrees(np.arctan2(
        float(tau_dip), float(tau_strike)
    ))
    
    return {
        'optimal_rake': optimal_rake,
        'tau_total': float(tau_total),
        'tau_strike': float(tau_strike),
        'tau_dip': float(tau_dip),
        'sigma_n': float(sigma_n)
    }


def calc_coulomb_stress(sigma_n: Union[float, np.ndarray],
                        tau: Union[float, np.ndarray],
                        mu: float = 0.6,
                        cohesion: float = 0.0) -> Union[float, np.ndarray]:
    """
    计算库仑破裂应力 (Coulomb Failure Stress)
    
    公式: CFS = |τ| - μ·σn - C
    
    参数:
        sigma_n: 有效正应力 (Pa, 压为正)
        tau: 剪应力 (Pa)
        mu: 摩擦系数
        cohesion: 内聚力 (Pa)
    
    返回:
        CFS (Pa): 正值表示趋向破裂，负值表示稳定
    """
    return np.abs(tau) - mu * sigma_n - cohesion