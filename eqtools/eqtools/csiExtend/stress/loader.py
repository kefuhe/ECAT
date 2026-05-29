"""
断层应力加载模块

提供 FaultStressLoader 类，用于:
    - 从区域应力参数计算断层面应力分布
    - 设置破裂核区域的过应力
    - 导出动力学模拟所需的数据
    - 便捷的可视化和报告生成
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Literal
from pathlib import Path

from csi import SourceInv

from .core import (
    StressParams,
    calc_stress_profile,
    calc_effective_stress,
    resolve_stress_on_fault,
    calc_coulomb_stress
)


@dataclass
class HypocenterParams:
    """
    震源（破裂核）参数
    
    基本参数:
        lon: 经度 (度)
        lat: 纬度 (度)
        depth: 深度 (km, 正值向下)
        radius: 破裂核半径 (km)
        overstress_ratio: 过应力比例 (默认 0.001)
    
    投影设置:
        plane: 投影平面 ('xz', 'yz', 'xy')，默认 'xz'
        align_strike: 走向对齐方式
            - 'pca': 自动 PCA 计算 (默认)
            - 'none': 不对齐
            - float: 指定角度
        discretized: 是否使用离散迹线 (默认 False)
        distance_metric: 距离计算方式 ('2d' 或 '3d')，默认 '2d'
    
    示例:
        >>> # 最简单的用法
        >>> hypo = HypocenterParams(lon=101.26, lat=37.77, depth=6.0, radius=3.0)
        >>> 
        >>> # 完整配置
        >>> hypo = HypocenterParams(
        ...     lon=101.26, lat=37.77, depth=6.0, radius=3.0,
        ...     overstress_ratio=0.001,
        ...     plane='xz', align_strike='pca', distance_metric='2d'
        ... )
    """
    lon: float
    lat: float
    depth: float
    radius: float
    overstress_ratio: float = 0.001
    plane: Literal['xz', 'yz', 'xy'] = 'xz'
    align_strike: Union[Literal['pca', 'none'], float] = 'pca'
    discretized: bool = False
    distance_metric: Literal['2d', '3d'] = '2d'


@dataclass
class FaultStressResult:
    """断层应力计算结果"""
    sigma_n: np.ndarray      # 正应力 (Pa)
    tau_rake: np.ndarray     # 沿滑动方向剪应力 (Pa)
    tau_total: np.ndarray    # 总剪应力 (Pa)
    tau_strike: np.ndarray   # 走向剪应力 (Pa)
    tau_dip: np.ndarray      # 倾向剪应力 (Pa)
    cfs: np.ndarray          # 库仑破裂应力 (Pa)
    patch_depths: np.ndarray # 面片深度 (m)


class FaultStressLoader(SourceInv):
    """
    断层应力加载器
    
    一站式完成断层应力计算、破裂核设置、可视化和数据导出。
    
    快速开始:
        >>> from eqtools.csiExtend.stress import StressParams, FaultStressLoader, HypocenterParams
        >>> 
        >>> # 1. 准备应力参数和断层
        >>> params = StressParams.from_convention(R=0.4, k=2.2, gamma_v=0.45, regime='SS')
        >>> fault = ATP('MyFault', lon0=101.0, lat0=37.5)
        >>> fault.readPatchesFromFile('fault.gmt')
        >>> fault.setTrace(delta_depth=0.1)  # 必须设置迹线！
        >>> 
        >>> # 2. 创建加载器并计算
        >>> loader = FaultStressLoader('MyFault', fault, params, theta_SHmax=45)
        >>> loader.compute_stress(rake=0.0, mu=0.6)
        >>> 
        >>> # 3. 设置破裂核并导出
        >>> hypo = HypocenterParams(lon=101.26, lat=37.77, depth=6.0, radius=3.0)
        >>> loader.run(hypo, output_dir='results/', save_figures=True)
    
    属性:
        fault: 断层几何对象
        stress_params: 应力参数
        result: 应力计算结果 (FaultStressResult)
        nucleation: 破裂核数据字典
    """
    
    def __init__(self, name: str,
                 fault_geometry,
                 stress_params: StressParams,
                 theta_SHmax: float = 0.0,
                 rho: float = 2700,
                 z_limit: Optional[float] = None,
                 utmzone: Optional[str] = None,
                 ellps: str = 'WGS84',
                 lon0: Optional[float] = None,
                 lat0: Optional[float] = None):
        """
        初始化断层应力加载器
        
        参数:
            name: 断层名称
            fault_geometry: 断层几何对象 (需要已设置迹线 xf/yf)
            stress_params: 区域应力参数 (StressParams 对象)
            theta_SHmax: 最大水平主应力方位角 (度，从北顺时针)
            rho: 岩石密度 (kg/m^3)，默认 2700
            z_limit: 应力饱和深度 (m)，None 表示不限制
            utmzone: UTM 投影带
            ellps: 椭球体，默认 'WGS84'
            lon0, lat0: 投影中心经纬度
        """
        if lon0 is None and hasattr(fault_geometry, 'lon0'):
            lon0 = fault_geometry.lon0
        if lat0 is None and hasattr(fault_geometry, 'lat0'):
            lat0 = fault_geometry.lat0
        if utmzone is None and hasattr(fault_geometry, 'utmzone'):
            utmzone = fault_geometry.utmzone
        
        super().__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)
        
        self.fault = fault_geometry
        self.stress_params = stress_params
        self.theta_SHmax = theta_SHmax
        self.rho = rho
        self.z_limit = z_limit
        
        self._check_fault_trace()
        self._extract_geometry()
        
        # 结果存储
        self._result: Optional[FaultStressResult] = None
        self._nucleation: Optional[Dict] = None
        self._rake: Optional[float] = None
        self._mu: Optional[float] = None
    
    @property
    def result(self) -> Optional[FaultStressResult]:
        """应力计算结果"""
        return self._result
    
    @property
    def nucleation(self) -> Optional[Dict]:
        """破裂核数据"""
        return self._nucleation
    
    def _check_fault_trace(self):
        """检查断层迹线"""
        has_xf_yf = hasattr(self.fault, 'xf') and hasattr(self.fault, 'yf')
        has_xi_yi = hasattr(self.fault, 'xi') and hasattr(self.fault, 'yi')
        
        if not has_xf_yf and not has_xi_yi:
            raise ValueError(
                "断层几何缺少迹线坐标！\n\n"
                "请先调用 fault.setTrace() 设置迹线:\n"
                "    fault.setTrace(delta_depth=0.1, sort='x')\n\n"
                "或手动设置:\n"
                "    fault.xf = np.array([...])  # 迹线 x 坐标 (km)\n"
                "    fault.yf = np.array([...])  # 迹线 y 坐标 (km)"
            )
        
        self._has_xf_yf = has_xf_yf
        self._has_xi_yi = has_xi_yi
        
        if has_xf_yf:
            self._trace_xf = np.asarray(self.fault.xf)
            self._trace_yf = np.asarray(self.fault.yf)
        if has_xi_yi:
            self._trace_xi = np.asarray(self.fault.xi)
            self._trace_yi = np.asarray(self.fault.yi)
    
    def _extract_geometry(self):
        """提取断层几何"""
        self.centers = self.fault.getcenters()
        self.patch_x = self.centers[:, 0]
        self.patch_y = self.centers[:, 1]
        self.patch_z = np.abs(self.centers[:, 2])
        
        try:
            self.centers_ll = self.fault.getcenters(coordinates='ll')
        except:
            lons, lats = self.xy2ll(self.patch_x, self.patch_y)
            self.centers_ll = np.column_stack([lons, lats, self.patch_z])
        
        self.patch_depths = self.patch_z * 1000
        self.strikes = np.degrees(self.fault.getStrikes())
        self.dips = np.degrees(self.fault.getDips())
        self.n_patches = len(self.patch_depths)
        self._mean_strike = np.mean(self.strikes)
        self._pca_cache = {}
    
    def _get_trace_coords(self, discretized: bool) -> Tuple[np.ndarray, np.ndarray]:
        """获取迹线坐标"""
        if discretized:
            if not self._has_xi_yi:
                raise ValueError("离散迹线 (xi, yi) 未设置，请改用 discretized=False")
            return self._trace_xi, self._trace_yi
        else:
            if not self._has_xf_yf:
                raise ValueError("原始迹线 (xf, yf) 未设置，请改用 discretized=True")
            return self._trace_xf, self._trace_yf
    
    def _compute_pca_direction(self, trace_x: np.ndarray, 
                                trace_y: np.ndarray) -> Tuple[np.ndarray, float]:
        """计算 PCA 主方向"""
        cache_key = (id(trace_x), id(trace_y))
        if cache_key in self._pca_cache:
            return self._pca_cache[cache_key]
        
        xy = np.column_stack([trace_x, trace_y])
        xy_centered = xy - xy.mean(axis=0)
        _, _, Vt = np.linalg.svd(xy_centered, full_matrices=False)
        direction = Vt[0]
        
        angle_rad = np.arctan2(direction[0], direction[1])
        angle_deg = np.degrees(angle_rad) % 360
        if angle_deg > 180:
            angle_deg -= 180
            direction = -direction
        
        self._pca_cache[cache_key] = (direction, angle_deg)
        return direction, angle_deg
    
    def _get_rotation_matrix(self, align_strike: Union[str, float],
                              discretized: bool) -> Tuple[np.ndarray, float]:
        """获取旋转矩阵"""
        if align_strike == 'none':
            return np.eye(2), 0.0
        
        if align_strike == 'pca':
            trace_x, trace_y = self._get_trace_coords(discretized)
            direction, strike_angle = self._compute_pca_direction(trace_x, trace_y)
        else:
            strike_angle = float(align_strike)
            angle_rad = np.radians(strike_angle)
            direction = np.array([np.sin(angle_rad), np.cos(angle_rad)])
        
        cos_a, sin_a = direction[0], direction[1]
        R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        return R, strike_angle
    
    def _transform_coordinates(self, x: np.ndarray, y: np.ndarray,
                                hypocenter: HypocenterParams) -> Tuple[np.ndarray, np.ndarray]:
        """变换坐标"""
        R, _ = self._get_rotation_matrix(hypocenter.align_strike, hypocenter.discretized)
        xy = np.column_stack([x, y])
        xy_rotated = xy @ R.T
        return xy_rotated[:, 0], xy_rotated[:, 1]
    
    def _transform_point(self, x: float, y: float,
                          hypocenter: HypocenterParams) -> Tuple[float, float]:
        """变换单点"""
        R, _ = self._get_rotation_matrix(hypocenter.align_strike, hypocenter.discretized)
        xy = np.array([x, y])
        xy_rotated = R @ xy
        return xy_rotated[0], xy_rotated[1]
    
    def _get_plane_coords(self, hypocenter: HypocenterParams) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """获取投影平面坐标"""
        patch_x_rot, patch_y_rot = self._transform_coordinates(
            self.patch_x, self.patch_y, hypocenter
        )
        hypo_x, hypo_y = self.ll2xy(hypocenter.lon, hypocenter.lat)
        hypo_x_rot, hypo_y_rot = self._transform_point(hypo_x, hypo_y, hypocenter)
        
        if hypocenter.plane == 'xz':
            return patch_x_rot, self.patch_z, hypo_x_rot, hypocenter.depth
        elif hypocenter.plane == 'yz':
            return patch_y_rot, self.patch_z, hypo_y_rot, hypocenter.depth
        elif hypocenter.plane == 'xy':
            return patch_x_rot, patch_y_rot, hypo_x_rot, hypo_y_rot
        else:
            raise ValueError(f"未知投影平面: {hypocenter.plane}")
    
    def find_nearest_patch(self, hypocenter: HypocenterParams) -> Tuple[int, float]:
        """找到最近面片"""
        patch_u, patch_v, hypo_u, hypo_v = self._get_plane_coords(hypocenter)
        distances_2d = np.sqrt((patch_u - hypo_u)**2 + (patch_v - hypo_v)**2)
        idx = np.argmin(distances_2d)
        return idx, distances_2d[idx]
    
    def calc_distance_from_nearest(self, hypocenter: HypocenterParams, 
                                    nearest_idx: int) -> np.ndarray:
        """计算到最近点的距离"""
        if hypocenter.distance_metric == '3d':
            ref = np.array([self.patch_x[nearest_idx], 
                           self.patch_y[nearest_idx], 
                           self.patch_z[nearest_idx]])
            dx = self.patch_x - ref[0]
            dy = self.patch_y - ref[1]
            dz = self.patch_z - ref[2]
            return np.sqrt(dx**2 + dy**2 + dz**2)
        else:
            patch_u, patch_v, _, _ = self._get_plane_coords(hypocenter)
            ref_u, ref_v = patch_u[nearest_idx], patch_v[nearest_idx]
            return np.sqrt((patch_u - ref_u)**2 + (patch_v - ref_v)**2)
    
    def get_patches_in_radius(self, hypocenter: HypocenterParams, 
                               nearest_idx: Optional[int] = None) -> np.ndarray:
        """获取半径内面片"""
        if nearest_idx is None:
            nearest_idx, _ = self.find_nearest_patch(hypocenter)
        distances = self.calc_distance_from_nearest(hypocenter, nearest_idx)
        return np.where(distances <= hypocenter.radius)[0]
    
    # ==================== 核心计算方法 ====================
    
    def compute_stress(self, rake: float = 0.0, mu: float = 0.6) -> FaultStressResult:
        """
        计算断层面应力分布
        
        参数:
            rake: 滑动角 (度)，0=走滑，90=逆冲，-90=正断
            mu: 摩擦系数
        
        返回:
            FaultStressResult 对象
        """
        stresses = calc_stress_profile(
            self.patch_depths, self.stress_params,
            rho=self.rho, z_limit=self.z_limit
        )
        eff = calc_effective_stress(stresses)
        
        sigma_n, tau_rake, tau_total = resolve_stress_on_fault(
            eff['sigma_v_eff'], eff['sigma_H_eff'], eff['sigma_h_eff'],
            self.theta_SHmax, self.strikes, self.dips, rake=rake
        )
        
        _, tau_strike, _ = resolve_stress_on_fault(
            eff['sigma_v_eff'], eff['sigma_H_eff'], eff['sigma_h_eff'],
            self.theta_SHmax, self.strikes, self.dips, rake=0
        )
        _, tau_dip, _ = resolve_stress_on_fault(
            eff['sigma_v_eff'], eff['sigma_H_eff'], eff['sigma_h_eff'],
            self.theta_SHmax, self.strikes, self.dips, rake=90
        )
        
        cfs = calc_coulomb_stress(sigma_n, tau_rake, mu=mu)
        
        self._result = FaultStressResult(
            sigma_n=sigma_n, tau_rake=tau_rake, tau_total=tau_total,
            tau_strike=tau_strike, tau_dip=tau_dip, cfs=cfs,
            patch_depths=self.patch_depths
        )
        self._rake = rake
        self._mu = mu
        
        return self._result
    
    def create_nucleation(self, hypocenter: HypocenterParams,
                          mu_s: Optional[float] = None,
                          rake: Optional[float] = None,
                          tapered: bool = False,
                          inner_radius: Optional[float] = None,
                          taper_type: str = 'cosine') -> Dict:
        """
        创建破裂核应力场
        
        参数:
            hypocenter: 震源参数
            mu_s: 静摩擦系数
            rake: 滑动角
            tapered: 是否使用渐变过渡
            inner_radius: 内环半径 (仅 tapered=True 时有效)
            taper_type: 过渡类型 ('gaussian', 'cosine', 'linear')
        
        返回:
            破裂核数据字典
        """
        mu_s = mu_s or self._mu or 0.6
        rake = rake if rake is not None else (self._rake or 0.0)
        
        if self._result is None:
            self.compute_stress(rake=rake, mu=mu_s)
        
        nearest_idx, dist_to_hypo = self.find_nearest_patch(hypocenter)
        distances = self.calc_distance_from_nearest(hypocenter, nearest_idx)
        
        sigma_n = self._result.sigma_n.copy()
        tau_base = self._result.tau_rake.copy()
        tau_critical = np.sign(tau_base) * sigma_n * mu_s * (1 + hypocenter.overstress_ratio)
        
        tau_nucleation = tau_base.copy()
        
        if tapered:
            inner_r = inner_radius or hypocenter.radius * 0.5
            if inner_r >= hypocenter.radius:
                raise ValueError(f"内环半径 ({inner_r}) 必须小于外环半径 ({hypocenter.radius})")
            
            inner_mask = distances <= inner_r
            taper_mask = (distances > inner_r) & (distances <= hypocenter.radius)
            
            tau_nucleation[inner_mask] = tau_critical[inner_mask]
            
            if np.any(taper_mask):
                r_norm = (distances[taper_mask] - inner_r) / (hypocenter.radius - inner_r)
                if taper_type == 'gaussian':
                    weights = np.exp(-2 * r_norm**2)
                elif taper_type == 'cosine':
                    weights = 0.5 * (1 + np.cos(np.pi * r_norm))
                elif taper_type == 'linear':
                    weights = 1 - r_norm
                else:
                    raise ValueError(f"未知过渡类型: {taper_type}")
                tau_nucleation[taper_mask] = tau_critical[taper_mask] * weights + tau_base[taper_mask] * (1 - weights)
        else:
            inner_r = hypocenter.radius
            inner_mask = distances <= hypocenter.radius
            taper_mask = np.zeros(self.n_patches, dtype=bool)
            tau_nucleation[inner_mask] = tau_critical[inner_mask]
        
        nucleation_mask = distances <= hypocenter.radius
        
        _, strike_used = self._get_rotation_matrix(hypocenter.align_strike, hypocenter.discretized)
        
        self._nucleation = {
            'sigma_n': sigma_n,
            'tau_initial': tau_base,
            'tau_critical': tau_critical,
            'tau_nucleation': tau_nucleation,
            'distances': distances,
            'nucleation_mask': nucleation_mask,
            'inner_mask': inner_mask,
            'taper_mask': taper_mask,
            'nearest_patch_idx': nearest_idx,
            'hypocenter': {
                'lon': hypocenter.lon,
                'lat': hypocenter.lat,
                'depth': hypocenter.depth,
                'radius': hypocenter.radius,
                'inner_radius': inner_r,
                'overstress_ratio': hypocenter.overstress_ratio,
                'tapered': tapered,
                'taper_type': taper_type if tapered else None,
                'plane': hypocenter.plane,
                'align_strike': hypocenter.align_strike,
                'strike_used_deg': strike_used,
                'distance_metric': hypocenter.distance_metric,
                'nearest_patch_depth_km': self.patch_z[nearest_idx],
                'dist_to_hypocenter_2d': dist_to_hypo,
                'n_inner_patches': int(inner_mask.sum()),
                'n_taper_patches': int(taper_mask.sum()),
                'n_total_patches': int(nucleation_mask.sum())
            }
        }
        
        return self._nucleation
    
    # ==================== 便捷方法 ====================
    
    def run(self, hypocenter: Optional[HypocenterParams] = None,
            rake: float = 0.0, mu_s: float = 0.6, mu_d: float = 0.4,
            tapered: bool = False, inner_radius: Optional[float] = None,
            output_dir: Optional[str] = None,
            save_figures: bool = True,
            save_data: bool = True,
            verbose: bool = True) -> Dict:
        """
        一键运行完整流程
        
        包括: 计算应力 → 设置破裂核 → 保存图片 → 导出数据
        
        参数:
            hypocenter: 震源参数 (可选)
            rake: 滑动角 (度)
            mu_s, mu_d: 静/动摩擦系数
            tapered: 是否使用渐变破裂核
            inner_radius: 内环半径 (tapered=True 时)
            output_dir: 输出目录，默认当前目录
            save_figures: 是否保存图片
            save_data: 是否保存数据文件
            verbose: 是否打印详细信息
        
        返回:
            导出的数据字典
        
        示例:
            >>> loader.run(hypo, output_dir='results/', save_figures=True)
        """
        output_dir = Path(output_dir) if output_dir else Path('.')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 计算应力
        if verbose:
            print(f"\n{'='*60}")
            print(f"断层应力计算: {self.name}")
            print(f"{'='*60}")
        
        self.compute_stress(rake=rake, mu=mu_s)
        
        if verbose:
            print(self.summary())
        
        # 2. 设置破裂核
        if hypocenter:
            self.create_nucleation(hypocenter, mu_s=mu_s, rake=rake,
                                   tapered=tapered, inner_radius=inner_radius)
            if verbose:
                self.print_nucleation_info()
        
        # 3. 保存图片
        if save_figures:
            self.plot_stress(output_dir=output_dir, verbose=verbose)
            if hypocenter:
                self.plot_nucleation(output_dir=output_dir, verbose=verbose)
        
        # 4. 导出数据
        data = self.export(hypocenter=hypocenter, mu_s=mu_s, mu_d=mu_d, rake=rake)
        
        if save_data:
            data_file = output_dir / f'{self.name}_dynamics.npz'
            np.savez(data_file, **data)
            if verbose:
                print(f"\n[v] 数据已保存: {data_file}")
        
        return data
    
    def plot_stress(self, output_dir: Optional[str] = None, 
                    cmap: str = 'cmc.roma_r',
                    verbose: bool = True) -> None:
        """
        绘制应力分布图
        
        参数:
            output_dir: 输出目录
            cmap: 颜色图
            verbose: 是否打印信息
        """
        if self._result is None:
            raise RuntimeError("请先调用 compute_stress()")
        
        output_dir = Path(output_dir) if output_dir else Path('.')
        
        plots = [
            ('tau_rake', self._result.tau_rake / 1e6, '剪应力 τ'),
            ('sigma_n', self._result.sigma_n / 1e6, '正应力 σn'),
            ('cfs', self._result.cfs / 1e6, '库仑应力 CFS'),
        ]
        
        for suffix, data, label in plots:
            self.fault.slip[:, 0] = data
            self.fault.plot(plot_on_2d=False, slip='strikeslip', cmap=cmap)
            
            fig_path = output_dir / f'{self.name}_{suffix}.png'
            self.fault.slipfig.savefig(
                prefix=str(output_dir / self.name) + f'_{suffix}',
                ftype='png', dpi=300, saveFig=['fault']
            )
            if verbose:
                print(f"[v] 已保存: {fig_path}")
    
    def plot_nucleation(self, output_dir: Optional[str] = None,
                        cmap: str = 'cmc.roma_r',
                        verbose: bool = True) -> None:
        """
        绘制破裂核分布图
        
        参数:
            output_dir: 输出目录
            cmap: 颜色图
            verbose: 是否打印信息
        """
        if self._nucleation is None:
            raise RuntimeError("请先调用 create_nucleation()")
        
        output_dir = Path(output_dir) if output_dir else Path('.')
        
        # 破裂核掩码图
        mask_data = np.zeros(self.n_patches)
        mask_data[self._nucleation['inner_mask']] = 1.0
        mask_data[self._nucleation['taper_mask']] = 0.5
        
        self.fault.slip[:, 0] = mask_data
        self.fault.plot(plot_on_2d=False, slip='strikeslip', cmap=cmap)
        self.fault.slipfig.savefig(
            prefix=str(output_dir / self.name) + '_nucleation_mask',
            ftype='png', dpi=300, saveFig=['fault']
        )
        if verbose:
            print(f"[v] 已保存: {output_dir / self.name}_nucleation_mask.png")
        
        # 破裂核应力图
        tau_max = np.abs(self._nucleation['tau_nucleation']).max() / 1e6
        self.fault.slip[:, 0] = self._nucleation['tau_nucleation'] / 1e6
        self.fault.plot(plot_on_2d=False, slip='strikeslip', cmap=cmap, 
                       norm=[0, tau_max * 1.1])
        self.fault.slipfig.savefig(
            prefix=str(output_dir / self.name) + '_tau_nucleation',
            ftype='png', dpi=300, saveFig=['fault']
        )
        if verbose:
            print(f"[v] 已保存: {output_dir / self.name}_tau_nucleation.png")
    
    def print_nucleation_info(self) -> None:
        """打印破裂核信息"""
        if self._nucleation is None:
            print("尚未设置破裂核")
            return
        
        h = self._nucleation['hypocenter']
        print(f"\n{'─'*50}")
        print(f"破裂核设置")
        print(f"{'─'*50}")
        print(f"位置: ({h['lon']}°, {h['lat']}°, {h['depth']} km)")
        print(f"半径: {h['radius']} km" + (f" (内环: {h['inner_radius']} km)" if h['tapered'] else ""))
        print(f"过应力比: {h['overstress_ratio']}")
        print(f"投影: plane={h['plane']}, align={h['align_strike']} -> {h['strike_used_deg']:.1f}°")
        print(f"距离度量: {h['distance_metric']}")
        print(f"面片统计: 内环={h['n_inner_patches']}, 过渡={h['n_taper_patches']}, 总计={h['n_total_patches']}")
        print(f"最近面片深度: {h['nearest_patch_depth_km']:.2f} km")
        print(f"{'─'*50}")
    
    def export(self, hypocenter: Optional[HypocenterParams] = None,
               mu_s: float = 0.6, mu_d: float = 0.4,
               rake: float = 0.0, fmt: str = 'dict') -> Union[Dict, np.ndarray]:
        """
        导出动力学模拟数据
        
        参数:
            hypocenter: 震源参数
            mu_s, mu_d: 静/动摩擦系数
            rake: 滑动角
            fmt: 输出格式 ('dict' 或 'array')
        
        返回:
            数据字典或数组
        """
        if self._result is None:
            self.compute_stress(rake=rake, mu=mu_s)
        
        data = {
            'n_patches': self.n_patches,
            'centers_xyz': self.centers,
            'centers_lonlatdep': self.centers_ll,
            'strike': self.strikes,
            'dip': self.dips,
            'rake': np.full(self.n_patches, rake),
            'sigma_n': self._result.sigma_n,
            'tau_strike': self._result.tau_strike,
            'tau_dip': self._result.tau_dip,
            'mu_s': np.full(self.n_patches, mu_s),
            'mu_d': np.full(self.n_patches, mu_d),
        }
        
        if hypocenter and self._nucleation:
            data.update({
                'tau_initial': self._nucleation['tau_initial'],
                'tau_nucleation': self._nucleation['tau_nucleation'],
                'nucleation_mask': self._nucleation['nucleation_mask'],
                'hypocenter': self._nucleation['hypocenter']
            })
        else:
            data['tau_initial'] = self._result.tau_rake
        
        if fmt == 'array':
            tau = data.get('tau_nucleation', data['tau_initial'])
            return np.column_stack([
                self.centers, self.strikes, self.dips,
                np.full(self.n_patches, rake),
                self._result.sigma_n, tau,
                np.full(self.n_patches, mu_s),
                np.full(self.n_patches, mu_d)
            ])
        
        return data
    
    def summary(self) -> str:
        """返回摘要字符串"""
        if self._result is None:
            return "尚未计算应力"
        
        r = self._result
        n_unstable = np.sum(r.cfs > 0)
        
        return f"""
面片数量: {self.n_patches}
深度范围: {self.patch_z.min():.2f} - {self.patch_z.max():.2f} km
平均走向: {self._mean_strike:.1f}°
应力体系: {self.stress_params.regime}, θSHmax = {self.theta_SHmax}°

应力统计 (MPa):
  σn:       {r.sigma_n.min()/1e6:8.2f} ~ {r.sigma_n.max()/1e6:.2f}
  τ_total:  {r.tau_total.min()/1e6:8.2f} ~ {r.tau_total.max()/1e6:.2f}
  τ_strike: {r.tau_strike.min()/1e6:8.2f} ~ {r.tau_strike.max()/1e6:.2f}
  τ_dip:    {r.tau_dip.min()/1e6:8.2f} ~ {r.tau_dip.max()/1e6:.2f}

库仑应力 (μ={self._mu}):
  CFS: {r.cfs.min()/1e6:8.2f} ~ {r.cfs.max()/1e6:.2f} MPa
  趋向破裂: {n_unstable}/{self.n_patches} ({100*n_unstable/self.n_patches:.1f}%)
""".strip()
    
    def __repr__(self):
        return f"FaultStressLoader('{self.name}', n_patches={self.n_patches})"