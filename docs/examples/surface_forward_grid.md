# 地表形变正演最小例子

这个例子从已有 slip model 生成规则 lon/lat 网格上的 ENU 地表位移。它适合检查滑动模型影响范围、导出绘图数据，或和 InSAR/GPS 观测位置做对照。

## 输入

- `output/slip_MainFault.gmt`：CSI patch GMT，header 中包含 slip 信息。
- `lon0/lat0`：和读取 fault 时一致的局部投影原点。
- 观测点模式：规则 lon/lat box 或自定义点文件。

## 规则网格

```python
import os

os.environ.setdefault("CUTDE_USE_BACKEND", "cpp")

from csi import TriangularPatches
from eqtools.csiExtend.surface_forward import (
    compute_multifault_surface_displacement,
    save_surface_forward_h5,
    save_surface_forward_txt,
)

lon0, lat0 = 87.5, 28.5

fault = TriangularPatches("MainFault", lon0=lon0, lat0=lat0, verbose=False)
fault.readPatchesFromFile("output/slip_MainFault.gmt", gmtslip=True)

result = compute_multifault_surface_displacement(
    {"MainFault": fault},
    box=[87.2, 87.8, 28.2, 28.8],
    npoints=300,
    nu=0.25,
    method="cutde",
    target_mem_gb=4.0,
    max_obs_batch=50000,
    output_coords="lonlat",
    return_each_fault=True,
    verbose=False,
)

save_surface_forward_h5("surface_displacement_enu.h5", result, include_by_fault=True)
save_surface_forward_txt("surface_displacement_enu.txt", result, include_by_fault=True)
```

## 自定义点

```python
import numpy as np

points = np.loadtxt("points_lonlat.txt")
lon = points[:, 0]
lat = points[:, 1]

result = compute_multifault_surface_displacement(
    {"MainFault": fault},
    lonlat=(lon, lat),
    method="cutde",
    output_coords="lonlat",
    return_each_fault=False,
)
```

## 检查

```python
total = result.disp_total_enu
print(total[:, 0].min(), total[:, 0].max())  # east
print(total[:, 1].min(), total[:, 1].max())  # north
print(total[:, 2].min(), total[:, 2].max())  # up
```

如果正演结果用于和 LOS 数据比较，需要继续用对应数据集的 ENU projection 做点积：

```text
scalar_observation = ENU_displacement dot projection
```

相关参考：
[Surface Displacement Forward](../reference/surface_displacement_forward.md),
[SAR Reader](../reference/sar_reader.md),
[BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md)。
