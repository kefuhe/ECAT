# BLSE/VCE 最小脚本骨架

这个例子展示固定 fault geometry 后，如何组织一个最小 BLSE/VCE 线性滑动脚本。真实案例应从 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 复制相近目录，再按自己的数据修改。

## 先生成配置

```bash
ecat-generate-config -o default_config.yml --gf-method cutde
ecat-generate-boundary -o bounds_config.yml -f MainFault
```

CLI 生成的是模板。至少需要检查：

- `faults` 名称是否和脚本里的 fault object 一致。
- `geodata` 顺序是否和脚本里的 `geodata = [...]` 一致。
- `alpha`、`sigmas`、`poly`、`bounds`、`rake` 和边界零滑设置。

## 最小脚本

```python
import os

os.environ.setdefault("CUTDE_USE_BACKEND", "cpp")

from csi import insar
from eqtools.csiExtend import print_fault_summary
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import (
    BayesianAdaptiveTriangularPatches as TriFault,
)
from eqtools.csiExtend.blse_multifaults_inversion import (
    BoundLSEMultiFaultsInversion,
)

lon0, lat0 = 87.5, 28.5

asc = insar("T012A", lon0=lon0, lat0=lat0, verbose=False)
asc.read_from_varres("InSAR/downsample/T012A_ifg", triangular=False, cov=True)

dsc = insar("T121D", lon0=lon0, lat0=lat0, verbose=False)
dsc.read_from_varres("InSAR/downsample/T121D_ifg", triangular=False, cov=True)

geodata = [asc, dsc]

fault = TriFault("MainFault", lon0=lon0, lat0=lat0, verbose=False)
fault.top = 0.0
fault.depth = 8.0
fault.generate_top_bottom_from_nonlinear_soln(
    clon=87.40,
    clat=28.67,
    cdepth=1.8,
    strike=332.0,
    dip=52.0,
    length=12.0,
    top=fault.top,
    depth=fault.depth,
)
fault.generate_mesh(top_size=1.0, bottom_size=1.5, show=False, verbose=0)
fault.initializeslip(values="depth")
fault.find_fault_fouredge_vertices()
print_fault_summary(fault)

inv = BoundLSEMultiFaultsInversion(
    "linear_slip",
    faults_list=[fault],
    geodata=geodata,
    config="default_config.yml",
    bounds_config="bounds_config.yml",
    verbose=True,
)

inv.run(penalty_weight=[100.0])
inv.returnModel()
inv.extract_and_plot_blse_results(plot_faults=True, plot_data=True)
```

## VCE

固定权重 BLSE 跑通后，再用 VCE 做权重诊断：

```python
inv.run_simple_vce()
inv.returnModel()
inv.extract_and_plot_blse_results(plot_faults=True, plot_data=True)
```

## 输出检查

- 模型是否能返回 slip，并且 patch 数量与 fault summary 一致。
- 数据、synthetic 和 residual 图是否方向一致。
- `bounds_config.yml` 中的边界零滑是否基于正确的 `top/bottom/left/right`。
- 如果使用 covariance，确认 `.cov` 与 `.txt/.rsp` 前缀对应。

相关参考：
[线性滑动反演配置](../reference/config_linear_slip.md),
[ECAT 约束管理器](../reference/constraint_manager.md),
[BLSE/VCE 参考](../reference/blse_vce.md),
[Fault Edges](../reference/fault_edges.md)。
