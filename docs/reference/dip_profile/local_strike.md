# 模式一：top 局部走向

这个模式让每个顶部节点的 strike 来自当前有序 top edge 的局部切向，适合明显弯曲且希望
下倾方向随断层弯曲而变化的非分层断层。

## 语义

```text
ordered top_coords
  -> segment geographic strike
  -> circular mean at interior nodes
  -> signed/continuous dip profile
  -> bottom_coords[i] from top_coords[i]
```

端点采用首段/末段走向，内部节点采用相邻走向的圆周平均。计算允许 `-90°` 与 `270°` 等价，
不会对跨 `0°/360°` 的走向做普通算术平均。

## 普通构模设置

三列控制点不包含独立 strike：

```python
import numpy as np

dip_points = np.array([
    [96.00, 21.00, 55.0],
    [96.20, 21.10, 65.0],
    [96.45, 21.20, 72.0],
])

fault.interpolate_top_dip_from_relocated_profile(
    dip_points,
    is_utm=False,
    interpolation_axis="auto",
)
fault.generate_bottom_from_segmented_relocated_dips(
    fault_depth=fault.depth,
    use_average_strike=False,
)
```

这里 `interpolation_axis="auto"` 只在 x/y 中选择主轴，不会自动改为弧长。普通构模入口
当前接受 `auto | x | y`；强弯曲且 x/y 不单调时，应先分段或改用下面的 Bayesian profile
resolver。

## Bayesian 设置

```python
import numpy as np

fault.set_dip_profile(
    sampled_controls=[
        [lon_0, lat_0, 70.0],
        [lon_1, lat_1, 80.0],
    ],
    fixed_controls=None,
    interpolation_axis="arc_length",
    transition_zones=None,
    is_utm=False,
)

fault.perturb_dips_with_preset_params(
    perturbations=np.zeros(2),
    angle_unit="degrees",
    use_average_strike=False,
)
fault.snapshot(capture_vertices=False, capture_layers=False)
```

候选解析时，所有 control 先投影到本候选的 top，再按累计弧长插值 dip；随后 strike 也从
该候选 top 重新计算。它不会复用无关的全局走向。

主 YAML 只声明参数切片和生成策略：

```yaml
faults:
  MainFault:
    geometry:
      update: true
      sample_positions: [0, 2]
    method_parameters:
      update_fault_geometry:
        method: perturb_dips_with_preset_params
        angle_unit: degrees
        use_average_strike: false
```

## 何时不能直接使用

- top 点序与期望走向相反：先反转点序，不要同时改 strike 和 dip 符号补偿。
- top 有尖锐回折或相邻线段接近反向：局部切向本身不稳定，应先平滑或分段。
- 地质走向不应等同于几何切线：改用[控制点走向插值](controlled_strike.md)。
- 近直线断层希望一个明确、可复现的统一走向：改用[单一代表性走向](representative_strike.md)。

## 结果判断

`top_strike` 应跟随 top 切向；而 `top -> bottom` 水平连接方向是其右手法向或左手法向，
两者接近垂直是正确结果。若 bottom trace 因倾角沿走向变化而不再平行于 top，也不自动表示
走向算错，应结合逐节点 dip 和最终单元法向判断。
