# 固定几何中的倾角剖面

本页说明如何把沿走向变化的倾角剖面只物化一次，再用于普通 BLSE/VCE。它不建立
Bayesian 候选参考，也不要求固定拓扑参数映射。若要比较多个几何或在 SMC-FJ 中采样倾角，
转到 [sampled/fixed/transition 组合](bayesian_mixed.md)。

## 先判断是否需要本页

`dip-profile`、固定拓扑和几何扰动是三个不同概念：

| 概念 | 回答的问题 |
| --- | --- |
| dip-profile | top 上各位置采用什么倾角，怎样在控制点和转换带之间插值 |
| 固定拓扑 | 多个候选是否保持同一组 `Faces`、patch 身份和参数映射 |
| 几何扰动 | sampled controls 是否随候选参数改变 |

只运行一个已经选定的几何时，使用本页的 `generate_mesh(...)` 路线即可。多个候选需要逐
patch 对照时使用 `generate_and_deform_mesh(...)`；是否 `snapshot()` 取决于 reference 是
独立边界还是生成型 dip profile，不能一概而论。

| 任务 | `snapshot()` | `set_densification()` 后重放 | mesh 入口 |
| --- | --- | --- | --- |
| 单个固定几何 BLSE/VCE | 不需要 | 不需要 | `generate_mesh(...)` |
| BLSE/VCE 独立 top/bottom 的固定拓扑比较 | 需要 | 启用候选期加密时需要 | 首次 `remap=True`，候选 `remap=False` |
| 生成型 dip-profile 的固定拓扑比较或 SMC-FJ | 不 snapshot 派生 bottom；使用 `set_dip_profile()` | 启用候选期加密时需要 | 准备期 `remap=True`，候选期 `remap=False` |

## 最小固定几何代码

下面的 controls 虽然沿用 `sampled_controls` 这个接口名称，但零扰动向量使所有控制点保持
参考倾角；本次运行不会采样它们。

```python
import numpy as np

fault.top = 0.0
fault.depth = 25.0
fault.trace(trace_lon, trace_lat, utm=False)
fault.set_top_coords_from_trace(
    sort_axis=0,
    sort_order="descend",
)

fault.set_dip_profile(
    sampled_controls=sampled_controls,
    fixed_controls=fixed_controls,
    interpolation_axis="arc_length",
    transition_zones=transition_zones,
    is_utm=False,
)

fault.perturb_dips_with_preset_params(
    perturbations=np.zeros(len(sampled_controls)),
    angle_unit="degrees",
    use_average_strike=False,
)

fault.generate_mesh(
    top_size=3.0,
    bottom_size=6.0,
    show=False,
    verbose=0,
)
fault.initializeslip(values="depth")
```

这里的零向量不是 MCMC 初值，也不是额外优化步骤；它只把冻结 profile 中的参考倾角解析到
当前 top，并据此生成一次 bottom。固定几何完成后即可创建 BLSE/VCE inversion。

## 按便利性选择一种加密方式

固定几何下有两种加密时机。二者使用相同的沿弧长、保留折点的数值内核，但状态所有者不同；
选择一种即可。新脚本优先使用方案 A，因为最终权威 top 可以在生成 bottom 前直接检查。

### 优先方案 A：事前重采样权威 trace

如果希望更密的地表迹线本身就是本次模型输入，应先重采样 trace，再建立 top 和 profile：

```python
fault.trace(trace_lon, trace_lat, utm=False)
fault.discretize_trace(every=2.0)
fault.set_top_coords_from_trace(
    discretized=True,
    sort_axis=0,
    sort_order="descend",
)
```

`every=2.0` 是目标间隔，单位 km。该路径会改变权威 top 的节点集合，所以应在声明 control、
transition 和 profile 之前完成。需要先裁剪、简化、平滑或统一方向时，参见
[断层迹线预处理工作流](../../workflows/02d_fault_trace_preprocessing.md)和
[地表迹线构建短例](../../examples/fault_trace_preprocessing.md)。完整 API 与命令行工具见
[断层迹线处理参考](../fault_trace_processing.md)。

### 便利方案 B：生成 bottom 时一次性加密

如果希望保留原始 trace 作为权威输入，只想让倾角插值、局部走向和 bottom 具有更密的求值
节点，可在固定几何的唯一一次物化中加入：

```python
fault.perturb_dips_with_preset_params(
    perturbations=np.zeros(len(sampled_controls)),
    angle_unit="degrees",
    discretization_interval=2.0,
    use_average_strike=False,
)
```

`discretization_interval` 必须是有限正数，单位 km。这会在当前物化过程中沿原折线插入节点，
并保证生成的 top 与 bottom 一一对齐。它适合保留稀疏权威 trace、只生成一次 bottom 的固定
几何；不要再为它增加 `snapshot()`、`set_densification()` 或第二次零扰动。

如果同一加密规则必须在每个 Bayesian 候选中重复执行，则不要依赖这段一次性写法，应改用
冻结 reference 上的 `set_densification(...)`，并遵守
[候选期加密生命周期](bayesian_mixed.md#稀疏-top-与候选加密)。

### 不要这样混用

```python
# 错误：同一个 candidate 存在两个加密所有者。
fault.set_densification(interval=2.0)
fault.perturb_dips_with_preset_params(
    perturbations=np.zeros(len(sampled_controls)),
    discretization_interval=2.0,
)
```

即使两个值相同，也不能依赖“第二次通常不再加点”这一数值偶然性。候选路径必须只有一个
可追溯的边界分辨率来源。

## 选择 bottom 的走向来源

走向策略属于 bottom 生成器，不属于 profile 定义，也不是 mesh 参数。固定模式和扰动模式
使用完全相同的三种选择。

### 曲线 top 跟随局部走向

```python
dip_generation = {
    "angle_unit": "degrees",
    "use_average_strike": False,
}
```

每个节点根据有序 top 的局部切向生成下倾方向。完整语义见
[top 局部走向](local_strike.md)。

### 用户给定统一参考走向

```python
reference_strike = 280.0

dip_generation = {
    "angle_unit": "degrees",
    "use_average_strike": True,
    "average_strike_source": "user",
    "user_direction_angle": reference_strike,
}

fault.perturb_dips_with_preset_params(
    np.zeros(len(sampled_controls)),
    **dip_generation,
)
```

`user_direction_angle` 是从北顺时针量取的 **strike**，并且必须与有序 top 的正方向一致；
它不是 `dip_direction`。对正倾角，右手侧下倾方位通常为

```text
dip_direction = (reference_strike + 90°) mod 360°
```

也可以把 `average_strike_source` 设为 `"pca"`，让程序从 top 主轴计算一个统一走向。两种
代表性走向的适用边界见[单一代表性走向](representative_strike.md)。

固定拓扑或 SMC-FJ 中，零扰动参考和所有候选必须使用同一份走向策略。不要只在参考 mesh
使用统一走向、却让候选改回局部走向。

## 建模后检查

1. 检查 top 点序和参考走向方向是否一致。
2. 绘制 top、bottom 及连接线，确认下倾侧符合预期。
3. 检查 top/bottom 节点数一致，mesh 无翻折或退化单元。
4. 使用实际 patch 的 `getpatchgeometry()` 检查最终 strike/dip，不用构造阶段元数据代替成品。
5. 固定几何改变后重新创建 inversion；不要在既有 BLSE/VCE 对象中替换 mesh 后继续复用旧
   Green's functions、Laplacian 或约束。

## 何时不用本页

- 要比较多个几何并保持 patch 一一对应：使用
  [BLSE 固定拓扑倾角搜索](../../workflows/04b_blse_dip_search.md)。
- 要由 SMC-FJ 采样 sampled controls：使用
  [sampled/fixed/transition 组合](bayesian_mixed.md)。
- 倾角随深度而不是沿走向变化：使用
  [倾角随深度变化](../fault_geometry_construction.md#layered-dip)。
