# 模式二：单一代表性走向

这个模式让所有顶部节点使用同一个 strike。它适合近直线或走向变化不大的断层，不适合用来
强制弯曲断层贴合每个局部切向。

## 两种来源

| 来源 | 设置 | 含义 |
| --- | --- | --- |
| 用户给定 | `average_strike_source="user"` | 使用明确的地理走向角 |
| PCA | `average_strike_source="pca"` | 由 top x/y 第一主轴得到代表性走向 |

PCA 主轴有 180° 二义性；实现用 `top[0] -> top[-1]` 定向。用户走向同样必须与这个正方向
同向，反向或垂直会报错。

## 推荐的显式设置

```python
fault.interpolate_top_dip_from_relocated_profile(
    dip_points,
    is_utm=False,
    interpolation_axis="auto",
)

fault.generate_bottom_from_segmented_relocated_dips(
    fault_depth=fault.depth,
    use_average_strike=True,
    average_strike_source="user",
    user_direction_angle=65.0,
    verbose=True,
)
```

`user_direction_angle` 是 strike，不是 dip direction；它按北为 `0°`、顺时针为正解释。
`65°`、`425°` 和 `-295°` 在三角函数意义下等价，但新脚本推荐写成 `[0°, 360°)`，便于
审阅。正倾角的下倾方位为 `strike + 90°`。

Bayesian preset 使用相同生成器选项：

```yaml
update_fault_geometry:
  method: perturb_dips_with_preset_params
  angle_unit: degrees
  use_average_strike: true
  average_strike_source: user
  user_direction_angle: 65.0
```

无需人工给角度时可以选择 PCA：

```python
import numpy as np

fault.perturb_dips_with_preset_params(
    perturbations=np.zeros(n_sampled_controls),
    use_average_strike=True,
    average_strike_source="pca",
)
```

## 与局部 top 的关系

统一走向会覆盖底边计算中的逐节点局部 strike。因此强弯曲 top 上，bottom 的下倾连接方向
可以明显偏离某些局部 top 的右手法向；这是该模式的定义，不是角度范围转换错误。

当前 `top_strike` 应理解为构造阶段的逐节点参考元数据。统一走向是在底边生成器内部应用的；
检查实际成品时，应结合 `verbose=True` 输出的代表性走向、top/bottom 坐标以及最终
`getpatchgeometry()`，不要仅凭 `top_strike` 判断最终采用的全局方向。

## 使用边界

- top 强弯曲：优先[局部走向](local_strike.md)。
- top 首末点几乎重合或几何近圆形：PCA 定向可能缺少稳定物理意义，应改为用户显式走向或
  对断层分段。
- 用户走向只保证与整体首末方向同向，不保证贴近每个局部切向。
- 若只是点序反了，应修正 top 点序；不要用 `strike + 180°` 掩盖点序问题。
