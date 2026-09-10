# 模式三：控制点走向插值

当已有分段地质走向或独立走向约束时，可以在普通非分层构模入口中同时提供 strike 和 dip
控制点。该模式当前不属于 Bayesian `DipProfileSpec` 的字段。

## 输入与圆周插值

四列数组顺序是：

```text
[lon, lat, strike, dip]
# 或 is_utm=True 时
[x_km, y_km, strike, dip]
```

```python
import numpy as np

strike_dip_points = np.array([
    [96.00, 21.00, 350.0, 55.0],
    [96.20, 21.10,   0.0, 65.0],
    [96.45, 21.20,  10.0, 72.0],
])

controlled = fault.interpolate_top_dip_from_relocated_profile(
    strike_dip_points,
    is_utm=False,
    interpolation_axis="auto",
)
fault.generate_bottom_from_segmented_relocated_dips(
    fault_depth=fault.depth,
    use_average_strike=False,
)
```

实现先把 strike 转成弧度并展开，再执行一维插值，最后取模到 `[0°, 360°)`。因此：

```text
350° -> 0° -> 10°
```

沿短路径跨过 `0°`，不会错误经过 `180°`。`-10°` 与 `350°`、`370°` 与 `10°` 等价。

## 方向校验

插值后的每个 strike 都与有序 top 的局部正走向比较：

\[
a_i=\cos(\theta_i^{\mathrm{control}}-\theta_i^{\mathrm{top}}).
\]

只有 \(a_i>0\) 才接受，也就是位于同一方向半平面。反向或垂直会报错。这个检查保证“没有
把正走向翻转”，但不会要求控制走向紧贴 top；相差接近 90° 但仍同向的输入可能产生明显
不同的底边方向，必须由用户和诊断图确认其地质含义。

## 容器形式

- NumPy：四列，列序固定。
- DataFrame：需要坐标列、`strike`、`dip`；可以保留其他说明列。
- CSV：首行写同名表头。

```csv
lon,lat,strike,dip
96.00,21.00,350.0,55.0
96.20,21.10,0.0,65.0
96.45,21.20,10.0,72.0
```

## 不要混用的情况

- `use_average_strike=True` 会在最终底边计算中覆盖插值后的逐点 strike。
- 输入列是 strike，不是 dip direction；两者相差 90°。
- 不要同时把 strike 加 180° 又把 dip 改为负值，否则倾向侧会翻转两次。
- Bayesian `set_dip_profile()` 接收位置和 dip；位置可由二维坐标或 reference-top 里程声明，
  但仍不接收 strike。若未来需要采样 strike controls，应先建立独立的参数布局、圆周扰动
  和诊断契约，不能把第四列静默塞入当前 profile。
