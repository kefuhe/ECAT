# 断层几何状态

ECAT/eqtools 中同一个断层几何会经历多个状态。理解这些状态，可以避免把 trace、top edge、bottom edge、mesh 和 patch GMT 混用。

## 常见状态

| 状态 | 典型字段或入口 | 含义 |
| --- | --- | --- |
| 地表 trace | `fault.trace(...)`, `xf/yf`, `lon/lat` | 二维地表迹线，不表示断层面。 |
| 离散 trace | `fault.discretize_trace(...)`, `xi/yi` | 沿 trace 的采样点，常用于生成顶部边界或检查间距。 |
| 顶部边界 | `top_coords`, `set_top_coords_from_trace(...)` | 三维 top edge，通常深度为 `top`。 |
| 底部边界 | `bottom_coords` | 三维 bottom edge，和 top edge 一起定义断层面宽度。 |
| 多层边界 | `layers`, `generate_layer_coords(...)` | 倾角随深度变化或 slab 几何中的中间等深线。 |
| mesh / patches | `Vertices/Faces`, `patch`, `generate_mesh(...)` | 可用于 Green's functions 和滑动反演的离散面元。 |
| CSI patch GMT | `writePatches2File(...)`, `readPatchesFromFile(...)` | 每个 GMT segment 表示一个矩形或三角 patch。 |

## 推荐操作顺序

简单地表 trace 加固定倾角：

```text
trace -> top_coords -> fixed dip + explicit dip direction -> bottom_coords -> mesh
```

地表 trace 加沿走向变化倾角：

```text
trace -> top_coords -> dip control points -> per-node strike/dip -> bottom_coords -> mesh
```

非线性几何结果：

```text
clon/clat/cdepth/strike/dip/length -> top/bottom -> mesh -> BLSE/VCE
```

多条等深线或 slab：

```text
contours/layers -> FaultGeometryEngine -> rectangular or triangular fault
```

已有 patch GMT：

```text
readPatchesFromFile -> fault summary -> edge/contour checks -> inversion or forward modeling
```

## 常见误区

- 不要直接在经纬度上做 trace 长度、延伸和重采样；先转成局部 `x/y` km。
- strike 和 dip direction 都是从北顺时针增加的地理方位角；右手规则下通常有 `dip_direction = strike + 90°`。
- trace/top 点序定义 strike 正点序。多倾角模式可使用一个显式/PCA 代表性 strike，也可使用
  top edge 自动局部 strike，或插值四列 `xydip` 中的控制点 strike；正倾角始终向最终采用的
  strike 右手侧下倾。
- 沿走向变化倾角和随深度变化倾角是不同问题；后者需要 layered-dip 几何。
- `fault.discretize(...)` 是 CSI legacy trace 离散接口；新项目优先使用 `fault.discretize_trace(...)`。
- 普通 polyline GMT 不等于 CSI patch GMT；只有后者能直接表示 fault patches。
- top/bottom 分别按距离重采样可能导致点数不一致；需要配对建 mesh 时，优先使用相同 `num_segments`。

## 继续阅读

- [地表迹线和倾角构建示例](../examples/fault_trace_preprocessing.md)
- [非线性几何结果到 fault object](../examples/fault_from_nonlinear_geometry.md)
- [Fault Geometry Construction](../reference/fault_geometry_construction.md)
- [Fault Summary](../reference/fault_summary.md)
- [Fault Edges](../reference/fault_edges.md)
- [Fault Contours](../reference/fault_contours.md)
