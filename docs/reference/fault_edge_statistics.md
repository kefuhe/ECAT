# 沿断层边界的统计展示

`StatisticsInFault` 汇总沿 top edge 的滑动或事件数，并输出 GMT 三维柱状图、曲线或
短线。统计值由输入、分箱、聚合和归一化决定；本页的方向选项仅控制三维展示坐标。

## 连续展示的最短调用

```python
from eqtools.csiExtend.statUtils.statisticsInFaultEdge import StatisticsInFault

stats = StatisticsInFault("statistics", source=fault, lon0=lon0, lat0=lat0,
                          outdir="statistics")
stats.StatinTop(
    slip="total_all", method="mean", cutmethod="pdcut", hinterval=2.0,
    xtick_angle_offset=30.0, zoffset=1.0, height_scale=7.0,
    strike_alignment="auto", outfile="slip_top.gmt",
)
```

先自行创建输出目录。`auto` 用有序 top 的局部切向选择每段走向的正向分支，保留局部
走向变化，并默认让相邻柱共享边界方向。近直线断层也可用 `strike_alignment="pca"`。
用户显式选择全局参考分支时使用：

```python
stats.StatinTop(
    value=event_weights, lonlat=event_lonlat, method="sum",
    strike_alignment="reference", reference_strike=85.0,
    boundary_frame="shared", outfile="events_top.gmt",
)
```

`lonlat` 是 `(n, 2)` 经度、纬度；事件计数的 `value` 可设为全 1。
这些事件按平面位置投到 top 迹线，不等同于先投到断层面后再统计。

## 方向参数与兼容性

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `strike_alignment` | `"legacy"` | 保持原有 canonical 角度与箱中点 x 查找，复现旧输出 |
| 同上 | `"auto"` | 按有序 top 的逐段切向选择正向分支 |
| 同上 | `"pca"` | 按 top x/y 的 PCA 主轴选择正向分支；用首末方向固定 PCA 符号 |
| 同上 | `"reference"` | 按 `reference_strike` 选择正向分支 |
| `reference_strike` | `None` | 仅 reference 模式接受；从北顺时针的地理方位，单位度 |
| `boundary_frame` | `"auto"` | legacy 使用 bin；其他模式使用 shared |
| 同上 | `"bin"` | 每根柱使用一个箱中点方向；不保证相邻接缝连续 |
| 同上 | `"shared"` | 只用于非 legacy 模式；相邻柱在共用边界使用同一方向 |

新增参数均为 keyword-only，原有位置参数顺序不变。未设置新选项时结果仍走 legacy。
推荐新图显式使用 `auto`，无需重新反演或改变断层网格。`curve`、`bar` 与
`hist` 使用同一套角度协议；`doStat=False` 保留原始平面输入坐标。

## 跨倾向时为什么要同时转换两个角度

对于反向于参考方向的局部走向，展示阶段使用：

```text
strike' = (strike + 180°) mod 360°
dip'    = 180° - dip
```

转换前后的全球下倾向量相同。倾角在 `[0, 180]` 内连续跨越直立位置，局部走向只选择
180° 分支，不会被替换为一个统一的参考方位。先做该转换，再加 `xtick_angle_offset`。
参考走向与某段局部走向垂直时分支不确定，程序报错；PCA 缺少稳定首末定向时应使用显式
参考或局部 auto。强弯曲断层优先 auto，不要用单一主轴强行选择全部分支。

shared 模式以每段上缘拥有的三角形定义局部角度，沿 top 累计弧长插值；走向先展开
360° 周期。左右柱对同一个边界点求得相同展示方向，沿同一条侧边延伸到各自高度。
高度差仍保留为台阶。这里没有平滑统计值、修改 mesh、替换 slip/rake 基底或重算
Green's functions。不要把有向展示角度再折回 canonical，否则倾向处的走向翻转会重现。

`zoffset` 是沿展示方向的距离，包含水平和竖向分量，单位 km；`height_scale` 是最大
归一化统计量对应的展示长度，单位 km；`xtick_angle_offset` 单位度。非零偏转在统一参考
框架中解释，因此选择相反参考分支可以改变展示朝向；它不能保证旧图两侧的位置都不变。

## 统计与输出的边界

- `total_all + mean` 表示全部面片滑动模长的分箱均值，不是累积或面积加权滑动。
- 默认以每组统计量的最大值归一化；需要固定比较尺度时使用 `norm_ref`。
- `cutmethod="pdcut"` 仍从该组投影距离最小值开始，以 `hinterval` 分箱；默认端点包含
  规则及尾段覆盖沿用旧逻辑。不同数据组不保证共用分箱边界。本次方向修复不改变计数。
- `cutmethod="hist"` 是计数直方图，不能用于求滑动均值。
- 三维 `.gmt` 与对应 `_2d.gmt` 保持现有格式和三位小数精度；方向模式不改变二维统计
  文件。严格连续指相邻共享底边及侧边方向，不要求不同高度的顶部端点重合。

相关约定：[断层角度](../concepts/fault_angle_conventions.md)、
[代表性走向](dip_profile/representative_strike.md)、[边界所有权](fault_edges.md)、
[地震投影](seismic_projection.md)。
