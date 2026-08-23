# BLSE 倾角 × 平滑参数敏感性分析

当单独平滑搜索和单独倾角搜索都已经完成，但倾角优选结果可能依赖平滑强度时，再运行
二维敏感性分析。独立模板是
[`scripts/test_dip_smoothing_search_BLSE.py`](https://github.com/kefuhe/eqtools/blob/main/scripts/test_dip_smoothing_search_BLSE.py)。

这不是新手默认步骤，也不是 Bayesian 联合后验。它是在固定数据、迹线、网格拓扑、
bounds、rake 和数据权重条件下，对一组 `dip × penalty weight` 组合逐一求解 BLSE。

## 推荐的实际顺序

更稳妥的科研顺序是：

1. 在参考倾角下先扫描平滑参数，查看 L-curve 和滑动分布；
2. 选一个或一小段合理平滑范围；
3. 在固定平滑参数下搜索倾角；
4. 如发现倾角与平滑明显耦合，再对局部范围做二维网格；
5. 最终结合 RMS、逐轨道残差、roughness、滑动分布和几何合理性选模型。

对应模板依次是：

```text
test_smoothing_search_BLSE.py
          ↓
test_dip_search_BLSE.py
          ↓ 仅在需要时
test_dip_smoothing_search_BLSE.py
```

## 设置候选范围

模板保留既有 BLSE loop 的宽 penalty 范围，方便首次诊断：

```python
penalty_weight_candidates = [
    1.0, 5.0, 10.0, 30.0, 50.0, 80.0, 100.0, 125.0,
    150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0,
    800.0, 1000.0,
]

mainfault_reference_dip = 65.0
mainfault_dips = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0]
```

但正式二维分析通常应缩小到前两步得到的局部范围。例如：

```python
penalty_weight_candidates = [50.0, 80.0, 100.0, 125.0, 150.0]
mainfault_dips = [55.0, 60.0, 65.0, 70.0]
```

脚本启动时会打印总求解次数。`8 dips × 17 penalties` 表示 136 次 BLSE；在更换为多断层
组合前应先估算规模。

## 为什么不是每个组合都重建 GF

不同倾角会改变物理 patch 坐标、GF、Laplacian 和约束对应的模型，因此每个倾角都新建
一个 inversion。相同倾角下只有 penalty weight 改变，可以安全复用同一套 G、L 和约束：

```text
参考倾角 remap=True：建立一次 patch 拓扑

dip 1 remap=False
  -> 新建 inversion，重算 G/L/constraints
  -> penalty 1, penalty 2, ...：重复求解

dip 2 remap=False
  -> 新建 inversion，重算 G/L/constraints
  -> penalty 1, penalty 2, ...：重复求解
```

这既避免不同倾角间残留旧 GF，也避免在同一倾角下重复进行昂贵的 GF 计算。所有候选
保持同一 patch 身份；若 patch 数变化，模板会立即报错。

## 输出与解释

默认输出：

| 文件 | 内容 |
| --- | --- |
| `dip_smoothing_search_results/dip_smoothing_search.csv` | 每个 dip–penalty 组合的实际权重、roughness、全局和逐数据集 RMS/VR |
| `dip_smoothing_search_results/dip_smoothing_grid.png` | RMS 和 VR 的二维网格 |
| `dip_smoothing_search_results/roughness_vs_rms_by_dip.png` | 每个倾角对应的 roughness–RMS 曲线 |

内层 penalty 循环每完成一次求解，就按
[循环统计公共骨架](../examples/script_templates.md#loop-statistics) 收集 rows，并同时附加
`dip_deg` 与 `penalty_weight`。这样结果表可保留全局和逐数据集统计，也允许用户继续加入
约束方案、mesh size 或数据组合等自己的实验维度，而不必改动 BLSE 求解接口。

模板故意不打印“全局最优模型”。如果在全部组合中直接最小化 RMS，结果通常会偏向最弱
平滑甚至过拟合。正确用法是判断：

- 合理平滑区间内，优选倾角是否稳定；
- L-curve 转折是否随倾角明显移动；
- 某个低 RMS 区域是否同时对应不合理粗糙滑动；
- 不同轨道是否支持相同倾角，而不是由单一数据集控制；
- 邻近组合的残差和主要滑动区是否连续变化。

原始 roughness 由当前几何的 Laplacian 计算。即使 patch 拓扑固定，物理 patch 面积和
Laplacian 仍会随倾角改变，因此二维图是敏感性诊断，不是具有统一概率含义的后验面。

## 多断层边界

公开模板默认只搜索一个断层的倾角，并给所有断层使用一个公共标量 penalty weight。
若同时搜索两个断层的倾角或分别搜索各断层平滑权重，维度会迅速增长：

```text
dip1 × dip2 × penalty1 × penalty2
```

这类设计应在具体案例脚本中显式组织，不加入默认模板。每个断层必须拥有独立参考 mesh，
并保存 `current_penalty_weight` 的解析结果，确认 `single/individual/grouped` 映射没有错位。

## 下一步

- 选定组合后，用普通 [BLSE/VCE workflow](04_linear_slip_blse_vce.md) 重新运行并导出完整结果。
- RMS/VR 和 poly 语义见 [Fit Statistics](../reference/fit_statistics.md)。
- 若需要几何概率分布而不是网格敏感性分析，使用
  [Bayesian 非线性几何反演](03_nonlinear_geometry_bayesian.md)。
