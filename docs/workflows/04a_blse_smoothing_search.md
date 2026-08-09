# BLSE 固定几何平滑参数搜索

当数据、断层几何、网格和约束已经检查通过，但平滑强度尚未确定时，在**同一个固定几何
模型**上扫描 penalty weight。独立模板是
[`scripts/test_smoothing_search_BLSE.py`](../../scripts/test_smoothing_search_BLSE.py)。

普通 BLSE、平滑搜索和倾角搜索是不同科研任务。不要为了减少脚本数量而把它们强制合并：

- 普通 BLSE 用于复现一个已选参数模型；
- 平滑搜索用于检查数据拟合与模型粗糙度的权衡；
- 倾角搜索用于在固定平滑强度下比较几何；
- 倾角 × 平滑搜索用于最后检查两者耦合。

## penalty weight 的含义

BLSE 目标中，penalty weight 控制 Laplacian 平滑项。数值越大，平滑约束越强。与
`alpha` 的关系是：

```text
penalty_weight = 1 / alpha
```

若配置使用 `alpha.log_scaled: true`，则 `alpha: -2` 表示：

```text
alpha = 10^-2 = 0.01
penalty_weight = 100
```

搜索模板直接使用 penalty weight，避免同时混入 `alpha`、`log10(alpha)` 和倒数三层
解释。运行时传入 penalty weight 会覆盖当前运行读取的 alpha 初始值，但不会修改 YAML。

## 复制并修改模板

把模板复制到线性反演目录，与 `default_config_BLSE.yml` 和 `bounds_config.yml` 放在一起：

```bash
python test_smoothing_search_BLSE.py
```

第一轮需要修改数据路径、固定断层几何和配置文件名。模板中的默认候选范围与既有
`test_slip_inv_BLSE.py --mode loop` 一致：

```python
penalty_weight_candidates = [
    1.0,
    5.0,
    10.0,
    30.0,
    50.0,
    80.0,
    100.0,
    125.0,
    150.0,
    200.0,
    250.0,
    300.0,
    400.0,
    500.0,
    600.0,
    800.0,
    1000.0,
]

preferred_penalty_weight = 100.0
```

`preferred_penalty_weight` 只在图上做参考标记，不代表程序自动选择的最优值。完成第一轮
宽范围扫描后，应把列表缩小到 L-curve 转折附近，再检查较密的局部范围。

主配置必须保持：

```yaml
alpha:
  enabled: true
```

若关闭 alpha，模板会明确报错，而不是运行一组实际相同的无平滑模型。数据权重、poly、
bounds 和 rake 在整个扫描中保持不变。

## 计算和输出

断层几何、GF、Laplacian 和约束只建立一次；每个 penalty weight 重新运行一次 BLSE，
并立即收集当前模型统计：

```text
固定 fault + geodata + config
  -> 建立一次 G、L、bounds 和 rake constraints
  -> penalty 1：run -> returnModel -> fit statistics
  -> penalty 2：run -> returnModel -> fit statistics
  -> ...
```

默认输出：

| 文件 | 内容 |
| --- | --- |
| `smoothing_search_results/smoothing_search.csv` | penalty、等价 log10(alpha)、解析后的实际权重、roughness、全局和逐数据集 RMS/VR |
| `smoothing_search_results/smoothing_search.png` | penalty–RMS、penalty–VR 和 roughness–RMS |

每个 penalty 的统计必须在该轮 `run()` 后立即收集。可直接复制
[循环统计公共骨架](../examples/script_templates.md#loop-statistics)，再把 `penalty_weight` 和
`equivalent_log10_alpha` 加入结果 row。`returnModel()` 返回的 roughness 是未乘 penalty 的
模型粗糙度，因此在固定几何和同一 Laplacian 下可用于比较不同 penalty；实际采用的权重以
`current_penalty_weight` 为准。

模板使用 `data_poly="config"`，因此拟合统计包含配置中实际求解的 offset/ramp。完整统计
定义见 [Fit Statistics](../reference/fit_statistics.md)。

## 如何选择

不要只取 RMS 最小值；减弱平滑通常就能降低 RMS。至少同时检查：

- roughness–RMS 的转折区；
- 各 InSAR 轨道或 GPS 数据的逐数据集 RMS/VR；
- 残差中是否仍有 ramp、周跳或局部系统结构；
- 滑动是否过度集中、出现棋盘格或被过度抹平；
- penalty 在转折点附近变化时，主要滑动区是否稳定。

选定 penalty 后，用普通 BLSE 模板重新运行并导出完整滑动和残差结果，不需要为全部候选
保存大批 GMT/PDF 文件。

## 下一步

- 已选好平滑强度，需要比较倾角：进入
  [BLSE 固定拓扑倾角搜索](04b_blse_dip_search.md)。
- 怀疑倾角选择随平滑强度明显变化：进入
  [倾角 × 平滑参数敏感性分析](04c_blse_dip_smoothing_search.md)。
- BLSE 配置、约束和结果解释：回到
  [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md)。
