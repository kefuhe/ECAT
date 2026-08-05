# 可运行脚本模板导航

`scripts/` 中的文件是可复制、可逐段修改的完整起始脚本；`docs/examples/` 的其他页面主要
提供短代码片段。第一次使用先按科研任务选择模板，再进入对应 workflow 理解输入、输出和
检查项，不需要从头阅读全部 reference。

## BLSE 模板怎么选

| 当前任务 | 模板 | 先读 |
| --- | --- | --- |
| 用已经选定的几何和平滑强度运行一次 BLSE | [`test_slip_inv_BLSE.py`](../../scripts/test_slip_inv_BLSE.py) 的 `--mode single` | [BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md) |
| 固定几何，只搜索平滑强度 | [`test_smoothing_search_BLSE.py`](../../scripts/test_smoothing_search_BLSE.py) | [固定几何平滑参数搜索](../workflows/04a_blse_smoothing_search.md) |
| 固定平滑强度，只搜索倾角 | [`test_dip_search_BLSE.py`](../../scripts/test_dip_search_BLSE.py) | [固定拓扑倾角搜索](../workflows/04b_blse_dip_search.md) |
| 已完成前两项，需要检查倾角和平滑耦合 | [`test_dip_smoothing_search_BLSE.py`](../../scripts/test_dip_smoothing_search_BLSE.py) | [倾角 × 平滑参数敏感性分析](../workflows/04c_blse_dip_smoothing_search.md) |

既有 `test_slip_inv_BLSE.py --mode loop` 继续保留，适合已有案例脚本直接做传统
`simple_run_loop()`。新的 smoothing-only 模板提供更明确的候选列表、逐数据集统计和
独立输出，二者不互相替代。

## 非线性、正演和分辨率检查模板

| 当前任务 | 模板 | 先读 |
| --- | --- | --- |
| 新项目做 Bayesian 非线性几何搜索 | [`test_nonlinear_geometry_smc.py`](../../scripts/test_nonlinear_geometry_smc.py) | [Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md) |
| 原样复现 legacy `explorefault` 案例 | [`test_nonlinear_bayesian.py`](../../scripts/test_nonlinear_bayesian.py) | [非线性几何配置](../reference/config_nonlinear_geometry.md) |
| 在规则点或自定义点计算 ENU 正演 | [`test_surface_displacement_forward.py`](../../scripts/test_surface_displacement_forward.py) | [地表位移正演](surface_forward_grid.md) |
| 在 SAR 有效像元计算并输出 LOS | [`test_sar_los_surface_forward.py`](../../scripts/test_sar_los_surface_forward.py) | [地表位移正演参考](../reference/surface_displacement_forward.md) |
| 检查 BLSE 空间分辨率 | [`test_BLSE_Inv_Checkboard.py`](../../scripts/test_BLSE_Inv_Checkboard.py) | [BLSE/VCE 工作流](../workflows/04_linear_slip_blse_vce.md) |

降采样通常不需要复制 Python：先用 `ecat-generate-downsample` 生成 YAML，再运行 `ecat-downsample`。完整命令见 [InSAR 降采样](../workflows/02_insar_downsampling.md)。`scripts/process_data_downsampling.py` 只是在源码树中调用同一 CLI 的薄入口。

旧 CovDiag checkerboard 变体仍列在 [`scripts/README.md`](../../scripts/README.md)，用于复现已有项目；新用户先从上表的单一 checkerboard 入口开始。

## 推荐学习顺序

```text
普通 BLSE 单次运行
  -> 确认数据、固定几何、bounds、rake、poly 和输出链条

固定几何平滑搜索
  -> 选择合理 penalty 范围

固定平滑倾角搜索
  -> 比较几何候选

倾角 × 平滑联合敏感性
  -> 仅在前两者显示明显耦合时运行
```

## 复制后先改哪些位置

模板采用一致的注释分块，优先修改：

1. `lon0/lat0` 和数据文件路径；
2. `geodata` 顺序；
3. `fault_name`、trace、top/depth、dip direction 和 mesh size；
4. `default_config_BLSE.yml`、`bounds_config.yml` 路径；
5. 当前任务对应的候选列表；
6. 输出目录。

`fault_name` 必须匹配配置 source 名，`geodata` 必须匹配配置数据顺序。模板中的文件路径
只是占位符，不能不检查就用于正式案例。

<a id="loop-statistics"></a>

## 循环中怎样获取统计信息

无论循环变量是倾角、平滑权重还是约束方案，都应在每次 `run()` 完成后、进入下一轮前
立即收集该轮统计。下面是可直接放进循环体的公共骨架：

```python
inversion.run(
    penalty_weight=penalty_weight,
    alpha=None,
    verbose=False,
)

roughness, solver_rms, solver_vr = inversion.returnModel(print_stat=False)

fit_rows = inversion.collect_fit_statistics(
    model=f"penalty_{penalty_weight:g}",
    data_poly="config",
    include_dataset=True,
    include_global=True,
)
global_fit = next(
    row for row in fit_rows if row["scope"] == "global_solver_vector"
)
dataset_rows = [row for row in fit_rows if row["scope"] == "dataset"]
fit_df = inversion.fit_statistics_to_dataframe(fit_rows)
```

上面的 `run()` 参数展示直接传入平滑权重的场景；倾角、mesh 或约束循环仍按各自 workflow
更新对象和配置，后续三行统计接口保持不变。

三个接口各司其职：

- `returnModel()` 返回当前模型的 roughness 以及组装后 solver vector 的 RMS/VR；
- `collect_fit_statistics()` 按 `data_poly="config"` 重建配置对应的 synthetic，并返回逐数据集
  rows 和独立计算的 `global_solver_vector` row；
- `fit_statistics_to_dataframe()` 只把已经获得的 rows 转成表格，不重新求解或重建模型。

公开搜索模板为便于直接阅读，默认把一个候选写成一行、把各数据集统计展开为列。模板不会
限制用户只能搜索某几个参数；当实验维度或数据集会变化时，可把循环变量附加到每个
statistics row，改用更便于扩展的长表：

```python
all_rows.extend(
    {
        "dip_deg": dip_deg,
        "penalty_weight": penalty_weight,
        "constraint_case": constraint_case,
        **row,
    }
    for row in fit_rows
)
```

没有参与当前实验的字段可以删掉，也可以加入 mesh size、数据组合或其他方案标识；不要改动
求解后立即采集统计这一顺序。下一轮 `run()` 会覆盖当前 `mpost` 和 penalty 状态，因此不要
等循环结束后再补取前面各轮统计，也不要把逐数据集 RMS/VR 的算术平均当作全局拟合。
字段、公式和 scope 的完整定义见 [Fit Statistics](../reference/fit_statistics.md)，penalty 的
解析语义见 [BLSE/VCE Reference](../reference/blse_vce.md#结构化拟合统计)。

## 文档层级怎么配合

| 层级 | 负责回答 |
| --- | --- |
| 本页 | 当前任务应该复制哪个脚本 |
| `scripts/README.md` | 仓库中有哪些公开脚本入口 |
| workflow | 输入、执行顺序、输出、科学检查和下一步 |
| example | 某个小任务的短代码怎么写 |
| reference | 类、方法、配置字段和完整接口语义 |
| casebook | 公开真实案例如何组织脚本、数据和结果 |

这样模板可以按场景独立演进，而公共计算和配置语义仍由 BLSE/reference 统一说明，避免在
每个脚本页面复制同一套长参数表。
