# BLSE/VCE 参考

本文说明线性滑动分布反演中 `BoundLSEMultiFaultsInversion` 的常用运行模式、参数关系和结果检查。完整工作流见 [BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md)，可运行案例见 [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md)。

## 数据流

```text
geodata + faults_list
  -> default_config.yml + bounds_config.yml
  -> 组装 G、d、poly、Laplacian 和约束矩阵
  -> BLSE / smoothing loop / VCE
  -> returnModel
  -> 滑动图、data/synth/resid、统计量
```

入口类：

```python
from eqtools.csiExtend.blse_multifaults_inversion import BoundLSEMultiFaultsInversion
```

最小结构：

```python
inv = BoundLSEMultiFaultsInversion(
    "linear_slip",
    faults_list,
    geodata,
    config="default_config.yml",
    bounds_config="bounds_config.yml",
)
```

配置字段见 [线性滑动反演配置](config_linear_slip.md)，约束模式见 [ECAT 约束管理器](constraint_manager.md)。

## 三种运行模式

| 模式 | 方法 | 主要用途 | 常见输出 |
| --- | --- | --- | --- |
| 固定权重 BLSE | `run(...)` | 复现一个指定平滑权重的模型 | 滑动模型、拟合图、统计量 |
| Smoothing loop | `simple_run_loop(...)` | 扫描平滑权重，查看 RMS 与粗糙度权衡 | `run_loop.dat`, `Roughness_vs_RMS.png` |
| VCE | `run_simple_vce(...)` | 迭代估计数据和正则化方差分量 | VCE 结果字典、最终权重、收敛信息 |

入门建议先跑固定权重 BLSE，再用 smoothing loop 或 VCE 做权重诊断。

## 固定权重 BLSE

`run(...)` 的核心是求解带边界和线性约束的最小二乘问题：

```python
inv.run(alpha=[-2.0])
inv.returnModel(print_stat=True)
inv.extract_and_plot_blse_results(plot_faults=True, plot_data=True)
```

如果 `alpha.log_scaled: true`，`alpha=[-2.0]` 表示实际 `alpha = 10 ** -2 = 0.01`。线性求解中的惩罚权重约为：

```text
penalty_weight = 1 / alpha
```

因此上例等价于：

```python
inv.run(penalty_weight=[100.0])
```

`alpha` 和 `penalty_weight` 不要同时传入。`sigma` 和 `data_weight` 也不要同时传入；二者也是倒数关系。

常用参数：

| 参数 | 含义 |
| --- | --- |
| `alpha` | 平滑尺度；是否按 `log10` 解释由 `penalty_log_scaled` 或配置中的 `alpha.log_scaled` 控制 |
| `penalty_weight` | 直接传入求解器的平滑惩罚权重 |
| `sigma` | 数据标准差；是否按 `log10` 解释由 `data_log_scaled` 或配置中的 `geodata.sigmas.log_scaled` 控制 |
| `data_weight` | 直接传入的数据权重 |
| `smoothing_constraints` | 可按断层传入 `(top, bottom, left, right)` 平滑边界 |
| `des_enabled` | 运行期覆盖 DES 开关 |

## Smoothing Loop

`simple_run_loop(...)` 用一组 `penalty_weight` 逐次运行 BLSE，并保存 RMS、粗糙度和 variance reduction：

```python
penalties = [1, 3, 10, 30, 100, 300]
df = inv.simple_run_loop(
    penalties,
    preferred_penalty_weight=30,
    output_file="run_loop_covdiag.dat",
    rms_unit="cm",
)
```

函数会把每个 penalty 转为 `alpha = log10(1 / penalty)` 后调用 `run(...)`。因此 loop 表格中的 `Penalty_weight` 是惩罚权重，不是 `alpha` 本身。

推荐检查：

- `Roughness_vs_RMS.png` 的拐点是否稳定。
- 选定 penalty 后的残差是否出现轨道系统误差。
- 选定 penalty 的滑动分布是否被过度平滑或出现不合理尖峰。
- 最终报告中保留 loop 表格，不只保留最终滑动图。

## VCE

VCE 是 variance component estimation，用于估计数据项和正则化项的相对权重。适合多数据集联合反演，尤其是不同 InSAR 轨道、GPS 与 InSAR 权重不容易手工确定时。

典型调用：

```python
vce_result = inv.run_simple_vce(max_iter=20, tol=1e-4)
inv.returnModel(print_stat=True)
inv.extract_and_plot_blse_results(plot_faults=True, plot_data=True)
```

返回结果通常包含：

| 键 | 含义 |
| --- | --- |
| `m` | 最终线性参数 |
| `var_d` | 数据方差分量 |
| `var_alpha` | 正则化方差分量 |
| `weights` | 最终权重比例 |
| `converged` | 是否收敛 |
| `iterations` | 迭代次数 |

VCE 可从配置读取 `geodata.sigmas` 和 `alpha` 的 `mode/update/initial_value`，也可以在调用时传入 `sigma_mode`、`sigma_groups`、`smooth_mode`、`smooth_groups` 等参数覆盖。分组组织方式见 [Sigmas 与 Alpha 配置模式](sigmas_alpha.md)。

## 结果导出

常用出口：

```python
inv.returnModel(print_stat=True)
inv.extract_and_plot_blse_results(
    plot_faults=True,
    plot_data=True,
    data_poly=None,
)
```

`extract_and_plot_blse_results(...)` 通常会生成：

- `output/*_slip.*` 类型的断层滑动图。
- `Modeling/<DataName>_fit_comparison.pdf`。
- GPS、InSAR 或其他数据类型的 data/synth/resid 文件。
- 控制台中的拟合统计和断层统计。

案例脚本也可额外调用断层和数据对象的方法，例如写出 `slip_<FaultName>.gmt`、`slipdir_<FaultName>.txt`、每条 InSAR 的 `data/synth/resid` 文本文件。Dingri 案例的对应代码见 [脚本对照：导出滑动、滑动方向和模型数据](../casebook/dingri_blse_vce.md#6-导出滑动滑动方向和模型数据)。

## 约束检查

BLSE/VCE 支持的约束主要包括：

- `strikeslip/dipslip` 上下界。
- `rake_angle` 转换得到的线性角度约束。
- Euler 线性约束。
- `strikeslip == 0`、`dipslip == 0` 这类零滑等式约束。
- `zero_edge_slip(...)` 这类边界零滑等式约束。
- 用户通过 `source_constraints` 添加的线性等式/不等式约束。

这些约束由统一约束管理器应用。固定权重 BLSE、smoothing loop 和 VCE 共用同一套约束矩阵；VCE 只估计权重，不改变约束写法。`rake_angle` 在线性 BLSE/VCE 中不是待求滑动参数，而是限制 `strikeslip/dipslip` 的角度范围；零滑和边界零滑的配置细节见 [ECAT 约束管理器](constraint_manager.md#零滑与边界零滑约束)。

## 推荐报告内容

每个 BLSE/VCE 案例建议报告：

- 数据类型、数据集名称和观测数量。
- 读取格式与协方差处理方式。
- 断层几何来源、网格尺寸和 `top/depth` 设置。
- GF 方法和 Laplacian 方法。
- bounds、rake、Euler、零滑、边界零滑或自定义线性约束。
- sigma/alpha 模式和最终权重。
- smoothing loop 或 VCE 诊断结果。
- 每个数据集的 RMS、normalized RMS 或 variance reduction。
- 地震矩、Mw 和主要滑动区。
- 滑动模型、滑动方向和 data/synth/resid 输出路径。

## 常见问题

- 若结果完全不平滑，检查 `alpha.enabled`、`update_Laplacian` 和 `penalty_weight` 是否被意外关闭或设得过小。
- 若某条 InSAR 轨道残差呈大尺度 ramp，检查 `geodata.polys` 和输出时的 `data_poly` 设置。
- 若 rake 约束没有效果，检查 `use_rake_angle_constraints: true`、`bounds_config.yml` 中断层名是否匹配、滑动参数化是否为 `ss_ds`。
- 若 VCE 权重异常，先用固定权重 BLSE 和 smoothing loop 检查数据、协方差和边界是否合理。
- 若 `alpha` 与预期相反，确认当前传入的是 `alpha` 还是 `penalty_weight`，以及 `log_scaled` 是否为 `true`。
