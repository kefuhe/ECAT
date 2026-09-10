# SMC 温度调度

ECAT 的 FULLSMC、SMC-FJ、旧版非线性入口和新版 nonlinear geometry SMC
使用同一套基于增量权重变异系数（coefficient of variation, COV）的温度调度。
本页说明公开配置、数学含义和使用边界；它不改变 prior、likelihood 或最终目标后验。

## 配置

```yaml
smc_tempering:
  target_cov: 1.0
  max_delta_beta: 0.5
```

省略整个 `smc_tempering` 段与显式写出以上默认值完全等价。

| 字段 | 默认 | 允许范围 | 含义 |
| --- | ---: | ---: | --- |
| `target_cov` | `1.0` | 有限正数 | 增量重要性权重的目标 COV；越大通常允许越大的温度步长，也会降低重采样前的有效样本量 |
| `max_delta_beta` | `0.5` | `0 < value <= 1` | 单个 stage 的最大温度增量；通常保持默认 |

`tolerance=1e-6` 和 `ddof=1` 是内部数值与统计约定，不属于公开配置。配置中写入
这两个字段会明确报错，而不是静默改变调度语义。

## 数学含义

当前粒子表示温度为 \(\beta_k\) 的过渡分布。对候选温度 \(\beta\)，增量权重为：

\[
w_i(\beta)=\exp\left[(\beta-\beta_k)
\left(\ell_i-\max_j\ell_j\right)\right],
\]

其中 \(\ell_i\) 是第 \(i\) 个粒子的 log likelihood。减去共同最大值只用于数值
稳定，不改变归一化权重。调度器寻找满足下式的 COV 控制步长：

\[
\operatorname{COV}(w)=\frac{s_w}{\bar w}
\approx \texttt{target_cov}.
\]

若 COV 单独允许的增量记为 \(\Delta\beta_{\rm COV}\)，实际增量受三项共同限制：

\[
\Delta\beta_k=\min\left(
\Delta\beta_{\rm COV},
\texttt{max_delta_beta},
1-\beta_k
\right).
\]

实现会先检查最大允许候选；若其 COV 已不超过目标，直接接受该候选，否则在固定的
旧温度 \(\beta_k\) 与候选上界之间二分求根。mutation acceptance 只影响提议尺度，
不会在运行中回写 `target_cov` 或 `max_delta_beta`。

## 与有效样本量的关系

活动路径使用样本标准差 `ddof=1`。当权重恰好达到目标 COV 时：

\[
\operatorname{ESS}
=\frac{N^2}{N+(N-1)\operatorname{COV}(w)^2}.
\]

对 \(N=100\) 个粒子，下面的值可帮助理解调度强度；它们不是质量保证：

| `target_cov` | 阈值处 ESS 约值 | 建议定位 |
| ---: | ---: | --- |
| `1.0` | `50.3` | 标准默认 |
| `1.25` | `39.3` | 温和实验 |
| `1.5` | `31.0` | 较激进，需要敏感性对照 |
| `2.0` | `20.2` | 高级实验，注意祖先粒子丢失 |

提高 `target_cov` 可能减少 stage，但会允许更不均匀的权重；mutation 需要跨越的分布
差异也可能增大，因此总耗时不保证下降。

## `max_delta_beta` 与短尾 stage

默认 `max_delta_beta: 0.5` 是历史保守上限。若某一步 \(\beta_k=0.482666\)，即使
COV 允许直达 1，默认上限也只能先到 `0.982666`，随后留下 `0.017334` 的最后增量。

`max_delta_beta: 0.75` 允许调度器在 COV 仍安全时直接到 1，从而消除这种由硬上限造成的
短尾 stage；若直达 1 的 COV 超过目标，调度器仍会求出中间温度。它不会无条件合并
最后阶段。

| 值 | 使用解释 |
| ---: | --- |
| `0.25` | 更保守，通常增加 stage |
| `0.5` | 标准默认 |
| `0.75` | 高级调度试验，可能减少上限造成的短尾 stage |
| `1.0` | 不施加额外的半程上限，仍受 COV 和最终温度限制 |

到达 \(\beta=1\) 后的最终 mutation 仍然需要执行，使粒子在最终 posterior 上重新平衡。

## 可复现性与接续

运行头、阶段 checkpoint 和最终 HDF5 记录解析后的 `target_cov`、
`max_delta_beta`、内部 tolerance 和 `ddof`。从 \(\beta>0\) 的 checkpoint 接续时，
配置必须和保存的 policy 一致。旧文件没有 metadata 时，只能按历史默认 `1.0/0.5`
接续；非默认调度应重新开始或使用带完整 metadata 的 checkpoint。

非默认值不会改变最终目标后验的定义，但会改变温度序列、重采样次数、随机数消耗路径
和有限粒子下的 Monte Carlo 结果。比较调度参数时，应固定 prior、粒子数、链长、MPI
rank、随机设置和其他求解参数，并同时检查 beta 序列、ESS、接受率、后验摘要和重复运行
稳定性。

## 相关页面

- [非线性几何反演配置](config_nonlinear_geometry.md)
- [Bayesian 联合反演](bayesian_joint_inversion.md)
- [Bayesian 非线性几何反演工作流](../workflows/03_nonlinear_geometry_bayesian.md)
- [联合 Bayesian 工作流](../workflows/05_joint_bayesian_geometry_slip.md)
