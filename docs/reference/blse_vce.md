# BLSE/VCE 参考

本文说明线性滑动分布反演中 `BoundLSEMultiFaultsInversion` 的常用运行模式、参数关系和结果检查。完整工作流见 [BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md)，可运行案例见 [Dingri 2020：BLSE/VCE 线性滑动反演](../casebook/dingri_blse_vce.md)。

## 阅读路径

- 只想跑通线性滑动反演：先读 workflow 和案例，本页作为方法和结果检查参考。
- 不确定 BLSE、smoothing loop、VCE 的区别：看 [三种运行模式](#三种运行模式)。
- 不确定应该复制普通 BLSE、平滑、倾角还是联合搜索脚本：看 [可运行脚本模板导航](../examples/script_templates.md)。
- 正在配置 sigma/alpha 或 poly：先查 [线性滑动反演配置](config_linear_slip.md) 和 [Sigmas 与 Alpha 配置模式](sigmas_alpha.md)，再回到本页看运行模式。
- 想复制常用 bounds/rake/patch 搭配：看 [约束配置与运行时调整短例](../examples/constraint_config_runtime.md)。
- 想选择平滑强度：看 [固定几何平滑搜索](../workflows/04a_blse_smoothing_search.md)；想比较倾角：看 [固定拓扑倾角搜索](../workflows/04b_blse_dip_search.md)。
- 想检查倾角与平滑是否耦合：看 [倾角 × 平滑参数敏感性分析](../workflows/04c_blse_dip_smoothing_search.md)。
- 正在检查物理约束是否生效：看 [约束检查](#约束检查) 和 [ECAT 约束管理器](constraint_manager.md)。
- 准备论文或报告：看 [推荐报告内容](#recommended-reporting)。

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

### 固定几何生命周期

一个 `BoundLSEMultiFaultsInversion` 实例对应一套固定的 fault geometry、mesh、GF、
Laplacian、参数列布局和约束。`run()`、`simple_run_loop()` 与 `run_simple_vce()`
只在这套固定基础上求滑动或更新权重，不会监听 fault 坐标变化并自动刷新全部矩阵。

比较多组倾角或其他几何时，把几何放在外层循环，并在每个候选完成 mesh 更新后创建
新的 inversion；同一候选内部的 penalty loop 或 VCE 迭代则复用该实例。不要在已经创建
inversion 后只修改 `Vertices`、`Faces`、top/bottom 或倾角再继续求解，否则可能混用新几何
与旧 GF、Laplacian 或约束布局。可复制组织方式见
[固定拓扑倾角搜索](../workflows/04b_blse_dip_search.md)。

## 协方差权重的统一度量

设第 \(k\) 个数据集的残差为 \(r_k=G_km-d_k\)，协方差为
\(C_k\)。BLSE 和 VCE 的数据目标必须是：

\[
\Phi_d=\sum_k r_k^\mathsf{T}C_k^{-1}r_k.
\]

ECAT 直接从每个数据集的协方差建立左白化度量：

这里默认 `d`、预测、`G` 和 `Cd` 已经使用同一个观测行顺序；GPS、optical 和多数据集
分块的具体排列见
[观测向量、协方差与设计矩阵排列合同](../concepts/observation_matrix_layout.md)。

\[
C_k=L_kL_k^\mathsf{T},\qquad W_k=L_k^{-1},\qquad
W_k^\mathsf{T}W_k=C_k^{-1}.
\]

因此 \(\lVert W_kr_k\rVert_2^2=r_k^\mathsf{T}C_k^{-1}r_k\)。实现不会先显式
计算 \(C_k^{-1}\)，也不会把所有数据集拼成一个全局白化矩阵；初始化时逐数据集验证并
因子化，求解时按观测行切片生成 \(W_kG_k\) 和 \(W_kd_k\)。这既保留非对角协方差，
也避免多个独立数据集产生不必要的全局稠密矩阵。

单位阵、缩放单位阵和对角协方差使用完全相同的统计定义，但实现会自动保留其结构：
单位阵只做无操作或标量缩放，对角阵按行乘以 \(1/\sqrt{C_{k,ii}}\)，一般非对角正定阵
继续使用 Cholesky 左白化。这个分派不需要配置开关，也不会用对角近似替代用户提供的
非对角协方差。结构化路径分别把白化的存储和计算从一般稠密阵的 \(O(n_k^2)\)、
\(O(n_k^2p)\) 降到 \(O(1)\) 或 \(O(n_k)\)、\(O(n_kp)\)。

在固定权重 BLSE 中，`data_weight = w` 产生
\(w^2 r^\mathsf{T}Pr\)；等价的 `sigma` 写法使用 \(w=1/\sigma\)。
`penalty_weight = \lambda` 同样是平滑残差行的直接乘数，因此平滑项为
\(\lambda^2\lVert L_0m\rVert_2^2\)，不是 \(\lambda\lVert L_0m\rVert_2^2\)。
当前实现按数据集依次白化并累计

\[
H=\sum_k(w_kW_kG_k)^\mathsf{T}(w_kW_kG_k)+L^\mathsf{T}L,
\qquad
q=-\sum_k(w_kW_kG_k)^\mathsf{T}(w_kW_kd_k).
\]

这与先拼接完整增广残差矩阵再计算 \(A^\mathsf{T}A\) 和
\(-A^\mathsf{T}b\) 是同一个目标；它只缩短矩阵装配过程，不改变 bounds、rake、
等式、不等式、DES 或最终模型定义。正常 CVXOPT 路径不再保留完整增广矩阵；仅当
CVXOPT 失败并进入 Clarabel 后备时，才从同一批数据和平滑矩阵延迟重建原残差系统。

VCE 第 \(k\) 个数据组和第 \(j\) 个平滑组求解：

\[
\min_m
\sum_k \frac{r_k^\mathsf{T}P_k r_k}{\sigma_k^2}
+\sum_j \frac{\lVert L_jm\rVert_2^2}{\alpha_j^2},
\]

随后用同一批白化量计算残差二次型、法矩阵和方差分量更新：

\[
e_k=W_kG_km-W_kd_k,\quad
e_k^\mathsf{T}e_k=r_k^\mathsf{T}C_k^{-1}r_k,\quad
(W_kG_k)^\mathsf{T}(W_kG_k)=G_k^\mathsf{T}C_k^{-1}G_k.
\]

基础 \(W_kG_k\) 和 \(W_kd_k\) 在 VCE 迭代前只计算一次；每轮只按
\(1/\sigma_k\) 缩放。当前 simple VCE 还会预先保存每个数据组和平滑组的
Gram/cross 块，并在第 \(t\) 轮直接组合

\[
H^{(t)}=
\sum_k\frac{(W_kG_k)^\mathsf{T}(W_kG_k)}{\sigma_k^{2(t)}}
+\sum_j\frac{L_j^\mathsf{T}L_j}{\alpha_j^{2(t)}},
\qquad
q^{(t)}=-\sum_k\frac{(W_kG_k)^\mathsf{T}W_kd_k}{\sigma_k^{2(t)}}.
\]

这与先拼接增广矩阵再计算 \(A^\mathsf{T}A\) 和 \(-A^\mathsf{T}b\) 严格等价；同一个
\(H^{(t)}\) 同时用于约束 QP 和本轮 VCE 方差更新，不会产生两套度量。合法协方差应当
为有限、对称正定矩阵；白化失败时 ECAT 会明确
报错，不会静默丢弃非对角项、改用伪逆或对角近似。当前多数据模型要求各数据集协方差
相互独立；若全局协方差含跨数据集非零块，会在求解前报错，避免静默丢失相关性。
如果为了诊断而改成单位阵或手工方差，应在结果中把它报告为新的观测误差模型，而不应
把结果差异解释成求解器自动修复了原协方差。

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
inv.extract_and_plot_blse_results(plot_faults=True, plot_data=True)
```

`extract_and_plot_blse_results()` 已经会物化当前模型，并默认打印一次拟合表。
只有在交互式地单独查看统计时才使用
`returnModel(print_fit_statistics=True)`，不要在标准
结果提取之前紧接着再调用一次。

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

当 `smoothing_constraints` 使 VCE 为本次运行构造新的 Laplacian 时，求解结果会保存
未缩放的 `smoothing_matrix`、与返回模型一致的 `model_smoothing_matrix` 以及
`smoothing_provenance`。求解后的 `GL_combined_poly` 直接发布这份实际使用的加权矩阵，
不会再从可能已经变化的 `fault.GL` 猜测重建。该状态合同不增加一次 VCE 求解，也不改变
`m`、sigma 或 alpha 的迭代。`returnModel()` 的原始模型粗糙度读取未缩放矩阵，求解目标
和 VCE 平滑组 `Qw` 读取加权矩阵；两者来自同一次求解，而不是从配置重新猜测。

### 显式平滑矩阵布局（进阶）

需要直接调用 `combine_GL_poly(GL_combined=..., penalty_weight=...)` 时，
`GL_combined` 必须是**尚未施加 penalty weight** 的块对角平滑矩阵，并且只包含
支持平滑的 source 块。各块按 `faults` 顺序排列；不要预先插入 polynomial 列或
不支持平滑的 source 列。方法会统一补齐这些零列、施加逐 source 权重，再生成与
全局线性模型向量完全对齐的 `GL_combined_poly`。显式矩阵未被当前 source 布局完整
消费时会直接报错，避免静默复用旧矩阵。省略 `GL_combined` 时仍从各 source 当前的
`fault.GL` 构造；关闭 alpha 时仍生成零平滑行，不改变“无平滑项”的语义。

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

函数把每个候选作为标量 `penalty_weight` 直接传给 `run(...)`；标量在当前 alpha 参数组空间广播。因此 `alpha.mode: single`、`individual` 和 `grouped` 都使用同一个候选惩罚权重，但不会错误地按断层数构造 alpha 向量。表中的 `Penalty_weight` 就是目标函数中实际使用的惩罚权重。对第 \(i\) 个
候选，求解矩阵和报告粗糙度分别为

\[
L_i=w_iL_0,
\qquad
R_i=\sqrt{\operatorname{mean}[(L_0m_i)^2]},
\]

其中 \(L_0\) 是当前固定 geometry、约束和参数列布局下未乘 penalty 的平滑矩阵。
`alpha.initial_value` 只定义普通 `run()` 未显式传权重时的默认值，不定义 \(L_0\)，也不会
改变 loop 的粗糙度尺度。

`simple_run_loop()` 是诊断事务：每个候选的模型、活动矩阵和统计量在同一轮内保持对应，
方法正常结束或候选求解异常时都会恢复调用前的活动模型、source 参数、平滑矩阵和权重。
事务内部若 geometry、协方差、数据权重和未缩放的 \(L_0\) 不变，会复用数据项
\(H_d,q_d\) 与 \(L_0^\mathsf{T}L_0\)；每个候选仍按自己的
\(w_i^2L_0^\mathsf{T}L_0\) 独立求解。缓存只存在于本次
`simple_run_loop()` 调用期间，不跨普通 `run()`、不跨 geometry 更新，也不在 DES
变换前后复用。

`preferred_penalty_weight` 只在图中标记候选，不会选择或激活模型。选定权重后应显式执行：

```python
inv.run(penalty_weight=30.0)
```

返回 DataFrame 和 CSV 中的 `RMS` 始终保存米制原值；`rms_unit="cm"` 或 `"mm"` 只改变
图的纵轴显示，不会原地修改结果表。

推荐检查：

- `Roughness_vs_RMS.png` 的拐点是否稳定。
- 选定 penalty 后的残差是否出现轨道系统误差。
- 选定 penalty 的滑动分布是否被过度平滑或出现不合理尖峰。
- 最终报告中保留 loop 表格，不只保留最终滑动图。

## VCE

VCE 是 variance component estimation，用于估计数据项和正则化项的绝对方差分量，并
由这些方差确定相对权重。适合多数据集联合反演，尤其是不同 InSAR 轨道、GPS 与 InSAR
权重不容易手工确定时。

第 \(t\) 轮使用当前方差分量求解：

\[
m^{(t)}=\underset{m\in\mathcal C}{\operatorname{argmin}}
\left[
\sum_g\frac{\lVert W_g(G_gm-d_g)\rVert_2^2}{\sigma_g^{2(t)}}
+\sum_h\frac{\lVert L_hm\rVert_2^2}{\alpha_h^{2(t)}}
\right].
\]

这里 \(W_g^\mathsf TW_g=C_g^{-1}\)，\(\mathcal C\) 是 bounds、等式和不等式约束
共同形成的可行域。求得模型后，数据组和平滑组分别计算

\[
u_g=\frac{Q_{w,g}}{\nu_{\mathrm{eff},g}},
\qquad v_g^{(t+1)}=v_g^{(t)}u_g,
\]

其中 \(v_g\) 表示 `sigma²` 或 `alpha²`。更新只改变下一轮行尺度，不改写本轮已求得的
`m`；因此结果把“本轮已求解尺度”和“下一轮提议尺度”作为两个明确状态返回。

典型调用：

```python
vce_result = inv.run_simple_vce(
    max_iter=20,
    tol=1e-4,
    report="compact",
)
inv.extract_and_plot_blse_results(plot_faults=True, plot_data=True)
```

返回结果通常包含：

| 键 | 含义 |
| --- | --- |
| `m` | 最终线性参数 |
| `solved_sigma2_by_group` | 返回模型实际求解时使用的数据组 `sigma²` |
| `solved_alpha2_by_group` | 返回模型实际求解时使用的平滑组 `alpha²` |
| `proposed_sigma2_by_group` | 最后一次更新后、可供下一轮使用的数据组 `sigma²` |
| `proposed_alpha2_by_group` | 最后一次更新后、可供下一轮使用的平滑组 `alpha²` |
| `sigma_groups` / `smooth_groups` | 结果中实际采用的成员映射 |
| `component_diagnostics` | 最终模型的组级 `Qw`、有效自由度和近似 reduced `Q` |
| `convergence_mode` | 有有效更新分量时为 `absolute_log`；全部固定时为 `fixed` |
| `convergence_metric` | 当前为 `max_abs_log_update_factor` |
| `convergence_measure` | 最后一次的 `max(abs(log(u)))`；与 `tol` 直接比较 |
| `converged` | 是否收敛 |
| `iterations` | 迭代次数 |

VCE 结果表中的符号遵循：`v=s²`，数据行的 `s=sigma`，平滑行的
`s=alpha`；`1/s` 才是增广最小二乘行的直接乘数。`log10(s)` 方便与
log-scaled 配置对照。`Qw` 是最终模型在该组完整协方差度量下的白化残差
二次型；`Approx. red.Q` 使用 VCE 最后一次线性化的有效自由度，受 bounds 和
活动约束影响，只作为诊断。当前实现还会在法矩阵奇异时使用伪逆，并在算得非正有效
自由度时采用显式数值保护，因此 `Approx. red.Q` 不能不加说明地作为正式 reduced
chi-square 写入论文。逐项公式、保护规则、单位和论文报告边界见
[拟合统计量](fit_statistics.md#covariance-aware-diagnostics)。

四个尺度字段始终都是以组名为键的字典，不因只有一个组而压缩成标量。无论本轮是否已经
满足容差，`proposed_*` 都记录本轮 `solved_* × u`；`m` 始终只对应 `solved_*`。
控制台表、拟合统计和当前模型权重只读取 `solved_*`；这不会追加一次求解或改变 `m`。
旧的 `var_d`、`var_alpha`、`weights` 和 `model_var_*` 结果别名已经移除，脚本若直接读取
低层结果字典，应迁移到上述具名字段。

### 乘法收敛判据

simple VCE 返回的是绝对方差分量，而不只是分量之间的相对比例。对所有确实在增广系统中
有行、且设置为更新的组，统一计算

\[
\Delta=\max_g\left|\log u_g\right|,
\qquad \Delta<\mathtt{tol}.
\]

`u_g=v_g^(t+1)/v_g^(t)` 是无量纲乘法因子。对数距离使放大和缩小对称，例如 `u=2`
与 `u=1/2` 距离收敛点相同。即使公共尺度不改变线性模型，残差矩仍然用于估计绝对方差，
所以不能用 `max(u)-min(u)`：单个更新分量的极差恒为零，多个相同但远离 1 的因子也会被
错误判为收敛。

默认 `tol=1e-4` 对应每轮方差乘法变化约小于 `0.01%`。它在 1 附近与历史
`abs(u-1)<1e-4` 几乎相同，但对倒数变化保持对称。没有可更新有效分量时模式为 `fixed`，
只求解一次；没有 Laplacian 行的空平滑组不参加停止判断。更新因子必须有限且严格为正，
否则无法定义对数尺度，算法会报告方差分量不可继续更新，而不会静默宣布收敛。

`report` 与 `verbose` 分工如下：

| 设置 | 作用 |
| --- | --- |
| `verbose=True` | 打印 VCE 初始化和逐轮收敛信息 |
| `report="compact"` | 打印一次最终方差分量表 |
| `report="full"` | 方差分量表后再打印一次当前模型拟合表 |
| `report="none"` | 不打印最终表，适合批处理 |
| `report=None` | `verbose=True` 时等价于 `compact`，否则等价于 `none` |

最终拟合表中的 `Eff. std`、`Qw`、`wRMS` 和 reduced 值定义见
[拟合统计量](fit_statistics.md#covariance-aware-diagnostics)。

VCE 可从配置读取 `geodata.sigmas` 和 `alpha` 的
`mode/update/initial_value`。`alpha.update: true` 表示由 VCE 迭代更新该
平滑组；`alpha.update: false` 表示仍使用平滑，但将该组 alpha 固定在
`alpha.initial_value`。这与 `alpha.enabled: false` 不同，后者关闭平滑项。

调用时也可以传入 `sigma_mode`、`sigma_groups`、`sigma_update`、
`sigma_values` 以及对应的 `smooth_mode`、`smooth_groups`、`smooth_update`、
`smooth_values`。每一类运行期覆盖都是一个原子合同：Sigma 至少同时给出
`sigma_mode/sigma_update/sigma_values`，Alpha 至少同时给出
`smooth_mode/smooth_update/smooth_values`；`mode: grouped` 时还必须给出对应
`*_groups`。若完全省略某一类运行期参数，则整套使用配置值。不能把部分新参数与其余旧配置混用。
分组组织方式及 `log_scaled` 含义见
[Sigmas 与 Alpha 配置模式](sigmas_alpha.md)。

`fault_ranges` 始终描述完整模型列，包括 Pressure、Sbarbot 等非平滑源；alpha 分组只覆盖
真正提供 Laplacian 的源。非平滑源仍正常参与 \(Gm\)、约束和结果分发，但不会被要求放入
`smooth_groups`，也不会在 VCE 分量表中出现虚构的 alpha。若同时向低层接口传
`smoothing_matrix` 和 `smoothing_constraints`，求解会在迭代前报错，因为两者代表两种
互斥的平滑来源。固定权重 BLSE 自动构造 Laplacian 时采用相同的行布局：保留所有源的
模型列，只为支持平滑的源建立实际平滑行，不用全零行表示非平滑源。

## 结果导出

常用出口：

```python
inv.extract_and_plot_blse_results(
    plot_faults=True,
    plot_data=True,
    data_poly="config",
    file_type="png",
    fault_outdir="output",
    data_outdir="Modeling",
)
```

该出口默认打印一次拟合表。若只需要数字、不绘图，改为单独调用
`returnModel(print_fit_statistics=True)`；不要把两个打印入口连续使用。

`data_poly="config"` 是推荐默认值：它逐数据集跟随已经解析的 `geodata.polys`。只有在明确诊断 source/slip-only 贡献时才传 `data_poly=None`；`data_poly="include"` 用于强制请求包含已求解改正项的预测。

`extract_and_plot_blse_results(...)` 通常会生成：

- `output/*_slip.<file_type>` 类型的断层滑动图。
- `Modeling/gps_<DataName>_map.<file_type>` 类型的 GPS data/synth 图。
- `Modeling/<DataName>_fit_comparison.<file_type>` 类型的 InSAR data/synth/residual 图。
- `Modeling/<DataName>_leveling_fit.<file_type>` 或 cross-fault offset 拟合图。
- 控制台中的拟合统计和断层统计。

该入口生成的 GPS/InSAR 合成观测与直接调用 `plot_data_fits()` 时相同。
`file_type` 只控制图像格式，不控制科学数据文件格式。
GPS、InSAR 的 data/synth/resid 文本仍由案例脚本按需要调用 CSI 的 `write2file()`
或 `writeDecim2file()` 明确导出；leveling 和 cross-fault offset 的 data/synth 文本
仍随各自拟合产品写入 `data_outdir`。

断层统计由 `inv.print_faults_summary()` 使用统一的 [Fault Summary / 断层概览和统计](fault_summary.md) 接口输出。它会报告 trace 长度、patch/mesh 数、面积、深度范围、平均走向倾角、slip 统计；位移单位模型报告 Moment/Mw，速率单位模型报告 moment rate。如果只想在脚本中拿结构化结果，使用 `inv.get_faults_summary()`。

案例脚本也可额外调用断层和数据对象的方法，例如写出 `slip_<FaultName>.gmt`、`slipdir_<FaultName>.txt`、每条 InSAR 的 `data/synth/resid` 文本文件。Dingri 案例的对应代码见 [脚本对照：导出滑动、滑动方向和模型数据](../casebook/dingri_blse_vce.md#export-slip-and-model-data)。

### 结构化拟合统计

`run()` 和 `run_simple_vce()` 都会在返回前把最终 `mpost` 分发到断层滑动和
poly 字段，因此完成求解后可以直接收集当前结果，不需要为了统计再次调用
`returnModel()`：

```python
inv.run(alpha=[-2.0])
rows = inv.collect_fit_statistics(
    model="BLSE",
    data_poly="config",
    include_dataset=True,
    include_global=True,
)
df = inv.fit_statistics_to_dataframe(rows)
```

`collect_fit_statistics()` 负责计算；`fit_statistics_to_dataframe()` 只负责把已有
rows 转为 DataFrame。逐数据集和全局统计的公式、poly 语义及通用循环模板见
[Fit Statistics](fit_statistics.md)。

## 约束检查

BLSE/VCE 支持的约束主要包括：

- `strikeslip/dipslip` 上下界。
- `rake_angle` 转换得到的线性角度约束。
- Euler 线性约束。
- `strikeslip == 0`、`dipslip == 0` 这类零滑等式约束。
- `zero_edge_slip(...)` 这类边界零滑等式约束。
- 用户通过 `source_constraints` 添加的线性等式/不等式约束。

这些约束由统一约束管理器组装。固定权重 BLSE、smoothing loop 和 VCE 使用相同
的约束配置写法；`rake_angle` 在线性 BLSE/VCE 中不是待求滑动参数，而是限制
`strikeslip/dipslip` 的角度范围。零滑和边界零滑的配置细节见
[ECAT 约束管理器](constraint_manager.md#zero-slip-constraints)。

### 求解器与约束保留

BLSE/VCE 首先使用 CVXOPT QP 路径。固定权重 BLSE 按数据块累计
\(H=A^\mathsf{T}A\)、\(q=-A^\mathsf{T}b\)；simple VCE 直接复用本轮由冻结
Gram/cross 块组合的同一 \(H,q\)。这仍是历史 `lsqlin` 在内部实际求解的二次目标，
并保留相同参数排列和约束。DES 开启时，BLSE 只从 DES 已变换的 \(G',D'\) 形成
\(H,q\)，不会与变换前矩阵混用。若 QP 因 Euler 列、滑动列和硬等式之间的尺度差异返回
`unknown` 或抛出数值异常，ECAT 才自动启用稳健后备：

1. 对具有独立 pivot 参数的硬等式做严格代数消元；这不是删除或软化约束。
2. 对变量列和约束行做可逆缩放。
3. 用 Clarabel 的二阶锥形式直接最小化 `||Gm-d||₂`，避免显式形成
   `G.T @ G` 后放大条件数。
4. 恢复原始参数，并用原始 `bounds/A/b/Aeq/beq` 再检查全部残差。

BLSE 和 simple VCE 正常求解时都不拼接完整增广残差矩阵；只有 QP 失败后，才从同一批
数据、活动平滑矩阵和 DES 状态延迟重建它并交给 Clarabel。因此加速路径没有删除稳健
后备所需的信息。后备路径只在原 QP 失败时运行，计算时间可能明显增加。若两个后端都不收敛，
求解会明确报错；ECAT 不会再通过移除等式或不等式约束来生成一个看似成功的
结果。正式研究仍应检查约束配置的物理可行性，不能把数值后备当作放宽先验。

<a id="recommended-reporting"></a>

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
- 断层概览统计，包括 trace 长度、mesh/patch 数、面积、深度范围、主要滑动区；位移模型报告地震矩和 Mw，速率模型报告矩率。字段见 [Fault Summary](fault_summary.md)。
- 滑动模型、滑动方向和 data/synth/resid 输出路径。

## 常见问题

- 若结果完全不平滑，检查 `alpha.enabled`、`update_Laplacian` 和 `penalty_weight` 是否被意外关闭或设得过小。
- 若某条 InSAR 轨道残差呈大尺度 ramp，检查 `geodata.polys` 和输出时的 `data_poly` 设置。
- 若 rake 约束没有效果，检查 `use_rake_angle_constraints: true`、`bounds_config.yml` 中断层名是否匹配、滑动参数化是否为 `ss_ds`。
- 若 VCE 权重异常，先用固定权重 BLSE 和 smoothing loop 检查数据、协方差和边界是否合理。
- 若出现稳健求解后备提示，说明原 QP 存在明显尺度或条件数问题；结果虽会按原矩阵复核约束，但循环计算会更慢，应同时检查 poly bounds、Euler 参数尺度和重复约束。
- 若 `alpha` 与预期相反，确认当前传入的是 `alpha` 还是 `penalty_weight`，以及 `log_scaled` 是否为 `true`。
