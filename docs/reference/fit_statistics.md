# 拟合统计量

本页定义 ECAT 在线性 BLSE/VCE、滑动 Bayesian 和几何 Bayesian 结果检查中使用的
拟合统计量，并说明模型激活、统计采集、控制台显示、绘图和文件导出的责任边界。
统计接口只检查已经求得的模型，不参与求解，也不会改变 Green's functions、协方差、
约束或后验样本。

## 统一的结果使用顺序

不同反演方法的求解入口可以不同，但用户理解结果时统一遵循五个阶段：

```text
求解或采样
  -> 激活一个可预测的物理模型
  -> 采集结构化统计
  -> 显示或绘图
  -> 写出科学数据和统计表
```

这五个阶段的关键约定是：

- “激活模型”负责把同一组 slip、poly、geometry 和 sigma 状态写入当前对象；
- “采集统计”只读取当前模型，不根据 `model=` 标签暗中切换 Bayesian 代表模型；
- “绘图”与“统计”使用相同的 synthetic 和 `data_poly` 约定；
- “格式化”和“写文件”不重新求解、不重新选择模型，也不重新解释 sigma；
- `extract_and_plot_*()` 是方便日常使用的一站式编排入口，底层仍保持上述责任分层。

当前公开接口的覆盖范围如下。这里明确写出差异，避免把尚未实现的统一能力写成事实：

| 场景 | 激活当前模型 | 结构化拟合统计 | 一站式结果入口 |
| --- | --- | --- | --- |
| 固定权重 BLSE | `run()`；交互查看可用 `returnModel()` | `collect_fit_statistics()` | `extract_and_plot_blse_results()` |
| 简化 VCE | `run_simple_vce()` | `collect_fit_statistics()` | `extract_and_plot_blse_results()` |
| SMC_FJ / FULLSMC / 联合 Bayesian | `returnModel(model=...)` | `collect_fit_statistics()` | `extract_and_plot_bayesian_results()` |
| 独立非线性几何 SMC | `returnModel(model=...)` | `collect_fit_statistics()` | `extract_and_plot_bayesian_results()` |
| 历史多断层 `explorefault` 路线 | `returnModel(model=...)` | `collect_fit_statistics()` | 历史 Bayesian 结果入口 |

这些路线现在共享 dataset row 的字段、RMS/VR 和可选协方差加权统计定义。线性及联合
Bayesian 对象还能提供 `global_solver_vector`；独立几何 SMC 没有一个固定的组装线性
求解向量，因此接受 `include_global` 以保持调用形状一致，但不会伪造全局行。新脚本仍应
优先使用新版 `NonlinearGeometrySMCInversion`；旧多断层路线保留用于复现已有参数组织。

三套 Bayesian 一站式结果入口统一使用 `model=` 选择代表模型，并使用
`print_fit_statistics=` 控制拟合表。联合 Bayesian 的 `best_model`、各类 `returnModel()`
历史上的 `print_stat` / `print_stats` 仍作为兼容关键字保留；公开脚本和新代码使用统一
拼写。相同关键字只统一结果责任，不会把不同求解器的参数注册或前向模型强行合并。

### 拟合表与尺度参数表的边界

逐数据集拟合表回答“当前模型如何拟合观测”；尺度参数表回答“这一活动模型使用什么
sigma/alpha，以及它来自固定、VCE 更新还是 Bayesian 采样”。两者共享同一个已激活模型，
但不应合并成一张宽表：

- 拟合表保留 `RMS`、`VR`、`Eff. std`、`Qw` 和 `wRMS`；
- 尺度表以物理 `Scale (s)` 为主值，并显式列出 `State`、`Sampling`、`log10(s)` 和
  `Row mult. (1/s)`；
- Bayesian 的 `Post. SD(s)` 从整列物理 posterior 样本精确计算；固定组留空；
- VCE 额外列出 `Variance (v)` 和组级 `Approx. red.Q`，但没有 posterior 标准差；
- 格式化器只读取活动结果和保存的 posterior，不重算 synthetic、likelihood、GF 或
  Laplacian，也不改变 HDF5 样本。

完整字段和 `log_scaled` 对照见
[Sigmas 与 Alpha 配置模式](sigmas_alpha.md#统一结果表中的名称)。

## 符号和残差

对第 \(k\) 个向量化数据集，记：

| 符号 | 含义 |
| --- | --- |
| \(d_k\) | 观测向量 |
| \(g_k\) | 当前模型对应的合成观测；线性情形通常为 \(G_km\) 并包含选定的 poly 改正 |
| \(r_k=g_k-d_k\) | 残差；ECAT 使用 synthetic minus observed |
| \(n_k\) | 当前数据向量中的标量观测数 |
| \(C_k\) | 基础协方差矩阵 |
| \(C_{\mathrm{eff},k}=\sigma_k^2C_k\) | 当前 sigma 尺度下的有效协方差 |
| \(C_k=L_kL_k^\mathsf T\) | Cholesky 分解 |
| \(W_k=L_k^{-1}\) | 左白化器，满足 \(W_k^\mathsf TW_k=C_k^{-1}\) |

残差符号不影响 RMS、VR 和二次型，但统一符号能够避免以后增加有符号偏差统计时产生
歧义。

## 原始观测空间统计

ECAT 使用 CSI 既有定义：

\[
\operatorname{RMS}_k
=\sqrt{\frac{r_k^\mathsf Tr_k}{n_k}},
\qquad
\operatorname{VR}_k
=100\left(1-\frac{r_k^\mathsf Tr_k}{d_k^\mathsf Td_k}\right).
\]

对应代码等价于：

```python
r = synthetic - observed
rms = np.sqrt(np.mean(r**2))
vr = (1 - np.sum(r**2) / np.sum(observed**2)) * 100
```

RMS 与观测使用相同单位。VR 为百分数，但这里的参考能量是零模型
\(d_k^\mathsf Td_k\)，不是减去观测均值后的总平方和，因此不要把它写成普通回归中的
\(R^2\)。VR 可以为负，表示当前模型的残差能量大于零模型的观测能量；结构化接口保留
负值。

对组装后的线性求解向量，同一定义作用于：

\[
r_{\mathrm{global}}=Gm_{\mathrm{post}}-d.
\]

全局 RMS/VR 不是逐数据集 RMS/VR 的算术平均。

<a id="covariance-aware-diagnostics"></a>

## 协方差加权统计

当当前模型已经发布本次求解实际使用的 \(C_k\) 和 \(\sigma_k\) 时，ECAT 计算：

\[
Q_{w,k}
=\left\lVert\frac{W_kr_k}{\sigma_k}\right\rVert_2^2
=r_k^\mathsf TC_{\mathrm{eff},k}^{-1}r_k,
\]

\[
\operatorname{wRMS}_k=\sqrt{\frac{Q_{w,k}}{n_k}}.
\]

`Qw` 是有效协方差度量下的残差总二次型；`wRMS` 是按观测数归一后的无量纲
白化残差尺度。二者都使用完整协方差，包括非对角相关项。

控制台中的 `Eff. std` 是便于阅读的边际尺度：

\[
s_{0,k}=\sqrt{\operatorname{mean}(\operatorname{diag}C_k)},
\qquad
s_{\mathrm{eff},k}=\sigma_k s_{0,k}.
\]

它继承观测单位，用来把 sigma 与数据的大致边际标准差联系起来；它不使用非对角项，
不参与白化、求解或 likelihood，也不是残差均值的标准误。相关协方差不能用这一列替代。

### 每一列的准确含义

| 显示列或结构化字段 | 公式 | 单位 | 主要用途 | 不能据此声称 |
| --- | --- | --- | --- | --- |
| `N` / `n_observations` | \(n_k\) | 个数 | 确认数据向量长度和归一化分母 | 有效自由度 |
| `RMS` / `rms` | \(\sqrt{r_k^\mathsf Tr_k/n_k}\) | 观测单位 | 查看物理量级上的平均残差 | 已考虑协方差相关性 |
| `VR (%)` / `vr` | \(100(1-r_k^\mathsf Tr_k/d_k^\mathsf Td_k)\) | % | 与零模型比较残差能量 | 普通回归 \(R^2\) 或 likelihood |
| `Eff. std` / `effective_marginal_std` | \(\sigma_k\sqrt{\operatorname{mean}(\operatorname{diag}C_k)}\) | 观测单位 | 阅读有效边际噪声尺度 | 完整协方差的单值等价物 |
| `Qw` / `weighted_quadratic` | \(r_k^\mathsf T(\sigma_k^2C_k)^{-1}r_k\) | 无量纲 | 检查求解目标中的数据失配 | 不含模型复杂度修正的正式 reduced chi-square |
| `wRMS` / `weighted_rms` | \(\sqrt{Q_{w,k}/n_k}\) | 无量纲 | 比较不同单位和协方差尺度的数据集 | 观测单位 RMS |
| `weighted_effective_dof` | \(\nu_{\mathrm{eff},g}\) | 个数 | 简化 VCE 组级线性化诊断 | 严格约束问题的精确自由度 |
| `Approx. red.Q` / `reduced_weighted_misfit` | \(Q_{w,g}/\nu_{\mathrm{eff},g}\) | 无量纲 | 检查 VCE 分组的尺度是否明显失衡 | 可直接作为所有模式的正式卡方检验 |

如果误差模型、sigma 和可预测模型都合适，独立高斯条件下 \(Q_w\) 的期望量级与相应
自由度相近。但相关结构、模型缺失、异常值、估计得到的 sigma、边界约束和有效自由度
近似都会改变这种解释。论文中可以直接报告 RMS、VR、\(Q_w\) 及其定义；若报告
`Approx. red.Q`，必须同时说明 ECAT 使用的是简化 VCE 的近似线性化自由度，而不是
严格约束卡方检验。

## VCE 组级统计和有效自由度

多个数据集共享一个 sigma 组 \(g\) 时：

\[
Q_{w,g}=\sum_{k\in g}Q_{w,k},
\qquad
n_g=\sum_{k\in g}n_k.
\]

简化 VCE 对最后一次线性化使用以下实现。令：

\[
H=\sum_g\frac{(W_gG_g)^\mathsf T(W_gG_g)}{\sigma_g^2}
+\sum_j\frac{L_j^\mathsf TL_j}{\alpha_j^2},
\]

\[
H_g=\frac{(W_gG_g)^\mathsf T(W_gG_g)}{\sigma_g^2},
\qquad
\nu_{\mathrm{eff},g}=n_g-\operatorname{tr}(H^{-1}H_g).
\]

若 \(H\) 不可逆，实现使用 Moore–Penrose 伪逆。若算得
\(\nu_{\mathrm{eff},g}\le 0\)，当前简化 VCE 使用
\(0.1n_g\) 作为数值保护值。平滑组使用相同结构，把 \(W_gG_g\) 换成 \(L_j\)，把
\(n_g\) 换成该平滑组的行数。

因此 `Approx. red.Q` 必须保留 `Approx.`：

- 它属于 VCE 组，不属于组内每个数据集；
- active bounds、等式和不等式约束没有进入严格的约束自由度推导；
- 非正自由度时还包含上述显式保护规则；
- 共享组的 reduced 值只出现在 VCE 分量表，数据集行不复制该值；
- 单数据集独占一个 VCE 组时，数据集行可以显示同一个近似 reduced 值。

在简化 VCE 中，数据组或平滑组的方差更新因子正是
\(u_g=Q_{w,g}/\nu_{\mathrm{eff},g}\)。所有有实际行且设置为更新的分量统一要求

\[
\max_g|\log u_g|<\mathtt{tol}.
\]

这是绝对方差分量的乘法收敛条件：即使所有因子拥有同一个公共尺度，它们仍须各自趋近
1。没有 Laplacian 行的空平滑组不参加停止判断。又因为 sigma 利用同一批残差估计，
`Approx. red.Q` 不是独立于反演的 goodness-of-fit 检验；在约束问题中还应保留
`Approx.` 限定。

固定 BLSE、SMC_FJ、FULLSMC 和独立几何 SMC 不会人为构造这一自由度，因此 reduced
列为空；只要当前模型已发布准确的协方差和 sigma，`Qw` 与 `wRMS` 仍然有效。

### 平滑组的 `Qw`

VCE 分量表同时显示数据组和平滑组。对平滑组 \(j\)：

\[
q_j=L_jm,
\qquad
Q_{w,j}^{(\alpha)}=\frac{q_j^\mathsf Tq_j}{\alpha_j^2}.
\]

它衡量当前模型相对该组平滑尺度的粗糙度，不是观测数据拟合，也没有观测单位 RMS。
因此 VCE 分量表与逐数据集拟合表应分开保留：前者回答方差分量和正则化平衡，后者
回答各观测数据拟合情况。

## 数据向量排列

逐数据集统计使用与反演相同的观测行合同：

| 数据类型 | 观测向量 | 合成向量 |
| --- | --- | --- |
| GPS，`vertical=True` | 全部 East、全部 North、全部 Up | 全部 East、全部 North、全部 Up |
| GPS，`vertical=False` | 全部 East，再全部 North | 全部 East，再全部 North |
| InSAR/SAR | `vel` | `synth` |
| optical offset | `east` + `north` | `east_synth` + `north_synth` |
| leveling | `vel` | `synth` |
| cross-fault offset | `data_vector` | `synth_vector` |

GPS 排列与 CSI 的 `d/G/Cd` 及 frame transform 行一致。RMS 和 VR 对同时重排的向量
不敏感，但加权统计必须保证残差每一行与协方差同一行对应。

`data_poly="config"` 时，绘图和统计使用同一逐数据集预测规则：

| 已配置的 poly | 统计使用的 synthetic |
| --- | --- |
| `None` | 仅 source/slip 预测 |
| 任意已配置改正 | `poly="include"` |

只有明确检查 slip-only 拟合时才使用 `data_poly=None`；需要无条件包含已求解改正项时
使用 `data_poly="include"`。

## 逐数据集、平均行和全局行

`collect_fit_statistics()` 可以返回三种 scope：

| `scope` | 计算对象 | 正确解释 |
| --- | --- | --- |
| `dataset` | 一个数据对象的实际观测向量 | 论文中常用的逐数据集拟合 |
| `dataset_average` | 各数据集 RMS 和 VR 的算术平均 | 仅作屏幕快速比较，不是总拟合 |
| `global_solver_vector` | 当前组装的 \(Gm-d\) | 完整线性求解向量的总体拟合 |

`dataset_average` 默认关闭。其 `rms` 和 `vr` 是逐数据集标量的算术平均，不按数据量
加权；该行同时保存的 `ss_res/ss_obs/n_observations` 是成员求和，不能反过来用这些和
重建该行的平均 RMS/VR。需要正式总体量时使用 `global_solver_vector`，不要把逐数据集
平均当作全局结果。

`include_global=True` 时，线性路线尝试使用：

```python
# BLSE
np.dot(self.G, self.mpost) - self.d

# Bayesian linear vector，维度一致时
np.dot(self.G_combined, self.mpost) - self.observations
```

若当前对象没有完整组装矩阵，或维度与当前模型不一致，ECAT 省略全局行，不用另一套
代理统计静默替代。

## 当前模型合同

拟合统计描述的是已经分发到 fault 和 data 对象的当前模型。`model=` 只是输出标签，
不会自行选择 Bayesian posterior summary：

| 路线 | 先激活模型 | 再采集统计 |
| --- | --- | --- |
| BLSE | `inv.run(...)` 已求解并分发最新 `mpost` | 直接调用 `collect_fit_statistics()` |
| VCE | `inv.run_simple_vce(...)` 已分发最终 VCE 解 | 直接调用 `collect_fit_statistics()` |
| 联合/滑动 Bayesian | `inv.returnModel(model="mean" | "median" | "MAP", print_fit_statistics=False)` | 用同名 `model` 作为报告标签 |
| 独立几何 SMC（新旧多断层） | `inv.returnModel(model="mean" | "median" | "MAP", print_fit_statistics=False)` | 用同名 `model` 采集 dataset rows |

例如 MAP 与 median 必须分别激活：

```python
inv.returnModel(model="MAP", print_fit_statistics=False)
map_rows = inv.collect_fit_statistics(model="MAP")

inv.returnModel(model="median", print_fit_statistics=False)
median_rows = inv.collect_fit_statistics(model="median")
```

只调用 `collect_fit_statistics(model="MAP")` 不会把 slip、poly 或 `mpost` 切换成 MAP。
`model="std"` 是参数离散度而不是可预测的物理模型，不应收集数据拟合统计。

BLSE 应使用 `run()` 返回后的同步状态。显式调用 `returnModel(mpost=...)` 时，传入向量会
成为新的活动解；`self.mpost`、分发到各源的参数、拟合统计和 solver RMS/VR 都使用该向量，
不会再出现只临时分发、随后用旧向量计算全局量的半状态。

`returnModel()` 返回的 `roughness` 是当前求解发布的未缩放平滑矩阵 \(L_0\) 下

\[
\operatorname{roughness}=\sqrt{\operatorname{mean}[(L_0m)^2]}.
\]

它适合在相同 \(L_0\) 下比较 smoothing loop。\(L_0\) 由当前固定 geometry、约束和
参数列布局定义，不由配置的 `alpha.initial_value` 定义。带 `1/alpha` 行乘数的活动目标矩阵保存在
`current_model_smoothing_matrix`；VCE 平滑组 `Qw` 使用后者对应的
\(\lVert L_hm/\alpha_h\rVert^2\)。报告动作不会从配置重建任一矩阵；
`simple_run_loop()` 结束后还会恢复调用前的完整活动结果，不能把诊断候选误当成已选模型。

## 结构化接口

| 方法 | 责任 |
| --- | --- |
| `collect_fit_statistics(...)` | 按需重建当前 synthetic，并计算 dataset/global rows |
| `fit_statistics_to_dataframe(rows)` | 把已有 rows 转成 DataFrame，不重新计算 |
| `format_fit_statistics_report(rows)` | 把已有 rows 渲染为文本，不重新计算 |
| `write_fit_statistics_report(...)` | 写出已有 rows，或在未传 rows 时按显式参数采集后写出 |
| `calculate_and_print_fit_statistics()` | 当前模型的交互式紧凑控制台表 |

独立几何 SMC 的采集器默认只读取 `returnModel()` 已经写回的完整 synthetic。若其他操作覆盖了
data 对象，可显式传 `rebuild_synth=True`：它只重复当前向量的预测，不会选择另一组样本，
并按观测/协方差使用的同一行序写回包含数据改正的完整预测。新项目推荐使用新版
`NonlinearGeometrySMCInversion`；旧入口继续保留这一能力用于既有案例复现。
线性/联合 Bayesian 还支持 `data_poly`、`faults` 和全局 solver row 等参数；不要向独立
几何对象传入这些仅属于分布式滑动路线的选项。

BLSE 求解后可复制：

```python
inv.run(alpha=[-2.0])

rows = inv.collect_fit_statistics(
    model="BLSE",
    data_poly="config",
    include_dataset=True,
    include_global=True,
    include_weighted=True,
)

df = inv.fit_statistics_to_dataframe(rows)
inv.write_fit_statistics_report("output", rows=rows)
```

`collect_fit_statistics()` 默认重建逐数据集 synthetic；全局行独立使用当前 solver vector。
结构化 rows 默认不新增 weighted 字段，只有 `include_weighted=True` 时才扩展，避免破坏
已有 DataFrame 消费者。

## 屏幕输出、提取和避免重复打印

模型激活与报告是两个责任：

- `run()` 和 `run_simple_vce()` 求解并分发当前线性模型；
- `returnModel()` 保留交互式入口，新代码用 `print_fit_statistics=True` 控制打印；
- `extract_and_plot_blse_results()` 默认物化当前结果、打印一次拟合表并生成图件；
- `extract_and_plot_bayesian_results()` 读取样本、激活指定代表模型，再组织统计和图件；
- 低层协方差/VCE 函数只返回结构化数值，不拥有第二份最终报告。

常用 VCE 路线：

```python
result = inv.run_simple_vce(
    max_iter=20,
    tol=1e-4,
    report="compact",
)
inv.extract_and_plot_blse_results(
    print_fit_statistics=True,
    plot_faults=True,
    plot_data=True,
)
```

`report="compact"` 只打印 VCE 分量表；`"full"` 立即追加当前模型拟合表；`"none"`
用于静默批处理。若紧接着又调用默认打印的 extraction，`full` 会使拟合表出现两次；
标准脚本通常使用 `compact + extraction`。

VCE 返回的 `m` 属于最后一次实际求解使用的方差，而
`proposed_sigma2_by_group/proposed_alpha2_by_group` 始终记录该轮残差提出的下一步更新；
即使已经满足容差，两者也可能存在容差范围内的差别。
报告只使用 `solved_sigma2_by_group/solved_alpha2_by_group`，避免模型与标签错配；这一
选择不会追加求解或改变 `m`。

每次新求解、数据重新组装或代表模型切换前，ECAT 会清除上一模型的 sigma/group/dof
报告上下文；基础协方差因子仍可复用。因而旧模型的加权统计不会被误贴到新 synthetic，
也不会因为结果报告而重复分解协方差。

## 循环实验

几何、平滑或约束循环应在每次成功求解后立即保存 dataset 和 global 行：

```python
all_rows = []

for value in test_values:
    inv.run(...)
    rows = inv.collect_fit_statistics(
        model=f"case_{value}",
        data_poly="config",
        include_dataset=True,
        include_global=True,
    )
    all_rows.extend({"test_value": value, **row} for row in rows)

df = inv.fit_statistics_to_dataframe(all_rows)
```

统计必须紧跟对应求解，避免下一轮修改当前 slip、poly 或 synthetic 后再回头采集旧标签。

## 论文和结果解释建议

至少同时说明：

- residual 定义和观测单位；
- 每个数据集的 \(n_k\)、RMS 和 VR；
- 使用单位阵、对角协方差还是完整协方差；
- 若报告 \(Q_w\)，给出 \(C_{\mathrm{eff}}=\sigma^2C\) 和白化定义；
- 若报告 `Eff. std`，明确它只是边际标准差的 RMS 描述量；
- 若报告 VCE `Approx. red.Q`，给出组成员、近似自由度公式、约束影响和非正自由度保护；
- 不把逐数据集 RMS/VR 的算术平均写成全局拟合；
- 明确统计对应 MAP、median、mean 还是固定权重线性解。
