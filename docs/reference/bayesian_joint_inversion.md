# Bayesian 联合反演参考 / Bayesian Joint Inversion

本文说明 `BayesianMultiFaultsInversion` 的联合 Bayesian 反演语义。它面向已经理解两步走路线、需要处理几何-滑动耦合、多源数据和高级约束的用户。

普通同震研究优先阅读 [Bayesian 非线性几何反演 / Nonlinear Geometry](../workflows/03_nonlinear_geometry_bayesian.md) 和 [BLSE/VCE 线性滑动分布反演 / Linear Slip](../workflows/04_linear_slip_blse_vce.md)。联合路线见 [Bayesian 联合几何-滑动分布反演 / Joint Bayesian Geometry-Slip](../workflows/05_joint_bayesian_geometry_slip.md)。

## 阅读路径

- 第一次做 ECAT 反演：不要从本页开始，先完成标准两步走 workflow。
- 已经完成两步走，想传播几何不确定性到滑动分布：先看 [入口定位](#入口定位) 和 [采样模式](#采样模式)。
- 不确定哪些参数被采样、哪些由线性求解得到：看 [参数分层](#参数分层)。
- 需要让断层几何随样本更新：看 [几何更新](#几何更新)，再进入 [可扰动断层几何](geometry_perturbation.md)。
- 正在检查案例配置：按 [配置检查顺序](#配置检查顺序) 逐项核对。
- 准备重跑或确认旧结果是否会被保留：先看 [样本文件和覆盖语义](#sample-file-overwrite)。

## 入口定位

| 入口 | 用途 |
| --- | --- |
| `BayesianMultiFaultsInversion` | 联合 Bayesian 主入口；管理多断层、多数据、SMC 采样和样本内求解 |
| `BayesianMultiFaultsInversionConfig` | 解析 `bayesian_sampling_mode`、`slip_sampling_mode`、sigma/alpha、GF、几何更新和约束 |
| `BayesianAdaptiveTriangularPatches` | 常用可扰动三角断层对象；提供几何基线、扰动方法和 mesh 派生状态有效性 |
| `ConstraintManagerSMC` | 管理 SMC 边界；在 `SMC_FJ + ss_ds` 中管理线性约束矩阵 |

联合 Bayesian 不是单独的“非线性几何搜索”。它可以让几何、滑动、sigma、alpha、poly 和多源约束在同一后验框架中耦合。

## 采样模式

| 字段 | 可选值 | 说明 |
| --- | --- | --- |
| `bayesian_sampling_mode` | `SMC_FJ`, `FULLSMC` | 解析后只保存这两个规范值 |
| `slip_sampling_mode` | `ss_ds`, `magnitude_rake`, `mag_rake`, `rake_fixed` | `mag_rake` 会归一化为 `magnitude_rake`；`SMC_FJ` 使用 `ss_ds` |

两种 Bayesian 模式的核心差异：

| 模式 | SMC 样本向量包含 | 滑动如何得到 | 约束特点 |
| --- | --- | --- | --- |
| `SMC_FJ` | 几何、sigma、alpha 等超参数 | 样本内通过线性求解得到 `strikeslip/dipslip` | 支持 rake、Euler、零滑、边界零滑和自定义 `A @ x` 线性约束 |
| `FULLSMC` | 几何、sigma、alpha 和滑动参数 | 直接从样本向量得到 | 主要通过参数边界和先验控制，不合并线性约束矩阵 |

`SMC_FJ` 维度通常较低，是联合几何-滑动反演的推荐高级路线。`FULLSMC` 适合小规模模型或需要显式研究滑动先验的场景，但计算成本和收敛诊断难度都更高。

## 协方差与似然度量

令数据残差为 \(r\)，基础精度矩阵为 \(P=C_d^{-1}\)。给定当前样本的
\(\sigma\) 后，两种联合模式使用同一个数据项（省略与当前样本无关的常数）：

\[
\log p(d\mid m,\sigma)
=-\frac{1}{2}\left[
r^\mathsf{T}\frac{P}{\sigma^2}r+n\log(\sigma^2)
\right].
\]

差别只在当前滑动 \(m\) 的来源：

| 模式 | 数据协方差进入计算的方式 | 一致性要求 |
| --- | --- | --- |
| `SMC_FJ` | 逐数据集准备 \(W_kG_k,W_kd_k\)，由其精确 Gram/cross 块构造条件 QP；求解后仍用同一批白化块直接计算残差分数 | 条件解和后验评分使用同一浮点度量，不另存 \(P_k\) |
| `FULLSMC` | 滑动已在样本向量中，逐数据集计算 \(\lVert W_kr_k/\sigma_k\rVert^2\) | 不构造条件线性系统，也不构造全局精度矩阵 |

其中 \(C_k=L_kL_k^\mathsf{T}\)、\(W_k=L_k^{-1}\)，所以
\(W_k^\mathsf{T}W_k=C_k^{-1}\)。每个数据集只在初始化时由 \(C_k\) 建立一次
`W_k/logdet_k`；几何、GF、Laplacian、面积、sigma 或 alpha 样本变化不会重新因子化
基础协方差。`SMC_FJ` 的当前几何会改变 \(G_k\)，因此每个候选仍需计算新的
\(W_kG_k\)，但不会重复建立 \(W_k\)。

固定几何 `SMC_FJ` 会在 target 构造时一次性准备

\[
B_k=(W_kG_k)^\mathsf{T}(W_kG_k),\qquad
c_k=(W_kG_k)^\mathsf{T}W_kd_k,\qquad
S_j=D_j^\mathsf{T}D_j,
\]

候选内只按当前 \(\sigma_k,\alpha_j\) 组合

\[
H=\sum_k\frac{B_k}{\sigma_k^2}+\sum_j\frac{S_j}{\alpha_j^2},
\qquad q=-\sum_k\frac{c_k}{\sigma_k^2}.
\]

其中未启用 alpha 时第二个求和为空，但数据 Hessian

\[
H_d=\sum_k\frac{B_k}{\sigma_k^2}
\]

仍然存在。`SMC_FJ` 对线性参数采用统一的 FJ/Laplace 曲率评分，不会因为关闭平滑而
切换成只在最优滑动处评价的 profile likelihood。

### 为什么消去线性参数会产生 log-determinant

对固定的几何和超参数，令完整线性参数为 \(m\)，维数为 \(p\)。无约束条件二次型可写成

\[
\Phi(m)=\Phi(\hat m)+(m-\hat m)^\mathsf{T}H(m-\hat m).
\]

`SMC_FJ` 的目的不是把高维 \(m\) 放进 SMC 样本，而是对它形成条件解并消去。形式上的
Gaussian 积分为

\[
\int_{\mathbb R^p}\exp\!\left\{-\Phi(m)/2\right\}\,dm
=\exp\!\left\{-\Phi(\hat m)/2\right\}(2\pi)^{p/2}|H|^{-1/2}.
\]

因此取对数后必须出现

\[
\ell_H=-\frac12\log|H|.
\]

它是线性参数邻域概率体积的曲率修正，不是数据协方差的 \(\log|C_k|\)，也不是平滑残差。
有平滑时 \(H=H_d+H_s\)；无平滑时 \(H=H_d\)。两种情况都使用同一个曲率合同。

以单一 sigma 为例，若 \(H_d=B/\sigma^2\)，则

\[
\log|H_d|=\log|B|-p\log\sigma^2.
\]

曲率项把数据尺度归一化中与 sigma 有关的系数从
\(N\log\sigma^2\) 修正为 \((N-p)\log\sigma^2\)。所以无平滑时省略
\(\log|H_d|\) 会改变 sigma 和几何的后验排序，不是可忽略常数。

实现使用 \(H=LL^\mathsf{T}\) 的 Cholesky 因子计算

\[
-\frac12\log|H|=-\sum_i\log L_{ii}.
\]

如果 \(H\) 不对称、非正定或在双精度下秩亏，程序会明确报错。此时平坦测度下的
全维 Gaussian 积分并不存在，不能用 `abs(det(H))`、伪行列式或静默退回 profile
评分掩盖问题。需要重新检查数据可辨识性，或明确定义一个 proper prior；数值 jitter
不能替代科学先验。

当前 bounds、rake、等式和不等式仍由条件 QP 强制执行。只有 bounds/不等式形成的
全维可行域，严格边缘化才可写成全空间 Gaussian 积分乘截断概率质量。精确等式约束会把
积分限制到仿射子空间；若以零空间基 (Z) 写成 (m=m_p+Zy)，严格曲率应由降维
Hessian (Z^\mathsf{T}HZ) 计算。ECAT 当前既不计算截断概率质量，也不执行这套等式
约束降维归一化，所以这里应解释为**受约束条件解加 FJ/Laplace 型全空间曲率近似**，
而不是严格受约束 marginal posterior。
可选 magnitude prior 也仍在条件解处评价。关闭 alpha 只移除 Laplacian 构造、平滑
二次项和 alpha 归一化项，不移除曲率项。

这就是原增广最小二乘的 \(A^\mathsf{T}A\) 与 \(-A^\mathsf{T}b\)，不是新的近似。
这里的 \(D_j\) 已经展开到完整线性参数列布局；非平滑 source 和 poly 列保留正确列宽，
但在对应平滑块中为零。条件求解和平滑后验项共同读取同一批冻结的 \(D_j\) 块：

\[
\log p_{\mathrm{smooth}}
=-\frac12\sum_j\left[
\frac{\lVert D_jm\rVert_2^2}{\alpha_j^2}
+n_j\log(\alpha_j^2)
\right],
\]

其中 \(n_j\) 是第 \(j\) 个平滑块的行数。因而 alpha 组的行归属、source/poly 列位置和
求解矩阵不会在评分阶段通过另一份 `GL_combined` 重新推断。
几何可变时不会缓存各个几何样本；当前候选更新 GF/Laplacian 后保守地瞬态重建数据块和
平滑块，候选结束即可释放。刚性变化仍可在几何派生状态层保留有效的 Laplacian，但当前
二次型工作区不跨几何候选缓存其 Gram 块；这避免在缺少明确几何版本键时误用陈旧矩阵。
后验数据项仍由直接白化残差计算，`smooth_prior_weight` 仍只改变后验平滑评分，
不会乘入 \(H\) 改变条件线性解。

单位阵、缩放单位阵和对角协方差由同一个 covariance metric 自动执行无操作、标量或逐行
白化；一般非对角正定阵继续使用 Cholesky 白化。用户不需要选择“快速协方差模式”，且
非对角阵不会被静默近似为对角阵。

以上边缘化结构对应 Fukuda–Johnson 混合线性—非线性 Bayesian 方法中“解析处理线性
参数、采样非线性参数”的基本思路。ECAT 的硬约束和额外评分扩展决定了它在一般场景下
采用上述明确标注的近似边界。原始方法背景见
[Fukuda & Johnson (2010)](https://doi.org/10.1111/j.1365-246X.2010.04564.x)。

独立的非线性几何 SMC 也逐数据集用 \(\lVert W_kr_k/\sigma_k\rVert^2\) 计算
似然，不经过 BLSE/SMC-FJ 的条件线性求解。这里的 Cholesky 与 SMC 提议分布为生成
相关随机扰动而做的 Cholesky 是两件不同的事。协方差必须为有限、对称正定矩阵；
`SMC_FJ` 不会在白化失败后静默改成对角权重。BLSE/VCE 中同一约定和完整公式见
[BLSE/VCE 参考：协方差权重的统一度量](blse_vce.md#协方差权重的统一度量)。

旧配置可能使用 `SMC_F_J`。该拼写只作为兼容输入保留，读取时立即归一化为
`SMC_FJ`；新配置、CLI 模板、配置导出、日志和运行时状态统一使用
`SMC_FJ`。

`FULLSMC + magnitude_rake` 和 `FULLSMC + rake_fixed` 当前只支持纯
`Fault` 源，并要求每个 Fault 恰好包含规范顺序 `[ss, ds]`。输入 `ds` 会被
规范化为 `sd`；只有 `s`/`d`、或包含 tensile/coupling 的 Fault 不适用于这两种
参数化。Fault、Pressure、Sbarbot 等源混合进入 `FULLSMC` 时使用 `ss_ds`，
避免把断层专属的 magnitude/rake 语义套到其他源参数上。

## 参数分层

| 层级 | 参数 | 在 `SMC_FJ` 中 | 在 `FULLSMC` 中 |
| --- | --- | --- | --- |
| 几何 | 断层位置、走向、倾角、网格扰动参数 | SMC 采样 | SMC 采样 |
| 数据权重 | `sigmas` | SMC 采样 | SMC 采样 |
| 平滑尺度 | `alpha` | SMC 采样 | SMC 采样 |
| 滑动 | `strikeslip/dipslip` 或 `magnitude/rake` | 线性求解 | SMC 采样 |
| 数据修正 | `poly`、ramp 或 Euler rotation 等 | 线性求解 | SMC 采样或按配置处理 |

在 `SMC_FJ` 中，约束矩阵 `A` 的列只对应线性参数块，即滑动和 poly 参数；不包含 geometry、sigma 或 alpha。这个边界很重要，否则容易把几何先验和线性约束混写。

## 几何更新

联合几何更新同时受顶层开关、断层开关和参数切片控制：

```yaml
nonlinear_inversion: true

faults:
  FaultA:
    geometry:
      update: true
      sample_positions: [0, 1]
    method_parameters:
      update_fault_geometry:
        method: perturb_bottom_coords_along_fixed_direction
        average_direction: 10.0
        angle_unit: degrees
        perturbation_direction: horizontal
      update_mesh:
        method: generate_and_deform_mesh
        top_size: 3.0
        bottom_size: 6.0
        num_segments: 25
        disct_z: 10
```

`sample_positions` 使用半开区间；`[0, 1]` 表示从全局几何样本向量取一个量，不表示
断层节点范围。所有启用断层的几何位置必须从 0 开始连续覆盖，也可以让多个断层显式
共享同一段位置。

运行时，采样器把当前切片作为 `perturbations` 传给 `update_fault_geometry`，所以
YAML 只写方法名和固定 kwargs。bounds 文件中同名 `geometry` 项限制相对于冻结
`GeometryReference` 的增量；它不是绝对坐标。参考基线、采样位置、变换方法和 bounds
四者必须一起解释。

初始化时会先把每个启用项解析为一个未执行的调用，并依次检查：配置结构、真实 fault
对象上的注册方法、Python kwargs 签名、方法声明的 reference 需求，以及采样切片长度。
这个 preflight 不执行扰动、不生成 mesh，也不改 reference。固定参数方法要求精确长度；
带 `fixed_nodes` 的方法允许一个广播标量，或按冻结 reference 中的可移动节点/倾角控制点
数量给值。全部节点固定时，空切片仍是合法 no-op；新扩展若尚未声明结构化契约，则保留
方法自身的运行时检查，不会被核心错误锁成固定参数个数。

构造 `FULLSMC` 或 `SMC_FJ` target 时还会执行一次 mesh replay 就绪检查。只有
`generate_and_deform_mesh` 这类声明了固定拓扑参数映射依赖的方法进入该检查；简单 mesh、
多层 mesh 和直接 whole-mesh 变换不会被错误要求提供 `param_coords`。检查确认映射来自一次
完整准备、顶点数和 face-row connectivity 仍与当前 mesh 对齐，并核对候选重放实际使用的
`num_segments`、`disct_z` 等映射参数。它不生成或变形 mesh，也不进入每个候选的执行循环；
MPI 下每个 rank 只在本地 target 构造时检查自己的状态，跟随断层不重复检查主断层映射。

无 mesh 后缀的方法需要独立 `update_mesh`；方法名含 `_simpleMesh`、`_DeformMesh` 或
`_multiLayerMesh` 时，方法内部已负责 mesh。完整配置步骤见
[联合 Bayesian workflow](../workflows/05_joint_bayesian_geometry_slip.md)，冻结参考和
`ref*` 接口见 [可扰动断层几何](geometry_perturbation.md)。

## 约束适用性

| 约束 | `SMC_FJ + ss_ds` | `FULLSMC` |
| --- | :---: | :---: |
| `strikeslip/dipslip` 边界 | 是 | 是 |
| `rake_angle` 线性角度约束 | 是 | 否 |
| `slip_magnitude/rake_angle` 采样边界 | 否 | 视 `slip_sampling_mode` 而定 |
| Euler 约束 | 是 | 否 |
| 零滑、边界零滑 | 是 | 否 |
| 自定义 `A @ x <= b` / `A @ x = b` | 是 | 否 |
| 震级先验 | 否 | 是 |

完整写法和模式守卫见 [ECAT 约束管理器 / Constraint Manager](constraint_manager.md)。

同一份 `bounds_config.yml` 和可选的 `interseismic_config.yml` 可以用于两种
Bayesian 模式，不需要为 `FULLSMC` 复制第二份配置。管理器会始终解析并应用
当前参数化可用的 bounds；只属于线性子问题的 rake sector、Euler、zero-slip、
source matrix 约束在 `FULLSMC` 中记为 inactive，不会悄悄转换为另一种
约束。可用以下轻量检查确认实际模式和 inactive 项：

```python
snapshot = inversion.get_constraint_snapshot()
print(snapshot["sampling_mode"])
print(snapshot["inactive_constraints"])
```

`slip_magnitude` 是物理幅值，lower bound 必须非负。若 `rake_fixed` 或
`magnitude_rake` 未给出 magnitude lower bound，`FULLSMC` 默认使用 `0`；显式
给出负下界会在初始化时被拒绝。

`walk(..., magprior=False)` 默认从 bounds 生成普通初始样本，保持常规 FULLSMC
入口简洁。`magprior=True` 只启用 magnitude-aware **初始样本**；`ss_ds` 模式下
还需显式提供 `rake_angle`、`rake_sigma` 和 `rake_range`。是否把震级项加入后验由
`magposteriors` 单独控制，不应把初始化策略当作后验约束。

## 初始化和失败语义

约束初始化是原子操作：bounds、rake、震间约束和 source constraints 全部解析、
组装并验证后才提交。任一步失败都会恢复初始化前状态，不会留下“部分约束已生效”
的对象。

| 情况 | 行为 |
| --- | --- |
| 未提供 bounds 文件，或默认可选文件不存在 | 保持无该文件的原有用法，继续初始化 |
| 文件存在但 YAML、selector、bounds 或矩阵无效 | 立即报错，不开始采样 |
| 配置了当前模式不消费的线性约束 | 保留配置并在 `inactive_constraints` 中列出 |

`SMC_FJ` 的每个样本都必须在完整的 bounds、等式和不等式约束下完成线性求解。
若求解器报错或返回非成功状态，该样本会清空临时线性解并获得低似然；ECAT 不会
删除等式约束后重试，也不会复用上一样本的 `mpost`。首次失败只在 rank 0 给出一条
紧凑告警。反复出现时应在采样前检查：

```python
snapshot = inversion.get_constraint_snapshot(validate=True)
print(snapshot["validation"])
```

## 配置检查顺序

配置联合 Bayesian 案例时，建议按这个顺序检查：

1. 数据对象顺序和 `geodata` 配置一致。
2. `bayesian_sampling_mode` 和 `slip_sampling_mode` 与科学问题一致。
3. 每个可扰动断层已建立 `geometry_ref`，且参考字段满足所选方法。
4. 顶层 `nonlinear_inversion` 和断层级 `geometry.update` 都已启用。
5. `sample_positions` 从 0 连续覆盖，并与方法期望的扰动参数个数一致。
6. `update_fault_geometry.method` 是该断层对象当前可发现的方法。
7. geometry bounds 的顺序、单位和参考零点已记录。
8. 扰动方法与独立/内置 mesh 更新职责没有重复或遗漏。
9. GF 后端和网格更新成本在计算预算内。
10. sigma/alpha 的 `mode`、初值、边界和数据组顺序一致。
11. 约束类型与当前采样模式匹配。

前 3--8 项属于无副作用的 geometry preflight。成功只表示配置调用能够被一致解释，不表示
极值候选的 mesh 一定具有满意质量；后者仍应通过少量代表候选的几何、GF、Laplacian 和
残差检查确认。preflight 在各 MPI rank 使用相同输入独立执行，不在候选循环内增加通信。

## 结果报告

<a id="sample-file-overwrite"></a>

### 样本文件和覆盖语义

`walk(..., filename=...)` 在 rank 0 把最终 sampler 写入指定 HDF5；底层 SMC 在启用最终
保存时还会在当前工作目录写 `samples_final.h5`，启用间隔保存时还可能写 stage 文件。
这些入口使用 HDF5 写入模式创建文件，同名文件会被替换，不表示从旧样本断点续跑。

公共联合模板显式使用 `save_at_interval=False`，但仍应同时管理 `--samples` 指向的文件和
当前目录的 `samples_final.h5`。正式重跑前应选择新的文件名或输出目录，或者先归档旧的
HDF5、日志和图件。

进程零退出码和文件存在只说明任务完成了最低限度的写出，不能替代 posterior、边界占用、
残差、sigma/alpha、约束和几何一致性检查。

### 激活代表模型后再统计

联合 Bayesian 采样保留的是后验样本集合。`mean`、`median` 和 `MAP` 是不同的
代表模型；统计之前必须先用 `returnModel()` 把目标代表模型写入当前
`mpost`、断层滑动和 poly 字段：

```python
inversion.returnModel(model="MAP", print_fit_statistics=False)
rows = inversion.collect_fit_statistics(
    model="MAP",
    data_poly="config",
    include_dataset=True,
    include_global=True,
    include_weighted=True,
)
df = inversion.fit_statistics_to_dataframe(rows)
```

`collect_fit_statistics(model="MAP")` 中的 `model` 只作为结果表标签，不会自行
从后验样本中选择 MAP。若要比较 MAP、median 和 mean，应对每一种模型分别调用
`returnModel()`，随后立即收集对应统计。完整公式和输出字段见
[Fit Statistics](fit_statistics.md)。

对非 `std` 代表模型，`returnModel()` 还会按 likelihood 使用的同一
single/individual/grouped 索引展开每个数据集的物理 sigma。因而
`include_weighted=True` 的 `Qw` 与 `wRMS` 使用的正是该代表模型的完整协方差度量；
它不会重新解释后验样本或额外计算一次候选似然。联合 Bayesian 不填造 VCE 式有效
自由度，所以 reduced 列保持为空。

激活后可从 `current_data_sigmas` 读取按数据集命名的完整物理 sigma，从
`current_smoothing_alphas` 读取按可平滑 source 命名的完整物理 alpha；固定组也包含在内。
`current_alpha_group_values` 则保留共享组层级。配置初值只能从
`config.sigmas["initial_value"]` 和 `config.alpha["initial_value"]` 读取或修改；已停用的
`sigmas` / `alpha` 快捷属性不会在配置态和结果态之间猜测。三类结果均已处理
`log_scaled`，不要再次应用 `10**`。

### 标准结果入口与脚本层导出

`extract_and_plot_bayesian_results(...)` 在 rank 0 执行完整的代表模型激活路径：读取样本、
调用 `returnModel()`、回填后验几何与滑动、按配置重建 synthetic，并按开关生成标准滑动图、
数据拟合图和 sigma/alpha KDE。它不导出四边、patch、滑动中心、滑动方向或 InSAR
data/synth/resid 文本，也不自动绘制几何改正图。

若脚本需要 geometry、sigma 和 alpha 的同一张联合 KDE，标准调用应关闭高层入口自带的
KDE，再显式调用一次：

```python
inversion.extract_and_plot_bayesian_results(
    rank=rank, filename=str(samples_file), model="median",
    plot_faults=True, plot_data=True,
    plot_sigmas=False,  # 关闭内置 KDE，避免与下面的联合 KDE 重复。
    data_poly="config",  # 沿用每个数据集解析后的 poly 设置。
    fault_outdir="output", data_outdir="Modeling", show=False,
)

if rank == 0:
    inversion.plot_kde_matrix(
        plot_geometry=True, plot_sigmas=inversion.sigmas_position is not None,
        plot_alpha=inversion.alpha_position is not None,
        fill=True, scatter=False, save=True, show=False,
        filename="output/kde_geometry_sigmas_alpha.png",
    )
```

`axis_labels=None` 使用内部顺序和默认标签。若手工提供标签，数量和顺序必须严格对应
geometry slice、被更新的 sigma、被更新的 alpha；更改分组或更新开关后也要同步修改。

代表模型激活后，公共模板再调用 CSI 现有的 `writeFourEdges2File()`、
`writePatches2File()`、`writeSlipCenter2File()`、`writeSlipDirection2File()`。InSAR 文本通过
`writeDecim2file(..., triangular=None)` 导出，`data`、`synth` 和 `resid` 均是该接口的
有效选择；`None` 会按 corner 形状自动判断三角形、完整四边形或旧式对角格式。synthetic
本身由上面的高层入口依据 `verticals`/`polys` 生成，脚本不再按 SAR/opticorr 类型自行
复制一套正演逻辑。

`plot_faults_geometry_correction(...)` 比较当前代表几何与 `geometry_ref`，因此必须放在代表
模型激活之后。它的 `filename` 是图件路径，`output_dir` 是参考/改正边界文本目录，两者不是
同一个参数。

报告联合 Bayesian 结果时，至少说明：

- `bayesian_sampling_mode` 和 `slip_sampling_mode`。
- 使用的断层对象类型和几何扰动方法。
- 几何扰动参数、边界、单位和是否使用坐标加密。
- 网格策略、GF 方法和是否随样本重建 GF。
- 数据协方差、sigma/alpha 配置和先验范围。
- 约束类型，以及哪些约束只在 `SMC_FJ` 中生效。
- 几何后验、滑动后验和数据残差诊断。

## 相关页面

- [Bayesian 联合几何-滑动分布反演 / Joint Bayesian Geometry-Slip](../workflows/05_joint_bayesian_geometry_slip.md)
- [Bayesian 联合反演中的可扰动断层几何 / Perturbable Fault Geometry](geometry_perturbation.md)
- [Bayesian 联合反演中的几何参考](../concepts/bayesian_geometry_reference.md)
- [联合 Bayesian 几何参考与配置短例](../examples/joint_bayesian_geometry_setup.md)
- [ECAT 约束管理器 / Constraint Manager](constraint_manager.md)
- [Sigmas 与 Alpha 配置模式 / Sigmas and Alpha](sigmas_alpha.md)
- [Fit Statistics](fit_statistics.md)
