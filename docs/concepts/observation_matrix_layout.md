# 观测向量、协方差与设计矩阵排列合同

本页解释 ECAT/CSI 中 `d`、预测向量、Green 函数、数据改正设计矩阵和 `Cd`
为什么必须使用同一个观测行顺序。它是数据读取、BLSE/VCE、Bayesian 似然和
frame/ramp 改正共同遵守的基础合同。

## 一条核心不变量

对第 (k) 个数据集，线性滑动模型可写为：

\[
d_k=G_km+H_kc+\varepsilon_k,
\qquad
\varepsilon_k\sim\mathcal N(0,C_{d,k}).
\]

非线性几何反演把 (G_km) 换成候选几何产生的预测 (g_k(\theta))，排列合同不变。

> 对任意观测行 (i)，`d[i]`、预测向量的第 (i) 项、`G[i, :]`、
> `H[i, :]` 以及 `Cd[i, :]`/`Cd[:, i]` 必须描述同一个标量观测。

如果其中一个对象单独改变顺序，即使数组形状仍然匹配，反演也会把一个位置或分量的
残差交给另一个位置或分量的协方差权重。这属于计算错误，不是显示顺序差异。

## 对象、形状和两类空间

| 对象 | 典型形状 | 行表示什么 | 列表示什么 |
| --- | --- | --- | --- |
| 观测向量 (d_k) | ((n_k,)) | 第 (i) 个标量观测 | 无 |
| 预测向量 (\mu_k) | ((n_k,)) | 与 `d_k[i]` 对应的模拟值 | 无 |
| Green 函数 (G_k) | ((n_k,p)) | 观测空间 | 滑动或其他模型参数空间 |
| 数据改正矩阵 (H_k) | ((n_k,q)) | 观测空间 | offset、ramp 或 frame 参数空间 |
| 协方差 (C_{d,k}) | ((n_k,n_k)) | 观测空间 | 同一观测空间 |
| 左白化矩阵 (W_k) | ((n_k,n_k)) | 白化后的观测空间 | 原观测空间 |

这里有一个重要区分：

- **观测行布局**决定 `d / prediction / G / H / Cd` 怎样彼此对齐；
- **参数列布局**决定 `G` 的模型参数和 `H` 的改正参数怎样解释。

白化只左乘观测行：

\[
\widetilde d_k=W_kd_k,
\qquad
\widetilde G_k=W_kG_k,
\qquad
\widetilde H_k=W_kH_k.
\]

它不改变 `G` 或 `H` 的参数列顺序。

## 各数据类型的观测行顺序

| 数据类型 | 反演向量顺序 |
| --- | --- |
| InSAR、leveling 等单标量数据 | 数据对象当前的一维观测顺序 |
| Optical offset | `East(all samples), North(all samples)` |
| GPS/GNSS 水平分量 | `East(all stations), North(all stations)` |
| GPS/GNSS 完整分量 | `East(all stations), North(all stations), Up(all stations)` |

GPS 对象本身仍以 `(n_stations, 3)` 保存 `vel_enu` 和 `synth`。这个二维存储形状不是
反演向量顺序；进入求解或似然前需要按分量优先展开。用户不应手工调用普通
`vel_enu.flatten()`，因为它会得到站点优先顺序：

```text
E1, N1, U1, E2, N2, U2, ...
```

而 CSI 的 GPS `d/G/Cd` 和 frame-transform estimator 使用：

```text
E1, E2, ..., EN, N1, N2, ..., NN, [U1, U2, ..., UN]
```

ECAT 的非线性入口通过统一适配层完成这一转换和模拟值写回。

## 数据改正矩阵怎样参与预测

数据改正项不是先修改 `Cd`，也不是替换原观测。对当前候选或线性模型：

\[
\mu_{k,\mathrm{total}}
=\mu_{k,\mathrm{fault}}+H_kc_k,
\qquad
r_k=\mu_{k,\mathrm{total}}-d_k.
\]

因此 (H_k) 的行必须与 `d_k` 完全一致，列则按公开参数名解释。例如两个 GPS 站点的
ENU translation 为：

\[
H=
\begin{bmatrix}
1&0&0\\
1&0&0\\
0&1&0\\
0&1&0\\
0&0&1\\
0&0&1
\end{bmatrix},
\qquad
c=[t_E,t_N,t_U]^\mathsf T.
\]

所以 (Hc=[t_E,t_E,t_N,t_N,t_U,t_U]^\mathsf T)，与 GPS 分量优先观测行一一对应。
只使用水平分量时，输出为 ((2N,2))，只保留 East/North 行与对应平移列。

## 如果确实需要重新排列

令 (Q) 为同一个置换矩阵。合法重排必须同时应用到所有观测空间对象：

\[
d'=Qd,
\quad
\mu'=Q\mu,
\quad
G'=QG,
\quad
H'=QH,
\quad
C_d'=QC_dQ^\mathsf T.
\]

随后应从新的 (C_d') 重新建立白化度量。只重排 `d`、只重排 `H`，或者只交换
`Cd` 的一条轴，都会改变科学问题。

## 多数据集怎样拼接

多个数据集先保持各自内部合同，再按规范化后的数据集顺序组成观测块：

\[
d=\begin{bmatrix}d_1\\d_2\\\vdots\\d_K\end{bmatrix},
\qquad
C_d=\operatorname{blockdiag}(C_{d,1},\ldots,C_{d,K}).
\]

同一数据集的 `dataname`、观测切片、GF 行、`Cd` 块、sigma 映射和输出标签必须指向
同一个块。当前独立数据集模型不表示跨数据集非零协方差。

## 用户检查清单

```python
n = len(d)
assert prediction.shape == (n,)
assert G.shape[0] == n
assert H.shape[0] == n          # 启用数据改正时
assert Cd.shape == (n, n)
```

这些检查只能证明形状一致，不能证明语义顺序正确。还应检查：

- GPS/optical 是否按分量块排列，而不是普通二维数组 `flatten()` 顺序；
- `buildCd(direction=...)` 是否与反演启用的 EN/ENU 分量一致；
- 删除或筛选观测后，`d/G/H/Cd` 是否由同一数据对象重新建立；
- 替换 `Cd` 后是否重新建立白化度量；
- 多数据集的输入顺序、sigma 分组和数据名称是否一致。

协方差怎样变成白化度量见 [BLSE/VCE](../reference/blse_vce.md)；offset、ramp 和 GPS
frame 参数的列含义见 [数据改正项与 Frame Transform](../reference/data_corrections.md)；
非线性几何入口见 [非线性几何配置](../reference/config_nonlinear_geometry.md)。
