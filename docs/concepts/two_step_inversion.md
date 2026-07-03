# 标准两步走反演逻辑

ECAT 推荐把常规同震反演分成两步：

1. **Bayesian 非线性几何反演**：用紧凑源估计断层顶边中点经纬度和深度、走向、倾角、长度、宽度、平均滑动或震级代理量。
2. **BLSE/VCE 线性滑动分布反演**：固定优选几何和网格，求解每个 patch 的分布式滑动。

## 为什么分两步

断层几何和分布式滑动都进入同一个非线性后验时，参数维度高、Green's functions 需要频繁更新，调试成本也高。两步走先把几何搜索限制在紧凑源参数上，再把优选几何交给线性滑动求解，可以让数据读取、几何构建、网格、约束、权重和结果诊断逐层检查。

这不是说联合 Bayesian 几何-滑动反演不重要，而是它应放在标准两步走跑通之后，用于传播几何不确定性到滑动分布。

## 每一步输出什么

| 阶段 | 主要输出 | 不是这个阶段的目标 |
| --- | --- | --- |
| 非线性几何 | 几何后验、优选紧凑源、sigma/alpha 后验、拟合诊断 | 最终分布式滑动模型 |
| 线性滑动 | patch slip、模型预测、残差、Mw、VCE 或平滑权重诊断 | 重新搜索几何参数 |

## 关键约定

- 非线性几何结果中的 `lon/lat/depth` 指**断层顶边中点**。
- 进入线性阶段时，`top/depth` 是滑动面网格的顶部和底部深度。
- `BLSE/VCE` 是线性滑动分布反演方法，不是非线性几何反演的替代品。
- 高级联合 Bayesian 应先有可复现的两步走基准。

## 继续阅读

- [标准两步走路线](../getting_started/quickstart_two_step.md)
- [Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md)
- [BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md)
- [Bayesian 联合几何-滑动分布反演](../workflows/05_joint_bayesian_geometry_slip.md)
