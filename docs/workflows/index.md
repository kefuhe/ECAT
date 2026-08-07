# Workflows / 科研工作流

本目录按科研任务组织页面。第一次使用先读
[安装与环境检查](../getting_started/installation.md) 和
[标准两步走路线](../getting_started/quickstart_two_step.md)；已经明确任务时，直接从下表进入对应 workflow。

## 标准主线

```text
01 数据读取
    └─ 需要时执行 02 降采样
             ├─ 02a Step1/Step2 调参
             ├─ 02b 自定义 reader adapter
             └─ 02c 交互调整断层迹线（可选）
          ↓
03 Bayesian 非线性几何反演
          ↓
04 BLSE/VCE 线性滑动分布反演
    ├─ 04a 固定几何平滑搜索（可选）
    ├─ 04b 固定拓扑倾角搜索（可选）
    └─ 04c 倾角 × 平滑敏感性（高级可选）
```

非线性几何阶段估计的经纬度和深度表示**断层顶边中点**。线性滑动阶段固定优选几何，再求解分布式滑动。

## 选择页面

| 你的输入或目标 | Workflow | 完成后应得到 |
| --- | --- | --- |
| 已有 InSAR/GNSS 点数据，或需要确认读入语义 | [01 InSAR 与 GNSS 数据读取](01_data_reading_insar_gps.md) | 单位、正负号、投影和协方差都明确的数据对象 |
| 有 GAMMA、GeoTIFF、GMTSAR、HyP3 或 offset 产品，需要生成反演点数据 | [02 InSAR 降采样](02_insar_downsampling.md) | 可供 CSI/ECAT 读取的降采样数据和协方差 |
| 需要复现旧两步脚本，或对标准降采样分阶段检查和调参 | [02a Step1/Step2 调参](02a_insar_downsampling_two_step.md) | 旧代码到 YAML/CLI 的映射、经检查的采样网格和最终输出 |
| 标准 reader 不适用，或需要复用时序数据的采样网格 | [02b Adapter 降采样](02b_adapter_downsampling.md) | 保持统一下游接口的自定义读入结果 |
| 需要在原始或改正后观测上调整已有断层迹线 | [02c 交互调整断层迹线](02c_interactive_trace_editing.md) | 不改观测和参考文件的两列 `lon lat` 新迹线 |
| 用紧凑源估计断层几何 | [03 Bayesian 非线性几何反演](03_nonlinear_geometry_bayesian.md) | 优选几何、不确定性和拟合诊断 |
| 已有非线性结果、地表迹线+倾角或其他固定几何，需要求分布式滑动 | [04 BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md) | 固定几何上的滑动分布、残差和结果报告 |
| 固定几何已检查，需要选择 BLSE 平滑强度 | [04a BLSE 固定几何平滑参数搜索](04a_blse_smoothing_search.md) | penalty–RMS/VR、L-curve 和逐数据集拟合表 |
| 迹线等几何已定，需要比较一组 BLSE 倾角候选 | [04b BLSE 固定拓扑倾角搜索](04b_blse_dip_search.md) | patch 身份一致的倾角—拟合统计表和诊断图 |
| 已完成平滑和倾角搜索，需要检查二者耦合 | [04c 倾角 × 平滑参数敏感性分析](04c_blse_dip_smoothing_search.md) | dip–penalty 二维诊断和分倾角 L-curve |
| 已完成并检查标准两步走，明确需要联合后验 | [05 Bayesian 联合几何-滑动分布反演](05_joint_bayesian_geometry_slip.md) | 几何、滑动和噪声参数的联合后验 |
| 需要在 Google Earth Pro 中叠加观测、降采样单元、断层、滑动或小震 | [06 Google Earth 科研导出](06_google_earth_export.md) | 带单位、正号和显示配置记录的 KMZ 显示副本 |
| 需要直接查看内置断层/GNSS，或在本地浏览器中叠加自己的观测和降采样单元 | [07 本地科研地图查看](07_research_map_viewer.md) | 不改科学源的本地交互研究地图 |

## 页面怎么配合

每个 workflow 负责说明输入来源、运行入口、预期输出、检查项和下一步。遇到具体字段或接口时再打开
[Reference Map](../reference/index.md)；只需要一个短片段时看
[Examples](../examples/index.md)；需要对照真实事件时看
[Casebook](../casebook/index.md)。

不要把 `reference/` 当作必须顺序读完的教程。长 reference 页面开头统一提供“阅读路径”，可按当前问题跳到对应小节。
