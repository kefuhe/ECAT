# 案例选择表

本页帮助用户从 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 中选择合适的学习入口。入门优先选择已经有脚本对照和 workflow 交叉引用的案例；其他案例可作为相同方法的扩展参考。

## 推荐阅读顺序

| 顺序 | 案例 | 适合学习的问题 | 先读 |
| --- | --- | --- | --- |
| 1 | InSAR Downsampling / Menyuan, Wushi GeoTIFF | 原始 GAMMA/GeoTIFF 如何读入、协方差估计和降采样 | [InSAR 降采样案例](insar_downsampling_gamma_geotiff.md), [InSAR 降采样两步走](../workflows/02a_insar_downsampling_two_step.md) |
| 2 | Wushi 2024 Mw7.0 | InSAR-only Bayesian 非线性几何反演 | [Wushi：InSAR-only 非线性几何反演](wushi_nonlinear_geometry.md) |
| 3 | Ridgecrest 2019 Mw6.4/Mw7.1 | GPS+InSAR、多事件覆盖关系、`geodata.faults` | [Ridgecrest：GPS+InSAR 非线性几何反演](ridgecrest_gps_insar.md) |
| 4 | Dingri 2020 Mw5.6 | 固定几何后的 BLSE、smoothing loop 和结果导出 | [Dingri 2020：BLSE/VCE 线性滑动反演](dingri_blse_vce.md) |

## ECAT-Cases 目录导向

| ECAT-Cases 目录 | 当前定位 | 学习价值 |
| --- | --- | --- |
| [`InSAR_Downsampling`](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling) | 降采样方法案例 | GAMMA/GeoTIFF 读入、`covarSAR-Step1.py` 与 `downsampleSAR-Step2_*.py` 的旧脚本对照 |
| [`Cases/Wushi_20240122M7_0`](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Wushi_20240122M7_0) | 入门非线性几何案例 | 两条 InSAR 轨道、`explorefault`、几何后验和 sigma 后验 |
| [`Cases/Ridgecrest_20190706Mw7_1`](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Ridgecrest_20190706Mw7_1) | 多数据、多事件非线性案例 | GPS 与 InSAR 混合，部分数据覆盖前震或主震，部分数据覆盖累计形变 |
| [`Cases/Dingri_Events/Dingri_20200320Mw5_6`](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Dingri_Events/Dingri_20200320Mw5_6) | 入门线性滑动案例 | `default_config.yml`、`bounds_config.yml`、BLSE 固定平滑、smoothing loop |
| [`Cases/Hotan_20200625M6_3`](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Hotan_20200625M6_3) | 单事件非线性几何扩展案例 | 和 Wushi 类似，可用于练习不同事件的配置改写 |
| [`Cases/Iran_20170405M6_1`](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Iran_20170405M6_1) | 单事件非线性几何扩展案例 | 反冲/逆冲机制下的几何先验、rake 和数据拟合检查 |
| [`Cases/Western_Xizang_20200722M6_3`](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Western_Xizang_20200722M6_3) | 多断层非线性几何扩展案例 | 多断层 alias、几何后验图和模型摘要 |
| [`Cases/Taiwan_20240405Mw7_4`](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Taiwan_20240405Mw7_4) | 多源数据整理参考 | GPS 转换、外部 InSAR/矢量产品和非线性几何配置 |
| [`Cases/Dingri_Events`](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Dingri_Events) | 同一区域多事件研究材料 | Dingri 2015/2020/2025 的数据组织、滑动结果和事件对比 |
| [`Cases/Sagaing_20250328Mw7_8`](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Sagaing_20250328Mw7_8) | 大震滑动结果参考 | 已整理的滑动分布输出，可用于结果格式和图件组织参考 |

## 选择建议

- 第一次跑通：先用 Wushi 非线性几何和 Dingri 2020 线性滑动。
- 想理解 InSAR 原始产品到反演输入：先看 `InSAR_Downsampling`，再回到 Wushi 或 Dingri。
- 想处理 GPS+InSAR 或多事件：看 Ridgecrest，重点检查 `geodata` 顺序和 `geodata.faults`。
- 想复用到新事件：从最相近的案例复制目录结构，再用 CLI 生成模板对照修改，不要直接套用模板默认参数。
