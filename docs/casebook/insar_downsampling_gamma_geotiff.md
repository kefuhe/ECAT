# InSAR 降采样案例

本文把降采样工作流连接到案例材料。

## GitHub 位置

- [ECAT-Cases / InSAR_Downsampling / GAMMA](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling/GAMMA)
- [ECAT-Cases / InSAR_Downsampling / GeoTiff](https://github.com/kefuhe/ECAT-Cases/tree/main/InSAR_Downsampling/GeoTiff)

## 本页范围

这些案例用于讲解：

- GAMMA 二进制产品读取
- GeoTIFF 产品读取
- quick-look 图检查
- 协方差估计
- `std` 降采样
- 输出为 CSI `.txt/.rsp/.cov`

## 文件来源与生成方式

[ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 提供示例原始数据、旧脚本和已整理的输出材料。新建降采样目录时，优先用 ECAT CLI 在当前目录生成 `downsample.yml` 模板：

```bash
ecat-generate-downsample \
  --mode sar \
  --sar-reader gamma_tiff \
  --sar-mode unwrapped_phase \
  --downsample-method std \
  -o downsample.yml
```

生成后需要按数据产品修改 `sar_config.files`、`general.origin`、`covar.mask_out`、`downsample.method` 和输出前缀。命令细节见 [CLI 命令参考](../reference/cli.md)，reader 和 mode 的含义见 [SAR Reader 参考](../reference/sar_reader.md)。

## 推荐做法

先选一个较小的 GeoTIFF 或 GAMMA 案例，用 `ecat-generate-downsample` 和 `ecat-downsample` 组织教程。旧脚本如 `downsampleSAR-Step2.py` 可以作为参考，但文档优先讲维护中的 CLI 路线。CLI 参数说明见 [CLI 命令参考](../reference/cli.md)。

如果希望按旧案例的 `covarSAR-Step1.py` / `downsampleSAR-Step2_*.py` 思路教学，先看 [InSAR 降采样两步走](../workflows/02a_insar_downsampling_two_step.md)。那一页把 Step 1 协方差准备、Step 2 正式降采样、`NoFault` 与 `WithFault` 的区别对应到了当前 YAML 配置和可复制脚本。

## 案例运行顺序

教程要把三步拆开讲，不要只给最终命令：

```bash
# 1. -s: show raw data
# 读入原始产品，检查文件、单位、正负号、投影范围和 quick-look 图。
ecat-downsample -f downsample.yml -s

# 2. -c: do covariance
# 根据 covar.mask_out 排除主形变区，估计 Covariance_estimator*.cov。
ecat-downsample -f downsample.yml -c

# 3. -d: do downsample
# 按 downsample.method 正式降采样，写出 CSI .txt/.rsp/.cov。
ecat-downsample -f downsample.yml -d
```

检查顺序也按这三步来：先确认原始图，再确认 `mask_out` 和协方差估计，最后看降采样单元是否保留近场梯度、远场是否不过密。

## 旧脚本到新 CLI 的映射

旧脚本核心逻辑：

```python
downsampler.stdBased(threshold=0.02, plot=False, verboseLevel="minimum")
downsampler.writeDownsampled2File(prefix=outName + "_ifg", rsp=True)
```

CLI 路线：

```bash
ecat-generate-downsample --mode sar --sar-reader gamma_tiff --sar-mode unwrapped_phase -o downsample.yml
ecat-downsample -f downsample.yml -s
ecat-downsample -f downsample.yml -c
ecat-downsample -f downsample.yml -d
```

对应关系：

| 旧脚本或人工步骤 | 新 CLI 阶段 | 作用 |
| --- | --- | --- |
| 读取 GAMMA/GeoTIFF 并画原始图 | `ecat-downsample -f downsample.yml -s` | 先确认读入没有问题，不产生最终反演文件 |
| 估计经验协方差 | `ecat-downsample -f downsample.yml -c` | 输出 `Covariance_estimator*.cov`，给后续降采样协方差矩阵使用 |
| `stdBased(...)` 和 `writeDownsampled2File(...)` | `ecat-downsample -f downsample.yml -d` | 生成 `<outName>_ifg.txt/.rsp/.cov`，作为非线性几何反演和 BLSE/VCE 的输入 |

旧脚本的两步走命名可直接对应到当前命令：`covarSAR-Step1.py` 对应 `-c`，`downsampleSAR-Step2_NoFault.py` 对应基础 `method: std`。`downsampleSAR-Step2_WithFault.py` 中的断层迹线可放入 `faults:` 用于叠加显示或 `trirb`，基础 `std` 的近场细化用面状 `focus_region` 表达；若要完全复现旧脚本里的 `reject_pixels_fault(...)`，需要用 `--copy-script` 复制处理脚本后手动加入。字段级对照见 [InSAR 降采样两步走](../workflows/02a_insar_downsampling_two_step.md#2-step-2-正式降采样)。

## 应保留输出

```text
raw quick-look figure
downsampled check figure
Covariance_estimator.cov
<outName>_ifg.txt
<outName>_ifg.rsp
<outName>_ifg.cov
<outName>_run_metadata.yml
```
