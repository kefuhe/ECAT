# 标准两步走路线

ECAT 教程按两步走路线组织反演阶段：

1. **Bayesian 非线性几何反演**：估计断层几何。
2. **BLSE/VCE 线性滑动分布反演**：固定几何，求解分布式滑动。

这里的“两步走”指反演阶段。完整上手顺序是：安装环境，确认 InSAR/GPS 数据读取，必要时做 InSAR 降采样，然后进入非线性几何反演和线性滑动分布反演。

## 完整上手顺序

| 顺序 | 要做的事 | 说明 |
| --- | --- | --- |
| 0 | [安装与环境检查](installation.md) | 确认 `eqtools`、`csi` 和 CLI 命令可用。 |
| 1 | [InSAR 与 GPS 数据读取](../workflows/01_data_reading_insar_gps.md) | 明确数据格式、单位、LOS 投影、误差和 `geodata` 顺序。 |
| 2 | [InSAR 降采样](../workflows/02_insar_downsampling.md) | 原始 SAR/offset 产品先转成 CSI `.txt/.rsp/.cov` 前缀；手动调参可按 [降采样两步走](../workflows/02a_insar_downsampling_two_step.md)。非标准读入或时序复用网格看 [自定义读入 Adapter 降采样](../workflows/02b_adapter_downsampling.md)。已有点位数据可跳过。 |
| 3 | [Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md) | 估计顶边中点位置、走向、倾角、长度、宽度等几何参数。 |
| 4 | [BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md) | 固定优选几何，反演分布式滑动并做权重诊断。 |

## 第一步：Bayesian 非线性几何反演

这一步用于估计紧凑断层模型参数。这里的经纬度和深度指**断层顶边中点**，不是断层面几何中心：

- 顶边中点经纬度
- 顶边中点深度
- 走向
- 倾角
- 长度
- 宽度
- 平均滑动量或震级代理量
- rake

这一步的输出不是最终分布式滑动模型，而是优选几何或若干候选几何。

配置文件可以先在当前反演目录生成模板：

```bash
ecat-generate-nonlinear -o default_config.yml
```

然后参照案例修改参数。`bounds` 中的 `Uniform` 目前写作 `[Uniform, start, range]`，实际上界是 `start + range`；它不是 `[下界, 上界]`。更完整的配置说明见 [非线性几何反演配置](../reference/config_nonlinear_geometry.md)。

推荐案例：

- [Wushi 非线性几何反演](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Wushi_20240122M7_0/Nonlinear)
- [Ridgecrest GPS+InSAR 非线性几何反演](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Ridgecrest_20190706Mw7_1/Nonlinear)

## 第二步：BLSE/VCE 线性滑动分布反演

选择几何后，建立固定断层网格，反演各子断层滑动：

- 先用固定平滑参数跑一个普通 BLSE 模型，确认数据、几何、边界和输出链条都正确。
- 再用 VCE 估计数据方差分量和平滑权重，或用 L-curve / smoothing loop 做权重诊断。
- BLSE 提供边界约束、rake 约束、平滑约束等约束最小二乘能力。

推荐案例：

- [Dingri 2020 BLSE/VCE 线性滑动反演](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Dingri_Events/Dingri_20200320Mw5_6/LinearInv)

## 文件从哪里来

可运行脚本、案例配置、输入数据和参考输出放在 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases)。这些文件是已经按具体事件整理过的案例材料。

新建自己的反演目录时，标准配置文件可以先由 ECAT CLI 在当前目录生成，再参照案例修改：

```bash
# InSAR/optical 降采样配置
ecat-generate-downsample --mode sar --sar-reader gamma --sar-mode unwrapped_phase -o downsample.yml

# 非线性几何反演主配置
ecat-generate-nonlinear -o default_config.yml

# 线性 BLSE/VCE 主配置和边界配置
ecat-generate-config -o default_config.yml --gf-method cutde
ecat-generate-boundary -o bounds_config.yml -f MyFault
```

CLI 生成的是模板，不是最终科学配置。需要继续修改数据路径、断层名称、几何边界、sigma/alpha、rake/Euler 约束和输出设置。命令细节见 [CLI 命令参考](../reference/cli.md)，配置字段见 [非线性几何反演配置](../reference/config_nonlinear_geometry.md) 和 [线性滑动反演配置](../reference/config_linear_slip.md)。

## 关键注意事项

- 非线性几何结果中的 `lon/lat/depth` 表示断层顶边中点三维坐标；进入线性阶段时对应 `clon/clat/cdepth`，不要和线性滑动面扩展后的 `top/depth` 混写。参见 [Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md) 和 [BLSE/VCE 线性滑动分布反演](../workflows/04_linear_slip_blse_vce.md)。
- InSAR 降采样输出的标准前缀用于 `read_from_varres(...)` 读入；`-s/-c/-d` 的含义见 [InSAR 降采样](../workflows/02_insar_downsampling.md)，按案例 Step1/Step2 手动调参见 [InSAR 降采样两步走](../workflows/02a_insar_downsampling_two_step.md)，自定义读入见 [自定义读入 Adapter 降采样](../workflows/02b_adapter_downsampling.md)。
- `sigmas` 和 `alpha` 的 `single/individual/grouped` 模式影响数据权重和平滑权重，参见 [Sigmas 与 Alpha 配置模式](../reference/sigmas_alpha.md)。
- BLSE/VCE 的边界、rake、Euler 和自定义线性约束由约束管理器统一处理，参见 [ECAT 约束管理器](../reference/constraint_manager.md)。

## 最小脚本骨架

非线性几何反演：

```python
from eqtools.csiExtend.exploremultifaults_smc import explorefault

expfault = explorefault(
    "geometry_search",
    lon0=lon0,
    lat0=lat0,
    config_file="default_config.yml",
    geodata=geodata,
)
expfault.setPriors(bounds=None, initialSample=None, datas=None)
expfault.setLikelihood(datas=None, verticals=None)
expfault.walk(nchains=100, chain_length=50, comm=comm, filename="samples_geometry.h5")
expfault.extract_and_plot_bayesian_results(filename="samples_geometry.h5")
```

线性滑动分布反演，先跑一个固定权重 BLSE：

```python
from eqtools.csiExtend.blse_multifaults_inversion import BoundLSEMultiFaultsInversion

inv = BoundLSEMultiFaultsInversion(
    "linear_slip",
    faults_list=[fixed_fault],
    geodata=geodata,
    config="default_config.yml",
    bounds_config="bounds_config.yml",
)

inv.run(penalty_weight=[100.0])
inv.returnModel()
inv.extract_and_plot_blse_results(plot_faults=True, plot_data=True)
```

权重诊断或多数据集权重估计时，再跑 VCE：

```python
from eqtools.csiExtend.blse_multifaults_inversion import BoundLSEMultiFaultsInversion

inv = BoundLSEMultiFaultsInversion(
    "linear_slip_vce",
    faults_list=[fixed_fault],
    geodata=geodata,
    config="default_config.yml",
    bounds_config="bounds_config.yml",
)

inv.run_simple_vce()
inv.returnModel()
inv.extract_and_plot_blse_results(plot_faults=True, plot_data=True)
```
