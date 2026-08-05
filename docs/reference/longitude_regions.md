# 经度约定与区域配置

本页说明降采样配置中的经度范围如何与 CSI 数据对象匹配。需要设置
`processing_region`、经纬度 `data_filters`、`covar.mask_out`、相位周跳/参考改正
区域或 `check_plots.*.coordrange` 时查阅本页。

## 阅读路径

- 第一次设置普通西半球 box：阅读“两种经度写法等价”和“适用字段”。
- 数据跨越 ±180°：阅读“跨日界线区域”。
- 区域仍然为空或协方差掩膜可疑：阅读“运行诊断”和“建议检查”。
- 所有 YAML 字段和执行顺序仍以 [Downsampling App](downsampling_app.md) 为准。

## 两种经度写法等价

同一地点可以写成 `[-180, 180]` 或 `[0, 360]` 经度。例如：

```text
-118.0° == 242.0°
```

CSI 标准数据对象可能把负经度保存到等价的 `0–360°` 分支，而 reader 的原始网格仍
保留源文件经度。ECAT 的区域判断会按 360° 周期匹配，因此同一个普通区域可以使用
任一连续写法：

```yaml
# 西经写法
box: [-118.239027, -116.911690, 35.246481, 36.328000]

# 与上面完全等价
box: [241.760973, 243.088310, 35.246481, 36.328000]
```

这种匹配只生成临时比较坐标，不会改写 CSI 的 `data.lon`、局部 `x/y`、观测值、
投影向量或降采样输出文件。已有使用 `0–360°` 的配置不需要转换。

## 适用字段

周期经度匹配适用于：

- `processing_region` 的 `box`、`polygon` 和 `polygon_file`；
- SAR/optical `data_filters` 的 `lonlat_box` 和 `lonlat_polygon`；
- `phase_cycle_correction` 与 `observation_correction` 的经纬度区域；
- `covar.mask_out`；
- `check_plots.raw.coordrange` 与 `check_plots.decim.coordrange`，包括 cell 和断层叠加线。

`downsample.std_config.focus_region` 会先把经纬度 polygon 投影到 CSI 局部 `x/y`，
也不要求用户手动更换经度分支。`coord_type: xy` 是非周期的局部坐标，不应用本页规则。

## 跨日界线区域

box 仍须满足 `minlon < maxlon`。跨越日界线的连续 box 使用展开写法，例如：

```yaml
box: [179.0, 181.0, -2.0, 2.0]
```

不要写成 `[179.0, -179.0, ...]`。polygon 顶点允许在 `179°/-179°` 之间切换，程序会
沿顶点顺序展开经度后再判断。polygon 顶点应按边界顺序排列，不能把相隔很远的区域
拼成一个 polygon。

## 运行诊断

启用 `data_filters` 或 `processing_region` 后，终端和 YAML 报告会记录进入该阶段的
经纬度范围。`processing_region` 还记录与实际数据分支对齐后的 geometry。若区域没有
保留任何点，异常会同时给出：

- 实际数据经纬度范围；
- 用户配置的 box；
- 与数据数值分支等价的 box。

运行协方差估计时，如果 `covar.mask_out` 需要转换到等价分支，终端会显示 configured
和 resolved box，运行 metadata 的 `_runtime.resolved_covariance_mask_out` 也会保留
解析结果。应检查掩膜位于 `processing_region` 内，并确认仍有足够背景点拟合协方差。

## 建议检查

1. `-s` 检查原始覆盖范围和观测量级。
2. `-c` 检查 processing-region 报告、resolved covariance mask 和协方差曲线。
3. `-d` 检查 decim 图的范围、cell 与断层叠加是否对齐。
4. 修改区域、数据过滤或投影原点后，重新运行协方差估计和正式降采样。

相关字段和执行顺序见 [Downsampling App](downsampling_app.md)，两步运行方法见
[InSAR 降采样两步运行](../workflows/02a_insar_downsampling_two_step.md)。
