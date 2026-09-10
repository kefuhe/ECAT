# Bayesian 前的曲率与转换带预分析

这一步用于在正式 Bayesian 采样之前检查 reference top 的转折尺度，并把人工关注的转折位置
转换成可审阅的 transition endpoint 建议。它是只读 setup 工具，不修改断层、不生成 mesh/GF，
也不自动替用户决定地质转换带。

## 计算对象与公式

分析对象始终是有序、冻结的 reference top，而不是某个候选几何。先按平面弧长 (s) 等间隔
重采样，再在以 km 指定的窗口上平滑 (x(s)) 和 (y(s))。有符号平面曲率为：

\[
\kappa(s)=
\frac{x'(s)y''(s)-y'(s)x''(s)}
{\left[x'(s)^2+y'(s)^2\right]^{3/2}}.
\]

- \(\kappa\) 的符号区分转弯方向；转换带搜索使用 \(|\kappa|\)。
- `spacing_km` 是诊断采样间距，不改变 reference top 节点或 Bayesian 参数。
- `smoothing_km` 是检查的物理尺度。较大值抑制短尺度数字化折点，较小值保留局部转折；它不是
  Bayesian prior。
- `min_prominence_ratio` 只筛掉候选峰噪声。阈值等于“全局最大 \(|\kappa|\) × ratio”，不参与
  最终半宽计算。

默认 `extent_method="peak_fraction"`。选择某个峰 \(s_c\) 后，两侧分别向外寻找第一次满足

\[
|\kappa(s)|=f|\kappa(s_c)|,\qquad 0<f<1
\]

的位置，得到 (s_-<s_c<s_+)。因此左右半宽可以不同：

\[
w_-=s_c-s_-,\qquad w_+=s_+-s_c.
\]

若任一侧没有阈值交点，分析会明确失败，不会静默改用固定半宽。`curvature_fraction` 越小，
通常得到的区间越宽；它和 `smoothing_km` 都需要结合地质尺度与图形判断。

宽缓或复合弯折还可显式选择 `extent_method="turning_coverage"`。它不再使用峰高阈值，而在
用户给定的 `turning_interval` 内计算累计绝对转角：

\[
F(s)=
\frac{\displaystyle\int_{s_a}^{s}|\kappa(u)|\,du}
{\displaystyle\int_{s_a}^{s_b}|\kappa(u)|\,du}.
\]

覆盖率 (c\) 的端点为：

\[
s_-=F^{-1}\!\left(\frac{1-c}{2}\right),\qquad
s_+=F^{-1}\!\left(1-\frac{1-c}{2}\right).
\]

例如 `turning_coverage=0.9` 保留区间内中央 90% 的绝对转角。`turning_interval` 是必须显式
给出的两项沿 top 位置；这样分析器不会把整条 top 上相邻的多个弯折静默合并。它只负责限定
积分域，不会写入正式 profile。若积分域内有多个保留峰，报告会发出人工审阅提示。

## 标准预分析流程

### 1. 建 top，并暂不声明 transition

先准备 sampled/fixed controls，但令 `transition_zones=None`。这会建立分析所需的最小 reference
top；此时不需要 mesh 或观测数据。

```python
fault.set_dip_profile(
    sampled_controls=sampled_controls,
    fixed_controls=fixed_controls,
    interpolation_axis="arc_length",
    transition_zones=None,
    is_utm=False,
)
```

### 2. 分析曲率并查看全部候选峰

```python
analysis = fault.analyze_reference_top_curvature(
    spacing_km=0.5,
    smoothing_method="savgol",
    smoothing_km=6.0,
    polyorder=3,
    min_prominence_ratio=0.05,
)

print(analysis.format_report())
```

报告对每个保留峰给出：从 START/END 量取的弧长、reference top 比例、fault-local x/y、
经纬度、原 reference segment、曲率、峰 prominence 和局部走向。`to_dict()` 可保存为 YAML 或
JSON；需要逐采样值时显式使用 `to_dict(include_samples=True)`。

### 3. 用关注位置选择峰，并形成建议

`nearest_peak` 适合已经知道大致转折位置的场景；anchor 只负责选择邻近峰，不直接成为
transition 中心。`search_radius_km` 防止误选到远处强峰。

```python
suggestion = analysis.suggest_transition(
    anchor={"s_km": 60.0},
    center_mode="nearest_peak",
    search_radius_km=15.0,
    curvature_fraction=0.2,
)

print(suggestion.format_report())
```

若只需要整条 top 上最强的峰，可改为：

```python
suggestion = analysis.suggest_transition(
    center_mode="largest_peak",
    curvature_fraction=0.2,
)
```

标准相对峰值法不合适时，可在一对相邻 controls 所限定的弧长范围内比较累计转角法：

```python
coverage_suggestion = analysis.suggest_transition(
    center_mode="largest_peak",
    extent_method="turning_coverage",
    turning_coverage=0.9,
    turning_interval=(
        {"s_km": lower_control_s},
        {"s_km": upper_control_s},
    ),
)
```

`turning_interval` 的每一端也可用 `{"s_from_end_km": value}`；数值标量等价于
`{"s_km": value}`。`peak_fraction` 仍是默认方法，两种方法不会根据曲率形状自动切换。

### 4. 画图核对尺度和区间

```python
from eqtools.viztools import plot_dip_transition_analysis

plot_dip_transition_analysis(
    fault,
    analysis,
    suggestion,
    coordinates="lonlat",
    save="dip_transition_preflight.png",
    show=False,
)
```

左图显示有序 reference top、归一化绝对曲率、所有保留峰、选中中心和两端点；右图显示有符号
曲率、绝对曲率和建议区间。相对峰值法额外显示阈值线，累计转角法显示其积分域。多个临近峰
通常表示该转折包含多个尺度：应调整
`smoothing_km` 并比较，而不是仅凭最大曲率自动接受一个很窄的区间。

### 5. 审阅后导出现有 transition 协议

```python
zone = suggestion.as_transition_zone(fault.geometry_ref.top_coords)

fault.set_dip_profile(
    sampled_controls=sampled_controls,
    fixed_controls=fixed_controls,
    interpolation_axis="arc_length",
    transition_zones=[zone],
    is_utm=False,
)
```

导出值仍是现有的显式端点格式：

```python
{
    "endpoints": [
        {"s_km": lower_s},
        {"s_km": upper_s},
    ]
}
```

分析结果包含 reference fingerprint。若 top 坐标或点序已经改变，`as_transition_zone()` 会
拒绝旧建议，要求重新分析；它不会把旧弧长位置错配到新的 reference。

最后仍应调用 [`plot_dip_profile_diagnostics()`](bayesian_mixed.md#只读诊断) 验证 controls、
transition 和连续 dip profile，再生成零扰动 bottom、冻结正式 reference、建立 mesh，并进入
Bayesian 配置。

## 怎样判断建议是否可用

至少检查以下四点：

1. START/END 顺序与断层的正走向一致，anchor 选中了预期转折。
2. 建议区间完整位于一对相邻 dip controls 之间，且不包含另一 control。
3. 在两个相邻的 `smoothing_km` 设置下，中心和区间没有不可解释的大幅跳变。
4. 建议尺度符合研究问题；短数字化折点不能仅因曲率较大就自动成为地质转换带。

`turning_coverage` 还应核对积分域只包含希望作为一个转换处理的弯折。若报告提示多峰且区间
明显过宽，应缩小 `turning_interval` 或增加 dip control，而不是继续叠加自动判据。

这项分析不会改变候选计算量。两种 extent 方法最终都只导出相同的显式 endpoint 协议。只有
用户审阅后写入 profile 的 transition endpoints 和候选期边界加密，才会参与后续 bottom
生成；mesh、GF 与缓存仍服从原有几何 pipeline。
