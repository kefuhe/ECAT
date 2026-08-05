# 断层走向、倾角与滑动基底约定

本页解释紧凑平面断层在 ECAT 中怎样输入 `strike/dip`、怎样转换成 CSI
实际使用的几何，以及为什么这个转换不自动修改 `rake`。第一次设置非线性几何边界时，
先读“最短规则”；需要跨 `90°` 搜索或把结果交给 BLSE/VCE 时，再读后续部分。

## 最短规则

普通用户可以只遵守下面四条：

1. `strike` 是从北顺时针增加的地理方位角；通常写在 `[0, 360)`。
2. 新版非线性几何 SMC 的 `dip` 推荐写在 `[0, 180]`。只搜索一侧时，优先把范围收紧到
   `(0, 90]`，最容易解释。
3. `90 < dip <= 180` 表示越过直立面后的另一侧。历史负值 `[-90, 0)` 仍兼容，但新配置
   不建议再用它表达连续跨直立搜索。
4. `rake`、`strikeslip/dipslip` 是相对于**当前输入走向/倾角基底**的滑动参数；几何折叠时
   不会被程序暗中换号或加减 `180°`。

例如，下面的原生写法允许几何连续跨过直立位置：

```yaml
prior_bounds_format: lower_upper

bounds:
  defaults:
    strike: [Uniform, 10.0, 40.0]
    dip: [Uniform, 70.0, 110.0]
```

如果地质上已经确定下倾侧，直接使用如 `dip: [Uniform, 30.0, 80.0]` 会更清晰、采样也通常
更容易。

## 三种角度表达

同一条计算链中需要区分三个词：

| 名称 | 含义 | 是否保留在样本和摘要中 |
| --- | --- | --- |
| input/sample | YAML、固定参数或 SMC 样本中的 `strike/dip` | 是 |
| compatibility | 历史负 `dip`，范围 `[-90, 0)` | 是，但新配置不推荐 |
| canonical solver geometry | 实际传给 CSI 建立 patch 的 `dip in [0, 90]` 及配套 `strike in [0, 360)` | 在模型摘要中单列 |

规范化公式为：

\[
(S_c,d_c)=
\begin{cases}
(S,d), & 0\le d\le90,\\
(S+180^\circ,180^\circ-d), & 90<d\le180,\\
(S+180^\circ,-d), & -90\le d<0,
\end{cases}
\]

其中所有 canonical `strike` 最后都会取模到 `[0, 360)`。例如：

| 输入 `(strike, dip)` | CSI solver geometry | 说明 |
| --- | --- | --- |
| `(20°, 70°)` | `(20°, 70°)` | 已是 canonical |
| `(20°, 110°)` | `(200°, 70°)` | 原生 0–180 表达的另一侧 |
| `(20°, -70°)` | `(200°, 70°)` | 历史带符号兼容表达 |

`0°` 和 `180°` 是水平端点。紧凑正演在显式给定宽度时可以表示水平面；但标准
nonlinear-to-mesh 桥接需要由顶边中点和 `top/depth` 生成上下边，会遇到 `sin/tan(dip)`
退化，因此会直接拒绝这两个端点。正式几何搜索应使用非退化的开区间。

## 为什么不自动转换 rake

角度规范化只回答“用哪一组 canonical 顶点建立同一个平面”。它不负责重新定义源机制。
当前模型的滑动分量是：

\[
s_s=M\cos(r),\qquad s_d=M\sin(r),
\]

或直接使用采样的 `strikeslip/dipslip`。这里的 `r`、`s_s`、`s_d` 始终相对于该样本定义的
局部走向/下倾基底解释。因此 `(dip=89°, rake=r)` 和 `(dip=91°, rake=r)` 是两个完整的联合
参数状态；它们不必表示同一个固定全球坐标滑动向量。

这也意味着：几何跨过 `90°` 时，固定数值 rake 对应的物理机制可能改变，似然可能出现明显
变化。它不是参数逐个搜索造成的错误，也不是程序漏做了一次 rake 转换，而是当前联合参数化的
定义。若研究目标要求跨侧时保持同一个全球坐标滑动向量，应单独设计相应参数化，不能靠在
现有 `rake` 上自动加减 `180°`。

## 从配置到正演的完整链条

新版与 legacy `explorefault` 都在建立 CSI patch 前调用同一个 strike/dip 规范化函数；
两者的配置格式和参数注册方式不同，但角度几何公式相同。下面以新版路径说明完整链条：

```text
YAML lower/upper 边界
  -> 配置解析为内部 lower/range
  -> ParameterSpec 固定参数索引
  -> 一整个 theta 样本
  -> build_fault_params 保留 input strike/dip/rake
  -> 仅 strike/dip 规范化为 solver geometry
  -> CSI planarfault.buildPatches
  -> patch 顶点与 getpatchgeometry() 派生几何
  -> input rake 或 ss/ds 生成滑动分量
  -> Green's functions、合成数据和联合 likelihood
```

SMC 评估的是完整 `theta`，不是把每个参数各自独立优化后再拼接。几何、滑动、数据改正项和
sigma 会作为同一个候选状态进入目标函数。

## 模型摘要和两步走衔接

模型摘要中的普通 fault 参数仍是样本值，以保证 HDF5、参数索引和 posterior 解释不变。
当角度需要折叠时，屏幕摘要会额外出现 `Solver geometry`，详细报告会给出：

```text
input  : strike=20,  dip=110
solver : strike=200, dip=70, side_flipped=True
```

从非线性结果构建 BLSE/VCE 的三角元或矩形元时，标准
`generate_top_bottom_from_nonlinear_soln(...)` 和
`buildPatches_from_nonlinear_soln(...)` 入口使用同一规范化函数。因此历史负 `dip` 不会在
两阶段交接处被解释成另一幅几何。最终线性 Green's functions 仍以 mesh/patch 顶点和
`getpatchgeometry()` 派生的 canonical 走向、倾角为准。

非线性紧凑源里的 `rake/slip` 只是紧凑源机制参数；BLSE/VCE 会在固定 mesh 上重新求解
分布式 `strikeslip/dipslip`。可按需要把非线性 rake 用作线性 rake 约束，但不能把它当作已经
求得的分布式滑动。

## 不同几何入口不要混用协议

本页公式针对“紧凑非线性平面源”和它的标准两阶段桥接入口。地表迹线建模有自己的明确接口：

- `generate_bottom_from_single_dip(...)`：只接受 `0 < dip_angle <= 90°`，并要求显式
  `dip_direction`；这里不使用负号或 `90–180°` 表示另一侧。
- 沿走向多倾角：推荐带符号 `[-90, 0) U (0, 90]`，也兼容 `(90, 180)`；先在
  `0–180` 空间跨直立插值，再生成逐节点底边。
- layered dip 的每个 `depth, dip` 是相对统一参考走向的绝对倾角，不是相对上一层的增量；
  连续负 dip 始终在同一侧，不能因网格分层而反复翻转。
- 几何扰动和外部 mesh 是高级路径，应按各自 reference 的输入协议使用，不能因为最终都有
  `dip` 字段就假设语义完全相同。

### 高级多倾角 Bayesian 扰动

`set_dip_control_points(...)` 配合 `perturb_dips_with_preset_params(...)` 时，控制点 dip 可以使用
带符号形式或与之等价的 `0–180°` 形式。setter 会立即校验坐标、长度和倾角域，并在只读
`DipControlPoints` 中统一保存连续 `(0, 180)` 值；proposal 随后只做加法，不再猜测原始表达，
因此可以连续跨过直立面：

```text
reference 77°, perturbation [-30°, 30°] -> proposal [47°, 107°]
reference -80° == 100°；二者加 -20° 后都得到 80°
```

候选必须保持在开区间 `(0, 180)`；`0°/180°` 不会自动回绕。生成对象上的
`top_strike/top_dip` 是用于构造底边的逐节点参考值，其中 `top_dip` 使用带符号表达；它们不是
最终 patch 的 canonical 角度。用于 Green's functions 或科学报告的最终走向、倾角应从 patch
顶点和 `getpatchgeometry()` 获得。

继续阅读：

- [非线性几何反演配置](../reference/config_nonlinear_geometry.md)
- [Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md)
- [非线性几何结果到 fault object](../examples/fault_from_nonlinear_geometry.md)
- [Fault Geometry Construction](../reference/fault_geometry_construction.md)
