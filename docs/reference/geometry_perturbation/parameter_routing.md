# 几何参数布局与配置路由

本页说明一个几何样本怎样被路由到正确断层和方法，不解释具体几何公式。

## 三道启用条件

几何更新必须同时满足：

1. 顶层 `nonlinear_inversion: true`；
2. 对应断层 `geometry.update: true`；
3. `geometry.sample_positions` 与方法的参数 cardinality 一致。

前两项不是两个可互相覆盖的开关，而是一个全局模式和一组 source selector。初始化时
会先解析成唯一的有效几何更新计划，参数布局、geometry bounds、target、后验重放、报告
和绘图都读取这份计划：

| 顶层 `nonlinear_inversion` | source `geometry.update` | 结果 |
| --- | --- | --- |
| `false` | `false` | 固定几何 |
| `true` | `false` | 该 source 固定；若所有 source 都如此，则提示后走固定几何路径 |
| `true` | `true` | 该 source 进入几何更新计划 |
| `false` | `true` | 配置冲突，初始化报错，不进入 MPI 或采样 |

最后一种写法不能解释成“顶层优先，所以忽略 source”。如果静默忽略，原始
`sample_positions` 可能与 sigma/alpha 起始位置重叠，并使报告误把尺度参数显示为几何参数。

```yaml
nonlinear_inversion: true

faults:
  MainFault:
    geometry:
      update: true
      sample_positions: [0, 1]
    method_parameters:
      update_fault_geometry:
        method: perturb_bottom_coords_along_fixed_direction
        average_direction: 10.0
        angle_unit: degrees
        perturbation_direction: horizontal
```

`sample_positions=[start, end]` 是全局几何样本向量的半开区间。启用断层的切片应从 0 开始
形成连续布局；显式复用同一切片表示多个断层共享同一几何参数，而不是错位。

## 参数个数契约

| 契约 | 允许的切片长度 | 常见用途 |
| --- | --- | --- |
| 固定 `exact` | 必须等于方法登记长度 | 旋转、平移、固定组合 |
| 动态节点选择 | 1（广播）或可移动节点数 | 含 `fixed_nodes` 的坐标方法 |
| 动态 dip profile（未分组） | 1（广播）或 sampled controls 数 | sampled/fixed 倾角 profile |
| 动态 dip profile（显式分组） | 唯一 group 标签数 \(K\) | 多个 sampled controls 共享倾角增量 |

全部 dip controls 为 fixed 时只接受空切片 `[k, k]`。未知的未来动态方法必须自行登记 schema，
核心不能猜成固定参数个数。

## bounds 与顺序

主配置决定怎样应用，bounds 决定允许多大。统一边界：

```yaml
geometry:
  MainFault: [-15.0, 15.0]
```

逐参数边界：

```yaml
geometry:
  MainFault:
    lb: [-10.0, -15.0, -5.0, -5.0]
    ub: [10.0, 15.0, 5.0, 5.0]
```

数组顺序严格跟随方法登记的 perturbation items；未分组 dip profile 跟随 sampled controls
声明顺序，显式分组则跟随 group 标签第一次出现的顺序。不要按空间排序重新排列 bounds，
也不要使用当前不支持的 `N x 2` 行式写法。

geometry 参数可能混合 km 与 degree，必须在案例注释和结果解释中逐项写明；不能只给整段
切片一个模糊单位。

## 配置预检

创建 target 前应完成：

- 方法存在且属于当前 fault；
- kwargs 都来自公开签名；
- reference fields 齐全；
- whole-mesh vertices/faces 成对且拓扑匹配；
- 切片、bounds 和动态 cardinality 对齐；
- 固定拓扑 remap 已准备并与候选 mesh 参数一致。

几何更新计划和样本布局在 inversion 构造时一起冻结。构造后若改变
`nonlinear_inversion`、`geometry.update`、`sample_positions` 或扰动方法，应重新创建
`BayesianMultiFaultsInversion`；target 构造会拒绝继续使用与旧布局不一致的新配置。计划中
还包含最终 method/mesh/GF/Laplacian kwargs，候选不再从可变 YAML 字典重新推导调用。
target 建立后若 fault 采用了新的 `GeometryReference`，旧 target 会统一失效；该规则属于
所有 reference-based geometry family 的共同生命周期，不是 dip profile 的特殊分支。
少数显式使用 `baseline_source="current_geometry"` 的历史方法仍保留原合同，本次不会暗中
改写其科学零点。

```python
fault.geometry_summary()
inversion.print_parameter_positions()
constraint_state = inversion.get_constraint_snapshot(validate=True)
print(constraint_state["validation"])
```

`geometry_summary()` 检查 fault-local reference、控制点/分组映射和 mesh 状态；
`print_parameter_positions()` 检查这些几何自由度进入全局 `S` 的位置，并同时区分
FULLSMC 直接采样与 SMC_FJ 条件线性 `L`。两者不是重复报告。只有在查询方法签名、kwargs
或 YAML 片段时才额外调用：

```python
fault.help("perturb_bottom_coords_along_fixed_direction")
```

参数布局的完整字段和 BLSE/VCE/FULLSMC/SMC_FJ 对照见
[参数列布局与诊断接口](../../concepts/observation_matrix_layout.md#参数列布局与诊断接口)。

`fault.snapshot()` 建立几何参考；`get_constraint_snapshot()` 只生成约束诊断副本，名称相近但
生命周期不同。

## dip profile 的单一配置源

sampled/fixed controls、axis、transition、可选 `perturbation_groups` 和输入坐标系只在 Python
setup 的 `set_dip_profile()` 中声明。YAML preset 只保留候选扰动单位、走向策略和 mesh
生成选项。分组标签按 sampled controls 声明顺序对齐；YAML `sample_positions` 只给出解析后
的 \(K\) 个参数所在的全局半开切片，不重复保存标签。
旧字段的逐项替换见[旧倾角剖面调用迁移](../dip_profile/migration.md)。
