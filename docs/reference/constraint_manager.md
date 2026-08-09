# 约束管理器

ECAT 用一个约束管理器统一保存参数边界、线性不等式
`A @ x <= b` 和线性等式 `Aeq @ x = beq`。BLSE、VCE 和
`SMC_FJ` 都从这里读取最终约束，不再接受求解入口上的临时约束矩阵。

普通用户应通过 inversion 对象的公开方法更新约束；不要直接修改
`inversion.constraint_manager` 的内部字典。

## 阅读路径

| 当前问题 | 阅读位置 |
| --- | --- |
| 只想复制常用 YAML/runtime 搭配 | [约束配置与运行时调整短例](../examples/constraint_config_runtime.md) |
| 想用配置和少量会话修改完成普通反演 | 最短使用路径 → 公开接口边界与操作语义 |
| 不清楚配置、会话 coarse 和 patch 谁覆盖谁 | 配置基线、会话修改与显式索引层级 |
| 需要按局部 patch 或 source component 约束 | Patch 规则 → Source 规则 |
| 需要 rake 公式和符号检查 | [Rake Constraints](rake_constraints.md) |
| 需要手工构造 `A/b/Aeq/beq` | 自定义线性矩阵（高级） |
| 需要检查最终生效状态 | 求解时读取与诊断 |

普通用户通常只需要前三层：

```text
workflow：确定何时设置约束
  -> example：复制一个最小搭配
      -> 本 reference：遇到字段、覆盖或模式问题时查阅
```

私有 registry、生命周期 metadata 和编译器方法属于内部维护协议，不是用户配置
字段或公开方法参数。用户只需要选择正确的 inversion facade 方法。

## 最短使用路径

默认选择很简单：

- 需要复现、分享或长期保存：写入 `bounds_config.yml`；
- 只在当前 inversion 上试验：初始化后调用公开 runtime 方法；
- 不确定最终是否生效：求解前读取 snapshot。

配置文件负责可复现的默认约束：

```yaml
# bounds_config.yml
lb: -10.0
ub: 10.0

strikeslip:
  FaultA: [-5.0, 5.0]
dipslip:
  FaultA: [-2.0, 2.0]
rake_angle:
  FaultA: [-30.0, 60.0]

patch_constraints:
  FaultA:
    - name: selected_segment
      selector: {patches: [10, 11, 12]}
      bounds:
        strikeslip: [0.0, 5.0]
      rake_angle: [-60.0, 60.0]
```

运行时只修改当前实验需要改变的部分：

```python
inversion.update_bounds(
    strikeslip_bounds={"FaultA": [-4.0, 4.0]},
)
inversion.update_fault_rake_limits(
    {"FaultA": [-20.0, 50.0]},
)
inversion.add_patch_constraints({
    "FaultA": [{
        "name": "runtime_segment",
        "selector": {"patches": [20, 21]},
        "bounds": {"dipslip": [-1.0, 1.0]},
    }],
})

snapshot = inversion.get_constraint_snapshot(validate=True)
```

在 BLSE/VCE 中，以上操作应在求解前完成；在 `SMC_FJ` 中，应在标准
`walk()` 入口构建 target 前完成。

配置和 runtime 只存在于声明与维护阶段。编译完成后，所有 backend 都只读取
manager 中的一套最终状态：

```text
配置基线 + 当前会话修改 + 局部 runtime 声明
                 ↓ 事务式解析、覆盖和校验
    一套 resolved lb/ub + named A/b/Aeq/beq
                 ↓
          BLSE / VCE / SMC consumer
```

因此 config/runtime 分层不会让求解器重复应用约束；它用于支持配置重载、
runtime clear、诊断和失败回滚。

## 模式与参数空间

| 模式 | 边界 | 线性矩阵约束 |
| --- | --- | --- |
| BLSE/VCE | 完整线性模型向量 | 支持 |
| `SMC_FJ + ss_ds` | 超参数边界 + 线性后缀边界 | 支持，矩阵只作用于线性后缀 |
| `FULLSMC + ss_ds` | 采样向量边界 | 不消费线性矩阵 |
| `FULLSMC + magnitude_rake` | magnitude/rake 采样边界，含 patch rake | 不消费线性矩阵 |
| `FULLSMC + rake_fixed` | magnitude 采样边界 | 不消费线性矩阵 |

线性 rake、zero-slip、Euler cap、backslip 和自定义 `A/b` 只在
BLSE/VCE 或 `SMC_FJ + ss_ds` 中生效。FULLSMC 会把配置中存在但
当前不消费的线性规则列入 `inactive_constraints`，不会假装它们已经生效。

配置开关也按 consumer 分工：

- BLSE/VCE 与 `SMC_FJ + ss_ds` 的线性 rake sector 由
  `use_rake_angle_constraints` 控制；
- `FULLSMC + magnitude_rake` 的 fault/patch sampled-rake bounds 属于 bounds，
  由 `use_bounds_constraints` 控制；
- runtime 显式添加 patch rake 时，若当前 mode 没有对应 consumer，会立即报错，
  而不是静默保存。

## 公开接口边界与操作语义

约束的稳定对外入口是 BLSE/Bayesian inversion 对象，不是
`constraint_manager` 的私有方法。公开动词具有统一含义：

| 动词 | 语义 |
| --- | --- |
| `update` | 只合并本次提供的 coarse/fault 条目 |
| `add` | 新增一条 runtime 声明或一个全局唯一的用户命名组 |
| `replace` | 替换完整 runtime 范围，或替换已经存在的同名用户组 |
| `set` | 设置一个单例领域约束的完整值 |
| `clear` | 只清除该方法明确负责的 runtime/领域范围 |
| `apply_*_from_config` | 事务式替换配置拥有的声明并重新编译 |

同一行为只保留一个推荐名字：

| 操作 | 含义 |
| --- | --- |
| `update_bounds(...)` | 局部更新当前 coarse bounds |
| `apply_constraints_from_config(path)` | 事务式重载配置拥有的声明 |
| `update_fault_rake_limits(mapping)` | 合并 runtime fault-level rake 扇区 |
| `replace_fault_rake_limits(mapping)` | 替换全部 runtime fault-level rake 扇区 |
| `clear_fault_rake_limits()` | 只清除 runtime fault-level rake |
| `add_patch_constraints(specs)` | 累积 runtime patch 规则 |
| `replace_patch_constraints(specs)` | 替换全部 runtime patch 规则 |
| `clear_patch_constraints()` | 只清除 runtime patch 规则 |
| `set_fixed_rake_constraints(mapping)` | 替换 fixed-rake 等式族 |
| `clear_fixed_rake_constraints()` | 清除 fixed-rake 等式族 |
| `add_linear_inequality_constraint(...)` | 新增用户命名不等式组 |
| `replace_linear_inequality_constraint(...)` | 替换已存在的用户不等式组 |
| `add_linear_equality_constraint(...)` | 新增用户命名等式组 |
| `replace_linear_equality_constraint(...)` | 替换已存在的用户等式组 |
| `remove_linear_constraint(name)` | 删除一个用户命名线性组 |
| `add_interseismic_backslip_constraint(...)` | 新增 hard-backslip 用户组 |
| `replace_interseismic_backslip_constraint(...)` | 替换已存在的 hard-backslip 用户组 |
| `add_deep_slip_loading_constraint(...)` | 新增一组 deep-slip 用户约束 |
| `replace_deep_slip_loading_constraint(...)` | 原子替换已存在的 deep-slip 用户约束组 |

`add` 要求名称尚不存在；`replace` 要求同类、同名、用户拥有的组已经存在。
等式和不等式共享同一名称空间。`rake_sector` 和 `fixed_rake` 是管理器保留名，
必须通过各自的领域方法修改，不能由 raw-matrix API 占用或删除。
配置生成的组、领域 helper 管理的组和用户命名组之间也不能靠同名相互覆盖；
名称冲突会报错并回滚。

公开 raw-matrix 方法自动维护用户命名组；内部生命周期和领域 metadata 由管理器
设置。需要 rake、patch、不可压缩、震间或 deep-slip 约束时，优先使用对应领域
方法，而不是手工管理最终矩阵。

## 配置基线、会话修改与显式索引层级

边界不是两套最终数组互相覆盖。管理器先形成当前 coarse 声明，再按具体程度
编译为一套 `lb/ub`：

```text
当前 coarse 声明
  ├─ 最近一次配置重载建立
  └─ update_bounds(...) 在当前会话中局部修改
        ↓
persistent explicit-index/data-correction bounds
        ↓
config patch
        ↓
runtime patch
        ↓
一套 resolved lb/ub
```

`update_bounds()` 不是与 config 并存且永久保留的第二套 coarse 字典。调用
`apply_constraints_from_config()` 后，当前 coarse 声明由新配置重新建立；因此
前一次 `update_bounds()` 写入的 coarse 值会被替换。这适合当前对象上的试验；
需要分享或长期保存的设置应写回 YAML。

显式 index/data-correction bounds 是独立、持久的已解析列声明：它比 coarse
更具体，但仍早于 patch selector 应用。它不属于用户通常需要手工填写的
`bounds_config.yml` 四类声明。

更具体的 patch 规则优先于 coarse 规则。config patch 先于 runtime patch；
如果二者或两个 patch 规则命中同一 fault、component 和 patch，默认报错，
只有后一个规则明确写出 `overwrite: true` 才允许覆盖。

需要特别区分两类运行时状态：

- `update_bounds()` 修改当前会话的 coarse 状态；以后重新加载 bounds 配置时会被
  新配置重建。
- runtime patch、runtime fault rake 和显式 index/data-correction bounds 是独立声明，
  配置重载后仍保留。

这项差异是明确的生命周期策略，不表示 solver 中存在两套约束。恢复配置基线时：

```python
inversion.clear_patch_constraints()
inversion.clear_fault_rake_limits()
inversion.apply_constraints_from_config("bounds_config.yml")
```

前两个方法只清除各自负责的 runtime 范围；最后一步重建配置拥有的声明。

线性 rake 的最终优先顺序为：

```text
config fault rake
  -> runtime fault rake
      -> config patch rake
          -> runtime patch rake
              -> one final rake_sector matrix
```

patch rake 是替换选中 patch 的 fault 默认值，不是再叠加第二套 rake 矩阵。
完整公式和角度限制见 [Rake Constraints](rake_constraints.md)。

## Bounds 字段

常用字段如下：

```yaml
lb: -10.0
ub: 10.0

strikeslip:
  FaultA: [-5.0, 5.0]
dipslip:
  FaultA: [-2.0, 2.0]
poly:
  FaultA: [-1000.0, 1000.0]

source_bounds:
  PressureA:
    pressure: [-1.0e6, 1.0e6]
  VolumeA:
    eps12: [-1.0e-4, 1.0e-4]

sigmas: [-3.0, 3.0]
alpha: [-3.0, 3.0]
```

对非 Fault 源使用 `source_bounds`，不要把它们写成
`strikeslip`/`dipslip`。`rake_angle` 在 FULLSMC magnitude/rake 参数化中
是普通采样边界，在 BLSE/VCE 和 `SMC_FJ + ss_ds` 中则编译为线性扇区约束。
这两种数学语义的角度校验不同，详见
[Rake Constraints](rake_constraints.md#两种-angle-校验)。

按源名组织的 bounds 字段采用统一加载契约：字段缺失、显式 `{}`、裸键或
`null` 都表示“本层没有声明”。加载器会在任何 BLSE/SMC 编译或摘要输出前把
后两种写法归一化为独立空映射。字段存在且非空时必须是 mapping；这保证配置
摘要、边界编译和约束编译读取的是同一个规范化对象，不因 `verbose` 开关改变
行为。`patch_constraints` 例外，因为它的公开语法同时允许 mapping 和 list。

## Patch 规则

`patch_constraints` 可按 patch id、edge、深度或空间范围选择局部 patch：

```yaml
patch_constraints:
  FaultA:
    - name: shallow_left_lateral
      selector:
        depth_range: [0.0, 15.0]
      bounds:
        strikeslip: [0.0, 5.0]
        dipslip: [-2.0, 2.0]
      rake_angle: [-60.0, 60.0]
```

selector 的完整写法见
[Fault Patch Indices](fault_patch_indices.md#selector-cookbook)。

运行时规则示例：

```python
inversion.add_patch_constraints({
    "FaultA": [{
        "name": "trial",
        "selector": {"patches": [10, 11, 12]},
        "bounds": {"strikeslip": [0.0, 5.0]},
        "rake_angle": [-60.0, 60.0],
        "overwrite": True,
    }],
})
```

`replace_patch_constraints()` 和 `clear_patch_constraints()` 只处理 runtime
patch，不删除 `bounds_config.yml` 中的 patch 声明。

同一个 `rake_angle` patch 字段按模式落地：

```text
BLSE/VCE 或 SMC_FJ + ss_ds
  -> 选中 patch 的 ss/ds convex sector

FULLSMC + magnitude_rake
  -> 选中 patch 的 sampled-rake lower/upper bounds
```

因此 `rake_angle: [-180, 180]` 不能用于线性 sector，但可用于
`FULLSMC + magnitude_rake` 的普通 sampled bounds。FULLSMC 的其他 slip
参数化不会消费 patch rake；共享配置会在 snapshot 中显示 inactive，runtime
显式调用则报错。

<a id="zero-slip-constraints"></a>

## Source 规则

`source_constraints` 由 source adapter 按参数分量生成矩阵：

```yaml
source_constraints:
  FaultA:
    - {name: ss_nonnegative, type: inequality, rule: "strikeslip >= 0"}
    - {name: ds_zero, type: equality, rule: "dipslip == 0"}
  PressureA:
    - {name: pressure_positive, type: inequality, rule: "pressure >= 0"}
  VolumeA:
    - {name: incompressible, type: equality, rule: "incompressible"}
```

Fault 参数始终按 canonical `sdtc` 分量顺序定位；`slipdir: ds` 和
`slipdir: sd` 不会产生不同列序。缺失的规则表示不约束，不应靠空矩阵占位。

全断层分量零滑动使用 `strikeslip == 0` 或 `dipslip == 0`；只固定 top、bottom 等边界 patch 时使用 `zero_edge_slip(...)` 或 `add_zero_edge_slip_constraint(...)`。可复制配置见 [线性滑动反演配置](config_linear_slip.md#普通线性约束)。

## Fixed rake 与领域约束

固定 rake 是独立的线性等式族：

```python
inversion.set_fixed_rake_constraints({"FaultA": -90.0})
inversion.clear_fixed_rake_constraints()
```

`set_fixed_rake_constraints({})` 是 no-op；删除必须显式调用 `clear`。

优先使用领域 helper，避免手工计算列号：

- `set_incompressibility_constraints(...)`
- `add_zero_edge_slip_constraint(...)`
- `add_patch_slip_constraint(...)`
- `add_data_correction_equality(...)`
- `add_interseismic_backslip_constraint(...)`
- `replace_interseismic_backslip_constraint(...)`
- `add_deep_slip_loading_constraint(...)`
- `replace_deep_slip_loading_constraint(...)`

震间约束见 [Interseismic Kinematics](interseismic_kinematics.md)；
data-correction 列解析见 [Data Corrections](data_corrections.md)。

## 自定义线性矩阵（高级）

先读取活动参数布局：

```python
layout = inversion.get_linear_parameter_layout()
print(layout["space"], layout["width"], layout["global_offset"])
for block in layout["blocks"]:
    print(block)
```

活动布局字段：

| 字段 | 含义 |
| --- | --- |
| `space` | `blse_full_linear`、`smc_fj_linear_suffix` 或 `inactive` |
| `width` | `A`/`Aeq` 必须具有的列数 |
| `global_offset` | 活动线性列 0 在完整/采样向量中的起点 |
| `blocks` | 按 source/component 排列的半开区间 |

BLSE/VCE 的 raw matrix 对应完整线性模型；`SMC_FJ` 的 raw matrix 对应
线性后缀，不能把完整采样向量列数直接用于 `A`。布局会与 source adapter、
`lsq_parameters`、assembled `G` 和已注册矩阵列数交叉校验。

```python
inversion.add_linear_inequality_constraint(
    A, b, name="my_cap", source="experiment_1"
)
inversion.replace_linear_inequality_constraint(
    A_new, b_new, name="my_cap", source="experiment_2"
)

inversion.add_linear_equality_constraint(
    Aeq, beq, name="my_tie"
)
inversion.remove_linear_constraint("my_tie")
```

当前不支持 CSI `custom=True` 产生、但未进入 ECAT source/poly position
映射的额外 GF 列；这种布局会在求解前明确报错。

## 批量事务（高级）

单个公开更新本身已经原子化。只有跨多个约束族必须一起成功或一起回滚时，
才使用：

```python
with inversion.constraint_transaction():
    inversion.update_bounds(lb=-8.0, ub=8.0)
    inversion.replace_fault_rake_limits(
        {"FaultA": [-30.0, 60.0]}
    )
    inversion.set_fixed_rake_constraints(
        {"FaultB": -90.0}
    )
```

外层事务只在结束时统一校验和同步。任一步失败时，bounds、命名组、
runtime 声明、interseismic 配置和 `state_revision` 都恢复到进入事务前。

## 求解时读取与诊断

BLSE 和 VCE 每次求解前读取同一管理器 revision。`SMC_FJ` 在 target
构建时冻结当时的 bounds 和矩阵；FULLSMC 也在 target 构建时生成一次有效
bounds 快照，并让 prior 与 proposal 使用同一数组。之后若约束发生变化，
旧 target 会拒绝继续，需要重新调用对应 `walk_*()` 或重建 target。

```python
snapshot = inversion.get_constraint_snapshot(validate=True)
print(snapshot["state_revision"])
print(snapshot["bounds"])
print(snapshot["constraint_totals"])
print(snapshot["runtime_overrides"])
print(snapshot["activation_flags"])
print(snapshot.get("sampling_mode"))
print(snapshot.get("inactive_constraints"))
print(snapshot.get("effective_bounds_defaults"))
print(snapshot["validation"])
```

FULLSMC 的 raw bounds 仍用 `NaN` 表示“未显式设置”。仅在采样快照中，
普通缺失端点补为 `[-10, 10]`，缺失的 magnitude 下界补为 `0`。
`effective_bounds_defaults` 会报告本次补齐的数量，方便区分用户声明与便利默认。

`validate=True` 会检查边界顺序、矩阵尺寸、有限值和 equality rank，可能对大矩阵
执行秩计算，因此不要放进 SMC likelihood 循环。

## 从旧接口迁移

下列旧接口名不再作为公开入口；已有脚本应迁移到对应的当前接口：

| 旧接口 | 当前接口 |
| --- | --- |
| `set_bounds(...)` | `update_bounds(...)` |
| `set_bounds_from_config(...)` | `apply_constraints_from_config(...)` |
| `set_inequality_constraints_for_rake_angle(...)` | `replace_fault_rake_limits(...)` |
| `update_rake_constraints(...)` | 分别调用 fault-sector 与 fixed-rake 方法 |
| `add_custom_inequality_constraint(...)` | `add_linear_inequality_constraint(...)` |
| `add_custom_equality_constraint(...)` | `add_linear_equality_constraint(...)` |
| `add_inequality_constraint(...)` | `add_linear_inequality_constraint(...)` |
| `add_equality_constraint(...)` | `add_linear_equality_constraint(...)` |
| `update_all_constraints(...)` | focused calls；需要整批回滚时使用 transaction |
| `add_interseismic_backslip_constraint(..., overwrite=True)` | `replace_interseismic_backslip_constraint(...)` |
| `add_deep_slip_loading_constraint(..., overwrite=True)` | `replace_deep_slip_loading_constraint(...)` |

震间 `backslip_constraints` 配置不再需要 `overwrite` 字段；配置重载本身就是
config-owned 状态的事务式替换。Patch 规则中的 `overwrite` 仍保留，因为它表达
同一次 selector precedence 中对重叠 patch 的显式覆盖，不是命名组 CRUD。

## 相关页面

- [Rake Constraints](rake_constraints.md)
- [BLSE/VCE](blse_vce.md)
- [Fault Patch Indices](fault_patch_indices.md)
- [Interseismic Kinematics](interseismic_kinematics.md)
- [Data Corrections](data_corrections.md)
