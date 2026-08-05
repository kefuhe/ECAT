# 约束配置与运行时调整短例

这个短例用于已经创建好 `inversion` 对象、准备在求解前设置约束的场景。
BLSE、VCE 和 `SMC_FJ + ss_ds` 使用相同的公开约束入口；FULLSMC 主要使用
参数边界和先验，不消费线性约束矩阵。

如果还没有完成数据、断层和 inversion 初始化，先看
[BLSE/VCE 最小脚本骨架](blse_minimal_run.md)。完整字段、覆盖顺序和高级矩阵接口见
[约束管理器](../reference/constraint_manager.md)。

## 场景一：只用配置文件

正式案例优先把可复现约束写入 `bounds_config.yml`：

```yaml
lb: -10.0
ub: 10.0

strikeslip:
  MyFault: [-5.0, 5.0]
dipslip:
  MyFault: [-2.0, 2.0]
rake_angle:
  MyFault: [-30.0, 60.0]

source_constraints:
  MyFault:
    - name: zero_top_slip
      type: equality
      rule: "zero_edge_slip(top, ss+ds)"
```

初始化 inversion 时传入该文件：

```python
from eqtools.csiExtend.blse_multifaults_inversion import (
    BoundLSEMultiFaultsInversion,
)

inversion = BoundLSEMultiFaultsInversion(
    "linear_slip",
    faults_list,
    geodata,
    config="default_config.yml",
    bounds_config="bounds_config.yml",
)
```

如果 inversion 已经存在，可事务式重新加载：

```python
inversion.apply_constraints_from_config("bounds_config.yml")
```

重新加载会重建配置负责的 coarse、patch 和线性约束；任一步校验失败都会恢复
加载前状态。

## 场景二：当前会话试验一个 coarse 边界

只想在当前 inversion 对象上试验一个较窄边界时：

```python
inversion.update_bounds(
    strikeslip_bounds={"MyFault": [-4.0, 4.0]},
    dipslip_bounds={"MyFault": [-1.5, 1.5]},
)
```

这会修改当前 coarse 声明。以后再次调用
`apply_constraints_from_config("bounds_config.yml")` 时，coarse 状态会由配置文件
重新建立；因此正式、可复现设置仍应写回 YAML。

## 场景三：试验局部 patch 约束

只调整一组 patch：

```python
inversion.add_patch_constraints({
    "MyFault": [{
        "name": "trial_segment",
        "selector": {"patches": [10, 11, 12]},
        "bounds": {
            "strikeslip": [0.0, 5.0],
            "dipslip": [-1.0, 1.0],
        },
        "rake_angle": [-60.0, 60.0],
    }],
})
```

如果该 selector 与配置文件或已有 runtime patch 规则重叠，默认会报错。只有确实
希望后一个规则覆盖前一个规则时，才在后一个规则中加入：

```text
"overwrite": True
```

循环试验不同局部分段时，用 `replace` 替换上一轮 runtime patch，而不是不断累积：

```python
inversion.replace_patch_constraints({
    "MyFault": [{
        "name": "second_trial",
        "selector": {"patches": [20, 21]},
        "bounds": {"strikeslip": [0.0, 3.0]},
    }],
})
```

恢复到只保留配置文件中的 patch 规则：

```python
inversion.clear_patch_constraints()
```

`clear_patch_constraints()` 不会删除 YAML 中的 `patch_constraints`。

## 场景四：试验 fault-level rake

合并一项 runtime fault rake：

```python
inversion.update_fault_rake_limits({
    "MyFault": [-20.0, 50.0],
})
```

替换全部 runtime fault-rake 映射：

```python
inversion.replace_fault_rake_limits({
    "MyFault": [-30.0, 45.0],
})
```

只清除 runtime fault rake，恢复配置和 patch rake：

```python
inversion.clear_fault_rake_limits()
```

线性 rake sector 的公式、跨零角度和最大张角限制见
[Rake Constraints](../reference/rake_constraints.md)。

## 求解前统一检查

所有 runtime 修改都应在 BLSE/VCE 求解前完成；`SMC_FJ` 应在 `walk()` 构建
target 前完成。

```python
snapshot = inversion.get_constraint_snapshot(validate=True)

print(snapshot["bounds"])
print(snapshot["runtime_overrides"])
print(snapshot["constraint_totals"])
print(snapshot["activation_flags"])
print(snapshot["validation"])
```

对于 SMC，还可检查：

```python
print(snapshot.get("sampling_mode"))
print(snapshot.get("inactive_constraints"))
```

看到配置字段并不等于当前模式一定消费了对应线性约束。FULLSMC 中不适用的配置
规则会列入 `inactive_constraints`；runtime 显式请求一个当前模式无法消费的规则
则会直接报错。

## 选择哪一种方式

| 目的 | 推荐方式 |
| --- | --- |
| 正式、可复现、需要分享 | 写入 `bounds_config.yml` |
| 当前对象上快速试验 fault/component 边界 | `update_bounds(...)` |
| 反复试验局部 patch | `replace_patch_constraints(...)` |
| 临时调整 fault rake | `update/replace_fault_rake_limits(...)` |
| 恢复 YAML 基线 | 清除相应 runtime 范围，必要时重新 `apply_constraints_from_config(...)` |
| 手工构造 `A/b/Aeq/beq` | 先读约束 reference 的高级接口，不在入门脚本中直接猜列号 |
