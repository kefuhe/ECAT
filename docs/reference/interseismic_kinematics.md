# 震间加载、Backslip 与 Coupling

本页说明 ECAT 如何在固定断层几何和线性滑动反演结果上计算震间运动学量，并如何添加可选的震间约束。这里的接口用于结果解释和约束辅助，不替代非线性几何反演，也不会把派生字段写回 `fault.slip`。

本页只讲 Euler/block direct-backslip 路径：长期加载 `b` 来自两侧 block 参数差，反演变量解释为 direct backslip `q`。如果你的长期加载由深部自由滑动 patch 直接代表，浅部变量解释为实际滑动或蠕滑 `s`，请使用 [深部滑动加载代理](deep_slip_loading_proxy.md)。两者的 coupling 公式不同，不要混用字段名。

| 路径 | loading | 反演变量 | Coupling 公式 | 推荐页面 |
| --- | --- | --- | --- | --- |
| Euler/block direct-backslip | `b = project(blocks[0] - blocks[1])` | `q = backslip_rate` | `coupling_ratio = -q / b` | 本页 |
| Deep-slip loading proxy | `b = matched deep slip` | `s = shallow_slip_rate` | `coupling_to_deep = (b - s) / b` | [深部滑动加载代理](deep_slip_loading_proxy.md) |

## 阅读路径

| 当前问题 | 建议阅读顺序 |
| --- | --- |
| 第一次配置 Euler/block direct-backslip | [配置分工](#配置分工) → [符号和公式](#符号和公式) → [Interseismic Config](#interseismic-config) |
| 需要分段 block pair、reference strike 或 motion sense | [Loading 与 Motion Sense 覆盖](#loading-与-motion-sense-覆盖的查阅与绘图) → [计算和导出](#计算和导出) |
| 添加 loading cap | [Euler Cap](#euler-cap) → [Cap 配置模式与排错](#cap-配置模式与排错) |
| 添加锁定、蠕滑或 prescribed backslip 硬约束 | [Backslip 约束](#backslip-约束) → [诊断报告](#诊断报告) |
| 只想解释求解结果 | [计算和导出](#计算和导出) → [诊断报告](#诊断报告) → [检查清单](#检查清单) |

## 配置分工

震间相关设置单独放在 `interseismic_config.yml`，主配置只保留一个文件指针：

```yaml
# default_config.yml
interseismic_config_file: interseismic_config.yml
units:
  observation: m/yr
```

`units.observation` 属于主配置，不属于 `interseismic_config.yml`。震间速度反演中它应与进入矩阵的数据单位一致，例如 `m/yr` 或 `mm/yr`。固定 Euler pole/vector 的输入始终按物理角速度解释，先转换为 Cartesian Euler vector `w`（radians/year），再由 ECAT 投影成 `m/yr` loading 并转换到 `units.observation`。由数据集 `eulerrotation` 估计的 block 参数则是反演矩阵单位中的系数；例如 `units.observation: mm/yr` 时，原始解向量中的 `eulerrotation` 数值约为物理 `w` 的 1000 倍，报告函数会乘以 `1e-3` 转回 radians/year 后再给出 Euler pole。

`interseismic_config.yml` 分为四层：

| 区块 | 职责 | 是否改变 loading |
| --- | --- | --- |
| `blocks` | 定义物理块体的 Euler 参数来源；同一 block 可让多个 dataset 的 `eulerrotation` 参数相等 | 是，作为 loading 的参数来源 |
| `fault_loading` | 指定每条断层两侧是哪两个 block，并在所有 patch 上计算长期构造加载率 `b` | 是 |
| `cap_constraints` | 可选地对部分 patch 添加 `|backslip| <= k * |loading|` 一类不等式 | 否 |
| `backslip_constraints` | 可选地对部分 patch 添加 `q=0`、`q+b=0`、`q+k*b=0` 等硬等式 | 否 |

关键原则：`blocks` 和 `fault_loading` 决定物理加载率；`cap_constraints` 和 `backslip_constraints` 只决定哪些 patch 被约束。不要用约束 selector 来间接定义构造加载率。

四层具有明确的单向依赖，不应互相代替：

```text
blocks
  -> 提供 Euler 参数来源或固定 Euler 值
fault_loading
  -> 选择有序 block pair，并在全部 patch 上定义 loading b
cap_constraints
  -> 可选：对选中 patch 添加 q 与 b 的不等式
backslip_constraints
  -> 可选：对选中 patch 添加 q 与 b 的硬等式
```

`blocks` 本身不选择 fault patch；`fault_loading` 的默认关系覆盖整条 fault，
`loading_overrides` 才对少量明确分段覆盖 block pair 或参考走向；cap/backslip 的
selector 只改变约束作用范围，不会让未选 patch 的 loading 变成零。

用户编写 YAML 时，顶层只使用以下字段：

| 顶层字段 | 主要子字段 | 不应承担的职责 |
| --- | --- | --- |
| `blocks` | `<block>.datasets`, `<block>.euler` | 不选择 fault 或 patch |
| `fault_loading` | `defaults`, `faults`, `loading_overrides`, `motion_sense_overrides` | 不固定 backslip 状态 |
| `cap_constraints` | `defaults`, `faults`, `selector`, `mode`, `max_coupling`, `hard_overlap` | 不定义 block pair、参考走向或 motion sense |
| `backslip_constraints` | `fault`, `state`, `selector`, `component`, `coupling/value` | 不定义长期 loading |
| `outputs` | 输出字段选择 | 不参与解算约束 |

此外可写 `version: 1`。`configured_blocks`、`configured_faults`、
`blocks_standard` 等是内部规范化结果，不需要手工写入 YAML。旧名
`loading_regions` 只作为兼容输入；新配置统一使用 `loading_overrides`。

约束层级按严格程度理解：

```text
fault_loading          -> 定义每个 patch 的长期加载率 b
backslip_constraints   -> hard equality，如 full_coupling / creep / prescribed_coupling
cap_constraints        -> 对仍自由估计的 q 与 b 添加比例或方向不等式
bounds_config.yml      -> 限制 q 本身的上下界和基本符号
```

默认情况下，`cap_constraints` 会自动跳过同一 fault、同一 component 上已经被
`backslip_constraints` 硬等式固定的 patch。也就是说，顶边 `full_coupling`
或底边 `creep` 已经精确定义状态时，不会再重复加入 cap 不等式。这避免了
`q+b=0` 与 `q+b<=0` 在 `max_coupling: 1.0` 时完全重合造成的数值退化。

## 符号和公式

线性滑动反演的当前解可抽象为：

```text
d = G x + epsilon
```

在当前震间接口中，反演得到的走滑分量按 direct backslip 解释：

```text
q = backslip_rate
```

两侧块体相对运动投影到断层走向方向后的长期加载率记为：

```text
b = block_slip_rate_signed = tectonic_loading_rate
```

派生字段为：

```text
slip_deficit_signed    = -q
slip_deficit_magnitude = abs(q)
coupling_ratio         = -q / b
coupling_magnitude     = abs(q) / abs(b)
creep_rate_signed      = b + q
creep_ratio            = abs(b + q) / abs(b)
```

当 `abs(b)` 小于数值阈值时，比值字段置为 `0`，避免除零。

当前 ECAT/CSI 走滑约定为左旋为正，因此通常有：

```text
dextral/right-lateral loading:    b < 0
sinistral/left-lateral loading:   b > 0
```

full coupling 对应：

```text
dextral:    q = -b > 0
sinistral:  q = -b < 0
```

## 加载投影与符号约定

`fault_loading.faults.<fault>.blocks` 是有序的代数约定，不是自动判定的东/西盘、上/下盘或块体拓扑：

```yaml
fault_loading:
  faults:
    MyFault:
      blocks: [Block_A, Block_B]
```

这表示 ECAT 在每个 patch 上计算：

```text
b = project(v_Block_A - v_Block_B, local_strike)
```

如果两个 block 顺序交换，`b` 会整体反号。为了让 ECAT 的走滑符号与 Blocks/celeri 常用约定一致，推荐把 `blocks` 理解为一个右手侧规则：

```text
blocks[0] = 位于 reference_strike 右手侧的 block
blocks[1] = 位于 reference_strike 左手侧的 block
```

这样：

```text
b = project(v_right_side - v_left_side, reference-strike-aligned local_strike)
```

在 ECAT 当前左旋为正的约定下，左旋加载通常得到 `b > 0`，右旋加载通常得到 `b < 0`。这也是把 Blocks/celeri 的 `east - west` 拓扑约定压缩到 ECAT 简单 fault-level block pair 时最不容易出错的写法。

需要注意，`east_side - west_side` 只能作为常见非东西向断层的简写。真正起决定作用的是“谁在 `reference_strike` 右手侧”。例如 NW-SE 右旋断层若使用东侧块体减西侧块体，`reference_strike` 通常应选 NW 向分支，使东侧块体位于走向右手侧；若误用相差 180° 的 SE 向分支，`b` 会整体反号。近东西向断层不建议用 east/west 描述两侧，应该直接按 `reference_strike` 的右手侧和左手侧来命名或检查。

当前 ECAT 不会自动根据断层迹线两侧的块体 polygon 判断谁减谁；复杂块体边界或同一断层不同段对应不同 block pair 时，应显式拆分断层或在脚本中维护清楚的分段关系。

ECAT 与专业块体模型程序的层级不同：

| 程序类型 | loading 关系 |
| --- | --- |
| ECAT 当前震间接口 | `one fault -> one default ordered block pair + optional manual loading_overrides`；patch 使用默认关系或显式 selector 命中的覆盖关系 |
| Blocks/celeri 类块体模型 | `one fault/network -> many segments -> each segment has a block pair -> patches inherit the segment pair` |

因此，ECAT 当前接口适合已经明确默认 block pair，并且只需少量手动分段覆盖的
线性滑动反演或局部 backslip/coupling 约束。它不会从 block polygon 自动推导
segment topology。若研究目标是完整块体网络、闭合 block polygon、同一长断层
不同段自动继承不同 block pair，建议直接使用更专业的震间块体模型反演程序，
或先在这些程序中确定 segment/block 关系，再把需要的断层几何和先验结果转入 ECAT。

Euler 速度先在 patch 中心投影到 EN 速度，再投影到走向：

```text
[E, N] = M(lon, lat) * omega
s      = [sin(strike), cos(strike)]
b      = (E_1 - E_2) * s_e + (N_1 - N_2) * s_n
```

其中 `strike` 来自断层对象的 `getStrikes()`，单位为 radians。`reference_strike` 的单位是 degrees，它只选择每个 patch 走向单位向量的正向分支：

```text
s_ref = [sin(reference_strike), cos(reference_strike)]
if dot(s, s_ref) < 0:
    s = -s
```

因此 `reference_strike` 不改变断层几何、不改变两个 block 的减法顺序，也不代表运动性质；它只决定 `b` 和走滑参数使用哪一个 180° 分支。若 `reference_strike` 差 180°，`b` 的符号也会翻转。

设置 `fault_loading` 时，建议按下面顺序检查：

1. 先选 `reference_strike`，使 `blocks[0]` 位于该走向的右手侧。
2. 再设置 `blocks: [right_side_block, left_side_block]`。
3. 再设置 `motion_sense`，让诊断和 cap 方向知道预期是右旋还是左旋。
4. 正式反演前运行 `print_interseismic_preflight_report()`，确认 loading 的符号和量级。

`motion_sense` 的职责更窄：它只用于 Euler cap 不等式方向和诊断报告中的符号期望，不参与 loading 计算。有效值包括 `dextral`/`right_lateral`/`right` 和 `sinistral`/`left_lateral`/`left`。在当前左旋为正的走滑约定下：

```text
dextral/right-lateral:    expected b < 0, cap uses q + k*b <= 0
sinistral/left-lateral:   expected b > 0, cap uses q + k*b >= 0
```

如果 `motion_sense` 与计算出的 loading 符号不一致，优先检查 `blocks` 顺序和 `reference_strike`，不要用 `motion_sense` 试图修正 block 顺序或投影方向。

倾角、倾向和上/下盘属于断层几何定义，不参与 `fault_loading.blocks` 的减法顺序。震间 coupling/backslip 约束目前主要围绕走滑分量 `q` 和走滑 loading `b`；若需要倾滑方向的零滑、边界零滑或普通线性约束，应使用 `bounds_config.yml:source_constraints` 或对应脚本接口，而不是用 `motion_sense` 重新解释倾向。

## Interseismic Config

最小模板可由 CLI 生成：

```bash
ecat-generate-interseismic -o interseismic_config.yml -f MyFault
```

典型结构：

```yaml
version: 1

blocks:
  Block_A:
    datasets: [GPS_Block_A]
    euler:
      mode: estimate
  Block_B:
    datasets: [GPS_Block_B]
    euler:
      mode: fixed_pole
      value: [100.2, 25.5, 0.45]
      units: [degrees, degrees, degrees_per_myr]

fault_loading:
  enabled: true
  defaults:
    reference_strike: 0.0
    motion_sense: sinistral
  faults:
    MyFault:
      blocks: [Block_A, Block_B]
      reference_strike: 90.0
      motion_sense: sinistral

cap_constraints:
  enabled: false
  faults:
    MyFault:
      selector: null

backslip_constraints:
  - fault: MyFault
    state: full_coupling
    selector: {edge: top}
    component: strikeslip
    name: top_full_coupling
```

`blocks` 先定义物理块体和该块体对应的数据集。`geodata.polys` 仍然是数据改正参数的统一入口；当某个数据集在 `geodata.polys` 中打开 `eulerrotation`，震间 block 模型可以把这三列解释为该数据集所属块体的 Cartesian Euler vector。

parser 会拒绝未知顶层键、未知 fault 名，以及 `blocks`、`fault_loading`、
`cap_constraints`、`backslip_constraints` 主要层级中的未知字段；报错信息包含完整
YAML 路径。解析后的 `config.interseismic_config` 也可以再次交给 parser，因此脚本
可深拷贝当前配置、修改 selector，再调用 `update_interseismic_config(...)`。重载采用
声明式替换：旧配置生成的 block/cap/backslip 矩阵组会被替换，manual/runtime 组保留。

`euler.mode` 有三种常用取值：

| mode | 含义 | 对 `datasets` 的处理 |
| --- | --- | --- |
| `estimate` | 块体 Euler 从数据集的 `eulerrotation` 三列估计 | 同一 block 下所有 datasets 默认共享同一个 Euler；内部自动选择一个 anchor dataset 作为矩阵锚点 |
| `fixed_pole` | 块体 Euler pole 固定为 `[lon, lat, omega]` | 若 datasets 打开了 `eulerrotation`，这些列会被固定到该 block Euler；若没有打开，表示数据已预先扣除该块体刚性运动 |
| `fixed_vector` | 块体 Cartesian Euler vector 固定为 `[wx, wy, wz]` | 同 `fixed_pole` |

`fixed_pole` 的顺序是 `[lon, lat, omega]`；解析后内部先得到 `[lon_rad, lat_rad, omega_rad_per_year]`，再转换为 Cartesian Euler vector `[wx, wy, wz]`。`fixed_vector` 直接写 `[wx, wy, wz]`，默认单位为 radians/year。约束管理器会添加 `interseismic_block_euler_constraints` 等式组，用于共享估计 Euler 或固定数据集 Euler；若主配置是 `mm/yr`，固定到数据集 `eulerrotation` 列的右端项会自动转换为矩阵单位，不要求用户手动乘 1000。

若 block 的 Euler 模式是 `estimate`，反演结果中的 `eulerrotation` 三列对应 Cartesian
Euler vector 的矩阵系数。它不是 GPS `translationrotation/full/strain` 里的局部归一化旋转项；解释为物理 Euler vector 时需要结合 `units.observation`。
反演后可用数据改正参数报告把它转成常用 Euler pole：

```python
report = inversion.collect_data_correction_parameters(datasets="GPS_Block_A")
inversion.print_data_correction_report(report)
```

该报告只解释参数，不改变 `fault_loading`、cap 或 backslip 约束。若同一数据集还打开了
`internalstrain`、`full` 或 `strain`，应检查这些长波项是否与 Euler/block loading 或 coupling
发生 trade-off。

## Loading 与 Motion Sense 覆盖的查阅与绘图

当一条 fault 使用 `loading_overrides` 做手动分段时，只看 YAML 很难确认每个 patch
最终继承了哪一组 block pair。ECAT 提供一组只读辅助接口，用于检查
`fault_loading` 的实际解析结果：

```python
pair_result = inversion.resolve_interseismic_loading_pairs("MyFault")

inversion.print_interseismic_pair_diagnostics("MyFault", result=pair_result)

inversion.plot_interseismic_loading_pairs(
    "MyFault",
    field="pair_id",
    cmap="tab20",
    plot_on_2d=False,
)

inversion.export_interseismic_loading_pair_table(
    "MyFault",
    "output/MyFault_loading_pairs.csv",
    result=pair_result,
)
```

这些接口只做查阅，不改变 `fault_loading`、cap、不等式、hard equality 或
`fault.slip`。`pair_id` 是稳定的配置标签：`0` 表示 fault 级默认 pair，
`loading_overrides[0]` 为 `1`，`loading_overrides[1]` 为 `2`，依此类推；
即使只查某个 patch 子集，编号也不会改变。CSV 表会列出 `patch_index`、
`pair_id`、`source`、`region_name`、两侧 block 名、`reference_strike`、
`motion_sense`、可计算时的 loading，以及 patch 中心坐标。

也可以绘制实际 loading：

```python
inversion.plot_interseismic_loading_pairs(
    "MyFault",
    field="loading",
    cmap="cmc.vik",
    cblabel="Tectonic loading rate",
)
```

建议使用顺序是：先画 `pair_id` 确认分段是否命中正确 patch，再看 loading 的正负号和量级，
最后再启用或调试 cap/backslip 约束。若没有当前解向量，pair assignment 仍可解析，
但 loading 数值会留空；正式解释 loading 前应先完成反演或显式传入 `solution`。

如果只想测试某一段是否应按另一种走滑模式约束，而不想改变该段的 block pair，
使用 `motion_sense_overrides`。它只覆盖 cap 方向和诊断期望，不重新计算 loading `b`：

```yaml
fault_loading:
  faults:
    MyFault:
      blocks: [Block_A, Block_B]
      reference_strike: 315
      motion_sense: dextral
      loading_overrides:
        - name: south_pair
          selector: {trace_segment: {start: {longitude: 103.0}, end: {longitude: 104.0}}}
          blocks: [Block_C, Block_B]
          reference_strike: 315
          motion_sense: dextral
      motion_sense_overrides:
        - name: trial_sinistral_zone
          selector: {trace_segment: {start: {longitude: 101.8}, end: {longitude: 102.4}}}
          motion_sense: sinistral
```

查阅和绘图：

```python
sense_result = inversion.resolve_interseismic_motion_sense("MyFault")
inversion.print_interseismic_motion_sense_diagnostics("MyFault", result=sense_result)

inversion.plot_interseismic_motion_sense(
    "MyFault",
    field="motion_sign",
    cmap="coolwarm",
    plot_on_2d=False,
)

inversion.export_interseismic_motion_sense_table(
    "MyFault",
    "output/MyFault_motion_sense.csv",
    result=sense_result,
)
```

`motion_sign` 中 `+1` 表示 dextral/right-lateral，`-1` 表示
sinistral/left-lateral。`source_id` 中 `0` 表示 fault 默认，
正值表示 `loading_overrides`，负值表示 `motion_sense_overrides`。

## Euler Cap

Euler cap 是可选不等式，作用在 direct backslip `q` 和 loading `b` 上。`k` 是 `cap_constraints` 中的 `max_coupling`，默认 1.0；旧键名 `factor` 仍作为输入别名解析为 `max_coupling`。

ECAT 支持两种 cap 模式：

| `mode` | 适用场景 | 约束含义 |
| --- | --- | --- |
| `motion_sense` | 默认；loading 可由待估 Euler 参数决定 | 按 `motion_sense` 生成一侧上限，需要 `bounds_config.yml` 同时给 `q` 设置基础符号 |
| `loading_sign` | 两侧 loading 已固定，且每个 patch 的 `b` 符号在求解前已知 | 按实际 `sign(b)` 自动生成 `0 <= -q/b <= k` |

这里有一个命名细节：`cap_constraints.mode: motion_sense` 中的
`motion_sense` 是“使用 fault loading 方向”的模式名，不是一个新的右旋/左旋
字段。右旋/左旋只能写在 `fault_loading.faults.<fault>.motion_sense`，或写在
`fault_loading.faults.<fault>.loading_overrides[].motion_sense` 做 loading 分段覆盖，
也可以写在 `fault_loading.faults.<fault>.motion_sense_overrides[]` 中只覆盖运动模式预期。
`cap_constraints` 只接受 `selector`、`mode`、`max_coupling`、`hard_overlap`、
`min_loading_abs` 等 cap 自身字段；误把 `motion_sense`、`reference_strike`
或 `blocks` 放到 cap 下会直接报错。

默认 `motion_sense` 模式的公式为：

```text
dextral:    q + k*b <= 0  ->  q <= -k*b
sinistral:  q + k*b >= 0  ->  q >= -k*b
```

如果 `bounds_config.yml` 同时约束 `q` 的基本符号，就得到常见的 `0 <= coupling <= 1` 区间：

```text
dextral:    0 <= q <= -b       when k = 1
sinistral: -b <= q <= 0        when k = 1
```

`loading_sign` 模式会对每个选中 patch 生成两行：

```text
s*q <= 0
-s*q <= k*abs(b)
where s = sign(b)
```

它直接表达 `0 <= -q/b <= k`，并在构造矩阵时检查两件事：`b` 不能依赖待估 Euler 参数，且 `|b|` 不能小于 `min_loading_abs`。如果这两个条件不满足，应继续使用默认 `motion_sense`，或改用更完整的块体模型/迭代约束方案。

### Cap 配置模式与排错

`cap_constraints.enabled: true` 只打开 cap 机制；真正生成约束行还需要
`cap_constraints.faults` 中存在目标 fault。推荐显式写出目标 fault：

```yaml
cap_constraints:
  enabled: true
  defaults:
    selector: null
    mode: motion_sense
    hard_overlap: skip
    max_coupling: 1.0
  faults:
    MyFault:
      selector: null       # null means all patches on MyFault
      mode: motion_sense   # or loading_sign when loading is fixed
      hard_overlap: skip    # default: skip hard equality patches
      max_coupling: 1.5    # optional; omit for 1.0
```

不要写成下面这样：

```yaml
cap_constraints:
  faults:
    MyFault:
      motion_sense: dextral  # invalid; put this under fault_loading
```

也可以省略 `faults` 键，让解析器对所有已配置 `fault_loading` 的 fault 使用
`defaults`：

```yaml
cap_constraints:
  enabled: true
  defaults:
    selector: null
```

不要把下面这种写法理解成“使用 defaults 作用到所有 fault”：

```yaml
cap_constraints:
  enabled: true
  defaults:
    selector: null
  faults: {}
```

`faults: {}` 是显式空映射，结果是没有任何 fault 进入 cap，Euler-cap 矩阵行数为
0。此时即使 `enabled: true`，`coupling_ratio > 1` 也不会被 cap 阻止。

反演前应查看：

```python
inversion.print_interseismic_preflight_report()
inversion.print_interseismic_constraint_report("MyFault")
```

重点看输出中类似下面的信息：

```text
Euler-cap rows: configured=..., matrix=...
Euler-cap matrix group: rows=..., cols=...
constraints: cap=active/configured (k=..., skipped_hard=...)
```

若 `configured=0`、`matrix=unavailable` 或报告说没有 `euler_cap_constraints`
group，说明 cap 没有实际进入求解矩阵。
`cap=active/configured` 中，`active` 是实际进入 cap 不等式的 patch 数；
`configured` 是 selector 原始选中的 patch 数；`skipped_hard` 是因为同一 patch
已经被 hard equality 固定而自动跳过的数量。

常见现象与原因：

| 现象 | 优先检查 |
| --- | --- |
| `cap_constraints.enabled: true`，但 `coupling_ratio` 仍大量大于 1 | `cap_constraints.faults` 是否为 `{}`；preflight 中 Euler-cap rows 是否为 0 |
| cap 行数正常，但仍出现 `coupling_ratio > 1` | `max_coupling` 是否大于 1；若 `max_coupling: 1.5`，`coupling_ratio` 到 1.5 以内是被允许的 |
| 默认 `motion_sense` 下 cap 行数正常，但 `coupling_ratio` 超过 `max_coupling` | `bounds_config.yml` 是否同时约束了 direct backslip `q` 的基础符号；若 loading 已固定，也可改用 `mode: loading_sign` |
| `max_coupling: 1.0` 加顶边 `full_coupling` 后求解退化 | 默认 `hard_overlap: skip` 会让 cap 自动跳过顶边 hard patches；若显式设为 `keep`，`q+b=0` 和 `q+b<=0` 会在同一 patch 上零松弛重合 |
| `loading_sign` 报错说 loading 依赖待估参数 | 该模式只适合固定 Euler/block loading；待估 Euler loading 的符号必须用 `motion_sense` 或更完整的迭代约束处理 |
| cap 只对部分 patch 生效 | `selector` 是否只选中了局部 patch；用 `print_interseismic_constraint_report(...)` 看 selected patches |
| loading 量级接近 0 | 先检查 `blocks` 顺序、`reference_strike`、GPS `eulerrotation` 参数和内部应变 trade-off |

若使用 `loading_overrides`，默认 `motion_sense` cap 会按 override 的
`motion_sense` 覆盖对应 patch 的不等式方向；未命中 override 的 patch 使用 fault 级
`motion_sense`。若只是测试某段走滑模式转换而不改变 block pair，优先使用
`motion_sense_overrides`。

是否强制这个区间取决于科学假设。若允许 over-coupling、反向 creep 或数据驱动的异常 patch，可以只设置宽 bounds，不启用 cap，或只对特定 patch group 启用 cap。

动态更新 cap selector：

```python
patch_ids = select_patch_indices(
    fault,
    {
        "trace_segment": {
            "start": {"point": [100.25, 25.57], "coord_system": "lonlat"},
            "end": {"point": [101.80, 23.80], "coord_system": "lonlat"},
        }
    },
)

inversion.update_euler_cap_constraint(
    "MyFault",
    selector={"patches": patch_ids},
    mode="motion_sense",
    max_coupling=1.5,
    enabled=True,
)
```

这只改变 cap 不等式的应用范围，不改变 `fault_loading` 计算出的 loading。
固定 loading 场景可以传入 `mode="loading_sign"`；若存在 loading 接近 0 的 patch，
可同时设置 `min_loading_abs` 让接口提前报错。

若分段边界需要循环测试，可用 trace marker helper 在脚本中生成 selector，
而不是把循环逻辑写进 YAML：

```python
from eqtools.csiExtend import (
    sample_trace_markers,
    update_fault_loading_override_from_trace_segment,
    update_fault_motion_sense_override_from_trace_segment,
)

markers = sample_trace_markers(
    fault,
    {"longitude": 101.5},
    {"longitude": 102.5},
    step_km=5.0,
)

trial_config, meta = update_fault_loading_override_from_trace_segment(
    inversion.config.interseismic_config,
    fault,
    "MyFault",
    "north_segment",
    markers[0],
    {"longitude": 102.5},
    depth_range=(0.0, 25.0),
    return_metadata=True,
)
```

如果搜索的是局部 dextral/sinistral 转换点，而 block pair 不变，改用：

```python
trial_config, meta = update_fault_motion_sense_override_from_trace_segment(
    inversion.config.interseismic_config,
    fault,
    "MyFault",
    "trial_sinistral_zone",
    markers[0],
    {"longitude": 102.5},
    depth_range=(0.0, 25.0),
    return_metadata=True,
)
```

`meta["trace_distance_km"]` 是沿 fault trace 的真实弧长距离，通常为 km；
它不是 GPS station distance，也不是经度或纬度方向上的单轴距离。

## Backslip 约束

`backslip_constraints` 和脚本接口支持以下状态：

| `state` | 公式 | 用途 |
| --- | --- | --- |
| `free` | 不生成矩阵行 | 自由估计 |
| `creep`, `zero_backslip` | `q = 0` | 完全 creep / 零 backslip |
| `full_coupling` | `q + b = 0` | 完全闭锁 |
| `prescribed_coupling` | `q + k*b = 0` | 指定 coupling fraction |
| `prescribed_backslip` | `q = value` | 指定 backslip rate |

脚本接口：

```python
inversion.add_interseismic_backslip_constraint(
    "MyFault",
    state="prescribed_coupling",
    selector={"edge": "bottom"},
    coupling=0.0,
)

inversion.add_interseismic_backslip_constraint(
    "MyFault",
    state="full_coupling",
    selector={"edge": "top"},
    name="top_full_coupling",
)
```

`add_interseismic_backslip_constraint()` 只新增；同名组已存在时会报错。如果要改变
已经命名的用户约束，应显式调用：

```python
inversion.replace_interseismic_backslip_constraint(
    "MyFault",
    state="prescribed_coupling",
    selector={"edge": "top"},
    coupling=0.8,
    name="top_full_coupling",
)
```

配置重载会事务式重建 config-owned backslip 组，因此
`backslip_constraints` 不使用 `overwrite` 字段。

`full_coupling` 和 `prescribed_coupling` 依赖 `fault_loading`，目前只支持 `component="strikeslip"`。`q=0` 或 `q=value` 可用于 `strikeslip` 或 `dipslip`。

Selector 由 [Fault Patch Indices](fault_patch_indices.md) 统一解析，常用形式包括：

```python
{"edge": "top"}
{"edges": ["top", "bottom"]}
{"patches": [0, 1, 2]}
{"depth_range": [0.0, 10.0]}
{"trace_segment": {"start": {"longitude": 101.8}, "end": {"longitude": 102.4}}}
```

完整可复制 selector 写法见 [Fault Patch Indices](fault_patch_indices.md#selector-cookbook)。震间分段推荐使用 `trace_segment` 的 `start/end` 写法；`trace_range` 是兼容旧配置和明确 `point1/point2` 输入的低层形式。

## 计算和导出

BLSE/VCE 完成求解后：

```python
result = inversion.calculate_interseismic_fields(
    "MyFault",
    slip_component="strikeslip",
)

loading = result["fields"]["tectonic_loading_rate"]
backslip = result["fields"]["backslip_rate"]
coupling = result["fields"]["coupling_ratio"]
creep = result["fields"]["creep_rate_signed"]
```

可用短别名：

```python
coupling = inversion.get_interseismic_field("MyFault", "coupling")
loading = inversion.get_interseismic_field("MyFault", "loading")
```

旧的 `locking_relative`、`locking_absolute`、`locking_rel` 和 `locking_abs` 不再作为字段或 alias 暴露。新脚本应显式使用 `coupling_ratio`、`coupling_magnitude`、`slip_deficit_signed` 或 `slip_deficit_magnitude`。

绘图和导出不会覆盖 `fault.slip`：

```python
inversion.plot_interseismic_field(
    "MyFault",
    field="coupling_ratio",
    cmap="viridis",
    cblabel="Coupling ratio",
)

inversion.write_interseismic_field_gmt(
    "MyFault",
    "coupling_ratio",
    "output/MyFault_coupling_ratio.gmt",
)

inversion.write_interseismic_field_centers(
    "MyFault",
    "tectonic_loading_rate",
    "output/MyFault_loading_centers.txt",
)
```

批量导出：

```python
inversion.export_interseismic_results(
    "MyFault",
    "output/interseismic_MyFault",
)
```

## 诊断报告

建议震间案例在应用约束后、正式反演前先打印全局 preflight 报告：

```python
inversion.print_interseismic_preflight_report()
```

preflight 报告面向快速排错，输出保持紧凑。它按 fault 列出：

| 项 | 作用 |
| --- | --- |
| block 顺序 | 明确 loading 使用 `first block - second block` |
| `reference_strike`、`motion_sense` | 检查走向分支和右旋/左旋符号期望 |
| loading `b` 统计 | 检查构造加载率的量级、median 和正负号 |
| cap patch 数 | 检查 Euler cap 选择了多少 patch |
| hard backslip/coupling 行数 | 检查 `backslip_constraints` 实际约束规模 |
| cap 与 hard/free overlap | 检查是否把同一批 patch 同时设为 cap、full coupling 或 free |

`state="free"` 不会关闭 Euler cap；如果某些 patch 需要完全自由，应从 cap selector 中排除它们。若 preflight 报告指出 loading 符号与 `motion_sense` 不一致，应优先检查 block 顺序和 `reference_strike`。

需要细查单条断层的 bounds、当前 backslip 和 Euler-cap 矩阵行时，再打印单断层报告：

```python
inversion.print_interseismic_constraint_report("MyFault")
```

单断层报告会列出：

| 项 | 作用 |
| --- | --- |
| `q`、`b` convention | 检查 direct backslip 和 loading 定义 |
| `motion_sense`、`reference_strike` | 检查右旋/左旋符号和投影方向 |
| selected patches | 检查诊断范围或 cap 范围 |
| loading/backslip stats | 检查数量级和正负号 |
| bounds stats | 检查 `bounds_config.yml` 是否给出合理符号和范围 |
| Euler-cap matrix group | 检查 cap 约束是否实际生成矩阵行 |

这些报告只做检查，不自动修改 solver。

## Bayesian 使用边界

Bayesian 入口也支持同一套接口。通常先选择一个代表模型，例如 `median`、`mean` 或 `MAP`，再解释震间字段：

```python
result = inversion.calculate_interseismic_fields(
    "MyFault",
    model="median",
    slip_component="strikeslip",
)
```

这等价于先运行 `returnModel(model="median", print_fit_statistics=False)`，再读取当前 `mpost`。如果非线性几何或 mesh 随 posterior 样本变化，不应直接平均不同样本的 patchwise coupling；应先定义固定 mesh 或投影规则。

## 检查清单

- `default_config.yml` 只保留 `interseismic_config_file` 指针，不再写旧 `euler_constraints`。
- `interseismic_config.yml:blocks` 定义了需要的块体 Euler 参数来源。
- `interseismic_config.yml:fault_loading` 对目标 fault 已启用，且两个 block 名称正确。
- 正式反演前运行 `print_interseismic_preflight_report()`，确认 block 顺序、loading 符号、cap 数量和 hard/free overlap。
- Dextral/full coupling 通常应看到 `b < 0` 且 `q=-b>0`；sinistral/full coupling 通常应看到 `b > 0` 且 `q=-b<0`。
- Cap selector 只控制不等式应用范围，不控制 loading 是否为 0。
- 顶边、底边或局部 patch 的 hard backslip 约束优先写在 `backslip_constraints` 或通过 `add_interseismic_backslip_constraint(...)` 添加。
- 输出 coupling 时使用 `coupling_ratio` 或 `coupling_magnitude`，不要在新脚本里使用旧 locking 名称。

## 相关页面

- [线性滑动反演配置](config_linear_slip.md)
- [ECAT 约束管理器](constraint_manager.md)
- [深部滑动加载代理](deep_slip_loading_proxy.md)
- [Fault Patch Indices](fault_patch_indices.md)
- [BLSE/VCE 线性滑动反演](../workflows/04_linear_slip_blse_vce.md)
