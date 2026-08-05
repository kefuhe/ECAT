# BLSE 固定拓扑倾角搜索

当断层迹线、上下边界和其他几何已经确定，但倾角仍需要用分布式滑动反演比较时，
可以对一组候选倾角分别运行 BLSE。ECAT 提供的标准模板是
[`scripts/test_dip_search_BLSE.py`](../../scripts/test_dip_search_BLSE.py)。

这是一种**条件模型比较**：每个候选倾角都对应一次完整、独立的 BLSE。它不是
Bayesian 倾角后验，也不能只凭最小 RMS 代替几何合理性、粗糙度、残差和机制检查。
建议先用 [固定几何平滑参数搜索](04a_blse_smoothing_search.md) 选出一个合理 penalty
范围，再固定其中一个值执行本页倾角搜索。


## 为什么使用固定拓扑

如果每个倾角都重新 Gmsh 剖分，patch 数量、位置和编号可能同时改变，RMS 差异会混入
离散化差异。标准模板采用：

```text
参考倾角
  -> remap=True：生成一次参考三角网格和顶点映射
每个候选倾角
  -> 更新真实 top/bottom 几何
  -> remap=False：只变形同一套网格
  -> 新建 BLSE 对象，重算 GF、Laplacian、边界和 rake 约束
  -> 保存逐数据集和全局 RMS/VR
```

固定拓扑的目的只是让不同倾角使用相同 patch 身份。物理坐标和 Green's functions 仍随
倾角改变，所以不能在候选之间复用同一个 inversion 对象。

## 准备输入

先准备：

- 两列 `lon lat` 的地表迹线；
- 已读取并检查的 InSAR/GPS 数据；
- 与普通 BLSE 相同的 `default_config_BLSE.yml` 和 `bounds_config.yml`；
- 顶部/底部深度、下倾方向、参考倾角和候选倾角；
- 一组固定的 `alpha` 或 `penalty_weight`。

断层对象名必须与两个配置文件中的 source 名一致，`geodata` 顺序也必须与主配置一致。
倾角候选在这里是送入 CSI/Okada 的物理倾角，范围为 `(0, 90]`。输入角度协议见
[断层角度约定](../concepts/fault_angle_conventions.md)。

## 复制并修改标准模板

把模板复制到自己的线性反演目录，放在配置文件旁，再修改其中标明的路径和参数：

```bash
python test_dip_search_BLSE.py
```

实际案例通常把脚本复制到配置文件旁运行。第一轮只需重点修改：

```python
lon0 = 87.5
lat0 = 28.5

output_dir = "dip_search_results"
config_file = "default_config_BLSE.yml"
bounds_file = "bounds_config.yml"
alpha = [-2.0]

mainfault_reference_dip = 65.0
mainfault_dips = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0]

fault_name = "MainFault"
fault_trace_file = "../Faults/main_fault_trace.txt"
fault_top = 0.0
fault_depth = 20.0
dip_direction = 180.0
top_size = 1.0
bottom_size = 2.0
```

`mainfault_reference_dip` 只用于建立共享拓扑，通常取候选范围中部或已有优选值；它不等于
最终选定倾角。`top_size/bottom_size` 决定参考 patch 离散化，所有候选随后复用它。

模板保持数据读取为顺序式代码，便于用户直接增删某一条观测。若使用 GPS、更多 InSAR
或不同 reader，先按 [InSAR 与 GPS 数据读取](01_data_reading_insar_gps.md) 建好对象，再
保持 `geodata` 顺序与配置一致。

## 输出与判断

默认输出：

| 文件 | 内容 |
| --- | --- |
| `dip_search_results/dip_search.csv` | 每个倾角的 patch 数、粗糙度、全局及逐数据集 RMS/VR |
| `dip_search_results/dip_search.png` | 全局 RMS 和 VR 随倾角的变化 |

模板使用 `collect_fit_statistics(data_poly="config")`，因此统计量包含配置中实际求解的
offset/ramp 等数据改正项。公式和 scope 区别见 [Fit Statistics](../reference/fit_statistics.md)。

每个倾角都会建立新的 inversion；应在该轮求解后立即按
[循环统计公共骨架](../examples/script_templates.md#loop-statistics) 取 rows，并给每一行附加
`dip_deg`，不能等进入下一倾角后再读取上一轮状态。

至少检查：

- 所有候选的 `n_patches` 相同；脚本发现变化会立即报错；
- 最小 RMS 附近是否形成稳定趋势，而不是单个边界点；
- 逐轨道 RMS/VR 是否一致改善；
- residual 是否仍有 ramp、解缠跳变或局部系统误差；
- 粗糙度、滑动分布、深度范围和震源机制是否合理；
- 候选范围或网格尺度轻微改变后结论是否稳定。

## 多断层怎么扩展

公开模板默认只构建一个断层，避免把最常见场景复杂化。多断层时，每个断层都必须拥有
自己的参考 mesh 和候选列表，不能共享顶点映射：

```python
from itertools import product

mainfault_reference_dip = 65.0
mainfault_dips = [55.0, 60.0, 65.0, 70.0, 75.0]

secondary_reference_dip = 75.0
secondary_dips = [75.0]  # 一个值表示该断层固定，只搜索 MainFault。

for mainfault_dip, secondary_dip in product(
    mainfault_dips,
    secondary_dips,
):
    # 分别对 mainfault 和 secondary_fault 调用 remap=False。
    # 然后按固定顺序建立一个新的 BLSE 对象。
    faults = [mainfault, secondary_fault]
```

若两个列表都有多个值，执行的是笛卡尔积。例如 `8 × 7` 会运行 56 次 BLSE。只需要一个
断层参与时，不要构建无关的第三断层；配置中未参与的 source 项不会替代脚本中的实际
source，但正式运行前仍应检查最终约束摘要。

多断层扩展必须保持：

- 每个断层先各自执行一次 `remap=True`；
- 每个候选只对对应断层执行 `remap=False`；
- `OrderedDict`、配置 source 名和 `faults_list` 顺序固定；
- 每个组合新建 `BoundLSEMultiFaultsInversion`；
- 组合数在运行前明确打印，避免意外进行过大的网格搜索。

## 下一步

- 尚未选择合理平滑强度：先看 [固定几何平滑参数搜索](04a_blse_smoothing_search.md)。
- 倾角结果随平滑强度变化明显：再看
  [倾角 × 平滑参数敏感性分析](04c_blse_dip_smoothing_search.md)。
- 普通 BLSE 构建、配置和输出：看 [BLSE/VCE 线性滑动分布反演](04_linear_slip_blse_vce.md)。
- bounds、rake 和 patch 约束：看 [ECAT 约束管理器](../reference/constraint_manager.md)。
- RMS/VR、poly 和全局统计：看 [Fit Statistics](../reference/fit_statistics.md)。
- 若需要完整几何概率分布而非条件网格比较：回到
  [Bayesian 非线性几何反演](03_nonlinear_geometry_bayesian.md)。
