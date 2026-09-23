# 坐标、刚体与组合几何扰动

本页按几何语义解释公开扰动族，不把每个 `perturb_*` 方法复制成静态清单。当前安装版本的
方法名、参数个数和 reference 依赖以 `fault.help()` 与 CLI 为准。

## 先按 target 选择族

| target | 典型方法族 | 几何意义 | 后续 mesh |
| --- | --- | --- | --- |
| `top` 或 `bottom` | fixed-direction、translation、rotation、endpoint | 只改一条边界 | 需要 MeshPolicy 或带 mesh 后缀 |
| `layer` | layer fixed-direction/translation/rotation | 改一层边界 | 通常由 multiLayer 路径重建 |
| `geometry` | geometry translation/rotation/fixed-direction | 直接作用完整几何或 vertices | 必须满足 whole-mesh pair |
| 多 target/多 stage | Rotate+Translate、bottom+geometry、endpoint+midpoint | 固定顺序组合多个操作 | 由组合方法合同决定 |
| dip profile | dip generator | 从 top、dip 与 strike 重新生成 bottom | 见独立倾角剖面页 |

“同一个平移量”不表示 target 等价：只平移 bottom 是 deform；整个 mesh 统一平移才是 rigid。
target 是科学语义的一部分，不能由方法名相近而互换。

## 固定方向、平移和旋转

- fixed-direction：标量或逐可动节点位移沿明确方向作用；角度单位和节点选择由方法签名定义。
- translation：通常消费二维平移量，方向由 fault-local x/y 基底解释。
- rotation：围绕 reference 确定的 pivot 旋转；`recalculate_pivot=False` 保持候选共享同一参考
  pivot，避免候选间基线漂移。
- endpoint：只改端点或由端点与中点重建曲线，参数顺序必须按方法注册信息核对。

对 top/bottom/layer 的局部操作会改变相对距离，通常报告 deform。对完整 mesh 的统一旋转或
平移保持形状、面积和拓扑，报告 rigid；GF 仍因空间位置变化而失效。

## 组合方法

组合方法不是把几个独立方法在脚本中随意串联。它公开固定的 Stage 顺序和参数布局，例如：

```text
sample slice
  -> split by registered perturbation_items
  -> Stage 1
  -> Stage 2
  -> ...
  -> one MeshPolicy
  -> materialize once
```

其必要性是让一个候选只从 reference 构造一次、只发布一次 mesh 变化性质，并让
`sample_positions`、bounds、结果列和方法参数保持同一顺序。不要在候选函数外再追加隐式
几何操作，否则预检无法描述完整 target。

## 倾角剖面后接刚体旋转和平移

公开方法
`perturb_dips_with_preset_params_and_rigid_transform()` 固定执行：

```text
frozen top/profile/density
  -> 解析 dip 并生成 candidate bottom
  -> 同时旋转 candidate top、bottom 和 trace top
  -> 同时平移 candidate top、bottom 和 trace top
  -> materialize（不在该方法内建 mesh）
```

局部样本切片为

\[
\boldsymbol\theta
=
[\Delta d_0,\ldots,\Delta d_{K-1},\varphi,\Delta x,\Delta y]^{\mathsf T}.
\]

其中 \(K\) 由 dip profile 决定：未分组时是一个广播增量或 sampled control 数，显式
`perturbation_groups` 时是唯一标签数。\(\Delta d_k\) 与旋转角 \(\varphi\) 使用
`angle_unit`，\(\Delta x,\Delta y\) 是 fault-local km。最后三个参数的位置固定，不能
与 dip 增量交换。

设候选工作 top 的第 \(i\) 个点为 \(\mathbf T_i\)，从 profile 得到的规范倾角和局部
走向为 \(\widetilde d_i,\widetilde a_i\)，垂向深度差为
\(H=\texttt{depth}-\texttt{top}\)。生成阶段使用

\[
\mathbf B_i
=
\mathbf T_i
+
\frac{H}{\sin\widetilde d_i}
\begin{bmatrix}
\cos\widetilde d_i\cos(-\widetilde a_i)\\
\cos\widetilde d_i\sin(-\widetilde a_i)\\
\sin\widetilde d_i
\end{bmatrix}.
\]

这里 \(\widetilde a_i\) 已包含反倾角的等价走向修正；代码不会用旋转后的全局方位反推
bottom。bottom 先由参考 top 的局部几何生成，随后 top 和 bottom 才作为同一刚体变换。

对冻结 reference top 确定的 pivot \(\mathbf c\)，水平旋转和平移为

\[
\mathbf R(\varphi)=
\begin{bmatrix}
\cos\varphi&-\sin\varphi\\
\sin\varphi&\cos\varphi
\end{bmatrix},
\qquad
\mathbf x_{r,xy}
=
\mathbf c+\mathbf R(\varphi)(\mathbf x_{xy}-\mathbf c),
\]

\[
\mathbf x_{f,xy}
=
\mathbf x_{r,xy}
+
\begin{bmatrix}\Delta x\\\Delta y\end{bmatrix},
\qquad
x_{f,z}=x_{r,z}.
\]

`pivot="start"|"end"|"midpoint"` 始终在冻结 reference top 上解析，常规加密或 control 事件
节点不会移动 pivot。其中 `midpoint` 是 frozen top 所有节点二维坐标的算术均值，不是沿迹线
弧长中点；设置 `force_pivot_in_coords=True` 时，再取离该均值最近的 frozen top 节点。显式
坐标由 `pivot_is_utm` 决定输入帧。旋转以后
`top_strike` 从最终 candidate top 重新计算，避免保留旋转前的走向元数据；倾角值不因刚体
变换改变。

完整扰动向量必须是有限数值。NaN、正无穷或负无穷会在任何坐标、mesh 或缓存有效性状态
改变之前报错；不会发布一个部分完成的候选。

Python 准备保持普通 dip-profile 入口：

```python
profile = fault.set_dip_profile(
    sampled_controls=sampled_controls,
    fixed_controls=fixed_controls,
    perturbation_groups=["wm", "wm", "east"],
    interpolation_axis="arc_length",
    transition_zones=transition_zones,
    is_utm=False,
)
fault.set_densification(interval=2.0)  # 可选

# 两个 dip groups + rotation + dx + dy，共 5 个零增量。
fault.perturb_dips_with_preset_params_and_rigid_transform(
    np.zeros(profile.perturbation_parameter_count + 3),
    pivot="midpoint",
    angle_unit="degrees",
)
fault.generate_and_deform_mesh(..., remap=True)
```

对应 Bayesian 配置只引用该公开方法；mesh 仍由独立 `update_mesh` 负责：

```yaml
faults:
  MainFault:
    geometry:
      update: true
      sample_positions: [0, 5]  # [dip_change[wm], dip_change[east], rotation, dx, dy]
    method_parameters:
      update_fault_geometry:
        method: perturb_dips_with_preset_params_and_rigid_transform
        pivot: midpoint
        angle_unit: degrees
        use_average_strike: false
      update_mesh:
        method: generate_and_deform_mesh
        num_segments: 50
        disct_z: 10
        remap: false
```

```yaml
geometry:
  MainFault:
    lb: [-10.0, -15.0, -8.0, -5.0, -5.0]
    ub: [ 10.0,  15.0,  8.0,  5.0,  5.0]
```

预检会把动态 dip 前缀展开为精确的 \(K+3\) 个角色与单位，并拒绝不匹配的切片；不需要
增加另一个 mode 开关。该方法不调用 snapshot、不建立 mesh，也不改变 prior 或 bounds。
## 混合模式的安全条件

1. 每个 Stage 必须声明 target 和固定参数顺序。
2. 后一 Stage 消费前一 Stage 的本候选 `GeometryState`，不是 current fault 或上一候选。
3. reference 只在 pipeline 起点复制一次。
4. 全部 Stage 完成后只执行一个 MeshPolicy。
5. 最终变化性质取能覆盖全部操作的最高等级；例如 rigid 后再局部 deform，整体是 deform。
6. `materialize()` 统一写回，缓存映射不由单个调用点自行修改。

未来铲状断层或新的 dip+endpoint 联合扰动应先组合现有 target、selector、direction provider
和 Stage；只有出现重复且稳定的新语义时才增加公开组合方法，不能为单一案例堆叠包装层。

## 发现与核对

```bash
ecat-list-fault-perturb-methods
```

```python
fault.help()
fault.help("perturb_bottom_coords_along_fixed_direction")
```

检查输出中的 reference fields、参数个数、候选分量角色与 mesh 责任。完整 YAML 路由见
[参数布局与配置](parameter_routing.md)，mesh/cache 后果见
[MeshPolicy 与派生状态](mesh_cache.md)。
