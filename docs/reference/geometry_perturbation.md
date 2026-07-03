# Bayesian 联合反演中的可扰动断层几何 / Perturbable Fault Geometry

本页说明 `BayesianAdaptiveTriangularPatches` 在联合 Bayesian 几何-滑动分布反演中的作用。这里的“几何扰动”不是普通两步走里的非线性几何参数估计，而是 SMC 样本内对断层几何、网格和相关矩阵进行一致更新的机制。

联合反演主线见 [Bayesian 联合几何-滑动分布反演 / Joint Bayesian Geometry-Slip](../workflows/05_joint_bayesian_geometry_slip.md)，采样模式见 [Bayesian 联合反演参考 / Bayesian Joint Inversion](bayesian_joint_inversion.md)。

## 阅读路径

- 普通两步走非线性几何搜索：不需要本页，读 [非线性几何反演配置](config_nonlinear_geometry.md)。
- 联合 Bayesian 中需要更新断层几何：先看 [核心对象](#核心对象) 和 [几何基线](#几何基线)。
- 不确定扰动方法怎么命名、多少参数：看 [扰动方法族](#扰动方法族) 和 [命名约定](#命名约定)。
- 正在写 YAML：看 [YAML 配置](#yaml-配置) 和 [更新标志](#更新标志)。
- 需要判断是否值得使用：看 [使用建议](#使用建议)。

## 核心对象

| 对象 | 作用 |
| --- | --- |
| `BayesianAdaptiveTriangularPatches` | 可扰动三角断层对象；组合几何、网格、扰动方法和状态标志 |
| `GeometryReference` | 冻结几何基线；每次样本扰动都从该基线出发 |
| `DensificationConfig` | 稀疏控制点加密配置；用于低维采样和高质量网格之间的桥接 |
| `PerturbationRegistry` | 自动发现 `perturb_*` 方法并生成帮助信息 |
| `update_fault_geometry` | 联合 Bayesian 配置中调用扰动方法的入口 |

关键原则：**每次样本都是 `frozen baseline + current perturbation`，不是在上一轮样本结果上累计漂移。**

## 几何基线

联合 Bayesian 采样前，断层对象需要有可复用的几何基线。常见做法包括：

```python
from eqtools.csiExtend.BayesianAdaptiveTriangularPatches import BayesianAdaptiveTriangularPatches

fault = BayesianAdaptiveTriangularPatches("FaultA", lon0=lon0, lat0=lat0)

# 从已有断层迹线或手动坐标构造 top/bottom，再设置 Bayesian 几何基线。
fault.set_edges_for_bayesian_optimization(segs=5, use_trace=True)
fault.snapshot()
```

`snapshot()` 会把当前几何状态冻结到 `GeometryReference`。如果扰动方法需要已有网格顶点或面片作为整体变换基线，应在生成初始网格后再捕获对应状态。

## 坐标加密

几何采样常希望使用少量控制点降低维度，但网格生成和走向计算又需要足够密的边界坐标。`DensificationConfig` 用于把稀疏控制点在每次扰动后临时加密：

```python
fault.set_densification(interval=1.0)     # 按弧长间距加密，单位通常为 km
# 或
fault.set_densification(num_segments=80)
```

加密后的密集坐标用于走向、倾角插值和网格生成；冻结基线仍保留低维控制点。这样可以在采样效率和几何表达之间取得平衡。

## 扰动方法族

可用扰动方法由断层类自动发现。常见 family：

| Family | 典型方法 | 主要作用 |
| --- | --- | --- |
| `Dip` | `perturb_DipsPresetParams_SimpleMesh` | 扰动倾角控制点并重建网格 |
| `Direction` | `perturb_BottomFixedDir_simpleMesh` | 沿固定或走向相关方向移动底边 |
| `Rotation` | `perturb_BottomRotation_simpleMesh` | 绕指定支点旋转几何 |
| `Translation` | `perturb_BottomTrans_simpleMesh` | 平移几何控制点 |
| `Composite` | `perturb_BottomFixedDir_RotateTransGeom_simpleMesh` | 组合偏移、旋转和平移 |
| `EndpointDutta` | `perturb_BottomEndpointsFixedDirandMidpointsDutta_simpleMesh` | 端点和中点扰动，适合特定几何构型 |

查看当前版本可用方法：

```bash
ecat-list-fault-perturb-methods
```

或在 Python 中：

```python
fault.help()
```

## 命名约定

扰动方法名通常包含目标、方法和网格策略：

```text
perturb_<Target>_<Method>
perturb_<Target><Method>_simpleMesh
perturb_<Target><Method>_DeformMesh
perturb_<Target><Method>_multiLayerMesh
```

| 后缀 | 含义 |
| --- | --- |
| 无网格后缀 | 只改几何坐标或控制参数，不重建网格 |
| `_simpleMesh` | 扰动后重建简单三角网格 |
| `_DeformMesh` | 扰动后变形已有 Gmsh 网格 |
| `_multiLayerMesh` | 扰动后重建多层网格 |

具体方法需要多少个扰动参数，由方法注册信息和帮助系统给出。配置中不要手写 `perturbations`；它由 SMC 样本自动提供。

## YAML 配置

联合 Bayesian 配置中，几何扰动通常写在断层的 `method_parameters.update_fault_geometry`：

```yaml
faults:
  FaultA:
    method_parameters:
      update_fault_geometry:
        method: perturb_BottomFixedDir_RotateTransGeom_simpleMesh
        disct_z: 10
        bias: 1.0
        angle_unit: degrees
        perturbation_direction: horizontal
```

`method` 是扰动方法名，其他字段是固定 kwargs。边界配置负责给样本中的几何扰动参数设置范围；主配置负责说明如何把这些参数应用到断层对象。

## 更新标志

几何扰动会影响不同下游对象。常见状态标志包括：

| 标志 | 含义 |
| --- | --- |
| `mesh_updated` | 网格拓扑或坐标已经改变，后续 GF 可能需要重建 |
| `laplacian_updated` | 平滑矩阵需要更新 |
| `area_updated` | patch 面积改变，震级、先验或面积权重需要重新计算 |

联合 Bayesian 运行时应根据这些标志保证样本内的几何、网格、GF、Laplacian 和面积项一致。不要为了省计算成本跳过必要更新；科研代码中几何-物理一致性优先。

## Pipeline 机制

部分新扰动方法使用 perturbation pipeline：把“做什么扰动”拆成 stage，把“如何生成网格”拆成 mesh policy。

```text
GeometryReference
  -> GeometryState.from_ref()
  -> Stage list
  -> MeshPolicy
  -> materialize()
```

这个机制主要服务于开发和复杂组合方法。普通用户只需要选择已注册的扰动方法；开发者若要新增组合方法，再查内部 pipeline 文档。

## 使用建议

- 先用两步走路线获得稳定几何和滑动结果，再设计联合 Bayesian 的扰动范围。
- 优先选择参数少、物理含义清楚的扰动方法。
- 对曲线断层，优先使用稀疏控制点加 `DensificationConfig`，不要直接采样过多节点。
- 每次改变扰动方法后，先跑小规模样本检查网格、GF 和残差。
- 报告结果时写清楚扰动方法、参数边界、单位、网格策略和是否重建 GF。

## 相关页面

- [Bayesian 联合几何-滑动分布反演 / Joint Bayesian Geometry-Slip](../workflows/05_joint_bayesian_geometry_slip.md)
- [Bayesian 联合反演参考 / Bayesian Joint Inversion](bayesian_joint_inversion.md)
- [ECAT 约束管理器 / Constraint Manager](constraint_manager.md)
- [CLI 命令参考 / CLI Reference](cli.md)
