# 地震目录投影与原面片事件计数

已有 CSI 三角断层和地震目录时，使用 `EqseqProj2Fault` 将事件分配到细分面片中心，
再按原面片统计。此操作用于后分析，不重新求解滑移，不改变输入断层的几何、物理滑移或目录。

## 最短使用流程

`fault` 是具有一致 `Vertices/Faces/patch` 的 CSI 三角断层，`catalog` 是读入并完成
时间、震级和空间选择的 `seismiclocations`。两者应使用相同坐标框架；选择条件由用户决定。

```python
from eqtools.csiExtend.statUtils.earthquakesProj2Fault import EqseqProj2Fault

projection = EqseqProj2Fault(
    "events", lon0=fault.lon0, lat0=fault.lat0,
    receiver=fault, seismic=catalog,
)
projection.splitReceiver(3)
projection.calproj(remove_fault_edge_events=True)
counts = projection.calculate_seismic_projection_statistics()
projection.write2file("events_projected.dat")
projection.save_seismic_projection_statistics("event_counts.gmt")
projection.plot_seismic_projection_statistics(
    cmap="viridis", cblabel="Event count", show=False, savefig=True,
)
```

不细分时使用 `splitReceiver(0)`。输出文件和绘图保存均由调用者显式请求。
投影目录保存在 `projection.seis_proj`；原目录保留原始坐标和事件字段。

## 计算含义与对应关系

投影沿用 CSI 的三维最近面片中心算法：在局部 `x/y/depth`（km）中寻找最近的细分
面片中心，并将事件坐标替换为该中心。它不是三角形内部包含测试，也不是事件到面片
平面的正交投影；细分层数控制中心离散精度。

| 数据 | 长度 | 对应关系与位置 |
| --- | --- | --- |
| `receiver_dense.original_indices` | 细分面片数 | 每个细分面片来自哪个输入面片；CSI 返回的来源数组只在这里保存一份 |
| `ipatch` | 筛选前事件数 | 事件对应哪个细分面片；局部变量，不是对象属性 |
| `seis_proj.original_patch_indices` | 保留事件数 | 每个保留事件对应哪个原面片，供按原面片聚合使用 |
| 返回的 `counts` | 原面片数 | 按输入面片顺序排列的整数事件数，无事件的面片为零 |

编号从零开始，表示 `Faces` 行号，不是顶点编号或 GMT 参数编号。
多轮细分的来源数组始终指向该次调用最初输入的面片。

`remove_fault_edge_events=True` 删除被分配到细分网格 `left/right/bottom` 边界面片的
事件，保留 top 边界；不会删除某个原边界面片下的全部子面片。
筛选通过 CSI 目录选择方法同步坐标、时间、震级和其支持的可选字段，并同步筛选
`original_patch_indices`。用户自行增加的其他事件属性仍需按其自身合同处理。
计数总和应等于保留事件数。

## 独立使用 CSI 细分断层

```python
refined, parent_face_ids = fault.subdivided_copy(3)
refined_with_slip, parent_face_ids = fault.subdivided_copy(3, slip_mode="inherit")
```

`parent_face_ids` 是返回的 NumPy 数组，不是新增断层属性。CSI 只负责网格及来源，
不处理事件。返回值为普通 `TriangularPatches`，保留坐标框架、迹线和 `vertical`
标记，不复制求解器、GF、参数编号、边界缓存或子类专有状态。

- 每轮一个面片生成四个子面片，共享边只生成一个中点；真实裂缝不自动焊接。
- 子面片顺序与 `splitPatch()` 相同，绕序跟随输入面片，不统一反转方向。
- 正层数默认初始化零滑移，与原 `splitReceiver()` 约定一致。
- `slip_mode="inherit"` 逐面片复制三分量滑移；输入滑移为 `None` 时仍为 `None`。
  继承是分片常量延拓，不代表重新反演。
- 零层数返回独立几何和滑移副本；输入滑移为 `None` 时保持 `None`。

只需几何时不要求输入有滑移。继承已有滑移时，其形状必须为 `(原面片数, 3)`。
层数必须为非负整数。`patch` 与 `Vertices/Faces` 不一致时明确报错，调用者应先通过
CSI 网格发布接口建立完整网格，不在每次投影时从坐标猜测连接关系。

新断层可使用已有 writer 显式保存。保留连接关系时优先使用 Gocad 网格格式；
GMT patch 文本带坐标舍入，不保证精确拓扑往返。几何文件不保存来源数组，后续需要
来源关系时应另存 `parent_face_ids`，例如 `numpy.save`。

## 生命周期与升级

`setReceiver()` 清除细分网格和投影；`setSeismic()` 清除投影并保留可复用细分网格。
成功再次 `splitReceiver()` 后需重新投影；失败不替换已有结果。直接修改输入对象的
网格或目录后，调用者应重新准备和投影，不自动追踪外部对象的全部修改。
边界由 CSI 按需计算和复用，eqtools 不维护第二套边界缓存。

原 `splitReceiver(n)`、`calproj()`、保存和绘图方法保持调用方式。以下旧方式需要迁移：

| 旧方式 | 当前方式 |
| --- | --- |
| `splitReceiver(n, use_parallel=..., n_workers=...)` | `splitReceiver(n)`；向量化共享边核心替代进程池，旧参数明确拒绝 |
| 统计后从 `receiver.slip[:, 0]` 读取事件数 | `counts = calculate_seismic_projection_statistics()`；原滑移不再作为计数容器 |
| 依赖细分对象携带完整反演子类或求解状态 | 原对象保留反演状态，细分返回的普通 CSI 对象用于后分析 |

`Contour3DExtraction` 的标量等值线与接收器合同不属于本接口，本次不改变其插值或投影语义。
相关说明：[断层几何构建](fault_geometry_construction.md)、[断层边界](fault_edges.md)、
[升级与依赖迁移](../getting_started/upgrading.md)。

若报错要求 receiver 提供 `subdivided_copy()`，先确认输入为 CSI 三角断层，并同时更新
CSI fork 与 eqtools。检查包的实际导入位置，避免源码已更新而仍加载旧安装副本。

计数图通过现有自定义数组绘图接口显示：临时滑移容器使用独立数组，结束或异常时
恢复原滑移引用，外部持有的物理滑移数组不会被改写。

保留的是输入断层当前实际投影，包括公开 `set_utmzone()` 的后续调整，而非仅依据
初始化时的 lon0/lat0 猜测投影。Gocad 的 UTM 写出使用米，可按以下方式读回 km 网格：

```python
import numpy as np
from csi import TriangularPatches

refined.writeGocadPatches("refined_mesh.ts", utm=True)
np.save("parent_face_ids.npy", parent_face_ids)
loaded = TriangularPatches("loaded", utmzone=fault.utmzone, ellps=fault.ellps,
                           lon0=fault.lon0, lat0=fault.lat0)
loaded.readGocadPatches("refined_mesh.ts", utm=True, factor_xy=1e-3, factor_depth=1e-3)
```

读回对象应使用与写出时相同的投影；若原对象调整过 UTM zone，也需对读回对象做同样调整。
该几何格式不保存滑移、来源关系或完整反演状态。
