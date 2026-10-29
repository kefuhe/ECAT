# 升级与依赖迁移

升级后先确认实际运行的包，再按明确的迁移合同处理脚本。安装问题的完整诊断见
[故障排查](troubleshooting.md)，源码与新增文件规则见[架构](../developer/architecture.md)。

## 先确定运行来源

在实际运行脚本的同一终端、解释器和 MPI 环境中执行：

```bash
python -c "import sys, eqtools, csi, ecat_viz; print(sys.executable); print(eqtools.__file__); print(csi.__file__); print(ecat_viz.__file__)"
python -m pip check
```

核对输出是否来自刚更新的源码、editable 安装或 wheel。复制源码后，普通安装不会自动
更新；多个 checkout、当前工作目录或 `PYTHONPATH` 也可能覆盖安装包。版本号相同不证明
实际文件相同。跨包更新按[安装页](installation.md)同时核对 ecat-viz、CSI 和 eqtools，
不要在脚本里临时改 `sys.path` 或伪造模块别名以绕过配套协议检查。

在已有源码目录执行 `python -m pip install .` 前，模块归组升级还应检查项目根的生成目录
`build/lib`。旧生成文件不会因为源文件移动而自动消失，可能被再次装进 wheel。
从干净 checkout 构建，或先另存/清理生成的 `build` 目录，再安装；不要把 `build/lib`
里的副本当作当前源码编辑。重新构建后核对 wheel 成员与当前发布清单。

## 按症状选择合同

| 症状或变化 | 应处理的位置 |
| --- | --- |
| 整个 `ecat_viz` / `csi` / `eqtools` 缺失，或 ABI/依赖冲突 | 先修安装和环境；见[故障排查](troubleshooting.md) |
| 已安装，但旧 trace/dip/mesh helper 模块不存在 | 按[几何旧路径表](../developer/geometry_support.md#geometry-import-migration)修改显式 import |
| 旧 data/covariance/constraint helper 模块不存在 | 按[观测与约束迁移表](../developer/observation_constraint_support.md#observation-constraint-import-migration)修改 import；主要 inversion 入口不变 |
| 旧 figure/statistics/parameter reporting 模块不存在 | 按[结果模块迁移表](../developer/architecture.md#reporting-import-migration)改用 `reporting`；对象方法和包级公开导出不变 |
| CPT 资源、字体或旧样式入口用法变化 | 查[Viztools 兼容与资源](../reference/viztools.md)，通用功能优先从 `ecat_viz` 导入 |
| import 已更新，旧 pickle 仍不能打开 | 单独处理序列化类路径；先在旧环境导出字段，再用当前类重建 |
| FULLSMC 配置了线性矩阵约束 | 查[约束模式范围](../reference/constraint_manager.md)；这是已有模式合同，不由导入迁移改变 |
| 模块能够导入，但行顺序、单位或结果不一致 | 查[观测与矩阵排列](../concepts/observation_matrix_layout.md)和相应 reference；保留证据，定位 owning package |

不要把所有 `ModuleNotFoundError` 当成迁移：若缺少的是另一个依赖，先处理错误中真正
缺失的模块名。也不要仅因为文件移动，就修改公式、数据单位、参数边界或科学预期值。

## 最小迁移与验证

1. 保存原环境、配置、输入和预期输出；根据上表只修改确需迁移的调用。
2. 检查目标方法签名和 reference。继续通过 inversion 对象设置约束，避免依赖内部存储。
3. 先运行无副作用的导入检查和小型验证，再运行自己的实际任务。
4. 对确定性求解比较模型、预测、统计和科学文件；对随机采样先确认 seed、restart、rank
   和采样合同一致，不能用两个随机后验不完全相同证明模块迁移错误。

观测与约束归组的导入检查可以写成：

```bash
python -c "from eqtools.csiExtend import DataCorrectionParameterRef; from eqtools.csiExtend.observations.covariance_utils import prepare_covariance_metric; from eqtools.csiExtend.constraints.constraint_manager_blse import ConstraintManagerBLSE; print('imports OK')"
```

结果模块也可用以下命令确认当前安装包含真实实现：

```bash
python -c "from eqtools.csiExtend import FigureProductMixin; from eqtools.csiExtend.reporting.parameter_layout_reporting import make_parameter_layout_report; from eqtools.csiExtend.reporting.nonlinear_geometry_plots import SMCPlotData; print('reporting imports OK')"
```

旧 `SMCPlotData` pickle 引用已删除的根层模块时，需要按[序列化迁移说明](../developer/architecture.md#reporting-import-migration)
在可信旧环境导出、用当前类重建；只改 import 或复制旧壳不能算作存档迁移完成。

计算结果改变但原因不明时，不通过改案例预期值消除失败。提交 issue 时附包来源、版本、
执行模式、最小可复现输入、预期与实际行为、完整 traceback 或差异量；避免提交隐私数据。
问题按发生责任归属到 [eqtools](https://github.com/kefuhe/eqtools/issues)、
[CSI](https://github.com/kefuhe/csi_cutde_mpiparallel/issues) 或通用绘图库对应的仓库。
本页负责排查路线，各 reference 和迁移表维护实际合同，issue 记录尚未解决的具体问题。

## 地震投影与计数接口

`EqseqProj2Fault.splitReceiver(n)` 使用 CSI 的共享边细分。旧 `use_parallel/n_workers`
参数已移除；`splitReceiver(n)`、投影、保存和绘图的普通调用继续使用。
`calculate_seismic_projection_statistics()` 现在返回原面片顺序的计数数组，不再覆盖
`receiver.slip`。依赖这一旧副作用的脚本应读取返回值。细分返回普通 CSI 三角断层，
不携带反演子类状态。完整合同见[地震投影与计数](../reference/seismic_projection.md)。

已有完整环境时，从 ECAT 根目录同步更新两包并检查底层接口：

```bash
python -m pip install --no-deps --force-reinstall ./csi_cutde_mpiparallel ./eqtools
python -c "from csi import TriangularPatches; assert callable(getattr(TriangularPatches, 'subdivided_copy', None)); print('subdivision API OK')"
```

单独源码仓库中分别在两包目录执行安装即可；仅更新 eqtools 不保证旧 CSI 提供新接口。
