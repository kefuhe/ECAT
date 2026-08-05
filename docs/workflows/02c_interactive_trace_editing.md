# 交互调整断层迹线

本流程在原始或改正后的 SAR/光学观测上调整一条断层迹线，并另存为现有降采样配置可直接读取的 `lon lat` 文本。参考迹线和观测始终只读；编辑器不会改写观测、参考文件或 YAML。

## 先选入口

| 当前数据 | 推荐入口 | 原因 |
| --- | --- | --- |
| GAMMA 等原始平台文件，已经有 `downsample.yml` | `ecat-downsample -f downsample.yml --edit-trace` | 复用现有 reader、单位、正号、改正和 mask，不重新解释平台格式 |
| ECAT 标准 `.nc/.h5`、GeoTIFF 或 CSI varres | `ecat-trace-edit` | 无需再写降采样 YAML |
| 只需浏览多类背景和图层，不改迹线 | `ecat-map` | 地图查看器保持只读，职责更清楚 |

从 ECAT 仓库根目录进入 eqtools 子项目，再安装可选交互依赖。若当前已经位于
独立 eqtools 仓库根目录，只执行第二行：

```bash
cd eqtools
python -m pip install -e ".[interaction]"
```

不安装该 extra 不影响反演、约束、降采样或只读地图查看。结构化二维观测默认以连续栅格叠加在灰白底图上；平移或缩放后只重绘当前视窗，不把观测改成散点，也不改变原始经纬度、数值或 mask。

## 路线 A：从降采样配置进入

配置中的 `fault_traces` 仍按原有方式组织；`stages` 包含 `raw` 的启用项会作为只读参考迹线出现：

```yaml
fault_traces:
  - enabled: true
    id: published_trace
    file: published_trace.txt
    stages: [raw, decim]
    marker:
      enabled: false
```

运行：

```bash
ecat-downsample -f downsample.yml --edit-trace
```

常用可选项：

```bash
ecat-downsample -f downsample.yml --edit-trace --trace-output adjusted_trace.txt
ecat-downsample -f downsample.yml --edit-trace --trace-component east
ecat-downsample -f downsample.yml --edit-trace --vmin -0.2 --vmax 0.2
```

该模式是检查/编辑模式，本次运行不会同时估计协方差或执行降采样。SAR 默认显示 `observation`；光学默认显示 `east`。显示倍率、色表、自动百分位、对称性和显式色限继承 `check_plots.raw`，不是 `check_plots.decim`；命令行 `--vmin/--vmax` 可临时覆盖色限。如果已启用整周修正、参考区归零或 ramp 改正，编辑器复用同一次运行产生的改正后网格。

GAMMA prefix 也可作一次性快捷入口：

```bash
ecat-downsample --sar-prefix geo_pair --sar-mode unwrapped_phase --edit-trace
```

快捷 prefix 只适合 GAMMA 约定文件；非 GAMMA 产品或需要完整改正配置时使用 YAML。

## 路线 B：直接打开标准观测

```bash
ecat-trace-edit scene_observation.nc --trace published_trace.txt --output adjusted_trace.txt
```

改正后变量需要显式选择：

```bash
ecat-trace-edit scene_observation.nc --variable corrected_observation --trace published_trace.txt --output adjusted_trace.txt
```

需要卫星背景时仍用同一个入口：

```bash
ecat-trace-edit scene_observation.nc --trace published_trace.txt --basemap satellite --opacity 0.70
```

多个 `--trace` 可以重复出现；它们都保持只读。GeoTIFF 可直接作为背景；CSI varres 需要显式写 `--kind csi_varres`，避免把普通文本猜成科学观测。

## 在界面中怎么做

1. 保持 `Browse` 检查范围和色标；在 `Display` 中切换灰白、街道、地形、卫星或无底图，并按需显隐观测、底图和 colorbar。
2. 点 `Copy reference as working` 从参考迹线开始；界面会自动进入 `Edit`。再点击黄色 working trace，显示当前线的临时编辑圆点。点 `New trace` 从空线开始；界面会自动进入 `Draw`。
3. `Browse` 只负责平移和缩放；`Draw` 逐点建立一条连续线；`Edit` 用于拖动节点、在线段中插点，以及选择待删除节点。离开 `Edit` 时临时圆点会隐藏；再次进入后重新点击黄色线即可，不需要切换 `Draw` 来刷新。
4. 删除时先在地图节点或坐标表中选中点，再点 `Delete selected vertex` 或按 `Delete`；剩余节点会按原顺序重新连线。点选本身只高亮，不会移动或放大视图。
5. 黄色工作线是浏览器几何编辑的提交入口，圆点只是 Bokeh 为当前选中线显示的临时手柄，坐标表是派生视图。一次拖动或删除完成后，工作线、坐标表和内部 working 坐标应同步；可从表中复制选中行或全部坐标。
6. 用 `Undo/Redo` 回退，完成后点 `Validate`。
7. 用 `Save As` 写新文件。目标已存在时默认拒绝覆盖；只有显式勾选覆盖才会替换目标。
8. 回到终端按 `Ctrl+C` 停止本地服务。

编辑器只允许一条活动 working trace，同时可以显示多条 reference。当前一次会话只显示一幅观测；多条影像联合查看仍交给 `ecat-map` 或 Google Earth。需要生成第二条迹线时，先保存第一条，再新建工作迹线。

## 输出和下一步

默认文本是 UTF-8、两列 `longitude latitude`，不自动平滑、重排、加密或闭合，可直接放回配置：

```yaml
fault_traces:
  - enabled: true
    id: adjusted_trace
    file: adjusted_trace.txt
    stages: [raw, decim]
```

也可把输出扩展名写为 `.geojson`，得到单个 `LineString`。GeoJSON 是查看/交换格式；现有断层构建和降采样流程优先使用两列文本。

保存后至少检查：节点顺序是否符合迹线方向、是否误跨越无效区、坐标是否仍为经纬度、源观测和参考迹线是否未变化。完整参数、快捷键、坐标和覆盖规则见
[交互迹线编辑器参考](../reference/interactive_trace_editor.md)；只需复制命令见
[交互迹线调整短例](../examples/interactive_trace_editing.md)。
