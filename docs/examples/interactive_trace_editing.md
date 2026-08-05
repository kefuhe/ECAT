# 交互迹线调整短例

## 原始 GAMMA 数据：复用降采样配置

```bash
ecat-downsample -f downsample.yml --edit-trace --trace-output adjusted_trace.txt
```

`downsample.yml` 中可复用的参考迹线：

```yaml
fault_traces:
  - enabled: true
    id: published_trace
    file: published_trace.txt
    stages: [raw]
```

关闭编辑器并保存后，把输出明确写回后续配置：

```yaml
fault_traces:
  - enabled: true
    id: adjusted_trace
    file: adjusted_trace.txt
    stages: [raw, decim]
```

编辑器不会自动改写这一字段。

## 标准 NetCDF/HDF5 观测：直接打开

```bash
ecat-trace-edit scene_observation.nc --variable corrected_observation --trace published_trace.txt --output adjusted_trace.txt
```

显示多条只读参考迹线：

```bash
ecat-trace-edit scene_observation.nc --trace trace_a.txt --trace trace_b.geojson --output adjusted_trace.txt
```

固定显示色限：

```bash
ecat-trace-edit scene_observation.nc --trace published_trace.txt --vmin -0.15 --vmax 0.15
```

只设置一个色限会报错，避免另一端被静默猜测。

卫星底图和较低观测透明度：

```bash
ecat-trace-edit scene_observation.nc --trace published_trace.txt --basemap satellite --opacity 0.70
```

启动后最短操作是：点 `Copy reference as working` 自动进入 `Edit`，再点击黄色 working trace 显示临时编辑圆点，然后拖动节点；选中不合适的节点后点 `Delete selected vertex` 或按 `Delete`，剩余线段会自动重连；最后 `Validate`、`Save As`。若从空迹线开始，点 `New trace` 会自动进入 `Draw`。离开 `Edit` 时圆点会隐藏；再次进入后重新点击黄色线即可。选择节点或坐标表行只高亮，不会突然放大地图。

启动后仍可在 `Display` 中随时切换底图、观测、colorbar 和透明度；这些操作只影响显示。完整步骤见
[交互调整断层迹线](../workflows/02c_interactive_trace_editing.md)，全部字段见
[交互迹线编辑器参考](../reference/interactive_trace_editor.md)。
