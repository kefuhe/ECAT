# 科研地图快速查看与项目短例

## 只看内置背景

直接运行，不需要 YAML：

```bash
ecat-map
```

页面按全球背景、区域断层、块体和 GNSS 分组；只默认显示轻量 PB2002 板块边界，其余
按需勾选。指定初始研究区：

```bash
ecat-map --region -72 -62 8 12
```

只查看一个已有地震目录：

```bash
ecat-map --catalog data/events.csv
```

终端保持运行，默认浏览器地址是 `http://127.0.0.1:8050`；结束时按 `Ctrl+C`。

左侧可用 `Hide all`、`Show active only`、活动层 alpha/色限和 `Fit active layer`。底图
下拉框还可选择 `outdoors` 地形、`satellite` 卫星影像或
`satellite-streets` 卫星影像加道路；这些在线底图需要网络。

## 加入自己的数据

准备一个地震 CSV 和一个断层 GeoJSON，然后复制：

```yaml
version: 2
project:
  name: quick_review  # 页面标题
view:
  region: [min_lon, max_lon, min_lat, max_lat]  # 建议按研究区填写
  basemap: open-street-map                      # 常用默认底图
layers:
  - id: events                   # 项目内唯一 ID
    name: Earthquakes
    kind: earthquake_catalog     # 数据是什么
    source: data/events.csv      # 路径相对本 YAML
    visible: true                # 启动时加载并显示
  - id: faults
    name: Active faults
    kind: vector
    source: data/faults.geojson
    visible: false
```

运行：

```bash
ecat-map research_map.yml
```

项目模式也会列出内置背景，不必把包内 Faults/GNSS 路径抄进 YAML。

添加标准 InSAR/optical 网格时，必须显式选择变量：

```yaml
  - id: corrected_observation
    name: Corrected observation
    kind: observation_grid
    source: data/observation.nc
    variable: corrected_observation
    mask: source_valid
    visible: false
    style:
      cmap: RdBu
      alpha: 0.8
      display_factor: 100.0
      display_unit: cm
      vmin: -20.0
      vmax: 20.0
      symmetry: true
```

第一次通常只改 `view.region`、每层 `source` 和 `visible`。标准观测还必须改
`variable`；`mask` 通常保留 `source_valid`。`style` 可整段省略，`vmin/vmax` 使用
`display_factor` 后的显示单位且只控制色标，不裁剪或修改源值。

查看运行选项：

```bash
ecat-map --help
```

完整流程见
[本地科研地图查看](../workflows/07_research_map_viewer.md)，字段和格式见
[Research Map Viewer Reference](../reference/research_map_viewer.md)。
