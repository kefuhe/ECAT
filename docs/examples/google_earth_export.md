# Google Earth 导出短例

本页只给可复制的常用片段。完整任务顺序见
[Google Earth 导出工作流](../workflows/06_google_earth_export.md)。

## 输入

任选一种：

- ECAT 标准观测网格 `observation.nc`；
- CSI 降采样文件对 `result.txt + result.rsp`；
- earthquake-clients CSV；
- 已回填结果的 CSI fault 或 `seismiclocations` 对象。

## 从 reader 配置直接生成全分辨率观测

在已有 `downsample.yml` 中加入：

```yaml
export:
  google_earth:
    enabled: true
    file: auto
    visible: true          # Google Earth 中初始勾选；不是 style 字段
    style:
      vmin: null
      vmax: null
      symmetry: true
```

```bash
ecat-downsample -f downsample.yml
```

输出为 `<outName>_google_earth.kmz`，只包含全分辨率最终观测；不需要重复配置
reader、source 或 variable。该导出不随 `-s/-c/-d` 执行。
`visible` 只控制图层初始是否勾选；颜色、透明度、单位和色标范围放在 `style`。


## 最小命令

标准全分辨率网格：

```bash
ecat-export-google-earth observation-grid observation.nc --variable observation --display-factor 100 --display-unit cm -o observation.kmz
```

CSI SAR 降采样单元：

```bash
ecat-export-google-earth varres result --data-type sar --component observation --units m -o result_cells.kmz
```

地震目录：

```bash
ecat-export-google-earth catalog events.csv -o events.kmz
```

复制后通常只改输入路径、输出 `-o`，以及标准网格的 `--variable` 或 varres 的
`--component/--units`。`--display-factor 100 --display-unit cm` 只控制显示；
如再设置 `--vmin -10 --vmax 10`，其单位也是 cm。两个范围参数必须同时给出。
标准网格的 `--mask` 默认
`source_valid`。只有确实需要覆盖同名输出时才加 `--force`。

不确定某个入口有哪些选项时：

```bash
ecat-export-google-earth observation-grid --help
ecat-export-google-earth varres --help
```

## Python：fault、滑动与 `seismiclocations`

```python
from eqtools.geoexport import (
    earthquakes_from_seismiclocations,
    patches_from_fault,
    trace_from_fault,
    write_kmz,
)

layers = [
    trace_from_fault(fault, trace="original"),
    patches_from_fault(
        fault,
        component="total",
        units="m",  # fault 对象不保证携带 slip 单位，需调用者明确
        style={
            "cmap": "viridis",
            "display_factor": 100.0,
            "display_unit": "cm",
        },
    ),
    earthquakes_from_seismiclocations(aftershocks),
]

write_kmz(layers, "fault_and_aftershocks.kmz")
```

`fault` 必须已经包含当前 patch 和回填后的 slip；导出器不会调用 solver、
`returnModel()` 或重新推断参数索引。`aftershocks` 保持 CSI `seismiclocations` 对象；
adapter 只读取 `lon/lat` 和可选的 `depth/mag/time`，不修改对象。

## 检查

- KMZ 可打开且图层数正确；
- colorbar 显示单位与倍率正确；
- fault trace、patch 和地震位置对齐；
- `manifest.json` 中 `display_only` 为 `true`；
- 原始数组、fault 和 `seismiclocations` 在导出后没有变化。

## 何时不用这个例子

- 需要归零、去 ramp 或区域周跳改正：先用观测改正流程；
- 输入仍是原始 GAMMA/GMTSAR/HyP3：先用对应 reader；
- 网格是 projected/curvilinear：当前不要强行导出 raster；
- 需要完整 YAML 或所有字段：查
  [Google Earth Export Reference](../reference/google_earth_export.md)。

各原始 reader 的可复制 Python 脚本见
[SAR 与光学观测读入参考](../reference/observation_data_readers.md)。
