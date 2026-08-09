# 06 把全分辨率观测导出到 Google Earth

本工作流把 ECAT reader 已经明确单位、正号和坐标的全分辨率观测导出成显示用 KMZ，
便于在 Google Earth Pro 中同时打开多轨影像，并与已有断层和地震目录检查空间关系。

> KMZ 是显示副本，不是反演输入或权威科学存储。定量计算必须回到标准
> NetCDF/HDF5、CSI `.txt/.rsp` 或内存中的 CSI 科研对象。

## 安装

从 ECAT 仓库根目录进入 eqtools 子项目，安装导出所需的轻量可选依赖。若当前
已经位于独立 eqtools 仓库根目录，只执行第二行：

```bash
cd eqtools
python -m pip install -e ".[geoexport]"
```

这不会安装或启动 Viewer 网页服务。

## 从读入配置直接导出全分辨率观测

无需再写第二份导出配置或重复指定 reader、variable 和 mask。在已有
`downsample.yml` 中启用：

```yaml
export:
  google_earth:
    enabled: true
    file: auto
    mask: source_valid
    visible: true         # Google Earth 中初始勾选；不是 style 字段
    style:
      vmin: null          # 显示色标下限；不是空间裁剪范围
      vmax: null          # 与 vmin 同时设置或同时留空
      symmetry: true      # 自动范围关于 0 对称
```
`visible` 管图层状态，`style` 只管颜色、透明度、显示单位和色标范围。


只运行配置中的改正和导出，不加 `-s/-c/-d`：

```bash
ecat-downsample -f downsample.yml
```

默认生成 `<outName>_google_earth.kmz`，只包含全分辨率观测，不自动加入
`.txt/.rsp` 降采样单元。若已做参考改正或整周修正，自动选择最终改正值。只有规则、
等间距的经纬网格能写成 GroundOverlay；curvilinear/projected 网格会明确报错，不会
插值或压成 bounding box。

`-s` 保持 quick-look 职责，`-c/-d` 保持数值计算职责；三者都不触发 KMZ。这样重复
估计协方差或调整降采样参数时不会反复覆盖显示文件。高级变量和样式字段见
[集成导出配置](../reference/google_earth_export.md#downsample-integration)。

## 先选择入口

| 手头数据 | 推荐入口 | 是否重新解释科学语义 |
| --- | --- | --- |
| 已有 raw reader 配置，想导出全分辨率观测 | 同一 YAML 的 `export.google_earth` | 否，自动复用当前 reader 和改正值 |
| `ecat-downsample` 导出的标准全分辨率 `.nc`/HDF5 | `observation-grid` | 否，读取文件中保存的变量、单位和正号 |
| 多个全分辨率观测及已有断层/地震文件 | `project` YAML | 否，只组织图层和显示样式 |

原始 GAMMA、GMTSAR、HyP3 或 optical 产品不由导出器再次猜测。先按
[InSAR 降采样](02_insar_downsampling.md) 的 reader 语义读取；手动 reader 脚本见
[SAR 与光学观测读入参考](../reference/observation_data_readers.md)。已经得到标准
NC/H5 时，再使用下面的独立命令。

## 已有文件的最短命令

### 全分辨率观测

```bash
ecat-export-google-earth observation-grid observation.nc --variable observation --display-factor 100 --display-unit cm -o observation.kmz
```

若文件同时含 `observation` 和 `corrected_observation`，必须明确选一个，导出器不会
猜测应该显示原始值还是改正值。

该命令可直接用于 PowerShell、cmd、bash 或终端。输出父目录不存在时
会自动创建；已有同名 KMZ 时必须显式加 `--force`。

### 地震目录

```bash
ecat-export-google-earth catalog events.csv -o events.kmz
```

CSV 至少需要 `longitude` 和 `latitude`；`magnitude`/`mag`、`depth`、`time` 和其他
已有列会写入 Google Earth 属性。

### 基本选项怎么改

第一次导出通常只需检查下面几项：

| 数据 | 通常需要修改 | 说明 |
| --- | --- | --- |
| 所有入口 | 输入路径、`-o` | `-o` 必须是 `.kmz`；父目录会自动创建 |
| 标准网格 | `--variable` | 明确选原始量、改正量或 optical component |
| 标准网格 | `--mask` | 默认 `source_valid`；仅需显示分析像元时用 `analysis_valid` |
| 定量着色 | `--display-factor`、`--display-unit` | 两者应配套设置；只改变显示，不改源值 |
| 定量着色 | `--vmin MIN --vmax MAX` | 可选；两者必须同时给出，单位是乘过 `display-factor` 后的显示单位 |
| 自动色标 | `--symmetry` / `--no-symmetry` | signed deformation 通常用 `--symmetry`；显式上下限优先 |
| 覆盖输出 | `--force` | 只在确认要替换同名 KMZ 时添加 |

查看某个入口的全部选项时，直接运行：

```bash
ecat-export-google-earth observation-grid --help
```

## 多图层项目

当需要把多轨全分辨率观测、断层线和地震目录放在一个 KMZ 中时，使用一个短
project YAML。它就是可复制的基础模板，路径相对于 YAML 所在目录。第一次通常只改
`output.path`、各层 `source`、观测 `variable/mask` 和显示单位：

```yaml
version: 2

output:
  path: results/research_context.kmz  # 输出；相对本 YAML
  document_name: Research context     # Google Earth 顶层名称

layers:
  - id: track_a                       # 项目内唯一且保持稳定
    name: Track A LOS                 # Google Earth 中显示的名称
    kind: observation_grid            # 标准全分辨率观测网格
    source: data/track_a.nc            # 相对本 YAML
    variable: corrected_observation    # 必须明确原始量或改正量
    mask: source_valid                 # 默认；也可用 analysis_valid 或 finite
    visible: true                      # Google Earth 初始是否勾选
    style:
      cmap: RdBu_r                     # Matplotlib colormap
      vmin: -10.0                       # display_factor 后的显示值
      vmax: 10.0
      symmetry: true                    # 显式 vmin/vmax 已优先
      display_factor: 100.0            # 例如 m -> cm；不改存储值
      display_unit: cm

  - id: events
    name: Earthquakes
    kind: earthquake_catalog
    source: data/events.csv
```

运行：

```bash
ecat-export-google-earth project google_earth.yml
```

完整字段、GeoJSON/GMT 图层和高级 Python 对象入口见
[Google Earth Export Reference](../reference/google_earth_export.md)。

## 与本地 Viewer 使用同一份标准网格

需要先在浏览器检查、再导出 Google Earth 时，两处写同一个 `variable` 和 `mask`：

```yaml
- id: corrected_los
  name: Corrected LOS
  kind: observation_grid
  source: data/track_a.nc
  variable: corrected_observation
  mask: source_valid
  visible: true
```

```bash
ecat-export-google-earth observation-grid data/track_a.nc --variable corrected_observation --mask source_valid -o track_a.kmz
```

两条路径读取同一组值、单位、正号和有效像元，并共享
`cmap/vmin/vmax/symmetry/alpha/display_factor/display_unit` 词汇。两端自动范围都从
完整有限显示值取 2–98 百分位后解析；Viewer 随后抽稀显示点，KMZ 则颜色化完整规则
网格。

## 输出后检查

在 Google Earth Pro 中至少检查：

1. 图层位置是否与海岸线、断层和地震目录一致；
2. 图像东西、南北方向是否正确，边界没有半像元缩进；
3. colorbar 单位与 `display_factor` 是否匹配；
4. LOS/offset 正号说明是否仍与标准文件一致；
5. 比较多幅影像时，各图层是否使用了相同的 `vmin/vmax` 和显示单位；单独设置
   `symmetry: true` 只保证零值居中，不保证不同图层使用相同范围。

KMZ 内的 `manifest.json` 记录存储单位、正号、显示倍率、解析色限、colormap、alpha、
normalization、变量和图层来源类型，可用于复核显示配置。Raster KMZ 是颜色化显示
产品，不包含逐像元数值；科研数值仍以标准 NC/H5 为准。

## 当前明确限制

- raster 只接受精确且两轴等间距的 `geographic_rectilinear` 网格；
- projected、rotated、curvilinear 和跨日界线 raster 会明确报错，不会自动插值或压成
  bounding box；
- 不直接读取原始平台文件，不执行相位转 LOS、参考区改正、ramp 估计或反演；
- 不渲染 beachball，不解释 `seismiclocations.CMTinfo`；
- 不提供 KML/KMZ 反向导入或数值 round-trip。

这些限制保护坐标和数值对齐。需要改正观测时，先回到
[Observation Correction and Grid Export](../reference/observation_correction_export.md)。

## 下一步

- 只想复制命令或 Python 片段：
  [Google Earth 导出短例](../examples/google_earth_export.md)。
- 需要完整 CLI、YAML、API 和错误语义：
  [Google Earth Export Reference](../reference/google_earth_export.md)。
