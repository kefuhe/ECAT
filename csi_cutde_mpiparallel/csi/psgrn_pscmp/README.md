# PSGRN/PSCMP — 黏弹性层状半空间形变计算引擎

## 概述

`csi.psgrn_pscmp` 模块封装了 PSGRN/PSCMP（Wang et al., 2006）黏弹性层状半空间
格林函数计算与正演形变模拟。计算分两步：

| 步骤 | 程序 | 功能 |
|------|------|------|
| 1. PSGRN | `psgrn` 可执行文件 | 预计算层状地球模型的格林函数表 |
| 2. PSCMP | `pscmp` 可执行文件 | 利用格林函数表计算指定断层源在观测点的位移 |

与 EDGRN/EDCMP（纯弹性）不同，PSGRN/PSCMP 支持黏弹性松弛效应，
可用于震后形变建模。

---

## 模块结构

```
csi/psgrn_pscmp/
├── __init__.py            # 公共 API 导出
│                          #   PSGRNConfig, LayerModel, PSGRNParameters
│                          #   PSCMPConfig, FaultSource, PSCMPParameters
│                          #   PscmpOptions
├── psgrn_config.py        # PSGRN 输入文件生成
│                          #   PSGRNConfig / PSGRNParameters / LayerModel
├── pscmp_config.py        # PSCMP 输入文件生成
│                          #   PSCMPConfig / PSCMPParameters / FaultSource
├── pscmp_options.py       # 配置参数 dataclass
│                          #   PscmpOptions — 所有 PSCMP 相关配置
├── PSGRNCmp.py            # exe 模式接口
│                          #   pscmpslip2dis() — 单 patch GF 计算
│                          #   get_pscmp_bin() — 二进制文件定位
└── README.md
```

运行时二进制文件位于 `csi/bin/` 下：

```
csi/bin/
├── windows/
│   └── pscmp2008.exe      # Windows PSCMP 可执行文件
└── ubuntu20.04/
    └── pscmp2008           # Linux PSCMP 可执行文件
```

---

## 调用链

### 从 Fault.buildGFs 到计算

```
Fault.buildGFs(method='pscmp', ...)
  └─ Fault.pscmpGFs(data, ...)
       ├─ PscmpOptions 参数合并
       ├─ _get_pscmp_patch_sources()      准备断层片源参数
       └─ ProcessPoolExecutor (n_jobs 并行)
            └─ _pscmp_patch_task()
                 └─ pscmpslip2dis()       写 .inp → 运行 pscmp → 读输出
                      ├─ PSCMPConfig.generate()   生成输入文件
                      ├─ subprocess.run(pscmp)     执行计算
                      └─ 解析输出 → (ss, ds, ts)   格林函数矩阵
```

### 与 EDCMP 的区别

| | PSCMP | EDCMP |
|---|---|---|
| 物理模型 | 黏弹性层状半空间 | 纯弹性层状半空间 |
| 计算引擎 | 仅 exe（进程调用） | exe + ctypes（内存计算） |
| 格林函数预计算 | PSGRN（必须先运行） | EDGRN（必须先运行） |
| 并行方式 | ProcessPoolExecutor | exe: ProcessPoolExecutor; ctypes: 共享内存 |
| 时间依赖 | 支持（震后松弛） | 不支持（瞬时弹性） |

---

## 配置体系

### PscmpOptions

`PscmpOptions` dataclass 封装了所有 PSCMP 相关配置参数：

```python
from csi.psgrn_pscmp.pscmp_options import PscmpOptions

opts = PscmpOptions(
    grn_dir='psgrnfcts',       # PSGRN 格林函数目录（相对于 workdir）
    output_dir='pscmpgrns',    # PSCMP 输出目录
    workdir='pscmp_ecat',      # 工作目录
    n_jobs=4,                  # 并行 worker 数
    cleanup_inp=True,          # 计算后删除 .inp 文件
    force_recompute=True,      # 强制重新计算
)
```

### 字段参考

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `grn_dir` | str | `"psgrnfcts"` | PSGRN 格林函数目录（相对于 workdir） |
| `output_dir` | str | `"pscmpgrns"` | PSCMP 输出目录（相对于 workdir） |
| `workdir` | str | `"pscmp_ecat"` | 工作目录 |
| `n_jobs` | int | `4` | 并行 worker 数 |
| `cleanup_inp` | bool | `True` | 计算后删除中间 .inp 文件 |
| `force_recompute` | bool | `True` | 即使输出文件已存在也重新计算 |

支持从 eqtools YAML 配置的 `options:` 字典构建：

```python
opts = PscmpOptions.from_kwargs(**yaml_options_dict)
```

### 自文档化 API

`PscmpOptions` 提供多种方式查看可用选项及其含义：

```python
from csi.psgrn_pscmp.pscmp_options import PscmpOptions

# 1. 文本表格
PscmpOptions.describe_options()

# 2. YAML 字符串（带行内注释，可直接粘贴到配置文件）
print(PscmpOptions.describe_yaml())
# 输出示例:
#   grn_dir: psgrnfcts       # PSGRN Green's function directory (relative to workdir)
#   output_dir: pscmpgrns    # PSCMP output directory (relative to workdir)
#   n_jobs: 4                # Number of parallel workers
#   ...

# 3. CommentedMap（用于程序化生成带注释的 YAML）
cm = PscmpOptions.to_commented_map()
```

也可通过统一入口查询：

```python
from csi.gf_options import describe_gf_options

describe_gf_options('pscmp')                  # 文本格式
describe_gf_options('pscmp', format='yaml')   # YAML 格式
```

### 构造时验证（`__post_init__`）

`PscmpOptions` 在构造时自动校验参数合法性：

| 校验规则 | 示例 |
|----------|------|
| `n_jobs` 必须 >= 1 | `PscmpOptions(n_jobs=0)` → ValueError |

`from_kwargs()` 会自动忽略未知键并发出 `UserWarning`：

```python
opts = PscmpOptions.from_kwargs(grn_dir='mygrn', bad_key=123)
# UserWarning: Unknown PscmpOptions keys ignored: ['bad_key']
```

### 与 gf_options 统一入口的关系

`csi.gf_options` 模块提供方法无关的统一解析入口：

```python
from csi.gf_options import resolve_gf_options

opts = resolve_gf_options('pscmp', {'grn_dir': 'mygrn', 'n_jobs': 2})
# → PscmpOptions(grn_dir='mygrn', n_jobs=2, ...)

opts = resolve_gf_options('pscmp', None)
# → PscmpOptions()  默认值
```

---

## 安装

```bash
cd csi_cutde_mpiparallel
pip install -e .
```

安装后 PSCMP 可执行文件直接可用（Windows 已包含预编译的 `pscmp2008.exe`）。

### PSGRN 格林函数预计算

使用 PSCMP 前必须先运行 PSGRN 生成格林函数表。可通过 eqtools CLI 工具：

```bash
# 生成 PSGRN 输入模板
python -m eqtools.cli_tools.psgrn_template_cli -o psgrn.inp

# 编辑 psgrn.inp（设置地球模型、深度范围等）

# 运行 PSGRN
python -m eqtools.cli_tools.psgrn_cli -i psgrn.inp
```

或通过 Python API：

```python
from csi.psgrn_pscmp import PSGRNConfig, PSGRNParameters, LayerModel

config = PSGRNConfig(PSGRNParameters(...), layers=[LayerModel(...)])
config.write('psgrn.inp')
```

---

## 验证

```bash
# 运行测试
pytest tests/test_edcmp_options.py -v -k "pscmp"
```

---

## 参考

- Wang R., Lorenzo-Martín F., Roth F. (2006). PSGRN/PSCMP — a new code for
  calculating co- and post-seismic deformation, geoid and gravity changes based
  on the viscoelastic-gravitational dislocation theory.
  *Computers & Geosciences*, 32, 527–541.
