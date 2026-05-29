# EDGRN/EDCMP — 层状半空间弹性形变计算引擎

## 概述

`csi.edgrn_edcmp` 模块封装了 EDGRN/EDCMP（Wang et al., 2003）层状半空间弹性形变计算。
支持两种后端引擎：

| 引擎 | 说明 | 速度 |
|------|------|------|
| `exe` | 调用 edcmp 可执行文件（进程间通过文件通信） | 慢（有磁盘 I/O） |
| `ctypes` | 通过 ctypes 直接调用 Fortran 共享库（内存计算） | 快（~10-100x） |

默认行为（`engine="auto"`）：优先使用 ctypes，不可用时回退到 exe。

---

## 模块结构

```
csi/edgrn_edcmp/
├── __init__.py                 # 公共 API 导出
│                               #   VALID_EDCMP_ENGINES
│                               #   normalize_edcmp_engine()
│                               #   normalize_edcmp_fallback_engines()
├── edcmp_backends.py           # 引擎发现、加载、计算调度
│                               #   EdcmpOptions — 配置参数 dataclass
│                               #   resolve_edcmp_engine() — 引擎自动选择
│                               #   compute_inmemory_edcmp_greens() — ctypes GF 计算
│                               #   compute_inmemory_edcmp_forward() — ctypes 正演计算
├── shared_memory_backend.py    # 多进程共享内存管理
│                               #   SharedGreenFunctions — 主进程创建/worker 附着
│                               #   create_shared_greens() / attach_shared_greens()
├── EDGRNcmp.py                 # exe 模式接口
│                               #   edcmpslip2dis() — exe 模式 GF 计算
│                               #   edcmpslip2dis_forward() — exe 模式正演
├── edcmp_config.py             # EDCMP 输入文件生成
│                               #   EDCMPParameters / EDCMPConfig / RectangularSource
├── edgrn_config.py             # EDGRN 输入文件生成
│                               #   EDGRNParameters / EDGRNConfig / EdgrnLayer
├── edcmp_coord.py              # CSI ↔ EDCMP 坐标系转换
│                               #   CSI: x=East, y=North, z=Up
│                               #   EDCMP: x=North, y=East, z=Down
├── tri2rectpoints.py           # 三角形断层 → 等效矩形分解
│                               #   triangle_to_rectangles()
│                               #   patch_local2d() / patch_local2d_inv()
├── README.md
└── build/                      # 构建工具与 Fortran 源码
    ├── build_edcmp4py_ctypes.py
    ├── edcmp4py_ctypes.f90
    ├── getdata.f
    ├── libedcmp4py.dll         # 预编译 Windows 共享库
    └── libedcmp4py.so          # 预编译 Linux 共享库
```

运行时二进制文件位于 `csi/bin/` 下：

```
csi/bin/
├── edcmp4py_ctypes.py          # ctypes Python 封装模块
├── windows/
│   ├── edcmp.exe               # exe 引擎
│   └── libedcmp4py.dll         # ctypes 共享库
└── ubuntu20.04/
    ├── edcmp                   # exe 引擎
    └── libedcmp4py.so          # ctypes 共享库
```

---

## 调用链

### 从 Fault.buildGFs 到计算

```
Fault.buildGFs(method='edcmp', ...)
  └─ Fault.edcmpGFs(data, ...)
       ├─ EdcmpOptions.resolve()          合并配置参数
       ├─ resolve_edcmp_engine()          选择 ctypes 或 exe
       ├─ _get_edcmp_patch_sources()      准备断层片源参数
       │    └─ triangle_to_rectangles()   三角形 → 矩形分解（如需要）
       └─ 分发到具体后端:
            ├─ _edcmpGFs_exe()            exe 模式
            │    └─ ProcessPoolExecutor
            │         └─ _edcmp_patch_task()
            │              └─ edcmpslip2dis()     写 .inp → 运行 edcmp → 读二进制输出
            └─ _edcmpGFs_inmemory()       ctypes 模式
                 ├─ n_jobs=1: 串行调用 compute_inmemory_edcmp_greens()
                 └─ n_jobs>1: 共享内存 + ProcessPoolExecutor
                      ├─ create_shared_greens()   主进程加载模型到共享内存
                      ├─ _init_shared_memory_worker()  worker 初始化
                      └─ _single_patch_inmemory_greens()
                           └─ compute_inmemory_edcmp_greens()
```

### 共享内存并行机制（ctypes + n_jobs > 1）

EDGRN 格林函数表（`edgrnhs.ss/ds/cl`）通常有几十到几百 MB。为避免每个 worker
进程重复加载，采用 `multiprocessing.shared_memory` 方案：

1. 主进程加载模型，将 `grnss/grnds/grncl` 数组写入共享内存块
2. 通过 `ProcessPoolExecutor(initializer=...)` 将元数据传给 worker
3. Worker 通过名称附着到共享内存，重建只读模型对象
4. 计算完成后主进程清理共享内存

---

## 配置体系

### EdcmpOptions

`EdcmpOptions` dataclass 封装了所有 EDCMP 相关配置参数，替代了原来分散在调用链中的
10+ 个独立关键字参数：

```python
from csi.edgrn_edcmp.edcmp_backends import EdcmpOptions

opts = EdcmpOptions(
    engine='auto',              # 'auto' | 'ctypes' | 'exe'
    fallback_engines=('exe',),  # auto 模式的回退链
    grn_dir='edgrnfcts',        # EDGRN 输出目录（相对于 workdir）
    output_dir='edcmpgrns',     # EDCMP 输出目录
    workdir='edcmp_ecat',       # 工作目录
    layered_model=True,         # True=层状模型, False=均匀半空间
    n_jobs=None,                # 并行 worker 数（None=自动）
    allow_triangle=True,        # 是否允许三角形断层分解
    triangle_rect_dx_km=0.1,    # 三角形→矩形分解精度 (km)
    triangle_rect_dy_km=0.1,
    force_recompute=True,       # 强制重新计算
)
```

支持从 eqtools YAML 配置的 `options:` 字典构建：

```python
opts = EdcmpOptions.from_kwargs(**yaml_options_dict)
```

### 自文档化 API

`EdcmpOptions` 提供多种方式查看可用选项及其含义，无需查阅源码：

```python
from csi.edgrn_edcmp.edcmp_backends import EdcmpOptions

# 1. 文本表格 — 打印所有字段的类型、默认值、说明
EdcmpOptions.describe_options()

# 2. YAML 字符串 — 带行内注释，可直接粘贴到配置文件
print(EdcmpOptions.describe_yaml())
# 输出示例:
#   engine: auto                    # Computation engine: "auto", "exe", or "ctypes"
#   fallback_engines:               # Fallback engine chain when engine="auto" (e.g. ["exe"])
#   ...

# 3. CommentedMap — 用于程序化生成带注释的 YAML 配置
cm = EdcmpOptions.to_commented_map()                # 默认值
cm = EdcmpOptions.to_commented_map(my_opts)         # 自定义实例的值
```

也可通过统一入口查询任意 GF 方法的选项：

```python
from csi.gf_options import describe_gf_options

describe_gf_options('edcmp')                  # 文本格式
describe_gf_options('edcmp', format='yaml')   # YAML 格式
describe_gf_options()                         # 所有可配置方法
```

### 构造时验证（`__post_init__`）

`EdcmpOptions` 在构造时自动校验参数合法性，非法值立即抛出 `ValueError`：

| 校验规则 | 示例 |
|----------|------|
| `engine` 必须是 `"auto"` / `"exe"` / `"ctypes"` | `EdcmpOptions(engine="bad")` → ValueError |
| `fallback_engines` 不能包含 `"auto"` | `EdcmpOptions(fallback_engines=["auto"])` → ValueError |
| `triangle_rect_dx_km` / `dy_km` 必须为正数 | `EdcmpOptions(triangle_rect_dx_km=0)` → ValueError |
| `n_jobs` 如非 None 则必须 >= 1 | `EdcmpOptions(n_jobs=0)` → ValueError |

`from_kwargs()` 会自动忽略未知键并发出 `UserWarning`，适合从 YAML dict 安全构建：

```python
# 未知键 'bad_key' 被忽略并警告，不会报错
opts = EdcmpOptions.from_kwargs(engine='ctypes', bad_key=123)
```

### 与 gf_options 统一入口的关系

`csi.gf_options` 模块提供了方法无关的统一解析入口 `resolve_gf_options()`，
它根据方法名自动选择对应的 Options 类：

```python
from csi.gf_options import resolve_gf_options

# dict → EdcmpOptions（自动构建 + 验证）
opts = resolve_gf_options('edcmp', {'engine': 'ctypes', 'n_jobs': 8})

# EdcmpOptions 实例 → 直接返回（类型检查）
opts = resolve_gf_options('edcmp', EdcmpOptions(engine='ctypes'))

# None → 返回默认 EdcmpOptions()
opts = resolve_gf_options('edcmp', None)

# 无配置选项的方法 → 返回 None
opts = resolve_gf_options('okada', None)
```

在 eqtools 反演框架中，YAML 配置的 `options:` 字段会经过
`_validate_gf_options()` → `resolve_gf_options()` 自动验证。

### 坐标系约定

CSI 和 EDCMP 使用不同的坐标系，`edcmp_coord.py` 负责转换：

| | CSI | EDCMP |
|---|---|---|
| x 轴 | East | North |
| y 轴 | North | East |
| z 轴 | Up (正) | Down (正) |

所有坐标转换在 `edcmp_backends.py` 的计算函数内部自动完成，
调用者只需使用 CSI 坐标系。

---

## 安装

```bash
cd csi_cutde_mpiparallel
pip install -e .
```

安装后 ctypes 引擎直接可用（Windows 已包含预编译的 `libedcmp4py.dll`）：

```python
from csi.edgrn_edcmp import VALID_EDCMP_ENGINES
from csi.edgrn_edcmp.edcmp_backends import resolve_edcmp_engine
print(resolve_edcmp_engine())  # → 'ctypes'
```

---

## 自行编译 ctypes 共享库

当预编译二进制不可用时（如 Linux 首次使用、或需要重新编译），可以用构建脚本编译。

### 前置条件

- `gfortran`（GCC Fortran 编译器）

**Windows (conda/MSYS2):**
```bash
conda install -c conda-forge m2w64-gcc-fortran
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install gfortran
```

### 编译

```bash
# 仅编译（在 build/ 目录下生成 .dll/.so）
python -m csi.edgrn_edcmp.build.build_edcmp4py_ctypes

# 编译并安装到 csi/bin/<platform>/
python -m csi.edgrn_edcmp.build.build_edcmp4py_ctypes --install
```

### Windows conda 环境注意事项

如果遇到 DLL 加载失败（`OSError: ... not found`），通常是缺少 MinGW 运行时 DLL
（`libgfortran-5.dll`, `libgcc_s_seh-1.dll` 等）。引擎会自动搜索 conda 环境中的
`Library/mingw-w64/bin` 目录，但如果仍然失败，可以手动将该目录加入 PATH：

```bash
set PATH=%CONDA_PREFIX%\Library\mingw-w64\bin;%PATH%
```

### WSL 编译注意事项

WSL 中编译的 `.so` 文件只能在 WSL 内使用，不能在 Windows Python 中加载。
反之亦然。如果同时在 Windows 和 WSL 中使用 csi，需要分别编译。

---

## 模块查找顺序

`_import_edcmp4py_module()` 按以下顺序查找 ctypes 模块：

1. 用户指定的 `module_dir` 参数或 `EDCMP4PY_MODULE_DIR` 环境变量
2. `csi/bin/` 目录（pip install 后的默认位置）
3. 全局 `sys.path`

大多数情况下不需要设置 `EDCMP4PY_MODULE_DIR`。

---

## 验证

```bash
# 检查引擎是否可用
python -c "from csi.edgrn_edcmp.edcmp_backends import resolve_edcmp_engine; print(resolve_edcmp_engine())"

# 运行测试（需要 EDGRN 输出的 Green's function 文件）
EDCMP4PY_TEST_GRN_DIR=/path/to/edgrn/output pytest tests/test_edcmp_backends.py -v
```

---

## 参考

- Wang R., Martin F. L., Roth F. (2003). Computation of deformation induced by earthquakes
  in a multi-layered elastic crust — FORTRAN programs EDGRN/EDCMP.
  *Computers & Geosciences*, 29, 195–207.
