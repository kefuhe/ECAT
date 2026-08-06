# 依赖与运行环境策略

本页面向 eqtools、CSI 和统一 ECAT 的维护者，说明安装文档、包元数据、平台运行时
和发布验证的边界。普通用户从[安装与环境检查](../getting_started/installation.md)
和[安装与运行故障排查](../getting_started/troubleshooting.md)进入。

## 版本兼容范围

当前稳定线保持：

```text
python>=3.10,<3.13
numpy>=1.23,<2
scipy>=1.10,<1.12
numba>=0.58,<0.60
```

Python 3.10是默认推荐和重点回归环境。3.11和3.12属于支持的安装目标，但不得因为
追求较新解释器而删除核心依赖、绕过 wheel ABI检查或放宽尚未验证的 NumPy/SciPy/
Numba范围。版本范围是兼容窗口，不应替换成维护者机器的精确环境导出。

## 依赖事实来源

- 独立 eqtools仓库以自身 `setup.py` 为 eqtools直接依赖事实来源；
- 独立 CSI仓库以自身 `setup.py` 为 CSI直接依赖事实来源；
- 统一 ECAT只聚合、去重两个包的直接依赖；
- `okada4py`、MPI运行时和 BLAS实现的安装要求必须与 Python包归属分开说明；
- PyMC、PyTensor、Theano以及维护者环境中的无关包不得进入基础清单。

如果新增诊断命令直接导入 `threadpoolctl`，应把它声明为实际使用该命令的包的直接
依赖，而不是只依赖 scikit-learn间接安装。

## Conda与 pip边界

Conda负责创建支持的 Python环境和解析地学、数值计算编译依赖；pip负责安装 CSI、
eqtools及其 extras源码。普通安装必须继续支持：

```bash
python -m pip install .
```

editable安装只用于维护者开发：

```bash
python -m pip install -e .
```

公共安装说明不以 `--no-deps` 绕过包元数据。若 pip准备调整现有环境，先检查环境
是否落在声明的兼容范围；不兼容时推荐新建 Python 3.10环境。

公共安装页必须把命令分成两层：

1. 默认路径只显示新用户正常需要的简短 `conda create`、`conda activate` 和
   `python -m pip install .`；
2. 在默认 `conda create` 紧邻位置提供少量、完整验证过的安装前配置，用于用户在
   创建环境前选择 MKL/OpenBLAS 和平台 MPI；这些配置必须明确为互斥替代命令；
3. VPN/残留代理和 solver/channel 这类高频、低风险问题可在对应安装步骤下给一条
   直接替代命令，详细原因和组合情况再链接到故障页；
4. 其他 `--override-channels`、永久 pin、外部 MPI ABI、线程调优等复杂内容只放在
   故障排查或高级对照测试中，并说明对应症状与副作用。

不要要求用户为一次正常安装永久修改 `.condarc`、代理变量、BLAS pin、MPI路径或
线程数。公共文档只保留可迁移的问题模式、检查方法和解决方案，不记录维护者个人
配置或本地绝对路径。

## 网络、VPN与渠道策略

VPN、系统代理、shell代理变量和 Conda `proxy_servers` 属于外部网络层，不是 Python
包依赖。故障页应区分：

- metadata下载前失败：优先检查网络、证书和残留代理；
- metadata已完成而停在 solver：优先检查 solver、channel priority和 pin；
- 只有残留代理被确认时才提供命令级 `env -u ...`；
- 只有渠道混用被确认时才提供本次命令的 `--override-channels` 和 strict priority；
- 不把 `ssl_verify: false` 作为常规修复。

安装与排错命令应尽量是命令级、可恢复的，不直接覆盖用户持久配置。

## BLAS与线程策略

NumPy/SciPy只声明数值接口依赖，不在 `setup.py` 中强制 MKL或 OpenBLAS。发布环境
可以使用不同 BLAS实现，但文档和安装器必须遵守：

- 不持久化 `OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 或 `OMP_NUM_THREADS`；
- 不写入用户系统环境变量或 Conda环境变量；
- 不自动创建 `libblas` 的永久 pin；
- 不把某台机器的最佳线程数作为所有用户默认值；
- 只提供单次命令测速和明确的恢复方法；
- `OMP_NUM_THREADS` 可能影响多个 OpenMP组件，不能作为默认 BLAS修复；
- MPI性能说明必须同时考虑进程数和每进程 BLAS线程数。

如果以后增加性能检查命令，它只能检测、测速和给出建议，默认不得修改环境。候选
线程数可以包含1、4、8、16，但输出必须说明结果只适用于当前机器和测试规模。

公开概念页必须区分 oneAPI工具集合、oneMKL、Intel MPI和 Intel编译器。安装
oneAPI或加载 `setvars` 不等于 NumPy/SciPy已经链接 MKL；只有运行时检测到的
`internal_api` 和动态库路径可以证明后端。编译器也不会重新优化已经安装的 wheel。

BLAS实现选择属于环境变体而不是 `install_requires`。如果文档提供 MKL与
OpenBLAS对照命令，应在第一次 `conda create` 时完成选择，或要求另建完整测试
环境；不得指导用户在已经安装 ECAT 的环境中原地替换二进制栈。默认命令仍由
Conda自动选择实现，显式 provider命令作为安装前可选配置紧邻默认命令。

## MPI策略

`mpi4py` 属于使用 MPI的 Python代码依赖；`mpiexec` 和动态库属于平台 MPI运行时。
普通用户不需要安装完整 oneAPI。只有集群明确规定 Intel MPI时，才加载该实现并
验证 `mpi4py` 与它一致。安装文档必须提供两进程检查，但不得把 MPI运行时错误描述
为 BLSE、CVXOPT或普通数据读取错误。

公共说明必须给出以下层级：MPI标准 → MPI实现 → 动态库/启动器 → mpi4py绑定，
并覆盖 Open MPI、MPICH、Intel MPI和 MS-MPI的主要平台、特点与常见配套。不得把
“相同目录”写成所有平台的硬性要求：Linux/WSL中同一 Conda prefix是最简单的
默认路线，Windows和集群可能合法使用系统级或 module提供的兼容运行时。真正的
判断条件是启动器与 mpi4py加载的 MPI实现或 ABI配套。

两进程检查必须同时输出 rank和 size。两次 `0 1`、两次 rank 0或 PMIx/PMI
singleton警告应明确归类为启动器/运行库错配。完整 oneAPI的 shell初始化可能
修改 `PATH` 和 `LD_LIBRARY_PATH`；文档只说明一般路径覆盖风险，不记录维护者个人
shell文件或机器路径。

## 发布验证

每次依赖或安装说明变化至少检查：

1. Python 3.10完成核心导入、依赖检查和重点回归测试；
2. Python 3.11和3.12能够求解依赖，并在有匹配 wheel的平台完成基础导入；
3. Windows和 Linux至少各验证一个支持环境；
4. 安装页列出的 Windows MKL+MS-MPI、Windows/Linux MKL+Intel MPI和
   Linux MKL+Open MPI组合至少完成 Conda dry-run；
5. MKL与 OpenBLAS至少各完成一次 NumPy线性代数和 CVXOPT/BLSE冒烟；
6. `python -m pip install .` 在推荐环境中不会无故替换 NumPy、SciPy或 Numba；
7. `mpiexec -n 2` 与当前 `mpi4py` 使用同一 MPI实现；
8. 安装页和排错页没有本地绝对路径、个人环境导出或不可公开案例；
9. 相对链接、MkDocs严格构建和 Markdown代码块检查通过。

将独立 eqtools/CSI更新同步到统一 ECAT时，应同步安装页、排错页、导航、依赖聚合
结果和维护者策略；不要在各仓库重新写一套含义不同的线程或版本说明。
