# 进程、MPI Rank、线程与 CPU 亲和性

ECAT 同时使用单进程数值计算和 MPI 多进程采样。第一次运行只需要知道：普通
BLSE/VCE 直接执行 `python script.py`；Bayesian SMC 使用
`mpiexec -n N python script.py` 启动多个进程。先按工作流默认命令跑通，再根据本页
和排障页调整性能，不需要在安装后立即设置一组永久线程变量。

ECAT 公开文档不假定用户采用 Intel MPI。实际环境可以是 Open MPI、MPICH、
Intel MPI 或 MS-MPI；除非已经用 `MPI.get_vendor()` 确认实现，否则不要复制
`I_MPI_*` 等厂商专属变量。

## 1. 首次运行的通用复制模板

以下三条命令不绑定具体 MPI 实现，适合作为 Windows、Linux 和 WSL 的公开主线。

先查看 mpi4py 实际加载的实现：

```bash
python -c "from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())"
```

再验证两个 rank 能加入同一通信器：

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"
```

正确输出应同时包含 `0 2` 和 `1 2`，顺序可以不同。然后从 4 个 rank 运行代表性
SMC 脚本：

```bash
mpiexec -n 4 python your_smc_script.py
```

这套模板不要求 Intel MPI，也不预设 pinning 或每 rank 线程数。只要 vendor、两个
rank 和案例运行都正常，就先保留简单命令；出现两个 `0 1`、启动器冲突、过量线程
或性能不升反降时，再进入
[安装与运行故障排查](../getting_started/troubleshooting.md#7-mpi-或-mpiexec-失败)。

## 2. 一分钟理解五个概念

| 概念 | 含义 | 在 ECAT 中的例子 |
| --- | --- | --- |
| Python 进程 | 一份独立运行的 Python 解释器和内存空间 | `python test.py` 通常启动一个进程 |
| MPI rank | MPI 通信器中某个进程的编号，从 0 到 `size - 1` | `mpiexec -n 4` 通常产生 rank 0、1、2、3 |
| 线程 | 一个进程内部共享内存的计算执行单元 | MKL/OpenBLAS 可在一次矩阵计算中启动多个线程 |
| 物理核心 | CPU 上实际的计算核心 | 可作为总并行预算的保守起点 |
| 逻辑 CPU | 操作系统可调度的硬件线程 | 启用 SMT/超线程时通常多于物理核心 |

一个 MPI rank 通常对应一个 Python 进程，但一个 rank 内部仍可能有多个 BLAS、
OpenMP、Numba 或 CUTDE 线程。因此：

```text
mpiexec -n 20 python test.py
```

表示启动 20 个 MPI 进程/rank，不是启动 20 个线程。rank 之间有各自的 Python
对象和内存；线程则共享所属进程的内存。ECAT 的 rank 0 通常还负责汇总、保存或
输出部分结果，但具体职责以对应脚本为准。

## 3. `nchains`、`chain_length` 与 rank

Bayesian SMC 中这三个数量属于不同层次：

| 参数 | 控制内容 | 是否等于 MPI 进程数 |
| --- | --- | --- |
| `nchains` | SMC 粒子/链的总数量 | 否 |
| `chain_length` | 每个 SMC 阶段内部的链长度 | 否 |
| `mpiexec -n N` | 启动的 MPI rank 数量 | 是，`N` 是 rank 数 |

ECAT 让不同 rank 分担粒子。rank 数不应超过 `nchains`；能整除 `nchains` 时工作量
通常更均匀。例如 `nchains: 100` 可以比较 4、10、20、25 个 rank，但不能据此
断定 25 一定最快，因为每个 rank 的内存、样本计算成本和 CPU affinity 也会影响
总耗时。

## 4. ECAT 三类计算怎样使用并行

| 任务 | 主要并行层 | 初次运行方式 |
| --- | --- | --- |
| BLSE/VCE 固定几何滑动反演 | 一个 Python 进程内的 BLAS/LAPACK 线程 | `python test_slip_inversion.py` |
| 非线性几何 SMC、固定几何 `FULLSMC` | MPI rank 分担粒子；每个 rank 内仍可能有数值线程 | `mpiexec -n 4 python test_smc.py` |
| `SMC_FJ` | MPI rank 分担粒子，每个粒子内部还执行受约束线性求解 | 先用少量 rank 核对内存和正确性 |

单进程 BLSE 的 8 或 16 线程不能直接套到 MPI 的每个 rank。若 20 个 rank 各自再
启动 16 个 BLAS 线程，理论并发上限可能达到 320，通常会产生严重争用。反过来，
如果 MPI runtime 已经为每个 rank 分配 CPU 并自动控制数值线程，用户也不需要再
手工设置 `MKL_NUM_THREADS=1`。

### 启动期配置诊断

配置读取、规范化和预检发现可继续运行的问题时，ECAT 由 rank 0 在实际发现阶段输出紧凑的
静态诊断，例如：

```text
CONFIG  WARN  CFG_UNUSED_RAKE_ANGLE           rake_angle: ignored because slip_sampling_mode='ss_ds'; it is used only by 'rake_fixed'
```

这类确定性的配置诊断与普通初始化信息使用同一个 `stdout`，每条诊断只显示一次并立即刷新；
因此不会因为 MPI 分别转发 `stdout` 和 `stderr` 而跑到数据、断层初始化信息之前。其他 rank
执行相同配置验证但不重复显示。致命配置错误仍通过异常停止运行，采样期的数值 warning 仍属于
运行期诊断，不会被伪装成配置提示。

### SMC 长任务进度表

SMC-FJ、FULLSMC 和非线性几何 SMC 由 rank 0 输出同一份阶段表：

```text
ATMIP  mode=fresh  chains=100x50  mpi_ranks=25  started=2026-08-27 09:14:03
TIME      ELAPSED    STATUS  CURRENT   BETA (FROM -> TO)       PREVIOUS / DETAIL
09:14:03    00:00:00  RUN     PRIOR      initial population      -
09:21:32    00:07:29  RUN     STAGE 02   0.000000 -> 0.001953    PRIOR 00:07:29
10:00:44    00:46:41  RUN     STAGE 03   0.001953 -> 1.000000    STAGE 02 00:39:12
10:38:49    01:24:46  DONE    ATMIP      stage=3 beta=1.000000   STAGE 03 00:38:05
```

每一行表示一次状态转换，而不是重复报告同一阶段：`CURRENT` 是该行输出后正在执行的工作，
所以长时间没有新行时，最后一行仍能回答“现在运行到哪里”；`ELAPSED` 是本次启动或恢复以来的
累计墙钟时间；`PREVIOUS / DETAIL` 给出刚完成工作的耗时。阶段完成时程序先记住耗时，下一阶段
开始时再把它写到 `PREVIOUS / DETAIL`，因此不会同时保留同一阶段的 `RUN` 和 `DONE` 两行。
最终行用 `DONE ATMIP` 给出最终 beta、阶段号、最后一次 MCMC 的耗时和总耗时。若在初始化阶段
失败，则相同表格用 `FAILED` 记录失败位置、已运行时间和异常摘要。首行的 `chains=100x50`
是 `nchains × chain_length` 的紧凑写法，不是 MPI rank 数。

这张表只由 rank 0 向同一个 `stdout` 追加普通文本；没有回车覆盖、ANSI 控制、终端探测或每个
rank 的重复进度。每条记录都会立即刷新，所以 Windows、Linux、WSL、MPI 转发、输出重定向和
批处理日志采用相同格式，普通初始化 `verbose` 信息也会在进度表开始前刷新。标准运行命令仍是
`python script.py` 或 `mpiexec -n N python script.py`，不要求用户了解或添加 `python -u`。
采样期 Python warning 可以正常出现在状态行之间，不会破坏已有记录。由于 beta 阶段数由采样器自适应
决定，进度表不显示容易误导的完成百分比或 ETA；当某行的目标 beta 为 `1.000000` 时，该行表示
正在完整目标后验下执行最后一次 mutation，并不表示采样已经完成，只有 `DONE ATMIP` 才表示结束。

## 5. CPU affinity / pinning 是什么

CPU affinity 表示某个进程允许在哪些逻辑 CPU 上运行；pinning 表示 MPI runtime、
作业调度器或用户把 rank 绑定到指定 CPU 集合。合理 pinning 可以减少 rank 迁移和
相互争抢，但“自动 pinning”不是 MPI 标准保证的统一行为。

不同 MPI 实现、版本、启动器参数、集群调度器和操作系统可能产生不同结果：

- 有的 MPI runtime 会自动为 rank 划分互不重叠的 CPU；
- 有的只限制每个 rank 可见的 CPU 数，但分区可能重叠；
- 有的主要交给操作系统调度，不自动修改 BLAS 线程数；
- 集群上的 `srun`、容器 CPU 限额或管理员策略还可能覆盖本机默认值。

因此公开文档只给判断方法，不把某台工作站的 `-n 20/-n 25` 或每 rank 线程数写成
所有用户的固定默认。

## 6. 前置环境变量怎样生效、由谁配置

Bash 命令前的 `名称=值` 是**仅对这一条命令及其子进程生效**的临时环境变量：

```bash
OMP_NUM_THREADS=1 mpiexec -n 4 python your_smc_script.py
```

这里由 Bash 把 `OMP_NUM_THREADS=1` 交给 `mpiexec`；本机运行时，`mpiexec`
通常再把它传给各 MPI rank，由实际加载的 OpenMP runtime 读取。它不是 ECAT YAML
字段，也不会修改 `setup.py`、Conda 包或下一次新终端。多节点集群还可能由调度器
或 MPI 参数决定哪些变量被传到远端节点。

PowerShell 的写法和作用范围不同：

```powershell
$env:MKL_NUM_THREADS = "1"
mpiexec -n 4 python .\test_smc.py
Remove-Item Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue
```

赋值会保留在当前 PowerShell 及其后续子进程中，直到删除、改写或关闭终端。

### 常见变量分别由谁读取

| 变量 | 主要读取者 | 控制内容 | 常见配置来源 |
| --- | --- | --- | --- |
| `MKL_NUM_THREADS` | oneMKL | 建议 oneMKL 使用的 OpenMP 线程数 | 用户临时测速；oneMKL 专用设置优先于通用 `OMP_NUM_THREADS` |
| `OMP_NUM_THREADS` | 当前 OpenMP runtime | OpenMP 并行区域的线程上限 | 用户、编译运行库或作业环境 |
| `OPENBLAS_NUM_THREADS` | OpenBLAS | OpenBLAS pthreads 构建的线程数 | 用户临时测速；OpenMP 构建可能改读 `OMP_NUM_THREADS` |
| `NUMBA_NUM_THREADS` | Numba parallel CPU runtime | Numba 并行 CPU 线程池大小 | 用户；独立于 MKL/OpenMP 线程数 |
| `MPLBACKEND` | Matplotlib | 交互或非交互绘图后端 | 用户临时设置、Matplotlib 配置或代码 |
| `CUTDE_USE_BACKEND` | CUTDE/ECAT backend helper | `cpp`、`cuda`、`opencl` 等计算后端选择 | ECAT 配置/脚本或用户；必须在 CUTDE 初始化前确定 |

这些变量也不是每种环境都生效：没有加载 oneMKL 时，`MKL_NUM_THREADS` 不控制
OpenBLAS；Numba 线程池独立于 MKL；`MPLBACKEND` 只影响绘图。先检查实际运行库，
再选择需要测试的变量。

### MPI 实现专属变量：先确认 vendor

ECAT 不默认采用 Intel MPI。只有下面命令返回 Intel MPI 时，`I_MPI_*` 才属于当前
环境的可用配置：

```bash
python -c "from mpi4py import MPI; print(MPI.get_vendor())"
```

| Intel MPI 专属变量 | 控制内容 | 公开文档中的定位 |
| --- | --- | --- |
| `I_MPI_PIN` | 开启或关闭 Intel MPI 进程 pinning；`1` 开启、`0` 关闭 | 仅供已确认 Intel MPI 的用户排查，不放进通用运行模板 |
| `I_MPI_PIN_DOMAIN` | 为混合 MPI/OpenMP 任务划分每 rank 的 CPU domain | 高级调优；普通本机首次运行不设置 |

Intel 官方说明 `I_MPI_PIN` 的默认值就是启用，因此在默认未被其他设置关闭时，
显式写 `I_MPI_PIN=1` 只是重申 Intel MPI 默认行为，不是所有 MPI 都需要的性能开关，
Open MPI、MPICH 和 MS-MPI 也不按这个 Intel 专用变量配置。`I_MPI_PIN` 只控制是否
pin 进程，不直接设置 MKL、OpenMP 线程数，也不单独规定每个 rank 获得几个 CPU；
更具体的 domain/list 由其他 Intel MPI 变量控制。参见
[Intel MPI process pinning variables](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-linux/2021-9/environment-variables-for-process-pinning.html)
和
[Intel MPI/OpenMP interoperability](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-reference-windows/2021-11/interoperability-with-openmp-api.html)。

oneMKL 把线程变量视为建议上限，具体例程仍可能为降低开销或改善数据局部性而使用
更少线程；oneMKL 专用设置优先于通用 OpenMP 设置，代码中的 oneMKL 线程 API
还可能再次覆盖环境变量。参见
[oneMKL threading controls](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2024-1/onemkl-specific-env-vars-for-openmp-thread-ctrl.html)。
Numba 和 Matplotlib 的变量分别见
[Numba environment variables](https://numba.readthedocs.io/en/latest/reference/envvars.html)
与 [Matplotlib backends](https://matplotlib.org/stable/users/explain/figure/backends.html)。

### 没写变量不等于没有默认行为

变量可能来自以下层次：

1. 用户在当前命令前临时设置；
2. 当前终端、shell profile、PowerShell profile 或 Conda activation script；
3. MPI launcher、集群调度器、容器或 CPU set；
4. 数值库、MPI runtime、Matplotlib 或 ECAT 代码自己的默认值；
5. Python/C API 在运行中覆盖环境变量。

因此，采用 Intel MPI 时，`env` 中没有 `I_MPI_PIN` 仍可能按 Intel 默认值启用
pinning；没有 `OMP_NUM_THREADS` 时，OpenMP runtime 仍会根据可见 CPU 或
affinity 选择默认容量。
同样，`threadpool_info()` 显示 MKL 为 1，只能证明当前有效上限，不能单凭这一项
断定是谁写入了 `MKL_NUM_THREADS=1`。

检查当前 shell 中显式存在的变量：

```bash
env | grep -E '^(I_MPI|MKL|OPENBLAS|OMP|NUMBA|CUTDE)_|^MPLBACKEND$'
```

Windows PowerShell：

```powershell
Get-ChildItem Env: | Where-Object Name -Match '^(I_MPI|MKL|OPENBLAS|OMP|NUMBA|CUTDE)_|^MPLBACKEND$'
```

没有输出表示当前 shell 没有显式保存这些变量，不表示相关运行库功能被关闭；运行库
仍会使用自己的默认值。

检查 MPI rank 实际继承到的值：

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; import os; keys=['MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']; print(MPI.COMM_WORLD.Get_rank(), {k: os.environ.get(k) for k in keys})"
```

这些命令只能显示显式环境值；最终线程上限、MPI affinity 和后端仍需结合
`threadpool_info()`、MPI vendor 和系统监视结果判断。已确认 Intel MPI 后，才按需
把 `I_MPI_PIN`、`I_MPI_PIN_DOMAIN` 加入检查列表。

## 7. Windows、原生 Linux 与 WSL 的一般差异

WSL 使用 Linux 用户空间和 Linux 命令，但 CPU、内存和文件系统仍由 Windows 主机
及 WSL 配置提供。它不是 Windows 原生 Python，也不等同于独立 Linux 工作站。

| 项目 | Windows 原生 | Linux 原生 | WSL |
| --- | --- | --- | --- |
| 常见命令行 | PowerShell | Bash | Bash |
| 临时变量 | `$env:MKL_NUM_THREADS = "8"` | `MKL_NUM_THREADS=8 command` | 与 Linux 相同 |
| 常见 MPI | MS-MPI、Intel MPI，也可使用其他配套实现 | Open MPI、MPICH、Intel MPI | 通常使用 Linux 版 Open MPI、MPICH 或 Intel MPI |
| CPU 拓扑 | 大型机器可能涉及 Windows processor groups | 由内核、affinity 和调度器呈现 | WSL 向 Linux 环境呈现主机分配的逻辑 CPU |
| 内存范围 | Windows 主机可用内存 | 主机或作业调度器额度 | 还受 WSL 配置和虚拟机可用内存影响 |
| 文件路径 | Windows 盘符路径 | Linux 路径 | Linux 路径与 Windows 挂载路径并存 |
| MPI 自动 pinning | 取决于所选 MPI 和启动参数 | 取决于 MPI、调度器和启动参数 | 同样取决于 Linux MPI runtime；不能假设所有 WSL 都相同 |

这意味着同一台电脑上的 Windows 与 WSL 也可能加载不同版本的 MKL/OpenBLAS、
OpenMP 和 MPI。某一侧默认更快不代表该操作系统普遍更快；应先核对实际加载的运行
库，再在同一侧比较默认与受控配置。

## 8. 新用户的渐进式运行顺序

1. **先用默认命令跑通。** 不预设线程变量；SMC 首次用 `-n 4`，BLSE 直接单进程。
2. **确认 MPI 正确。** 两进程测试应得到不同 rank，而不是两个独立的 `0/1`。
3. **记录默认基线。** 用同一案例记录完整 wall time、峰值内存和结果摘要。
4. **只有慢或资源紧张时才调参。** 单进程比较 BLAS 线程；MPI 比较 rank 数，并检查每 rank 线程和 affinity。
5. **保留最简单的有效命令。** 如果 MPI 默认已经合理 pin rank 并控制 BLAS，就继续直接使用 `mpiexec -n N`。
6. **模型规模变化后复测。** 特别是 `SMC_FJ`，rank 增加会复制数据、GF 和求解工作区，最佳值可能受内存限制。

MPI 正确性、默认/手动配置的计时方法、线程池检测和性能测试矩阵见
[安装与运行故障排查](../getting_started/troubleshooting.md#6-mpi-进程与-blas-线程相互叠加)。
BLAS、MPI 实现、mpi4py 与 oneAPI 的软件层级见
[Python 数值计算、BLAS、MPI 与 oneAPI](compute_runtime_stack.md)。
