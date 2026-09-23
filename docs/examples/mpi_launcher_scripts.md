# Windows 与 WSL 的 MPI 启动脚本

仓库提供两个可复制的长任务启动器：

- Windows PowerShell：[`run_ecat_mpi_windows.ps1`](../../scripts/run_ecat_mpi_windows.ps1)
- WSL/Linux Bash：[`run_ecat_mpi_wsl.sh`](../../scripts/run_ecat_mpi_wsl.sh)

它们只负责启动层：设置每个 MPI rank 的数值线程上限、切换到案例目录、执行
`mpiexec`、传递 Python 参数并保留退出状态。它们不修改 ECAT YAML、SMC 参数、随机数、
目标函数、MPI 实现或系统永久环境变量。默认使用 `MPLBACKEND=Agg` 保存图件而不弹出窗口。

## 1. 运行前检查

先激活要使用的 Conda 环境，再确认 Python、mpi4py 和 `mpiexec` 属于兼容的 MPI
运行栈：

```bash
conda activate ecat
python -c "from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())"
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"
```

第二条命令应包含 `0 2` 和 `1 2`。如果出现两次 `0 1`，先按
[MPI 故障排查](../getting_started/troubleshooting.md#7-mpi-或-mpiexec-失败)修复启动器和
mpi4py 的配套关系，不要用脚本参数掩盖环境问题。

## 2. 一个线程和两个线程的区别

`ThreadsPerRank` 或 `THREADS_PER_RANK` 控制一个 MPI 进程内部的 MKL、OpenBLAS 或
OpenMP 线程上限，不增加 SMC 粒子数，也不会让一个 rank 同时处理两个候选：

| 组合 | 并行结构 | 典型用途 |
| --- | --- | --- |
| `25 ranks × 1 thread` | 25 个独立候选计算；每个候选的 BLAS 运算单线程 | 进程并发优先、内存带宽紧张或建立受控基线 |
| `20 ranks × 2 threads` | 20 个独立候选计算；每个候选的大矩阵运算最多用 2 线程 | 数据和线性参数较多、矩阵乘法或分解占主导时比较 |
| `10 ranks × 4 threads` | 进程复制较少、候选内部线程较多 | 内存限制明显且 BLAS 多线程有效时比较 |

`ranks × threads/rank` 是线程预算上限，不等于实时 CPU 占用。MPI 进程会复制 Python
对象、GF、协方差和求解工作区；线程主要共享进程内存。因此大模型不能只追求更多 rank。
若 `nchains=100`，4、10、20、25 和 50 都能整除链数，但仍应由阶段耗时和峰值内存决定。

第一次调优建议比较：

```text
25 ranks × 1 thread
20 ranks × 2 threads
```

两次必须使用相同源码、配置、数据、随机种子和输出选项。比较 `PRIOR`、连续几个正式
stage 的中位耗时和峰值内存；beta 路径或有效候选数量不同时，不能只用总时间下结论。

## 3. Windows PowerShell

从仓库根目录把启动器复制到案例目录：

```powershell
Copy-Item scripts\run_ecat_mpi_windows.ps1 path\to\case\run_ecat_mpi.ps1
Set-Location path\to\case
conda activate ecat
```

受控单线程基线：

```powershell
.\run_ecat_mpi.ps1 -PythonScript .\test_smc.py `
    -Ranks 25 -ThreadsPerRank 1
```

两个线程的候选配置：

```powershell
.\run_ecat_mpi.ps1 -PythonScript .\test_smc.py `
    -Ranks 20 -ThreadsPerRank 2
```

### 启动器参数和 Python 参数不要混写

`-PythonScript`、`-Ranks` 和 `-ThreadsPerRank` 是 PowerShell 启动器的参数；Python
脚本自己的 `-r`、`--no-plot` 等参数必须放在 `-ScriptArguments` 后面。PowerShell
允许用唯一前缀缩写参数，因此裸写的 `-r` 会被识别成启动器的 `-Ranks`，而不是传给
Python。下面分别是全新运行和把 `-r` 传给 Python 做续算的单行命令：

```powershell
# 全新运行
.\run_ecat_mpi.ps1 -PythonScript .\test_smc.py -Ranks 25 -ThreadsPerRank 2

# Python 脚本使用 -r 续算
.\run_ecat_mpi.ps1 -PythonScript .\test_smc.py -Ranks 25 -ThreadsPerRank 2 -ScriptArguments "-r"
```

传递多个 Python 参数时使用数组：

```powershell
.\run_ecat_mpi.ps1 -PythonScript .\test_smc.py `
    -Ranks 20 -ThreadsPerRank 2 `
    -ScriptArguments @("-r", "--no-plot")
```

多行命令使用 PowerShell 反引号 `` ` `` 续行，而不是 Bash 的反斜杠 `\`。反引号必须是
该行最后一个字符，后面不能再有空格或注释；不确定时优先复制上面的单行命令。常见错误是：

```powershell
# 错误：-r 会被当成 -Ranks，而它后面没有整数
.\run_ecat_mpi.ps1 -PythonScript .\test_smc.py -r `
    -Ranks 25 -ThreadsPerRank 2
```

如果本机策略不允许直接执行 `.ps1`，只为这次进程临时绕过：

```powershell
powershell -ExecutionPolicy Bypass -File .\run_ecat_mpi.ps1 `
    -PythonScript .\test_smc.py -Ranks 20 -ThreadsPerRank 2
```

脚本开始时保存已有线程环境，结束或失败时恢复，因此不会把本次设置遗留给当前
PowerShell。需要交互绘图时增加 `-InteractivePlots`；长时间采样通常保留默认的无窗口模式。

## 4. WSL/Linux Bash

复制脚本并赋予执行权限：

```bash
cp scripts/run_ecat_mpi_wsl.sh path/to/case/run_ecat_mpi.sh
cd path/to/case
chmod +x ./run_ecat_mpi.sh
conda activate ecat
```

受控单线程基线：

```bash
RANKS=25 THREADS_PER_RANK=1 PYTHON_SCRIPT=test_smc.py \
  ./run_ecat_mpi.sh
```

两个线程的候选配置：

```bash
RANKS=20 THREADS_PER_RANK=2 PYTHON_SCRIPT=test_smc.py \
  ./run_ecat_mpi.sh
```

位置参数会原样传给 Python 脚本：

```bash
RANKS=20 THREADS_PER_RANK=2 PYTHON_SCRIPT=test_smc.py \
  ./run_ecat_mpi.sh -r --no-plot
```

也可以直接编辑脚本顶部的 `RANKS`、`THREADS_PER_RANK`、`CASE_DIR` 和
`PYTHON_SCRIPT` 默认值。当前环境只提供 `python3` 或 `mpirun` 时，分别设置：

```bash
PYTHON_EXE=python3 MPIEXEC=mpirun PYTHON_SCRIPT=test_smc.py \
  ./run_ecat_mpi.sh
```

默认情况下，Bash 中导出的变量只存在于启动脚本及其 MPI 子进程，脚本结束后不会修改
父终端环境。需要交互绘图时使用 `INTERACTIVE_PLOTS=1`。

## 5. 怎样修改而不改变计算含义

可以修改：

- `Ranks` / `RANKS`：MPI 进程数；
- `ThreadsPerRank` / `THREADS_PER_RANK`：每进程数值线程上限；
- 案例目录、Python 脚本名和传入脚本的命令行参数；
- 是否使用非交互绘图后端。

不应把以下内容放进启动器：

- sigma、alpha、几何边界或 rake 约束；
- SMC 的 `nchains`、`chain_length`、先验或恢复状态；
- MPI vendor 专属 pinning 参数，除非已经确认当前实现并完成独立测试；
- 会删除、覆盖或自动重命名既有结果的文件操作。

科学配置继续由 Python 和 YAML 拥有，启动器只负责可重复的执行环境。MPI rank、线程、
物理核心、逻辑 CPU 和内存之间的完整关系见
[进程、MPI Rank、线程与 CPU 亲和性](../concepts/parallel_process_rank_thread.md)。
