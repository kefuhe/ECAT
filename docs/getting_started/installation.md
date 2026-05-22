# 安装与环境检查

本文给出运行本手册教程所需的推荐环境。ECAT 面向科研使用，依赖会随平台、Python 版本和 Green's function 后端有所差异；安装目标不是逐字复刻某一个开发环境，而是先建立可运行环境，再按案例缺什么补什么。

## 推荐策略

建议使用 Anaconda 或 Miniconda 管理环境：

```bash
conda create -n myecat python=3.10
conda activate myecat
python -m pip install -U pip setuptools wheel build numpy
```

Python 3.10 是当前较稳妥的选择。Python 3.11/3.12 也可以尝试，但需要确保 `okada4py` 等编译扩展包有对应 wheel，或本机具备可用编译环境。

## 安装 ECAT

从 GitHub 克隆 [ECAT](https://github.com/kefuhe/ECAT)：

```bash
git clone https://github.com/kefuhe/ECAT.git
cd ECAT
```

ECAT 对外发布为一个代码包，包内包含 `eqtools` 和 `csi` 两个 Python 子包。推荐使用仓库中的安装脚本安装这两个子包：

```bash
# Linux / macOS
chmod +x install.sh
./install.sh

# Windows
.\install.bat
```

安装脚本会进入 `eqtools/` 和 `csi_cutde_mpiparallel/` 子目录并执行 `pip install .`。如果需要开发模式，也可以进入对应子目录后手动执行：

```bash
python -m pip install -e eqtools
python -m pip install -e csi_cutde_mpiparallel
```

## 依赖安装

ECAT 仓库提供了平台相关的 requirements 文件：

```text
requirements/conda-requirements-win-64.txt
requirements/pip-requirements-win-64.txt
requirements/conda-requirements-linux-64.txt
requirements/pip-requirements-linux-64.txt
```

这些文件是维护者在特定 Windows/Linux 平台上的测试环境快照，不一定需要在所有机器上逐包、逐版本完整安装。推荐理解为“参考环境”，而不是强制锁死的通用环境。

如果希望尽量贴近维护者环境，可以使用：

```bash
# Windows
conda create -n myecat --file requirements/conda-requirements-win-64.txt
conda activate myecat
python -m pip install -r requirements/pip-requirements-win-64.txt

# Linux
conda create -n myecat --file requirements/conda-requirements-linux-64.txt
conda activate myecat
python -m pip install -r requirements/pip-requirements-linux-64.txt
```

如果某个包或特定 build 在你的平台上解析失败，可以复制 requirements 文件，删除报错的那一行后继续创建环境。后续运行案例时，如果出现 `ModuleNotFoundError` 或后端缺失，再按报错补装即可：

```bash
conda install -c conda-forge <package>
# 或
python -m pip install <package>
```

`*-full.txt` 文件更接近完整环境导出，适合调试和复现维护者机器，不建议初学者第一步就完整安装。

如需 `conda-forge`：

```bash
conda config --add channels conda-forge
conda config --set channel_priority flexible
```

## 安装 okada4py

[okada4py](https://github.com/kefuhe/okada4py) 是 ECAT 中部分 Okada Green's function 工作流使用的依赖。它不是纯 Python 包，包含 C/C++ 编译扩展；Windows 用户本地编译最容易失败。

优先使用 GitHub Releases 中与你的平台和 Python 版本匹配的预编译 wheel：

[okada4py Releases](https://github.com/kefuhe/okada4py/releases)

下载后在已激活的 `myecat` 环境中安装：

```bash
python -m pip install path/to/okada4py-<version>-<python-tag>-<abi-tag>-<platform-tag>.whl
```

wheel 是平台和 Python 版本相关的，例如：

```text
okada4py-12.0.2-cp310-cp310-win_amd64.whl
okada4py-12.0.2-cp310-cp310-linux_x86_64.whl
```

`cp310` 对应 CPython 3.10，`win_amd64` 对应 64 位 Windows。若提示 `not a supported wheel on this platform`，说明 wheel 与当前 Python 版本、ABI、操作系统或 CPU 架构不匹配，需要换对应 wheel，或从源码安装。

只有在没有合适 wheel 时，才建议从源码编译：

```bash
git clone https://github.com/kefuhe/okada4py.git
cd okada4py
python -m pip install -U pip setuptools wheel build numpy
python -m pip install .
```

Windows 源码编译需要 Microsoft C++ Build Tools；Linux 通常需要 `build-essential` 和 Python 开发头文件；macOS 需要 Xcode Command Line Tools。初学者尤其是 Windows 用户，建议优先使用 release wheel。

安装后检查：

```bash
python -c "import okada4py; print(okada4py.__file__)"
```

## 可选并行与性能依赖

初学者先跑小案例时，不需要一开始就配置完整 MPI 或 oneAPI。等需要运行大规模 Bayesian 采样或多进程生产计算时，再单独处理并行环境。

可选安装：

```bash
conda install -c conda-forge mpi4py
conda install scikit-learn-intelex
```

Linux 上如需 Intel MPI / oneAPI，可参考 Intel oneAPI 官方安装流程，并在 shell 启动文件或当前终端中加载环境，例如：

```bash
source ~/intel/oneapi/setvars.sh intel64
```

MPI 检查见下方“快速检查”。如果 MPI 配置失败，不影响先用非 MPI 模式检查 ECAT 基本功能。

## 案例仓库

案例材料放在 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases)。需要运行案例时，另行克隆案例仓库：

```bash
git clone https://github.com/kefuhe/ECAT-Cases.git
```

[ECAT](https://github.com/kefuhe/ECAT) 负责代码、方法文档、接口说明和模板；[ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 负责数据、脚本、参考输出和图件。

## 快速检查

ECAT 子包：

```bash
python -c "import eqtools; print('eqtools import ok')"
python -c "import csi; print('csi import ok')"
```

CLI 命令：

```bash
ecat-generate-downsample --help
ecat-downsample --help
ecat-generate-nonlinear --help
ecat-generate-config --help
ecat-generate-boundary --help
```

如果命令行入口暂时不可用，可以使用模块形式：

```bash
python -m eqtools.cli_tools.generate_downsample_config --help
python -m eqtools.cli_tools.process_data_downsampling --help
python -m eqtools.cli_tools.generate_nonlinear_config --help
```

如果使用 MPI：

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

## 常见安装判断

- `ModuleNotFoundError`：按缺失包名用 `conda install -c conda-forge ...` 或 `python -m pip install ...` 补装。
- conda requirements 解析失败：删除报错包的固定版本行，继续创建环境；不同平台不必强行使用同一个 build。
- okada4py Windows 编译失败：优先下载匹配 Python 版本和平台的 release wheel。
- `okada4py._okada92` 缺失：说明编译扩展没有正确安装；重新安装匹配 wheel，或在具备编译工具的环境中源码安装。
- MPI 不可用：先用非 MPI 小案例检查 ECAT 基本功能，再单独处理 `mpi4py`、MPI runtime 或 oneAPI/Intel MPI。
