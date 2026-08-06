# 独立开发仓库与 ECAT 集成

eqtools 与 CSI 可以在各自开发仓库中独立维护和验证；对外发布时，再将确认稳定的
代码、包元数据和公共文档选择性集成到统一 ECAT 仓库。本页只说明公开、可复用的
维护流程，不依赖某台机器的绝对路径。

## 三层职责

| 层级 | 事实来源 | 主要职责 |
| --- | --- | --- |
| 独立 eqtools 仓库 | `setup.py`、`eqtools/`、包级测试和内部设计文档 | 开发 eqtools，执行 editable 增量安装和功能验证 |
| 独立 CSI 仓库 | `setup.py`、`csi/`、CSI 测试和后端文档 | 开发 CSI，验证 Green's function 与基础反演对象 |
| 统一 ECAT 仓库 | `eqtools/`、`csi_cutde_mpiparallel/`、顶层 `docs/` 和 `requirements/` | 对外发布、首次完整安装、统一用户手册和环境清单 |

统一 ECAT 是公开安装和文档的最终权威入口，但不是独立开发仓库的自动镜像。同步时
允许有意删减内部计划、审计、测试材料或尚未进入稳定公开接口的代码。

## 安装文档的所有权

`docs/getting_started/installation.md` 是 ECAT 集成层页面，必须同时说明 CSI、
eqtools、统一依赖清单和安装脚本。包级 README 只保留简要入口，不替代完整安装页。

`docs/getting_started/troubleshooting.md` 与
`docs/concepts/parallel_process_rank_thread.md`、
`docs/concepts/compute_runtime_stack.md`、
`docs/developer/dependency_environment_policy.md` 也属于安装发布边界，必须与安装页
同步，不能在各仓库保留含义不同的代理、BLAS、MPI 或线程说明。

eqtools 的 workflow、example 和 reference 页面可以在独立仓库先维护，再同步到
ECAT 顶层 `docs/`。涉及安装命令时统一使用以下语义：

```bash
# ECAT 用户先进入 eqtools 子项目；独立 eqtools 仓库中省略 cd
cd eqtools
python -m pip install .
python -m pip install ".[viewer]"
```

这样同一个普通包级命令既适用于 ECAT 子目录，也适用于独立 eqtools 根目录。
维护者需要直接编辑源码时才改用 `python -m pip install -e .`。

## 推荐同步顺序

1. 在独立 eqtools 或 CSI 仓库完成聚焦测试和真实工作流检查。
2. 比较独立仓库与 ECAT 对应子目录，只同步准备公开的代码、资源、`setup.py` 和
   必要 README；不要整目录覆盖。
3. 将稳定的 eqtools 用户文档选择性同步到 ECAT 顶层 `docs/`，保留 ECAT 自己的
   安装入口、总导航和发布边界说明。
4. 若 `install_requires` 或 extras 改变，从 ECAT 根目录重新生成并检查唯一环境清单。
5. 在 ECAT 结构中分别验证首次安装和包级增量安装。
6. 运行文档链接、导航、代码块和 MkDocs 严格构建检查。

## 依赖事实来源

独立包只维护自己的直接依赖声明：

```text
eqtools/setup.py                 -> eqtools install_requires / extras_require
csi_cutde_mpiparallel/setup.py   -> CSI install_requires
```

ECAT 集成层将二者聚合为：

```text
requirements/ecat-requirements.txt
```

生成工具会分别审计 CSI 和 eqtools 的源码导入；某个依赖即使已由另一个包声明，也
不能掩盖当前包自己的元数据缺项。这样两个独立开发仓库的 editable 安装都能得到
各自需要的直接依赖。

依赖所有权按源码直接使用关系确定，而不是按生成清单的输出顺序确定。两边都 import
的包必须分别保留在两个 `setup.py` 中；生成后的唯一清单把它们放入 shared 分组并
只输出一次。只被一个源码树 import 的包进入相应的 CSI-only 或 eqtools-only 分组。
生成检查也会拒绝“本包没有 import、却被错误加入本包基础依赖”的反向错误。

同步前可在 ECAT 根目录直接审计两个独立 checkout，而不生成或修改统一清单：

```bash
python scripts/generate_requirements.py --audit-only \
  --csi-project <path-to-csi-checkout> \
  --eqtools-project <path-to-eqtools-checkout>
```

统一清单只从两个包的直接依赖元数据生成，完整环境快照不进入公开发布文件。修改
依赖后，在 ECAT 根目录运行：

```bash
python scripts/generate_requirements.py
python scripts/generate_requirements.py --check
```

## 发布前检查

- Python 3.10完成重点回归，3.11和3.12仍符合包元数据声明并能完成基础安装检查；
- `okada4py` 的 wheel/源码安装说明与支持平台一致；
- CSI 与 eqtools 能在统一环境中导入；
- mesh、SAR/InSAR、BLSE/VCE 和 SMC 基础依赖仍在 base 环境；
- 基础清单只包含核心功能直接使用的运行依赖，可选工具仍由 extras 或专项说明管理；
- `cd eqtools && python -m pip install .` 能完成普通增量安装；
- editable安装只作为维护者开发入口；
- extras 从 eqtools 项目根目录使用 `.[extra]` 安装；
- 安装脚本和公共文档不持久化 BLAS线程数、不强制 MKL/OpenBLAS，也不创建永久
  `libblas` pin；
- 安装页保留简短默认命令，并在同一步骤给出经过 dry-run 验证、互斥的安装前
  MKL/MPI配置，以及 VPN残留代理和 solver/channel两个高频直接替代；复杂组合再
  链接排错页或运行栈概念页；
- Windows的 MKL+MS-MPI、MKL+Intel MPI和 Linux/WSL的 MKL+Open MPI、
  MKL+Intel MPI至少完成依赖 dry-run；
- 计算运行栈页区分 oneAPI、oneMKL、Intel MPI、编译器、MPI实现和 mpi4py绑定；
- 并行基础页区分进程、rank、线程、CPU affinity 和环境变量归属；
- MPI两进程检查输出 rank与 size，并检查启动器和动态库是否配套；
- `python scripts/generate_requirements.py --check` 通过；
- 公共文档不存在本地绝对路径、私有案例目录或失效相对链接。
