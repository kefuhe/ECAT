# Dingri 2020：BLSE/VCE 线性滑动反演

这是建议的第一个线性滑动分布反演入门案例。

## GitHub 位置

[ECAT-Cases / Cases / Dingri_Events / Dingri_20200320Mw5_6 / LinearInv](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Dingri_Events/Dingri_20200320Mw5_6/LinearInv)

关键文件：

```text
test_tri_inv_BLSE_CovDiag.py
default_config.yml
bounds_config.yml
```

<a id="file-sources"></a>

## 文件来源与生成方式

`test_tri_inv_BLSE_CovDiag.py` 是 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 中的案例脚本，负责把数据读取、断层网格构建、配置加载、求解和结果导出串起来。这个脚本不是 CLI 自动生成的。

`default_config.yml` 和 `bounds_config.yml` 可以用 ECAT CLI 在当前线性反演目录生成模板，再按案例修改：

```bash
ecat-generate-config -o default_config.yml --gf-method cutde
ecat-generate-boundary -o bounds_config.yml -f Dingri_2020
```

生成的模板需要继续设置 `geodata`、`polys`、`alpha`、`sigmas`、`strikeslip/dipslip`、`rake_angle` 等字段。命令细节见 [CLI 命令参考](../reference/cli.md)，主配置字段见 [线性滑动反演配置](../reference/config_linear_slip.md)，约束逻辑见 [ECAT 约束管理器](../reference/constraint_manager.md)。

## 为什么选这个案例

- 规模适合作为 BLSE/VCE 入门。
- 几何在脚本中显式构建。
- 读取两条 InSAR 轨道。
- 展示固定平滑和 penalty loop 两种模式。

## 运行方式

以下命令假设当前工作目录是已克隆的 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 仓库根目录：

```bash
cd Cases/Dingri_Events/Dingri_20200320Mw5_6/LinearInv
python test_tri_inv_BLSE_CovDiag.py --mode single
python test_tri_inv_BLSE_CovDiag.py --mode loop
```

## 脚本对照

这里的流程不是抽象步骤，而是 `test_tri_inv_BLSE_CovDiag.py` 里应逐段讲解的代码。读者需要知道每一段代码在反演链条里负责什么。

### 1. 读取降采样 InSAR 数据

脚本先读取两条已经降采样的 InSAR 轨道。`read_from_varres(...)` 传入的是文件前缀，不是单独的 `.txt/.rsp/.cov` 文件；`buildDiagCd()` 表示本案例使用对角协方差。

```python
sar_t012a_file = os.path.join("..", "InSAR", "Dingri_2020_T012A", "downsampled", "S1_T012A_ifg")
sar_t121d_file = os.path.join("..", "InSAR", "Dingri_2020_T121D", "downsampled", "S1_T121D_ifg")

sar_t012a = insar(name="T012A", lon0=lon0, lat0=lat0, verbose=verbose)
sar_t012a.read_from_varres(sar_t012a_file, triangular=False)
sar_t012a.buildDiagCd()

sar_t121d = insar(name="T121D", lon0=lon0, lat0=lat0, verbose=verbose)
sar_t121d.read_from_varres(sar_t121d_file, triangular=False)
sar_t121d.buildDiagCd()

insardata = [sar_t012a, sar_t121d]
geodata = [sar_t012a, sar_t121d]
```

降采样输出进入线性反演时，脚本读取的是标准前缀对应的 `.txt/.rsp/.cov` 文件组；协方差可按完整矩阵或对角阵进入求解。格式约定见 [InSAR 降采样](../workflows/02_insar_downsampling.md#read-downsampled-output)。

### 2. 建立固定三角断层网格

几何来自前一步非线性几何反演或人工选定的优选模型。`clon/clat/cdepth` 是非线性结果中的断层顶边中点三维坐标；`top/depth` 是线性滑动面向上、向下扩展后的顶部和底部深度。

```python
fault_em1 = TriFault(name="Dingri_2020", lon0=lon0, lat0=lat0, verbose=verbose)
fault_em1.top = 0.0
fault_em1.depth = 8.0
fault_em1.generate_top_bottom_from_nonlinear_soln(
    clon=87.39976,
    clat=28.66787,
    cdepth=1.7692,
    strike=332.2241,
    dip=52.0271,
    length=12,
)
fault_em1.generate_mesh(top_size=1.0, bottom_size=1.5, show=False, verbose=0)
fault_em1.initializeslip(values="depth")
```

这里的 `cdepth` 与 `clon/clat` 一起表示非线性反演得到的顶边中点三维坐标；`top/depth` 才是线性滑动网格向上、向下扩展后的顶部和底部深度。完整桥接逻辑见 [Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md#geometry-results-to-linear-inversion)。

### 3. 加载配置并构造 BLSE 反演对象

`default_config.yml` 控制数据、多项式、Green's functions、平滑等主配置；`bounds_config.yml` 控制滑动边界、rake 约束、sigma/alpha 边界等约束配置。

```python
trifaults_list = [fault_em1]

inversion = BoundLSEMultiFaultsInversion(
    "inv",
    trifaults_list,
    geodata,
    verbose=True,
    config="default_config.yml",
    bounds_config="bounds_config.yml",
    des_enabled=False,
)
```

配置字段的含义见 [线性滑动反演配置](../reference/config_linear_slip.md)，rake、Euler 和自定义线性约束见 [ECAT 约束管理器](../reference/constraint_manager.md)。

<a id="single-mode-fixed-alpha"></a>

### 4. `single` 模式：用固定平滑参数求解

`--mode single` 运行一次 BLSE。这里 `alpha=[np.log10(1/100.0)]` 表示配置值在 `log10` 尺度上为 `-2`，实际 `alpha = 0.01`，对应的平滑惩罚权重约为 `100`。

```python
if args.mode == "single":
    inversion.run(penalty_weight=None, alpha=[np.log10(1 / 100.0)])
    inversion.returnModel(print_stat=False)
```

`returnModel(...)` 会把求解得到的滑动分量分发回断层对象，后续才能写出滑动图和断层面文件。

### 5. `loop` 模式：扫描平滑权重

`--mode loop` 不输出最终滑动模型，而是扫描一组 penalty weight，用于检查数据拟合和模型粗糙度的权衡。本案例实际使用的是 smoothing loop；VCE 是另一种权重估计路线，见 [BLSE/VCE 参考](../reference/blse_vce.md)。

```python
elif args.mode == "loop":
    penalty_weight = [1.0, 5.0, 10.0, 30.0, 50.0, 80.0, 100.0, 125.0,
                      150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0,
                      800.0, 1000.0]
    inversion.simple_run_loop(
        penalty_weight,
        preferred_penalty_weight=10.0,
        output_file="run_loop_covdiag.dat",
        verbose=True,
    )
```

<a id="export-slip-and-model-data"></a>

### 6. 导出滑动、滑动方向和模型数据

`single` 模式求解完成后，脚本先画图，再导出断层滑动和每条 InSAR 轨道的 data/synth/resid。

```python
inversion.extract_and_plot_blse_results(rank=rank, plot_faults=True, plot_data=True)

for trifault in trifaults_list:
    trifault.writeFourEdges2File(dirname="output/stat_infos")
    trifault.writePatches2File(f"output/slip_{trifault.name}.gmt", add_slip="total")
    trifault.writeSlipDirection2File(
        filename=f"output/slipdir_{trifault.name}.txt",
        scale="total",
        factor=0.4,
    )

for sardata in insardata:
    for itype in ["data", "synth", "resid"]:
        sardata.writeDecim2file(f"{sardata.name}_{itype}.txt", itype, outDir="Modeling")
```

这一段解释结果清单从哪里来：`output/` 保存断层模型和滑动方向，`Modeling/` 保存每条 InSAR 轨道的数据、合成值和残差。

## 结果清单

- 最终滑动图
- 断层平面图
- data/synthetic/residual 输出
- `output/slip_Dingri_2020.gmt`
- `output/slipdir_Dingri_2020.txt`
- loop 模式下的 `run_loop_covdiag.dat`

## 跑通判据

`--mode single` 跑通后，至少应能看到：

- `output/slip_Dingri_2020.gmt`
- `output/slipdir_Dingri_2020.txt`
- `output/stat_infos/Dingri_2020_top.gmt`
- `output/stat_infos/Dingri_2020_bottom.gmt`
- `Modeling/T012A_data.txt`
- `Modeling/T012A_synth.txt`
- `Modeling/T012A_resid.txt`
- `Modeling/T121D_data.txt`
- `Modeling/T121D_synth.txt`
- `Modeling/T121D_resid.txt`

`--mode loop` 跑通后，应生成 `run_loop_covdiag.dat`，用于检查平滑权重、数据拟合和模型粗糙度的权衡。
