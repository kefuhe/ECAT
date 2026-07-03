# Wushi：InSAR-only 非线性几何反演

这是建议的第一个非线性几何反演案例。

## GitHub 位置

[ECAT-Cases / Cases / Wushi_20240122M7_0](https://github.com/kefuhe/ECAT-Cases/tree/main/Cases/Wushi_20240122M7_0)

关键目录：

```text
InSAR/downsample/
Nonlinear/
```

## 文件来源与生成方式

`Nonlinear/` 目录中的运行脚本和事件配置来自 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 的 Wushi 案例材料，适合直接作为复现实例阅读。

如果新建一个非线性几何反演目录，可以先在当前目录生成标准配置模板：

```bash
ecat-generate-nonlinear -o default_config.yml
```

生成的 `default_config.yml` 只是模板，需要参照 Wushi 案例修改 `bounds`、`fixed_params`、`geodata`、`fault_aliasnames`、`nchains` 和 `chain_length`。命令细节见 [CLI 命令参考](../reference/cli.md)，字段含义见 [非线性几何反演配置](../reference/config_nonlinear_geometry.md)。

## 为什么选这个案例

- InSAR-only，避免一开始处理 GPS 格式问题。
- 使用两条 Sentinel-1 轨道。
- 展示 Bayesian 非线性几何反演路线。
- 已有样本、模型摘要、KDE 图和数据拟合产品。

## 数据入口

非线性脚本读取：

```text
InSAR/downsample/S1T034D_ifg
InSAR/downsample/S1T056A_ifg
```

典型代码：

```python
from csi.insar import insar

sar = insar("S1T056A_ifg", lon0=lon0, lat0=lat0, verbose=False)
sar.read_from_varres("../InSAR/downsample/S1T056A_ifg", cov=True)
```

## 运行方式

以下命令假设当前工作目录是已克隆的 [ECAT-Cases](https://github.com/kefuhe/ECAT-Cases) 仓库根目录：

```bash
cd Cases/Wushi_20240122M7_0/Nonlinear
mpiexec -n 4 python test_nonlinear_mag_rake.py -r
python test_nonlinear_mag_rake.py
```

## 脚本对照

### 1. 设置投影原点

```python
lon0 = 78.56
lat0 = 41.19
```

`lon0/lat0` 必须与后续 InSAR 数据、断层对象和绘图保持一致。这个点不是反演参数，而是 CSI/ECAT 对局部坐标转换使用的参考原点。

### 2. 读取降采样 InSAR 前缀

```python
basedir = ".."
varres_t034d = os.path.join(basedir, "InSAR", "downsample", "S1T034D_ifg")
varres_t056a = os.path.join(basedir, "InSAR", "downsample", "S1T056A_ifg")

coDscsar = insar("S1T034D_ifg", lon0=lon0, lat0=lat0, verbose=verbose)
coDscsar.read_from_varres(varres_t034d, cov=True)

coAscsar = insar("S1T056A_ifg", lon0=lon0, lat0=lat0, verbose=verbose)
coAscsar.read_from_varres(varres_t056a, cov=True)

geodata = [coAscsar, coDscsar]
```

`read_from_varres(...)` 读取的是同名前缀，对应 `S1T056A_ifg.txt/.rsp/.cov` 这一组文件。降采样输出如何进入反演见 [InSAR 降采样](../workflows/02_insar_downsampling.md#read-downsampled-output)。

### 3. 用配置文件构造 `explorefault`

```python
expfault = explorefault(
    "invrc",
    lat0=lat0,
    lon0=lon0,
    config_file="default_config.yml",
    geodata=geodata,
    verbose=verbose,
)
nchains = expfault.nchains
chain_length = expfault.chain_length
```

`default_config.yml` 中的 `nchains` 和 `chain_length` 控制采样规模和计算成本；`geodata.sigmas.values` 和 `geodata.verticals` 的顺序要与脚本中的 `geodata` 一致。配置字段见 [非线性几何反演配置](../reference/config_nonlinear_geometry.md)。

### 4. 运行采样和绘图

```python
expfault.setPriors(bounds=None, initialSample=None, datas=None)
expfault.setLikelihood(datas=None, verticals=None)

if args.run:
    expfault.walk(
        nchains=nchains,
        chain_length=chain_length,
        comm=comm,
        filename="samples_mag_rake_multifaults.h5",
        save_every=2,
    )

if not args.no_plot:
    expfault.extract_and_plot_bayesian_results(
        rank=rank,
        filename="samples_mag_rake_multifaults.h5",
        plot_faults=True,
        plot_sigmas=True,
        plot_data=True,
        save_data=True,
    )
```

运行命令中的 `-r` 对应 `expfault.walk(...)`，不带 `-r` 时主要用于读取已有样本并重新绘图。

### 5. 解释 KDE 和几何摘要

```python
expfault.plot_kde_matrix(
    plot_sigmas=True,
    plot_faults=False,
    filename="kde_matrix_sigmas.png",
)

expfault.plot_kde_matrix(
    plot_sigmas=False,
    plot_faults=True,
    filename="kde_matrix_faults.png",
)
```

`kde_matrix_faults.png` 用来检查几何参数后验分布，`kde_matrix_sigmas.png` 用来检查数据标准差超参数。sigma 参数图的配置含义见 [Sigmas 与 Alpha 配置模式](../reference/sigmas_alpha.md)。

### 6. 进入线性滑动反演

非线性输出中的 `model_results_median.txt` 或等价摘要提供优选几何。进入 BLSE/VCE 时，`lon/lat/depth` 要作为断层顶边中点三维坐标传给 `clon/clat/cdepth`；`top/depth` 则是线性滑动面扩展后的顶部和底部深度。完整桥接逻辑见 [Bayesian 非线性几何反演](../workflows/03_nonlinear_geometry_bayesian.md#geometry-results-to-linear-inversion)。

## 跑通判据

一次完整运行后，至少应能看到：

- `samples_mag_rake_multifaults.h5` 或 `samples_final.h5`
- `model_results_median.txt`
- `kde_matrix_faults.png` 或对应断层几何 KDE 图
- `kde_matrix_sigmas.png`
- `Modeling/S1T034D_ifg_data.txt`
- `Modeling/S1T034D_ifg_synth.txt`
- `Modeling/S1T034D_ifg_resid.txt`
- `Modeling/S1T056A_ifg_data.txt`
- `Modeling/S1T056A_ifg_synth.txt`
- `Modeling/S1T056A_ifg_resid.txt`

如果只重新绘图而不重新采样，应确认脚本能读取已有 HDF5 样本并重新生成 KDE 与 data/synth/resid 输出。
