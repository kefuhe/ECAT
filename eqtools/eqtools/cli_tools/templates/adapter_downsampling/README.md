# adapter_downsampling template

This template is for data that need a custom input stage before ECAT
downsampling. The boundary is:

```text
custom files or arrays -> input_adapter.py -> CSI data object
CSI data object -> standard ECAT downsampling runtime
```

Edit `input_adapter.py` for your data source. Keep `run_adapter_downsampling.py`
and `run_timeseries_downsampling.py` thin so they continue to use the package
runtime.

## Generate a config and copy this template

```powershell
ecat-generate-downsample -m sar --sar-reader gamma --sar-mode unwrapped_phase `
  -o downsample.yml --copy-adapter-template
```

For custom input that does not use the standard reader, keep:

```yaml
input_adapter:
  enabled: true
general:
  origin: manual
  lon0: 101.0
  lat0: 37.5
```

`origin: manual` is required when you fully bypass the standard reader, because
the template resolves the projection origin before calling `input_adapter.py`.

## Single scene

```powershell
python run_adapter_downsampling.py -f downsample.yml -s
python run_adapter_downsampling.py -f downsample.yml -c
python run_adapter_downsampling.py -f downsample.yml -d
```

The flags have the same meaning as `ecat-downsample`: `-s` quick-look, `-c`
covariance estimation, and `-d` final downsampling.

## Time series

Add a time-series block:

```yaml
timeseries:
  mode: independent        # independent | reference_grid
  reference_epoch: 2022-01-17
  epochs: [2022-01-05, 2022-01-17, 2022-01-29]
```

`independent` runs the configured `std`, `data`, `trirb`, or `from_rsp` method
for every epoch. `reference_grid` affects only the downsampling step: with
`-d`, it first runs the configured method on `reference_epoch`, then reuses that
reference `.rsp` with `from_rsp` for the other epochs. With `-s` or `-c`, there
is no grid to reuse, so epochs are processed independently.

Implement `input_adapter.load_epoch_data()` before using:

```powershell
python run_timeseries_downsampling.py -f downsample.yml -d
```
