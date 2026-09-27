# ecat-viz

General scientific plotting on NumPy and Matplotlib. Install from this directory:

```bash
python -m pip install .
# Optional file readers:
python -m pip install ".[raster]"
```

Import name: `ecat_viz`. CSI and eqtools are not required. Python 3.10–3.12
is supported; validation currently uses Python 3.10 and Matplotlib 3.8.

```python
import matplotlib.pyplot as plt
from ecat_viz import PlotStyle, finish_fig, get_cmap, list_cmaps

with PlotStyle('science', figsize='single', fontsize=8):
    fig, ax = plt.subplots()
    ax.scatter([0, 1, 2], [2, 1, 3], c=[0, 1, 2], cmap=get_cmap('viridis'))
    finish_fig(fig, 'figure.png', show=False, dpi=300)

print(list_cmaps())  # bundled CPT names
cmap = get_cmap('cpt:precip3_16lev_change', samples=15)
```

## Ownership and modules

| Module | Responsibility |
| --- | --- |
| Top-level `ecat_viz` | `PlotStyle`, presets, fonts, widths, formatters, save/show, color API |
| `ecat_viz.colors` | `get_cmap`, `load_cpt`, `list_cmaps` |
| `ecat_viz.cpt` | Historical CPT signatures and parser; `plot_cmaps` previews |
| `ecat_viz.raster` | Prepared arrays, DataArray, GeoTIFF and NetCDF display |
| `ecat_viz.axes3d` | Generic Matplotlib 3-D axis appearance |

Fault objects, boundary/dip diagnostics, slip distributions, inversion semantics,
units, projections and model calculations remain owned by CSI/eqtools. The
library accepts Matplotlib axes and prepared arrays; it does not interpret faults.

## Colors and explicit input selection

```python
from pathlib import Path
from ecat_viz import get_cmap, load_cpt
from ecat_viz import cpt

get_cmap('viridis')                           # Matplotlib name
get_cmap('cpt:precip3_16lev_change')            # bundled CPT, continuous
load_cpt(Path('custom.cpt'))                  # explicit local file
load_cpt('cpt:precip3_16lev_change', kind='listed')
cpt.get_cmap('precip3_16lev_change.cpt', method='list', N=15)
positions, cmap = cpt.get_listed_cmap('precip3_16lev_change.cpt')
```

`get_cmap` accepts a Matplotlib Colormap unchanged, or resamples when requested.
`load_cpt` accepts explicitly requested HTTP(S) URLs with a 30-second timeout.
Names are not silently registered, directories are not searched automatically,
and CPT limits do not create a data normalization. The historical parser keeps
its existing interpolation and ignores CPT B/F/N records. Listed `samples`
truncates original colors; it does not interpolate to a different palette.

## Sizes, configuration and GeoTIFF coordinates

Every public preset/style-directory/width entry initializes shared state before a user operation.
User presets remain removable and explicit overrides survive later calls. Named widths are always
in inches; `unit="cm"` applies to numeric/tuple sizes and explicit height. Dimensions must be
finite and positive. A full size tuple ignores fraction, height and aspect.

The first existing user configuration is loaded once with unchanged search priority. Invalid JSON
or width sections warn without applying partial overrides. `save_column_width()` validates first,
preserves unrelated keys, and atomically replaces the file. Parse/validation/write failures raise
and preserve the original file; concurrent writers must coordinate their own updates.

`plot_geotiff()` uses native CRS coordinates: north-up rasters keep imshow; rotated/sheared/flipped
rasters use affine pixel edges with pcolormesh. It does not reproject. Coordinates cannot be
overridden via x/y/extent or origin=lower; use `plot_raster()` for caller-supplied coordinates.

## Behavior and migration

`with PlotStyle(...)` restores rcParams, including nesting and exceptions.
`apply()/reset()` change process-wide state and must be paired. Matplotlib is
not isolated across threads; the registry lock only protects internal updates.
Font baking preserves explicit fonts, font files, TeX and text properties.
`show_fig(fig)` limits this figure's DPI, then calls `plt.show()` for all windows.
`update_style_library()` reloads bundled, SciencePlots and registered directory
styles after a Matplotlib library reload; it preserves presets, widths and rcParams.
Keep plotting in one thread/process, or explicitly manage Matplotlib yourself.

Existing `eqtools.viztools`, `eqtools.plottools` and `eqtools.getcpt.get_cpt`
function imports remain compatibility entries sharing this implementation.
There is one registry. CSI imports the general library directly. New scripts
should import `ecat_viz` for general plotting and eqtools for fault diagnostics.

The old physical `eqtools/cpt` and `eqtools/viztools/styles` directories are
removed. Replace filesystem/resource lookups and `get_cpt.basedir` mutation
with `load_cpt(Path(...))`, `list_cmaps()` or `register_style_directory(Path(...))`.
For resource bytes, use `importlib.resources.files('ecat_viz').joinpath('cpt',
'NAME.cpt')` and its `open()`/`read_bytes()`; avoid converting resource handles
into persistent filesystem strings. Private internal module names are not a
supported extension API. Existing configuration locations and priority are
retained in this release (`~/.config/eqtools/viztools.json` first).

See LICENSE, COPYING-GPL-3.0, NOTICE and individual CPT headers for provenance
and licensing; bundled resources are not all MIT-licensed.
