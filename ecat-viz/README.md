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

## Choosing an entry point

Use `with PlotStyle(...)` for a local figure. Start with `science`, `science-serif`,
`chinese`, `notebook` or `presentation`; inspect `list_presets()` for the full list.
Overlay a color cycle with `PlotStyle(['science', 'colors-bright'])`. Explicit
style parameters override presets; `rcparams={...}` has final priority. Invalid
explicit parameters raise before the global style changes. A Colormap and its
data normalization remain separate Matplotlib choices.

For a single output, `fig.savefig('result.pdf', dpi=600)` is sufficient. Use
`save_fig` for multiple formats and `finish_fig` when one function manages the
save/show/close lifecycle. Figures and axes stay ordinary Matplotlib objects.

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

Discovery and loading include both `.cpt` and `.2cpt`: `list_cmaps()` currently
lists 84 palettes (81 `.cpt`, 3 `.2cpt`; the extra resource is a provenance text).
For example, `get_cmap('cpt:GMT_topo')` loads `GMT_topo.2cpt`. If both suffixes
share a stem, `.cpt` takes priority; an explicit suffix selects that exact file.
The newer `kind`/`samples` API and historical `method`/`N` API are separate
signatures; changing imports does not translate their parameters.

## Publication sizes and reusable configuration

Every public preset/style-directory/width entry initializes shared state before a user operation.
User presets remain removable and explicit overrides survive later calls. Named widths are always
in inches; `unit="cm"` applies to numeric/tuple sizes and explicit height. Dimensions must be
finite and positive, including NumPy integer/real scalars. A full size tuple
ignores fraction, height and aspect.

```python
from ecat_viz import publication_figsize, register_column_width, save_column_width

size = publication_figsize(10, unit='cm', aspect=0.75)  # returns inches
register_column_width('my_journal', 3.25)  # this session; inches
save_column_width('my_journal', 3.25)      # persist to the default user config
with PlotStyle('science', figsize='my_journal'):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    finish_fig(fig, 'journal.pdf')
```

`save_column_width(..., config_path=Path(...))` writes a selected file; only the
default search locations are automatically read in a fresh process. Their order
is `~/.config/eqtools/viztools.json`, `~/.config/eqtools/plottools.json`, then the
legacy `~/.config/statutils/plottools.json` and `~/.plottools.json` (which warn).
These historical names remain valid without installing eqtools.

```json
{"column_widths": {"my_journal": 3.25}}
```

The first existing user configuration is loaded once with unchanged search priority. Invalid JSON
or width sections warn without applying partial overrides. `save_column_width()` validates first,
preserves unrelated keys, and atomically replaces the file. Parse/validation/write failures raise
and preserve the original file. Width names are case-insensitive: equal aliases
merge on save; conflicting aliases warn and apply no widths on load, and raise
on save. Correct the ambiguous JSON explicitly before saving. Unknown top-level
keys are preserved; concurrent writers must coordinate their own updates.

## Fonts and style extensions

```python
from ecat_viz import list_chinese_fonts, register_style_directory

print(list_chinese_fonts(refresh=True))  # best sans/serif, or None when absent
# For your own .mplstyle files:
# register_style_directory(Path('styles'))
```

Refresh bypasses memory and disk caches and replaces the current result. Fonts
must first be visible to Matplotlib's active `fontManager`; after installing a
font, restart Python or explicitly use `fontManager.addfont('font.ttf')`.
The optional JSON cache expires after seven days; corrupt data or a read-only
home triggers a live probe without blocking plotting. Old pickle caches are
ignored and need no migration. The caller owns font installation and glyph
coverage. Explicit font properties and files are preserved by font baking.

## Arrays and optional file readers

```python
import numpy as np
from ecat_viz import plot_raster, plot_geotiff

data = np.arange(12, dtype=float).reshape(3, 4)
fig, ax, image = plot_raster(data, cmap='viridis', show=False)
finish_fig(fig, 'array.png')
# With the raster extra and a georeferenced file:
# fig, ax, image = plot_geotiff('grid.tif', axis='geo', show=False)
```

Use `plot_raster(data, x=x, y=y)` for 1-D coordinates or 2-D meshes, or `extent`
for an image rectangle. A Matplotlib `norm` owns the color scale when supplied;
do not combine it with explicit limits or symmetric scaling. Without `norm`,
the default percentile is 99: central 99% for an asymmetric scale, or the 99th
percentile of `abs(data - center)` for `symmetric=True`. Use `percentile=None`
for the full finite range. Values, units and signs are the caller's responsibility.

`plot_geotiff()` uses native CRS coordinates: north-up rasters keep imshow; rotated/sheared/flipped
rasters use affine pixel edges with pcolormesh. It does not reproject. Coordinates cannot be
overridden via x/y/extent or origin=lower; use `plot_raster()` for caller-supplied coordinates.
Geographic labels require appropriate geographic coordinates. Missing/projected
CRS, identity-like transforms and index-like bounds still warn; a valid rotated
geographic affine is supported and no longer triggers a false warning.

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
