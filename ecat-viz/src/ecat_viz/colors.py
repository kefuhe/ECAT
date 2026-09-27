"""Explicit Matplotlib and bundled CPT colormap selection, without registration."""
from os import PathLike

import matplotlib as mpl
from matplotlib.colors import Colormap

from . import cpt


def list_cmaps():
    """Return sorted bundled CPT names without extensions (use ``cpt:NAME``)."""
    return cpt.list_cpts()


def load_cpt(source, *, name=None, kind='continuous', samples=None):
    """Load a bundled CPT, explicit file path, or explicitly requested HTTP(S) URL.

    Continuous maps retain the historical interpolation unless ``samples`` is
    supplied. Listed maps use original CPT colors; samples truncates that list.
    No normalization, B/F/N colors, registration, or display is implicit.
    Use a Path for local files, including files in the current directory.
    """
    if kind not in {'continuous', 'listed'}:
        raise ValueError("kind must be 'continuous' or 'listed'")
    if samples is not None and (type(samples) is not int or samples <= 0):
        raise ValueError('samples must be a positive integer or None')
    if kind == 'listed':
        return cpt.get_listed_cmap(source, name=name, N=samples)[1]
    return cpt.get_cmap(source, name=name,
                        method='cdict' if samples is None else 'list', N=samples)


def get_cmap(name, *, kind='continuous', samples=None):
    """Resolve a Matplotlib name/object, ``cpt:NAME``, or explicit Path.

    Bare names always select Matplotlib maps. CPT names require ``cpt:``;
    explicit paths/URLs go to :func:`load_cpt`. Colormap objects are unchanged
    when samples is None. No global Matplotlib registration is performed.
    """
    if isinstance(name, PathLike) or (isinstance(name, str) and name.startswith('cpt:')):
        return load_cpt(name, kind=kind, samples=samples)
    if kind != 'continuous':
        raise ValueError('kind applies to CPT inputs; use cpt:NAME or load_cpt')
    if samples is not None and (type(samples) is not int or samples <= 0):
        raise ValueError('samples must be a positive integer or None')
    cmap = name if isinstance(name, Colormap) else mpl.colormaps[name]
    return cmap if samples is None else cmap.resampled(samples)
