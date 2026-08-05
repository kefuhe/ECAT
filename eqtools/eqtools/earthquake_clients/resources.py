"""Read-only discovery of curated data shipped with earthquake-clients."""

from pathlib import Path


_DATA_ROOT = Path(__file__).resolve().parent / "data"
_CATEGORIES = {"Faults", "Blocks", "GNSS"}
_FORMAT_PRIORITY = {
    ".geojson": 0,
    ".json": 1,
    ".gmt": 2,
}


def packaged_resource_files(category):
    """Return one preferred packaged file for each resource stem.

    Parameters
    ----------
    category : {"Faults", "Blocks", "GNSS"}
        Curated package-data category.

    Returns
    -------
    tuple of pathlib.Path
        Existing package files in stable name order. GeoJSON/JSON is preferred
        to GMT when both encodings share a case-insensitive stem.

    Notes
    -----
    This function never scans or overrides from the current working directory.
    User-owned files belong in a viewer project YAML.
    """

    category = str(category).strip()
    if category not in _CATEGORIES:
        raise ValueError(
            f"Unknown packaged resource category {category!r}; "
            f"expected one of {sorted(_CATEGORIES)}."
        )
    directory = _DATA_ROOT / category
    if not directory.is_dir():
        return ()

    preferred = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            continue
        rank = _FORMAT_PRIORITY.get(path.suffix.casefold())
        if rank is None:
            continue
        key = path.stem.casefold()
        current = preferred.get(key)
        if current is None or rank < current[0]:
            preferred[key] = (rank, path.resolve())
    return tuple(preferred[key][1] for key in sorted(preferred))


__all__ = ["packaged_resource_files"]
