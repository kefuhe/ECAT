"""Project factories for curated, lazily loaded background resources."""

from dataclasses import dataclass, replace
import re

from ..earthquake_clients.resources import packaged_resource_files
from .models import LayerSpec, ViewerProject


@dataclass(frozen=True)
class BackgroundInfo:
    """Small, pre-load description of one packaged scientific resource."""

    stem: str
    label: str
    group: str
    group_label: str
    scope: str
    description: str
    citation: str
    url: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    units: str | None = None
    reference_frame: str | None = None
    quick_default: bool = False


_GLOBAL = ("global_context", "Global tectonic context")
_REGIONAL = ("regional_faults", "Regional fault data")
_BLOCKS = ("regional_blocks", "Regional blocks")
_GNSS = ("gnss", "GNSS velocity fields")

_BACKGROUND_INFO = {
    "afead_v2022": BackgroundInfo(
        "AFEAD_v2022",
        "AFEAD v2022 — Eurasia active faults",
        *_REGIONAL,
        "Eurasia",
        "Continental-scale active-fault database; detailed and relatively heavy.",
        (
            "Zelenin et al. (2022), Earth System Science Data, "
            "doi:10.5194/essd-14-4489-2022"
        ),
        "https://doi.org/10.5194/essd-14-4489-2022",
        bbox=(-35.9, 180.0, -20.0, 87.7),
    ),
    "cafd400_v2023_1": BackgroundInfo(
        "CAFD400_V2023_1",
        "CAFD v2023 — China active faults (1:4M)",
        *_REGIONAL,
        "China and adjacent regions",
        "China Active Faults Database at 1:4,000,000 scale.",
        (
            "Wu et al. (2024), Earth System Science Data, "
            "doi:10.5194/essd-16-3391-2024"
        ),
        "https://doi.org/10.5194/essd-16-3391-2024",
        bbox=(72.0, 134.5, 1.3, 55.0),
    ),
    "cn-faults": BackgroundInfo(
        "CN-faults",
        "China faults — CAFD v2023",
        *_REGIONAL,
        "China and adjacent regions",
        "GMT-China distribution derived from the China Active Faults Database.",
        (
            "Wu et al. (2024), Earth System Science Data, "
            "doi:10.5194/essd-16-3391-2024"
        ),
        "https://docs.gmt-china.org/latest/dataset/CN-faults/",
        bbox=(72.0, 134.5, 15.0, 55.0),
    ),
    "cn-faults-gmtchina": BackgroundInfo(
        "CN-faults-gmtchina",
        "China faults — GMT-China regional set",
        *_REGIONAL,
        "China",
        "Regional fault-line distribution provided with the ECAT backgrounds.",
        "See the GMT-China geospatial-data distribution metadata.",
        "https://github.com/gmt-china/china-geospatial-data",
        bbox=(73.6, 134.2, 19.8, 51.5),
    ),
    "gem_active_faults_harmonized": BackgroundInfo(
        "gem_active_faults_harmonized",
        "GEM — Global Active Faults",
        *_GLOBAL,
        "Global",
        "Harmonized global active-fault context; load only when needed.",
        "Styron and Pagani (2020), GEM Global Active Faults Database.",
        "https://github.com/GEMScienceTools/gem-global-active-faults",
        bbox=(-180.0, 180.0, -66.2, 86.9),
    ),
    "pb2002_boundaries": BackgroundInfo(
        "PB2002_boundaries",
        "PB2002 — Plate boundaries",
        *_GLOBAL,
        "Global",
        "Lightweight plate-boundary context intended for maps.",
        (
            "Bird (2003), Geochemistry, Geophysics, Geosystems, "
            "doi:10.1029/2001GC000252"
        ),
        "https://peterbird.name/oldftp/PB2002/",
        bbox=(-180.0, 180.0, -66.2, 86.9),
        quick_default=True,
    ),
    "pb2002_orogens": BackgroundInfo(
        "PB2002_orogens",
        "PB2002 — Orogens",
        *_GLOBAL,
        "Global",
        "Diffuse orogenic zones in the PB2002 plate model.",
        (
            "Bird (2003), Geochemistry, Geophysics, Geosystems, "
            "doi:10.1029/2001GC000252"
        ),
        "https://peterbird.name/oldftp/PB2002/",
        bbox=(-180.0, 180.0, -36.6, 77.0),
    ),
    "pb2002_plates": BackgroundInfo(
        "PB2002_plates",
        "PB2002 — Plate outlines",
        *_GLOBAL,
        "Global",
        "Plate outlines for the 52-plate PB2002 model.",
        (
            "Bird (2003), Geochemistry, Geophysics, Geosystems, "
            "doi:10.1029/2001GC000252"
        ),
        "https://peterbird.name/oldftp/PB2002/",
        bbox=(-180.0, 180.0, -90.0, 90.0),
    ),
    "pb2002_steps": BackgroundInfo(
        "PB2002_steps",
        "PB2002 — Boundary steps",
        *_GLOBAL,
        "Global",
        "Detailed PB2002 boundary steps and kinematic attributes; relatively heavy.",
        (
            "Bird (2003), Geochemistry, Geophysics, Geosystems, "
            "doi:10.1029/2001GC000252"
        ),
        "https://peterbird.name/oldftp/PB2002/",
        bbox=(-180.0, 180.0, -66.2, 86.9),
    ),
    "cn-blocks": BackgroundInfo(
        "CN-Blocks",
        "China tectonic blocks",
        *_BLOCKS,
        "China",
        "Regional tectonic-block outlines distributed with ECAT.",
        "See the source properties and accompanying ECAT resource metadata.",
        bbox=(64.0, 116.2, 14.0, 50.0),
    ),
    "wang_2020_china_eurasian": BackgroundInfo(
        "wang_2020_china_eurasian",
        "Wang et al. 2020 GNSS — stable Eurasia",
        *_GNSS,
        "Continental China and reference sites",
        "Horizontal GNSS velocities transformed to a stable-Eurasia frame.",
        (
            "Wang and Shen et al. (2020), Journal of Geophysical Research, "
            "doi:10.1029/2019JB018774"
        ),
        "https://doi.org/10.1029/2019JB018774",
        bbox=(-4.0, 180.0, -12.9, 79.0),
        units="mm/yr",
        reference_frame="stable Eurasia",
    ),
    "wang_2020_china_itrf2008": BackgroundInfo(
        "wang_2020_china_itrf2008",
        "Wang et al. 2020 GNSS — ITRF2008",
        *_GNSS,
        "Continental China and reference sites",
        "Horizontal GNSS velocities in the ITRF2008 frame.",
        (
            "Wang and Shen et al. (2020), Journal of Geophysical Research, "
            "doi:10.1029/2019JB018774"
        ),
        "https://doi.org/10.1029/2019JB018774",
        bbox=(-4.0, 180.0, -12.9, 79.0),
        units="mm/yr",
        reference_frame="ITRF2008",
    ),
}


_BACKGROUND_GROUPS = (
    ("fault", "Fault", "vector", "Faults", {}),
    ("block", "Block", "vector", "Blocks", {}),
    (
        "gnss",
        "GNSS",
        "gnss_velocity",
        "GNSS",
        {"display_scale": 0.02},
    ),
)


def _background_id(prefix, stem, used):
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.-").lower()
    token = token or "resource"
    base = f"background.{prefix}.{token}"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}.{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def packaged_background_layers():
    """Return stable, hidden layer declarations for packaged backgrounds.

    Returns
    -------
    tuple of LayerSpec
        Fault, block and GNSS resources. Files are registered without being
        parsed; the existing viewer catalog loads one only after it is shown.
    """

    layers = []
    used = set()
    for prefix, label, kind, category, style in _BACKGROUND_GROUPS:
        for source in packaged_resource_files(category):
            info = _BACKGROUND_INFO.get(source.stem.casefold())
            layers.append(
                LayerSpec(
                    id=_background_id(prefix, source.stem, used),
                    name=info.label if info else f"{label}: {source.stem}",
                    kind=kind,
                    source=source,
                    visible=False,
                    style=style,
                )
            )
    return tuple(layers)


def create_background_project(*, region=None, basemap="open-street-map"):
    """Create the no-YAML quick-view project."""

    layers = tuple(
        replace(
            layer,
            visible=bool(
                (info := background_info_for_layer(layer))
                and info.quick_default
            ),
        )
        for layer in packaged_background_layers()
    )
    return ViewerProject(
        name="ECAT background data",
        path=None,
        layers=layers,
        region=region,
        basemap=basemap,
    )


def with_packaged_backgrounds(project):
    """Return a project containing its own layers plus packaged backgrounds."""

    if not isinstance(project, ViewerProject):
        raise TypeError("project must be a ViewerProject.")
    return ViewerProject(
        name=project.name,
        path=project.path,
        layers=tuple(project.layers) + packaged_background_layers(),
        region=project.region,
        basemap=project.basemap,
    )


def background_info_for_layer(layer):
    """Return curated pre-load metadata for a packaged background layer."""

    if not isinstance(layer, LayerSpec) or not layer.id.startswith("background."):
        return None
    return _BACKGROUND_INFO.get(layer.source.stem.casefold())


def background_matches_region(layer, region):
    """Return whether a packaged resource can intersect a study region."""

    info = background_info_for_layer(layer)
    if info is None or region is None:
        return False
    if info.group == "global_context":
        return True
    if info.bbox is None:
        return False
    west, east, south, north = map(float, region)
    data_west, data_east, data_south, data_north = info.bbox
    return not (
        east < data_west
        or west > data_east
        or north < data_south
        or south > data_north
    )


__all__ = [
    "BackgroundInfo",
    "background_info_for_layer",
    "background_matches_region",
    "create_background_project",
    "packaged_background_layers",
    "with_packaged_backgrounds",
]
