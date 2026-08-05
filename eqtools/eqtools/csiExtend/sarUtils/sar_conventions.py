"""Explicit SAR scalar, angle-geometry, and projection conventions.

The public contract used by all readers is:

``scalar_observation = ENU_displacement dot projection``

Readers therefore normalize two independent inputs before passing data to CSI:
the scalar sign and the projection-vector direction.  Acquisition side is
geometry metadata; it never defines the sign of the scalar raster.
"""

from dataclasses import dataclass
from enum import Enum


class AngleUnit(Enum):
    DEGREE = "degree"
    RADIAN = "radian"


class AngleDirection(Enum):
    CLOCKWISE = "clockwise"
    COUNTERCLOCKWISE = "counterclockwise"


class AzimuthReference(Enum):
    NORTH = "north"
    EAST = "east"


class IncidenceReference(Enum):
    ZENITH = "zenith"
    ELEVATION = "elevation"


class ByteOrder(Enum):
    """Byte order for raw float32 raster files."""

    NATIVE = "native"
    LITTLE = "little"
    BIG = "big"


class AzimuthAngleRole(Enum):
    """Physical direction encoded by the normalized azimuth angle."""

    HEADING = "heading"
    SENSOR_TO_GROUND = "sensor_to_ground"
    GROUND_TO_SENSOR = "ground_to_sensor"


class AcquisitionLookSide(Enum):
    """Side of the platform heading on which the imaged ground swath lies."""

    RIGHT = "right"
    LEFT = "left"


class ObservationType(Enum):
    """Physical type of the raw scalar raster."""

    UNWRAPPED_PHASE = "unwrapped_phase"
    LOS_DISPLACEMENT = "los_displacement"
    AZIMUTH_OFFSET = "azimuth_offset"


class RawValueConvention(Enum):
    """Positive direction or encoding of the raw scalar raster."""

    UNWRAPPED_PHASE = "unwrapped_phase"
    TOWARD_SENSOR = "toward_sensor"
    AWAY_FROM_SENSOR = "away_from_sensor"
    ALONG_HEADING = "along_heading"
    OPPOSITE_HEADING = "opposite_heading"


class ProjectionAxis(Enum):
    """Physical axis represented by a supplied ENU projection vector."""

    LOS = "los"
    AZIMUTH = "azimuth"


class ProjectionDirection(Enum):
    """Positive direction of a supplied ENU projection vector."""

    GROUND_TO_SENSOR = "ground_to_sensor"
    SENSOR_TO_GROUND = "sensor_to_ground"
    ALONG_HEADING = "along_heading"
    OPPOSITE_HEADING = "opposite_heading"


class SarProductPreset(Enum):
    """Internal bundles behind the short reader ``mode`` interface."""

    GENERIC_UNWRAPPED_PHASE = "generic_unwrapped_phase"
    GENERIC_LOS_DISPLACEMENT = "generic_los_displacement"
    GENERIC_RANGE_OFFSET = "generic_range_offset"
    GENERIC_AZIMUTH_OFFSET = "generic_azimuth_offset"
    GAMMA_UNWRAPPED_PHASE = "gamma_unwrapped_phase"
    GAMMA_LOS_DISPLACEMENT = "gamma_los_displacement"
    GAMMA_RANGE_OFFSET = "gamma_range_offset"
    GAMMA_AZIMUTH_OFFSET = "gamma_azimuth_offset"
    GAMMA_TIFF_UNWRAPPED_PHASE = "gamma_tiff_unwrapped_phase"
    GAMMA_TIFF_LOS_DISPLACEMENT = "gamma_tiff_los_displacement"
    GAMMA_TIFF_RANGE_OFFSET = "gamma_tiff_range_offset"
    GAMMA_TIFF_AZIMUTH_OFFSET = "gamma_tiff_azimuth_offset"
    GMTSAR_UNWRAPPED_PHASE = "gmtsar_unwrapped_phase"
    GMTSAR_LOS_DISPLACEMENT = "gmtsar_los_displacement"
    GMTSAR_RANGE_OFFSET = "gmtsar_range_offset"
    GMTSAR_AZIMUTH_OFFSET = "gmtsar_azimuth_offset"
    HYP3_UNWRAPPED_PHASE = "hyp3_unwrapped_phase"
    HYP3_LOS_DISPLACEMENT = "hyp3_los_displacement"


def coerce_enum(enum_cls, value, field_name):
    if isinstance(value, enum_cls):
        return value
    if value is None:
        raise ValueError(f"{field_name} cannot be None.")
    key = str(value).replace("-", "_").lower()
    for item in enum_cls:
        if item.value == key or item.name.lower() == key:
            return item
    allowed = ", ".join(item.value for item in enum_cls)
    raise ValueError(
        f"Unsupported {field_name}: {value!r}. Expected one of: {allowed}."
    )


def default_raw_value_convention(observation_type):
    observation_type = coerce_enum(
        ObservationType, observation_type, "observation_type"
    )
    if observation_type == ObservationType.UNWRAPPED_PHASE:
        return RawValueConvention.UNWRAPPED_PHASE
    if observation_type == ObservationType.LOS_DISPLACEMENT:
        return RawValueConvention.TOWARD_SENSOR
    if observation_type == ObservationType.AZIMUTH_OFFSET:
        return RawValueConvention.ALONG_HEADING
    raise ValueError(f"Unsupported observation_type: {observation_type}.")


@dataclass
class SarObservationSpec:
    """Scalar-observation semantics, independent of acquisition geometry."""

    observation_type: ObservationType = ObservationType.UNWRAPPED_PHASE
    raw_value_convention: RawValueConvention = None
    wavelength: float = None

    def __post_init__(self):
        self.observation_type = coerce_enum(
            ObservationType, self.observation_type, "observation_type"
        )
        if self.raw_value_convention is None:
            self.raw_value_convention = default_raw_value_convention(
                self.observation_type
            )
        else:
            self.raw_value_convention = coerce_enum(
                RawValueConvention,
                self.raw_value_convention,
                "raw_value_convention",
            )
        validate_observation_spec(self)


@dataclass
class AngleGeometrySpec:
    """Meaning of an azimuth-angle raster and acquisition look side."""

    azimuth_angle_role: AzimuthAngleRole = AzimuthAngleRole.SENSOR_TO_GROUND
    acquisition_look_side: AcquisitionLookSide = AcquisitionLookSide.RIGHT

    def __post_init__(self):
        self.azimuth_angle_role = coerce_enum(
            AzimuthAngleRole, self.azimuth_angle_role, "azimuth_angle_role"
        )
        self.acquisition_look_side = coerce_enum(
            AcquisitionLookSide,
            self.acquisition_look_side,
            "acquisition_look_side",
        )


def validate_observation_spec(spec):
    if spec.observation_type == ObservationType.UNWRAPPED_PHASE:
        valid = {RawValueConvention.UNWRAPPED_PHASE}
    elif spec.observation_type == ObservationType.LOS_DISPLACEMENT:
        valid = {
            RawValueConvention.TOWARD_SENSOR,
            RawValueConvention.AWAY_FROM_SENSOR,
        }
    elif spec.observation_type == ObservationType.AZIMUTH_OFFSET:
        valid = {
            RawValueConvention.ALONG_HEADING,
            RawValueConvention.OPPOSITE_HEADING,
        }
    else:
        raise ValueError(f"Unsupported observation_type: {spec.observation_type}.")

    if spec.raw_value_convention not in valid:
        allowed = ", ".join(item.value for item in sorted(valid, key=lambda item: item.value))
        raise ValueError(
            f"{spec.observation_type.value} does not accept "
            f"raw_value_convention={spec.raw_value_convention.value!r}; "
            f"expected one of: {allowed}."
        )


@dataclass
class SarReaderConfig:
    """Reader-level scalar conventions shared by all SAR product readers."""

    preset: SarProductPreset = None
    zero2nan: bool = True
    wavelength: float = 0.0554658
    observation_type: ObservationType = ObservationType.UNWRAPPED_PHASE
    raw_value_convention: RawValueConvention = None

    def observation_spec(self, wavelength=None):
        return SarObservationSpec(
            observation_type=self.observation_type,
            raw_value_convention=self.raw_value_convention,
            wavelength=self.wavelength if wavelength is None else wavelength,
        )


@dataclass
class AngleProjectionSarConfig(SarReaderConfig):
    """Conventions for products that provide azimuth/incidence rasters."""

    azimuth_reference: AzimuthReference = AzimuthReference.NORTH
    azimuth_unit: AngleUnit = AngleUnit.DEGREE
    azimuth_direction: AngleDirection = AngleDirection.CLOCKWISE
    incidence_reference: IncidenceReference = IncidenceReference.ZENITH
    incidence_unit: AngleUnit = AngleUnit.DEGREE
    azimuth_angle_role: AzimuthAngleRole = AzimuthAngleRole.SENSOR_TO_GROUND
    acquisition_look_side: AcquisitionLookSide = AcquisitionLookSide.RIGHT
    is_lonlat: bool = True

    def geometry_spec(self):
        return AngleGeometrySpec(
            azimuth_angle_role=self.azimuth_angle_role,
            acquisition_look_side=self.acquisition_look_side,
        )


@dataclass
class DirectProjectionSarConfig(SarReaderConfig):
    """Conventions for products that provide ENU projection vectors."""

    input_projection_axis: ProjectionAxis = ProjectionAxis.LOS
    input_projection_direction: ProjectionDirection = ProjectionDirection.GROUND_TO_SENSOR
    acquisition_look_side: AcquisitionLookSide = AcquisitionLookSide.RIGHT


@dataclass
class GammasarConfig(AngleProjectionSarConfig):
    byte_order: ByteOrder = ByteOrder.NATIVE

    def __post_init__(self):
        self.byte_order = coerce_enum(ByteOrder, self.byte_order, "byte_order")


@dataclass
class GammaTiffConfig(AngleProjectionSarConfig):
    azimuth_angle_role: AzimuthAngleRole = AzimuthAngleRole.HEADING
    observation_type: ObservationType = ObservationType.LOS_DISPLACEMENT
    raw_value_convention: RawValueConvention = RawValueConvention.TOWARD_SENSOR


@dataclass
class GmtsarConfig(DirectProjectionSarConfig):
    pass


@dataclass
class Hyp3TiffConfig(AngleProjectionSarConfig):
    azimuth_reference: AzimuthReference = AzimuthReference.EAST
    azimuth_unit: AngleUnit = AngleUnit.RADIAN
    azimuth_direction: AngleDirection = AngleDirection.COUNTERCLOCKWISE
    incidence_reference: IncidenceReference = IncidenceReference.ELEVATION
    incidence_unit: AngleUnit = AngleUnit.RADIAN
    azimuth_angle_role: AzimuthAngleRole = AzimuthAngleRole.GROUND_TO_SENSOR
    acquisition_look_side: AcquisitionLookSide = AcquisitionLookSide.RIGHT
    observation_type: ObservationType = ObservationType.LOS_DISPLACEMENT
    raw_value_convention: RawValueConvention = RawValueConvention.TOWARD_SENSOR
    is_lonlat: bool = False


def _angle_config(config_cls, observation_type, raw_value_convention, preset):
    return config_cls(
        observation_type=observation_type,
        raw_value_convention=raw_value_convention,
        preset=preset,
    )


def _direct_config(preset, observation_type, raw_value_convention, axis, direction):
    return GmtsarConfig(
        observation_type=observation_type,
        raw_value_convention=raw_value_convention,
        input_projection_axis=axis,
        input_projection_direction=direction,
        preset=preset,
    )


def config_from_preset(preset):
    """Return the complete reader config behind an internal product preset."""

    preset = coerce_enum(SarProductPreset, preset, "preset")
    angle_presets = {
        SarProductPreset.GENERIC_UNWRAPPED_PHASE: (
            AngleProjectionSarConfig,
            ObservationType.UNWRAPPED_PHASE,
            RawValueConvention.UNWRAPPED_PHASE,
        ),
        SarProductPreset.GENERIC_LOS_DISPLACEMENT: (
            AngleProjectionSarConfig,
            ObservationType.LOS_DISPLACEMENT,
            RawValueConvention.TOWARD_SENSOR,
        ),
        SarProductPreset.GENERIC_RANGE_OFFSET: (
            AngleProjectionSarConfig,
            ObservationType.LOS_DISPLACEMENT,
            RawValueConvention.AWAY_FROM_SENSOR,
        ),
        SarProductPreset.GENERIC_AZIMUTH_OFFSET: (
            AngleProjectionSarConfig,
            ObservationType.AZIMUTH_OFFSET,
            RawValueConvention.ALONG_HEADING,
        ),
        SarProductPreset.GAMMA_UNWRAPPED_PHASE: (
            GammasarConfig,
            ObservationType.UNWRAPPED_PHASE,
            RawValueConvention.UNWRAPPED_PHASE,
        ),
        SarProductPreset.GAMMA_LOS_DISPLACEMENT: (
            GammasarConfig,
            ObservationType.LOS_DISPLACEMENT,
            RawValueConvention.TOWARD_SENSOR,
        ),
        SarProductPreset.GAMMA_RANGE_OFFSET: (
            GammasarConfig,
            ObservationType.LOS_DISPLACEMENT,
            RawValueConvention.AWAY_FROM_SENSOR,
        ),
        SarProductPreset.GAMMA_AZIMUTH_OFFSET: (
            GammasarConfig,
            ObservationType.AZIMUTH_OFFSET,
            RawValueConvention.ALONG_HEADING,
        ),
        SarProductPreset.GAMMA_TIFF_UNWRAPPED_PHASE: (
            GammaTiffConfig,
            ObservationType.UNWRAPPED_PHASE,
            RawValueConvention.UNWRAPPED_PHASE,
        ),
        SarProductPreset.GAMMA_TIFF_LOS_DISPLACEMENT: (
            GammaTiffConfig,
            ObservationType.LOS_DISPLACEMENT,
            RawValueConvention.TOWARD_SENSOR,
        ),
        SarProductPreset.GAMMA_TIFF_RANGE_OFFSET: (
            GammaTiffConfig,
            ObservationType.LOS_DISPLACEMENT,
            RawValueConvention.AWAY_FROM_SENSOR,
        ),
        SarProductPreset.GAMMA_TIFF_AZIMUTH_OFFSET: (
            GammaTiffConfig,
            ObservationType.AZIMUTH_OFFSET,
            RawValueConvention.ALONG_HEADING,
        ),
        SarProductPreset.HYP3_UNWRAPPED_PHASE: (
            Hyp3TiffConfig,
            ObservationType.UNWRAPPED_PHASE,
            RawValueConvention.UNWRAPPED_PHASE,
        ),
        SarProductPreset.HYP3_LOS_DISPLACEMENT: (
            Hyp3TiffConfig,
            ObservationType.LOS_DISPLACEMENT,
            RawValueConvention.TOWARD_SENSOR,
        ),
    }
    if preset in angle_presets:
        return _angle_config(*angle_presets[preset], preset)

    direct_presets = {
        SarProductPreset.GMTSAR_UNWRAPPED_PHASE: (
            ObservationType.UNWRAPPED_PHASE,
            RawValueConvention.UNWRAPPED_PHASE,
            ProjectionAxis.LOS,
            ProjectionDirection.GROUND_TO_SENSOR,
        ),
        SarProductPreset.GMTSAR_LOS_DISPLACEMENT: (
            ObservationType.LOS_DISPLACEMENT,
            RawValueConvention.TOWARD_SENSOR,
            ProjectionAxis.LOS,
            ProjectionDirection.GROUND_TO_SENSOR,
        ),
        SarProductPreset.GMTSAR_RANGE_OFFSET: (
            ObservationType.LOS_DISPLACEMENT,
            RawValueConvention.TOWARD_SENSOR,
            ProjectionAxis.LOS,
            ProjectionDirection.GROUND_TO_SENSOR,
        ),
        SarProductPreset.GMTSAR_AZIMUTH_OFFSET: (
            ObservationType.AZIMUTH_OFFSET,
            RawValueConvention.ALONG_HEADING,
            ProjectionAxis.AZIMUTH,
            ProjectionDirection.ALONG_HEADING,
        ),
    }
    if preset in direct_presets:
        return _direct_config(preset, *direct_presets[preset])

    raise ValueError(f"Unsupported preset: {preset}.")
