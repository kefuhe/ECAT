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


class InputAzimuthRole(Enum):
    """
    Physical meaning of the input azimuth after angle normalization to ENU.
    """
    HEADING = "heading"
    RIGHT_LOOK_AWAY = "right_look_away"
    LEFT_LOOK_AWAY = "left_look_away"
    RIGHT_LOS_TOWARD = "right_los_toward"
    LEFT_LOS_TOWARD = "left_los_toward"


class LookSide(Enum):
    RIGHT = "right"
    LEFT = "left"


class ObservationType(Enum):
    """
    Physical type of the scalar raster read from a SAR product.

    `phase_los` is an unwrapped phase raster in radians; it is converted to
    LOS displacement with wavelength / (-4*pi). `los_displacement` is already
    a LOS/range-direction displacement or offset. Range-offset products use
    this same observation type, and are converted to the same target positive
    direction as LOS displacement: toward the satellite. `azimuth_offset` is
    an along-track scalar.
    """
    PHASE_LOS = "phase_los"
    LOS_DISPLACEMENT = "los_displacement"
    AZIMUTH_OFFSET = "azimuth_offset"


class InputValueConvention(Enum):
    """
    Sign convention of the raw scalar observation read from the product.
    """
    UNWRAPPED_PHASE = "unwrapped_phase"
    TOWARD_SATELLITE = "toward_satellite"
    AWAY_FROM_SATELLITE = "away_from_satellite"
    ALONG_HEADING = "along_heading"
    OPPOSITE_HEADING = "opposite_heading"


class InputProjectionRole(Enum):
    """
    Physical axis represented by a supplied ENU projection vector.
    """
    SAME_AS_OBSERVATION = "same_as_observation"
    LOS = "los"
    AZIMUTH = "azimuth"


class InputProjectionConvention(Enum):
    """
    Positive direction of a supplied ENU projection vector.
    """
    SAME_AS_VALUE = "same_as_value"
    CANONICAL = "canonical"
    TOWARD_SATELLITE = "toward_satellite"
    AWAY_FROM_SATELLITE = "away_from_satellite"
    ALONG_HEADING = "along_heading"
    OPPOSITE_HEADING = "opposite_heading"


class SarProductPreset(Enum):
    """
    Common product/observation convention bundles.

    Reader `mode` values are short aliases for these presets. Use `mode` for
    interactive scripts, full presets for reproducible configs, and explicit
    config objects for products outside the built-in conventions. Presets fix
    both raw angle conventions and raw value semantics before
    `extract_raw_grd()` and `read_observation()` run.
    """
    GENERIC_PHASE_LOS = "generic_phase_los"
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
    raise ValueError(f"Unsupported {field_name}: {value!r}.")


def default_input_value_convention(observation_type):
    observation_type = coerce_enum(ObservationType, observation_type, "observation_type")
    if observation_type == ObservationType.PHASE_LOS:
        return InputValueConvention.UNWRAPPED_PHASE
    if observation_type == ObservationType.LOS_DISPLACEMENT:
        return InputValueConvention.TOWARD_SATELLITE
    if observation_type == ObservationType.AZIMUTH_OFFSET:
        return InputValueConvention.ALONG_HEADING
    raise ValueError(f"Unsupported observation_type: {observation_type}.")


@dataclass
class SarObservationSpec:
    observation_type: ObservationType = ObservationType.PHASE_LOS
    input_azimuth_role: InputAzimuthRole = InputAzimuthRole.RIGHT_LOOK_AWAY
    look_side: LookSide = LookSide.RIGHT
    input_value_convention: InputValueConvention = None
    wavelength: float = None

    def __post_init__(self):
        self.observation_type = coerce_enum(
            ObservationType, self.observation_type, "observation_type"
        )
        self.input_azimuth_role = coerce_enum(
            InputAzimuthRole, self.input_azimuth_role, "input_azimuth_role"
        )
        self.look_side = coerce_enum(LookSide, self.look_side, "look_side")
        if self.input_value_convention is None:
            self.input_value_convention = default_input_value_convention(self.observation_type)
        else:
            self.input_value_convention = coerce_enum(
                InputValueConvention,
                self.input_value_convention,
                "input_value_convention",
            )
        validate_observation_spec(self)


def validate_observation_spec(spec):
    if spec.observation_type == ObservationType.PHASE_LOS:
        valid = {InputValueConvention.UNWRAPPED_PHASE}
    elif spec.observation_type == ObservationType.LOS_DISPLACEMENT:
        valid = {
            InputValueConvention.TOWARD_SATELLITE,
            InputValueConvention.AWAY_FROM_SATELLITE,
        }
    elif spec.observation_type == ObservationType.AZIMUTH_OFFSET:
        valid = {
            InputValueConvention.ALONG_HEADING,
            InputValueConvention.OPPOSITE_HEADING,
        }
    else:
        raise ValueError(f"Unsupported observation_type: {spec.observation_type}.")

    if spec.input_value_convention not in valid:
        allowed = ", ".join(item.value for item in sorted(valid, key=lambda item: item.value))
        raise ValueError(
            f"{spec.observation_type.value} does not accept "
            f"input_value_convention={spec.input_value_convention.value!r}; "
            f"expected one of: {allowed}."
        )


@dataclass
class SarReaderConfig:
    """
    Common reader-level SAR product conventions.

    Subclasses add either angle-raster projection fields or direct ENU
    projection fields. Observation fields are used to convert raw scalar values
    into the CSI scalar-observation contract.
    """
    preset: SarProductPreset = None
    zero2nan: bool = True
    wavelength: float = 0.0554658
    observation_type: ObservationType = ObservationType.PHASE_LOS
    input_value_convention: InputValueConvention = None

    def observation_spec(self, wavelength=None):
        if wavelength is None:
            wavelength = self.wavelength
        role = getattr(self, "input_azimuth_role", InputAzimuthRole.RIGHT_LOOK_AWAY)
        look_side = getattr(self, "look_side", LookSide.RIGHT)
        return SarObservationSpec(
            observation_type=self.observation_type,
            input_azimuth_role=role,
            look_side=look_side,
            input_value_convention=self.input_value_convention,
            wavelength=wavelength,
        )


@dataclass
class AngleProjectionSarConfig(SarReaderConfig):
    """
    Convention fields for products that provide azimuth/incidence rasters.
    """
    azimuth_reference: AzimuthReference = AzimuthReference.NORTH
    azimuth_unit: AngleUnit = AngleUnit.DEGREE
    azimuth_direction: AngleDirection = AngleDirection.CLOCKWISE
    incidence_reference: IncidenceReference = IncidenceReference.ZENITH
    incidence_unit: AngleUnit = AngleUnit.DEGREE
    input_azimuth_role: InputAzimuthRole = InputAzimuthRole.RIGHT_LOOK_AWAY
    look_side: LookSide = LookSide.RIGHT
    is_lonlat: bool = True


@dataclass
class DirectProjectionSarConfig(SarReaderConfig):
    """
    Convention fields for products that provide ENU projection vectors.
    """
    input_projection_role: InputProjectionRole = InputProjectionRole.SAME_AS_OBSERVATION
    input_projection_convention: InputProjectionConvention = InputProjectionConvention.CANONICAL
    look_side: LookSide = LookSide.RIGHT


@dataclass
class GammasarConfig(AngleProjectionSarConfig):
    pass


@dataclass
class GammaTiffConfig(AngleProjectionSarConfig):
    input_azimuth_role: InputAzimuthRole = InputAzimuthRole.HEADING
    observation_type: ObservationType = ObservationType.LOS_DISPLACEMENT
    input_value_convention: InputValueConvention = InputValueConvention.TOWARD_SATELLITE


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
    input_azimuth_role: InputAzimuthRole = InputAzimuthRole.RIGHT_LOS_TOWARD
    look_side: LookSide = LookSide.RIGHT
    observation_type: ObservationType = ObservationType.LOS_DISPLACEMENT
    input_value_convention: InputValueConvention = InputValueConvention.TOWARD_SATELLITE
    is_lonlat: bool = False


def config_from_preset(preset):
    """
    Return a reader config for a common SAR product/observation convention.

    Presets are intentionally explicit because they affect both angle
    normalization during raw-grid extraction and scalar-value conversion during
    `read_observation()`.
    """
    preset = coerce_enum(SarProductPreset, preset, "preset")

    if preset == SarProductPreset.GENERIC_PHASE_LOS:
        return AngleProjectionSarConfig(
            observation_type=ObservationType.PHASE_LOS,
            input_value_convention=InputValueConvention.UNWRAPPED_PHASE,
            preset=preset,
        )
    if preset == SarProductPreset.GENERIC_LOS_DISPLACEMENT:
        return AngleProjectionSarConfig(
            observation_type=ObservationType.LOS_DISPLACEMENT,
            input_value_convention=InputValueConvention.TOWARD_SATELLITE,
            preset=preset,
        )
    if preset == SarProductPreset.GENERIC_RANGE_OFFSET:
        return AngleProjectionSarConfig(
            observation_type=ObservationType.LOS_DISPLACEMENT,
            input_value_convention=InputValueConvention.AWAY_FROM_SATELLITE,
            preset=preset,
        )
    if preset == SarProductPreset.GENERIC_AZIMUTH_OFFSET:
        return AngleProjectionSarConfig(
            observation_type=ObservationType.AZIMUTH_OFFSET,
            input_value_convention=InputValueConvention.ALONG_HEADING,
            preset=preset,
        )
    if preset == SarProductPreset.GAMMA_UNWRAPPED_PHASE:
        return GammasarConfig(
            observation_type=ObservationType.PHASE_LOS,
            input_value_convention=InputValueConvention.UNWRAPPED_PHASE,
            preset=preset,
        )
    if preset == SarProductPreset.GAMMA_LOS_DISPLACEMENT:
        return GammasarConfig(
            observation_type=ObservationType.LOS_DISPLACEMENT,
            input_value_convention=InputValueConvention.TOWARD_SATELLITE,
            preset=preset,
        )
    if preset == SarProductPreset.GAMMA_RANGE_OFFSET:
        return GammasarConfig(
            observation_type=ObservationType.LOS_DISPLACEMENT,
            input_value_convention=InputValueConvention.AWAY_FROM_SATELLITE,
            preset=preset,
        )
    if preset == SarProductPreset.GAMMA_AZIMUTH_OFFSET:
        return GammasarConfig(
            observation_type=ObservationType.AZIMUTH_OFFSET,
            input_value_convention=InputValueConvention.ALONG_HEADING,
            preset=preset,
        )
    if preset == SarProductPreset.GAMMA_TIFF_UNWRAPPED_PHASE:
        return GammaTiffConfig(
            observation_type=ObservationType.PHASE_LOS,
            input_value_convention=InputValueConvention.UNWRAPPED_PHASE,
            preset=preset,
        )
    if preset == SarProductPreset.GAMMA_TIFF_LOS_DISPLACEMENT:
        return GammaTiffConfig(
            observation_type=ObservationType.LOS_DISPLACEMENT,
            input_value_convention=InputValueConvention.TOWARD_SATELLITE,
            preset=preset,
        )
    if preset == SarProductPreset.GAMMA_TIFF_RANGE_OFFSET:
        return GammaTiffConfig(
            observation_type=ObservationType.LOS_DISPLACEMENT,
            input_value_convention=InputValueConvention.AWAY_FROM_SATELLITE,
            preset=preset,
        )
    if preset == SarProductPreset.GAMMA_TIFF_AZIMUTH_OFFSET:
        return GammaTiffConfig(
            observation_type=ObservationType.AZIMUTH_OFFSET,
            input_value_convention=InputValueConvention.ALONG_HEADING,
            preset=preset,
        )
    if preset == SarProductPreset.GMTSAR_UNWRAPPED_PHASE:
        return GmtsarConfig(
            observation_type=ObservationType.PHASE_LOS,
            input_value_convention=InputValueConvention.UNWRAPPED_PHASE,
            input_projection_role=InputProjectionRole.SAME_AS_OBSERVATION,
            input_projection_convention=InputProjectionConvention.TOWARD_SATELLITE,
            preset=preset,
        )
    if preset == SarProductPreset.GMTSAR_LOS_DISPLACEMENT:
        return GmtsarConfig(
            observation_type=ObservationType.LOS_DISPLACEMENT,
            input_value_convention=InputValueConvention.TOWARD_SATELLITE,
            input_projection_role=InputProjectionRole.SAME_AS_OBSERVATION,
            input_projection_convention=InputProjectionConvention.SAME_AS_VALUE,
            preset=preset,
        )
    if preset == SarProductPreset.GMTSAR_RANGE_OFFSET:
        return GmtsarConfig(
            observation_type=ObservationType.LOS_DISPLACEMENT,
            input_value_convention=InputValueConvention.TOWARD_SATELLITE,
            input_projection_role=InputProjectionRole.SAME_AS_OBSERVATION,
            input_projection_convention=InputProjectionConvention.SAME_AS_VALUE,
            preset=preset,
        )
    if preset == SarProductPreset.GMTSAR_AZIMUTH_OFFSET:
        return GmtsarConfig(
            observation_type=ObservationType.AZIMUTH_OFFSET,
            input_value_convention=InputValueConvention.ALONG_HEADING,
            input_projection_role=InputProjectionRole.SAME_AS_OBSERVATION,
            input_projection_convention=InputProjectionConvention.SAME_AS_VALUE,
            preset=preset,
        )
    if preset == SarProductPreset.HYP3_LOS_DISPLACEMENT:
        return Hyp3TiffConfig(
            observation_type=ObservationType.LOS_DISPLACEMENT,
            input_value_convention=InputValueConvention.TOWARD_SATELLITE,
            preset=preset,
        )
    if preset == SarProductPreset.HYP3_UNWRAPPED_PHASE:
        return Hyp3TiffConfig(
            observation_type=ObservationType.PHASE_LOS,
            input_value_convention=InputValueConvention.UNWRAPPED_PHASE,
            preset=preset,
        )

    raise ValueError(f"Unsupported preset: {preset}.")
