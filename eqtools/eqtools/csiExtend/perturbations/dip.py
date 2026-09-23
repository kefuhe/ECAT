"""Generate non-layered fault geometry from one frozen dip-profile contract.

The public setup boundary distinguishes sampled controls, fixed controls, and
transition zones explicitly. Candidate methods never receive a second copy of
those declarations: they consume ``GeometryReference.dip_profile`` and add
only the sampled perturbation values. Direct, preset, meshed, and diagnostic
paths therefore share the same projection/interpolation semantics.

Layered-dip fault classes own a separate profile model and are deliberately not
routed through this module.
"""

from __future__ import annotations

import numpy as np

from ..bayesian_perturbation_base import track_mesh_update
from ..dip_profile import (
    build_dip_profile_spec,
    transform_dip_profile_coordinates,
    resolve_dip_profile as _resolve_dip_profile,
)


class DipPerturbationMixin:
    """Non-layered along-strike dip perturbations."""

    def _project_reference_dip_profile(self):
        """Return the frozen lon/lat profile in fault-local x/y kilometres."""
        self._require_geometry_ref("dip_profile")
        return transform_dip_profile_coordinates(
            self.geometry_ref.dip_profile,
            self.ll2xy,
        )

    def _build_dip_profile(
        self,
        sampled_controls,
        fixed_controls,
        interpolation_axis,
        transition_zones,
        perturbation_groups,
        *,
        is_utm,
    ):
        """Build one temporary profile through the shared setup resolver."""
        self._require_geometry_ref("top_coords")
        return build_dip_profile_spec(
            sampled_controls,
            fixed_controls,
            interpolation_axis=interpolation_axis,
            transition_zones=transition_zones,
            perturbation_groups=perturbation_groups,
            reference_top_xy=self.geometry_ref.top_coords,
            xy_to_declaration_frame=None if is_utm else self.xy2ll,
        )

    def resolve_dip_profile(
        self,
        perturbations=None,
        *,
        angle_unit="degrees",
        top_coords=None,
    ):
        """Resolve the frozen profile without changing fault or mesh state.

        ``perturbations`` follows the frozen profile layout: without explicit
        groups it is scalar or one additive value per sampled control; with
        groups it contains exactly one value per unique label. Omission
        resolves the reference profile. ``top_coords`` is a fault-local top
        edge; omission uses the frozen reference top edge. The returned
        immutable object is the same representation consumed by
        :class:`DipGeneratorStage` and by diagnostics.
        """
        self._require_geometry_ref("top_coords", "dip_profile")
        profile = self._project_reference_dip_profile()
        if top_coords is None:
            top_coords = self.geometry_ref.top_coords
        if perturbations is None:
            perturbations = np.zeros(
                profile.perturbation_parameter_count,
                dtype=float,
            )
        return _resolve_dip_profile(
            profile,
            top_coords,
            perturbations,
            angle_unit=angle_unit,
        )

    def perturb_dips(
        self,
        sampled_controls,
        perturbations,
        *,
        fixed_controls=None,
        interpolation_axis="auto",
        transition_zones=None,
        perturbation_groups=None,
        angle_unit="degrees",
        discretization_interval=None,
        is_utm=False,
        use_average_strike=False,
        average_strike_source="pca",
        user_direction_angle=None,
    ):
        """Generate a bottom edge from an explicit temporary dip profile.

        Control rows are ``[coordinate_1, coordinate_2, reference_dip]``. The
        first two columns are longitude/latitude unless ``is_utm=True``. A row
        may instead contain ``dip`` plus ``s_km`` or ``s_from_end_km``; the
        same along-top position mappings are accepted by transition centres
        and endpoints. Every position is resolved against the frozen reference
        top and then projected onto the candidate top before interpolation.
        This direct method does not modify the frozen reference profile.
        ``discretization_interval``, when provided, is a finite positive
        spacing in km for this call only. It cannot be combined with an active
        reference-owned ``DensificationConfig``.
        """
        from .pipeline import DipGeneratorStage, NoMeshPolicy, run_pipeline

        profile = self._build_dip_profile(
            sampled_controls,
            fixed_controls,
            interpolation_axis,
            transition_zones,
            perturbation_groups,
            is_utm=is_utm,
        )
        if is_utm:
            local_profile = profile
        else:
            coordinates = np.column_stack([profile.controls.x, profile.controls.y])
            if (
                np.any(np.abs(coordinates[:, 0]) > 360.0)
                or np.any(np.abs(coordinates[:, 1]) > 90.0)
            ):
                raise ValueError(
                    "dip-profile coordinates exceed lon/lat bounds; set "
                    "is_utm=True for fault-local projected coordinates"
                )
            local_profile = transform_dip_profile_coordinates(
                profile,
                self.ll2xy,
            )

        stages = [DipGeneratorStage(
            dip_profile=local_profile,
            perturbations=perturbations,
            angle_unit=angle_unit,
            densify_top=True,
            discretization_interval=discretization_interval,
            use_average_strike=use_average_strike,
            average_strike_source=average_strike_source,
            user_direction_angle=user_direction_angle,
        )]
        run_pipeline(self, stages, mesh_policy=NoMeshPolicy())
        return self.bottom_coords

    @track_mesh_update(
        description="Resolve the frozen dip profile and regenerate the bottom edge.",
        params_info={
            "perturbations": (
                "Scalar, one change per sampled control, or one per frozen "
                "dip-profile perturbation group"
            ),
        },
        reference_requirements={"fields": ("top_coords", "dip_profile")},
        perturbation_cardinality={"kind": "scalar_or_sampled_dip_controls"},
        perturbation_items=({"role": "dip_change", "unit_from": "angle_unit"},),
    )
    def perturb_dips_with_preset_params(
        self,
        perturbations,
        *,
        angle_unit="degrees",
        discretization_interval=None,
        use_average_strike=False,
        average_strike_source="pca",
        user_direction_angle=None,
        _resolved_perturbation_layout=None,
    ):
        """Regenerate bottom geometry from ``geometry_ref.dip_profile``.

        Controls, roles, interpolation axis, transition zones, and any shared
        perturbation groups come only from the frozen profile. Candidate values
        are expanded through that frozen layout before mapping to sampled
        controls. For Bayesian replay, configure density with
        ``set_densification(...)`` and leave
        ``discretization_interval`` unset; the method-level interval is only a
        one-call convenience and cannot coexist with reference-owned density.
        """
        from .pipeline import DipGeneratorStage, NoMeshPolicy, run_pipeline

        stages = [DipGeneratorStage(
            dip_profile=self._project_reference_dip_profile(),
            perturbations=perturbations,
            angle_unit=angle_unit,
            perturbation_layout=_resolved_perturbation_layout,
            densify_top=True,
            discretization_interval=discretization_interval,
            use_average_strike=use_average_strike,
            average_strike_source=average_strike_source,
            user_direction_angle=user_direction_angle,
        )]
        run_pipeline(self, stages, mesh_policy=NoMeshPolicy())
        return self.bottom_coords

    @track_mesh_update(
        update_mesh=True,
        description="Resolve the frozen dip profile and rebuild a simple mesh.",
        params_info={
            "perturbations": (
                "Scalar, one change per sampled control, or one per frozen "
                "dip-profile perturbation group"
            ),
            "kwargs": "Simple-mesh generation parameters",
        },
        reference_requirements={"fields": ("top_coords", "dip_profile")},
        perturbation_cardinality={"kind": "scalar_or_sampled_dip_controls"},
        perturbation_items=({"role": "dip_change", "unit_from": "angle_unit"},),
    )
    def perturb_DipsPresetParams_SimpleMesh(
        self,
        perturbations,
        *,
        angle_unit="degrees",
        discretization_interval=None,
        disct_z=None,
        bias=None,
        min_dz=None,
        use_average_strike=False,
        average_strike_source="pca",
        user_direction_angle=None,
        _resolved_perturbation_layout=None,
    ):
        """Regenerate bottom geometry and rebuild the simple triangular mesh."""
        from .pipeline import DipGeneratorStage, SimpleMeshPolicy, run_pipeline

        stages = [DipGeneratorStage(
            dip_profile=self._project_reference_dip_profile(),
            perturbations=perturbations,
            angle_unit=angle_unit,
            perturbation_layout=_resolved_perturbation_layout,
            densify_top=True,
            discretization_interval=discretization_interval,
            use_average_strike=use_average_strike,
            average_strike_source=average_strike_source,
            user_direction_angle=user_direction_angle,
        )]
        run_pipeline(
            self,
            stages,
            mesh_policy=SimpleMeshPolicy(
                disct_z=disct_z,
                bias=bias,
                min_dz=min_dz,
            ),
        )


__all__ = ["DipPerturbationMixin"]
