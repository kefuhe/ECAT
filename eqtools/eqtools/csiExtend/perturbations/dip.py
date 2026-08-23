"""Generate bottom edges from perturbed dip-control values.

Direct methods accept controls at the call boundary; preset methods consume
canonical lon/lat controls stored in ``GeometryReference``.  Both delegate the
scientific calculation to ``DipGeneratorStage``.  The no-suffix method changes
coordinates only, while ``*_SimpleMesh`` also rebuilds the mesh.
"""
import numpy as np
import warnings
from ..bayesian_perturbation_base import track_mesh_update, DipControlPoints


class DipPerturbationMixin:
    """Mixin providing dip-angle perturbation capabilities."""

    def _project_reference_dip_control_points(self):
        """Return canonical reference dip controls in fault-local x/y (km).

        ``geometry_ref.dip_control_points`` is stored in longitude/latitude by
        the public reference setters. Pipeline stages operate in the fault's
        local projected coordinates, so preset consumers perform this one
        conversion internally instead of exposing another coordinate-frame
        switch to callers.
        """
        self._require_geometry_ref('dip_control_points')
        reference = self.geometry_ref.dip_control_points
        ctrl_x, ctrl_y = self.ll2xy(reference.x, reference.y)
        return DipControlPoints(
            x=np.asarray(ctrl_x, dtype=np.float64),
            y=np.asarray(ctrl_y, dtype=np.float64),
            dip=reference.dip.copy(),
        )

    #--------------------------------------Perturbing Dip--------------------------------------#
    def perturb_dips(self, x_coords, y_coords, dips,
                     perturbations, fixed_nodes=None, angle_unit='degrees',
                     discretization_interval=None, interpolation_axis='x',
                     is_utm=False, buffer_nodes=None, buffer_radius=None,
                     use_average_strike=False, average_strike_source='pca', user_direction_angle=None):
        """Generate a bottom edge from perturbed dip control points.

        The input reference dips are first represented continuously in
        ``(0, 180)`` and the proposal increments are added there.  The resulting
        dips are then interpolated onto the current top edge before rebuilding
        ``bottom_coords`` from depth and strike.

        Parameters
        ----------
        x_coords, y_coords : array-like
            Control-point coordinates. They are interpreted as longitude and
            latitude when ``is_utm=False`` and as fault-local projected x/y in
            km when ``is_utm=True``.
        dips : array-like
            Reference dips. Signed and 0--180 representations are accepted and
            converted to continuous 0--180 coordinates before perturbation.
        perturbations : array-like
            Additive dip changes. A scalar broadcasts to all movable control
            points; otherwise the number of values must match the movable
            controls after applying ``fixed_nodes``.
        fixed_nodes : sequence of int, optional
            Indices of control points whose reference dips remain fixed.
        angle_unit : {'degrees', 'radians'}, default 'degrees'
            Unit of ``perturbations``. Reference ``dips`` are in degrees.
        discretization_interval : float, optional
            Arc-length interval in km for temporarily rediscretizing the top
            edge before dip interpolation.
        interpolation_axis : {'auto', 'x', 'y', 'arc_length'}, default 'x'
            One-dimensional coordinate used for dip interpolation. ``'x'`` and
            ``'y'`` use fault-local projected easting and northing. ``'auto'``
            uses PCA to select one of those two axes; it does not select
            arc-length interpolation. ``'arc_length'`` projects each control
            point onto the current top-edge polyline and interpolates by
            cumulative distance along that polyline. It is intended for curved
            traces that are not monotonic in x or y.
        is_utm : bool, default False
            Coordinate frame of ``x_coords`` and ``y_coords``. False means
            longitude/latitude; True means fault-local projected x/y in km.
        buffer_nodes : array-like, optional
            Buffer-node locations in longitude/latitude. Buffer augmentation is
            supported only when the resolved interpolation axis is ``'x'`` or
            ``'y'``; it cannot be combined with ``'arc_length'``.
        buffer_radius : float or array-like, optional
            Buffer radius or per-node radii in km. Used together with
            ``buffer_nodes``.
        use_average_strike : bool, default False
            Use one representative strike when projecting the bottom edge.
        average_strike_source : {'pca', 'user'}, default 'pca'
            Source of the representative strike.
        user_direction_angle : float, optional
            Representative geographic strike in degrees clockwise from North
            when ``average_strike_source='user'``. It must follow the positive
            top-edge direction.

        Returns
        -------
        np.ndarray
            Updated ``self.bottom_coords``.

        Notes
        -----
        ``'arc_length'`` uses nearest-segment projection. Control points should
        therefore lie near the intended top edge and map to distinct positions
        along it. Outside the outermost controls, the nearest endpoint dip is
        retained.
        """
        from .pipeline import (
            run_pipeline, DipGeneratorStage, NoMeshPolicy,
        )

        if dips is None:
            raise ValueError("dips parameter is required")

        ctrl_x = np.asarray(x_coords, dtype=np.float64)
        ctrl_y = np.asarray(y_coords, dtype=np.float64)
        if not is_utm:
            if np.any(np.abs(ctrl_x) > 360) or np.any(np.abs(ctrl_y) > 90):
                warnings.warn(
                    "Coordinate values exceed lon/lat range (|x|>360 or |y|>90). "
                    "If passing UTM coordinates, set is_utm=True.",
                    stacklevel=2,
                )
            ctrl_x_utm, ctrl_y_utm = self.ll2xy(ctrl_x, ctrl_y)
        else:
            ctrl_x_utm, ctrl_y_utm = ctrl_x.copy(), ctrl_y.copy()

        dcp = DipControlPoints(
            x=ctrl_x_utm,
            y=ctrl_y_utm,
            dip=np.asarray(dips, dtype=np.float64).copy(),
        )

        stages = [
            DipGeneratorStage(
                dip_control_points=dcp,
                perturbations=perturbations,
                fixed_nodes=fixed_nodes,
                angle_unit=angle_unit,
                densify_top=True,
                discretization_interval=discretization_interval,
                interpolation_axis=interpolation_axis,
                buffer_nodes=buffer_nodes,
                buffer_radius=buffer_radius,
                use_average_strike=use_average_strike,
                average_strike_source=average_strike_source,
                user_direction_angle=user_direction_angle,
            ),
        ]
        run_pipeline(self, stages, mesh_policy=NoMeshPolicy())

        return self.bottom_coords

    @track_mesh_update(
                       description="Perturb dips using preset reference values (No mesh update).",
                       params_info={"perturbations": "Scalar or one change per movable dip control", "fixed_nodes": "List of fixed indices"},
                       reference_requirements={"fields": ("top_coords", "dip_control_points")},
                       perturbation_cardinality={
                           "kind": "scalar_or_dip_controls",
                           "fixed_nodes_parameter": "fixed_nodes",
                       },
                       perturbation_items=({"role": "dip_change", "unit_from": "angle_unit"},))
    def perturb_dips_with_preset_params(self, perturbations, *,
                                        discretization_interval=None, interpolation_axis='x',
                                        fixed_nodes=None, angle_unit='degrees',
                                        buffer_nodes=None, buffer_radius=None,
                                        use_average_strike=False, average_strike_source='pca', user_direction_angle=None):
        """Apply dip perturbations using preset control points from geometry_ref.

        Reads dip control points from ``self.geometry_ref.dip_control_points``.
        Reference setters store those points in longitude/latitude; this method
        always converts them to fault-local projected x/y internally. It does
        not require a mesh: only ``top_coords`` and dip control points must be
        set.

        Parameters
        ----------
        perturbations : np.ndarray
            Perturbation values added after each reference dip is converted to
            the continuous 0--180 coordinate. This permits a proposal such as
            ``77 + [-30, 30] -> [47, 107]`` to cross the vertical orientation
            continuously and makes equivalent references ``100`` and ``-80``
            behave identically.
        discretization_interval : float, optional
            Interval for discretizing the fault trace (UTM km).
        interpolation_axis : str, default 'x'
            One of ``'auto'``, ``'x'``, ``'y'``, or ``'arc_length'``. ``'x'``
            and ``'y'`` refer to fault-local projected easting and northing
            after coordinate conversion. ``'auto'`` uses PCA to choose one of
            those axes. ``'arc_length'`` projects the controls onto the current
            top edge and interpolates along its cumulative distance; it is the
            preferred option for a curved trace that is not monotonic in x or
            y. It cannot be combined with buffer augmentation.
        fixed_nodes : list, optional
            Indices of control points to hold fixed.
        angle_unit : str, default 'degrees'
            Unit of *perturbations*: 'degrees' or 'radians'.
        buffer_nodes : np.ndarray, optional
            Extra buffer coordinates (lon/lat) appended before interpolation.
        buffer_radius : float, optional
            Search radius for buffer node influence.
        use_average_strike : bool, default False
            Use average strike direction for dip projection.
        average_strike_source : str, default 'pca'
            Source for the single strike ('pca' or 'user').
        user_direction_angle : float, optional
            Explicit strike azimuth in degrees clockwise from North when
            ``average_strike_source='user'``.

        Returns
        -------
        np.ndarray
            Updated ``self.bottom_coords`` after dip perturbation.
        """
        dcp = self._project_reference_dip_control_points()

        return self.perturb_dips(
            dcp.x,
            dcp.y,
            dcp.dip,
            perturbations,
            fixed_nodes=fixed_nodes,
            angle_unit=angle_unit,
            discretization_interval=discretization_interval,
            interpolation_axis=interpolation_axis,
            is_utm=True,
            buffer_nodes=buffer_nodes,
            buffer_radius=buffer_radius,
            use_average_strike=use_average_strike,
            average_strike_source=average_strike_source,
            user_direction_angle=user_direction_angle,
        )

    @track_mesh_update(update_mesh=True,
                       description="Perturb dips using preset reference values and rebuild simple mesh.",
                       params_info={"perturbations": "Scalar or one change per movable dip control", "kwargs": "Mesh generation parameters (disct_z, bias...)"},
                       reference_requirements={"fields": ("top_coords", "dip_control_points")},
                       perturbation_cardinality={
                           "kind": "scalar_or_dip_controls",
                           "fixed_nodes_parameter": "fixed_nodes",
                       },
                       perturbation_items=({"role": "dip_change", "unit_from": "angle_unit"},))
    def perturb_DipsPresetParams_SimpleMesh(self, perturbations, *,
                                            discretization_interval=None, interpolation_axis='x',
                                            fixed_nodes=None, angle_unit='degrees',
                                            buffer_nodes=None, buffer_radius=None,
                                            disct_z=None, bias=None, min_dz=None,
                                            use_average_strike=False, average_strike_source='pca', user_direction_angle=None):
        """Perturb dips using preset control points and rebuild the mesh.

        Same reference, perturbation, and interpolation contracts as
        :meth:`perturb_dips_with_preset_params`, including automatic conversion
        of canonical lon/lat reference controls and the
        ``{'auto', 'x', 'y', 'arc_length'}`` choices for
        ``interpolation_axis``. It also rebuilds the triangular mesh via
        ``SimpleMeshPolicy`` after geometry update. In particular,
        ``'arc_length'`` is suitable for curved top edges and cannot be combined
        with ``buffer_nodes``/``buffer_radius``.

        Additional Parameters
        ---------------------
        disct_z : float, optional
            Vertical discretization interval (km) for mesh generation.
        bias : float, optional
            Depth-bias factor for variable layer spacing.
        min_dz : float, optional
            Minimum layer thickness (km).
        """
        from .pipeline import (
            run_pipeline, DipGeneratorStage, SimpleMeshPolicy,
        )

        pipeline_dcp = self._project_reference_dip_control_points()

        stages = [
            DipGeneratorStage(
                dip_control_points=pipeline_dcp,
                perturbations=perturbations,
                fixed_nodes=fixed_nodes,
                angle_unit=angle_unit,
                densify_top=True,
                discretization_interval=discretization_interval,
                interpolation_axis=interpolation_axis,
                buffer_nodes=buffer_nodes,
                buffer_radius=buffer_radius,
                use_average_strike=use_average_strike,
                average_strike_source=average_strike_source,
                user_direction_angle=user_direction_angle,
            ),
        ]
        run_pipeline(self, stages,
                     mesh_policy=SimpleMeshPolicy(disct_z=disct_z, bias=bias, min_dz=min_dz))

        return

    #--------------------------------------Legacy aliases--------------------------------------#
    def set_xy_dip_ref(self, x_dip_ref, y_dip_ref, dip_ref, is_utm=False):
        """Legacy alias for :meth:`set_dip_control_points`."""
        self.set_dip_control_points(x_dip_ref, y_dip_ref, dip_ref, is_utm=is_utm)

    def set_xy_dip_ref_from_coords(self, coords, dips, is_utm=False):
        """Legacy alias for :meth:`set_dip_control_points_from_coords`."""
        self.set_dip_control_points_from_coords(coords, dips, is_utm=is_utm)

    def set_xy_dip_ref_from_file(self, filename, header=0, is_utm=False):
        """Legacy alias for :meth:`set_dip_control_points_from_file`."""
        self.set_dip_control_points_from_file(filename, header=header, is_utm=is_utm)
    #-------------------------------------------------------------------------------------------------------------------#
