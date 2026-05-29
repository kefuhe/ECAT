"""
DipPerturbationMixin — Dip angle perturbation methods.

Extracted from BayesianAdaptiveTriangularPatches for modularity.
Methods now delegate to the perturbation pipeline internally.
"""
import numpy as np
import warnings
from ..bayesian_perturbation_base import track_mesh_update, DipControlPoints


class DipPerturbationMixin:
    """Mixin providing dip-angle perturbation capabilities."""

    #--------------------------------------Perturbing Dip--------------------------------------#
    def perturb_dips(self, x_coords, y_coords, dips,
                     perturbations, fixed_nodes=None, angle_unit='degrees',
                     discretization_interval=None, interpolation_axis='x',
                     is_utm=False, buffer_nodes=None, buffer_radius=None,
                     use_average_strike=False, average_strike_source='pca', user_direction_angle=None):
        """
        Apply random perturbations to dips.

        Parameters:
        x_coords (np.ndarray): x coordinates.
        y_coords (np.ndarray): y coordinates.
        dips (np.ndarray): Original dips.
        perturbations (np.ndarray): Direct perturbation values. If provided, the function will use these values directly.
        fixed_nodes (list, optional): A list of indices of fixed nodes. If provided, the function will not perturb these nodes.
        angle_unit (str, optional): Unit of dip perturbations, 'radians' or 'degrees'. Default is 'degrees'.
        discretization_interval (float, optional): The interval for discretization of the fault trace. Default is None.
        interpolation_axis (str, optional): The axis for interpolation. Default is 'x'.
        is_utm (bool, optional): Whether the coordinates are in UTM. Default is False.
        buffer_nodes (np.ndarray, optional): Coordinates of buffer nodes. If provided, the function will interpolate these nodes after perturbing the dips.
        buffer_radius (float, optional): Radius of the buffer. If provided, the function will interpolate these nodes after perturbing the dips.
        use_average_strike (bool, optional): Whether to use the average strike direction. Default is False.
        average_strike_source (str, optional): The source of the average strike direction. Default is 'pca'.
        user_direction_angle (float, optional): User-specified direction angle. Default is None.

        Returns:
        np.ndarray: Perturbed coordinates.
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

    @track_mesh_update(description="Perturb dips using preset reference values (No mesh update).",
                       params_info={"perturbations": "Array of dip changes", "fixed_nodes": "List of fixed indices"})
    def perturb_dips_with_preset_params(self, perturbations, discretization_interval=None, interpolation_axis='x',
                                        fixed_nodes=None, angle_unit='degrees', is_utm=False,
                                        buffer_nodes=None, buffer_radius=None,
                                        use_average_strike=False, average_strike_source='pca', user_direction_angle=None):
        """Apply dip perturbations using preset control points from geometry_ref.

        Reads dip control points (stored as lon/lat) from
        ``self.geometry_ref.dip_control_points``.  Does NOT require a mesh —
        only ``top_coords`` and dip control points must be set.

        Parameters
        ----------
        perturbations : np.ndarray
            Perturbation values to add to each control-point dip.
        discretization_interval : float, optional
            Interval for discretizing the fault trace (UTM km).
        interpolation_axis : str, default 'x'
            Axis along which to interpolate dips. Refers to the UTM
            coordinate system (after internal ll2xy conversion):
            'x' = UTM-easting, 'y' = UTM-northing.
        fixed_nodes : list, optional
            Indices of control points to hold fixed.
        angle_unit : str, default 'degrees'
            Unit of *perturbations*: 'degrees' or 'radians'.
        is_utm : bool, default False
            If True, stored dip control-point coords are already UTM (km)
            and will NOT be converted via ll2xy.  Default False means they
            are lon/lat and will be converted internally.
        buffer_nodes : np.ndarray, optional
            Extra buffer coordinates (lon/lat) appended before interpolation.
        buffer_radius : float, optional
            Search radius for buffer node influence.
        use_average_strike : bool, default False
            Use average strike direction for dip projection.
        average_strike_source : str, default 'pca'
            Source for strike estimation ('pca' or 'endpoints').
        user_direction_angle : float, optional
            Override direction angle (degrees clockwise from north).

        Returns
        -------
        np.ndarray
            Updated ``self.bottom_coords`` after dip perturbation.
        """
        self._require_geometry_ref('dip_control_points')
        dcp = self.geometry_ref.dip_control_points

        return self.perturb_dips(
            dcp.x,
            dcp.y,
            dcp.dip,
            perturbations,
            fixed_nodes,
            angle_unit,
            discretization_interval,
            interpolation_axis,
            is_utm,
            buffer_nodes,
            buffer_radius,
            use_average_strike,
            average_strike_source,
            user_direction_angle
        )

    @track_mesh_update(update_mesh=True,
                       description="Perturb dips using preset reference values and rebuild simple mesh.",
                       params_info={"perturbations": "Array of dip changes", "kwargs": "Mesh generation parameters (disct_z, bias...)"})
    def perturb_DipsPresetParams_SimpleMesh(self, perturbations, discretization_interval=None, interpolation_axis='x',
                                            fixed_nodes=None, angle_unit='degrees', is_utm=False,
                                            buffer_nodes=None, buffer_radius=None,
                                            disct_z=None, bias=None, min_dz=None,
                                            use_average_strike=False, average_strike_source='pca', user_direction_angle=None):
        """Perturb dips using preset control points and rebuild the mesh.

        Same as :meth:`perturb_dips_with_preset_params` but also rebuilds
        the triangular mesh via ``SimpleMeshPolicy`` after geometry update.

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

        self._require_geometry_ref('dip_control_points')
        dcp = self.geometry_ref.dip_control_points

        ctrl_x, ctrl_y = dcp.x, dcp.y
        if not is_utm:
            ctrl_x, ctrl_y = self.ll2xy(ctrl_x, ctrl_y)

        pipeline_dcp = DipControlPoints(
            x=np.asarray(ctrl_x, dtype=np.float64),
            y=np.asarray(ctrl_y, dtype=np.float64),
            dip=dcp.dip.copy(),
        )

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
