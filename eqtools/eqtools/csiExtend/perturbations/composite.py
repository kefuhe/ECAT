"""
CompositePerturbationMixin — Combined perturbation methods (rotation + translation + fixed direction).

Extracted from BayesianAdaptiveTriangularPatches for modularity.
Migration: methods now delegate to the perturbation pipeline internally.
"""
import numpy as np
from ..bayesian_perturbation_base import track_mesh_update


class CompositePerturbationMixin:
    """Mixin providing composite perturbation capabilities (combinations of direction, rotation, translation)."""

    #-------------------------------------Perturbing geometry by translation and rotation-------------------------------------#
    @track_mesh_update(update_mesh=True, update_laplacian=True, update_area=True, expected_perturbations_count=3,
                       description="Rotate and Translate the entire geometry.",
                       params_info={"perturbations": "[rot_angle, dx, dy]", "pivot": "'start'/'end'/'midpoint'"})
    def perturb_RotateTransGeom(self, perturbations, pivot='midpoint', is_utm=False,
                                                     angle_unit='degrees', force_pivot_in_coords=False):
        """
        Apply a global rotation and translation to all coordinates.

        Parameters:
        perturbations (list): A list containing the rotation angle and translation distances. Format: [rotation_angle, dx, dy].
        pivot (str or array-like, optional): The pivot point for rotation. Default is 'midpoint'.
        is_utm (bool, optional): Whether the coordinates are in UTM. Default is False.
        angle_unit (str, optional): The unit of the rotation angle. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot point to be within the coordinates. Default is False.

        Returns:
        np.ndarray: The rotated and translated vertices.
        """
        from .pipeline import (
            run_pipeline, Target, RotateStage, TranslateStage, NoMeshPolicy,
        )
        self._ensure_vertices_ref()
        stages = [
            RotateStage([Target("vertices")], angle=perturbations[0],
                        pivot=pivot, pivot_source=Target("top"),
                        pivot_is_utm=is_utm, force_pivot_in_coords=force_pivot_in_coords),
            TranslateStage([Target("vertices")], dx=perturbations[1], dy=perturbations[2]),
        ]
        state = run_pipeline(self, stages, mesh_policy=NoMeshPolicy(), angle_unit=angle_unit)
        return state.vertices
    #---------------------------------------------------------------------------------------------------------------#

    #--------------------------------------Combined: FixedDir + Rotation + Translation--------------------------------------#
    @track_mesh_update(update_mesh=False, update_laplacian=False, update_area=False, expected_perturbations_count=4,
                       description="Combined Perturbation: FixedDir + Rotate + Translate (No mesh update).",
                       params_info={"perturbations": "[d_bottom, rot, dx, dy]"})
    def perturb_BottomFixedDir_RotateTransGeom(self, perturbations, average_direction=None, recalculate_pivot=False,
                                                          angle_unit='degrees', force_pivot_in_coords=False,
                                                          fixed_nodes=None, use_average_strike=True, pivot='midpoint'):
        from .pipeline import (
            run_pipeline, Target, OffsetStage, RotateStage, TranslateStage,
            StrikeNormalDirection, AllNodes, ExcludeNodes, NoMeshPolicy,
        )
        d_bottom, rot, dx, dy = perturbations[0], perturbations[1], perturbations[2], perturbations[3]
        node_sel = ExcludeNodes(fixed_nodes) if fixed_nodes else AllNodes()
        direction = StrikeNormalDirection(
            use_average=use_average_strike,
            average_direction=average_direction,
            angle_unit=angle_unit,
        )
        stages = [
            OffsetStage(Target("bottom"), node_sel, direction, values=np.array([d_bottom])),
            RotateStage([Target("top"), Target("bottom")], angle=rot, pivot=pivot,
                        force_pivot_in_coords=force_pivot_in_coords),
            TranslateStage([Target("top"), Target("bottom")], dx=dx, dy=dy),
        ]
        run_pipeline(self, stages, mesh_policy=NoMeshPolicy(), angle_unit=angle_unit)
        return 
    
    @track_mesh_update(update_mesh=True, update_laplacian=False, update_area=False, expected_perturbations_count=4,
                       description="Combined Perturbation (FixedDir+Rot+Trans) and rebuild simple mesh.",
                       params_info={"perturbations": "[d_bottom, rot, dx, dy]"})
    def perturb_BottomFixedDir_RotateTransGeom_simpleMesh(self, perturbations, average_direction=None, recalculate_pivot=False,
                                                          angle_unit='degrees', force_pivot_in_coords=False,
                                                          fixed_nodes=None, use_average_strike=True, pivot='midpoint',
                                                          disct_z=None, bias=None, min_dz=None):
        """
        Apply perturbations to the bottom coordinates and update the simple mesh.

        Approach: First perturb the bottom coordinates along the fixed direction, then rotate, and finally translate.
        """
        from .pipeline import (
            run_pipeline, Target, OffsetStage, RotateStage, TranslateStage,
            StrikeNormalDirection, AllNodes, ExcludeNodes, SimpleMeshPolicy,
        )
        d_bottom, rot, dx, dy = perturbations[0], perturbations[1], perturbations[2], perturbations[3]
        node_sel = ExcludeNodes(fixed_nodes) if fixed_nodes else AllNodes()
        direction = StrikeNormalDirection(
            use_average=use_average_strike,
            average_direction=average_direction,
            angle_unit=angle_unit,
        )
        stages = [
            OffsetStage(Target("bottom"), node_sel, direction, values=np.array([d_bottom])),
            RotateStage([Target("top"), Target("bottom")], angle=rot, pivot=pivot,
                        force_pivot_in_coords=force_pivot_in_coords),
            TranslateStage([Target("top"), Target("bottom")], dx=dx, dy=dy),
        ]
        run_pipeline(self, stages,
                     mesh_policy=SimpleMeshPolicy(disct_z=disct_z, bias=bias, min_dz=min_dz),
                     angle_unit=angle_unit)
    
    @track_mesh_update(update_mesh=True, update_laplacian=False, update_area=False, expected_perturbations_count=6,
                       description="Complex Multi-layer Perturbation (FixedDir + Rot + Trans).",
                       params_info={"perturbations": "[mid_offset, vert_offset, bot_offset, rot, dx, dy]"})
    def perturb_BottomFixedDir_RotateTransGeom_multiLayerMesh(self, perturbations, average_direction=None, recalculate_pivot=False,
                                                              angle_unit='degrees', force_pivot_in_coords=False,
                                                              fixed_nodes=None, use_average_strike=True, pivot='midpoint',
                                                              disct_z=None, bias=None):
        """
        Apply perturbations to the bottom coordinates and update the multi-layer mesh.

        Approach: First perturb bottom/layer coordinates along the fixed direction, then rotate, and finally translate.
        Assumption: One middle layer; the middle layer is perturbed in both horizontal and vertical directions, the bottom layer is perturbed only in the horizontal direction.

        Parameters:
        perturbations (list): The angles of perturbation.
                              0: mid_layer offset along fixed direction
                              1: mid_layer offset along vertical direction
                              2: bottom offset along fixed direction
                              3: rotation of total geometry
                              4-5: dx, dy of total geometry
        average_direction (float or array-like, optional): Strike-normal azimuth
            in the unit specified by angle_unit.
        recalculate_pivot (bool, optional): Whether to recalculate the pivot point. Default is False.
        angle_unit (str, optional): The unit of the angle. Default is 'degrees'.
        force_pivot_in_coords (bool, optional): Whether to force the pivot to be one of the points in coords. Default is False.
        fixed_nodes (list, optional): A list of indices of fixed nodes.
        use_average_strike (bool, optional): Whether to use the average strike direction. Default is True.
        pivot (str or tuple, optional): The center point of the perturbation. Default is 'midpoint'.
        disct_z (float, optional): Mesh density in the z direction. Default is None.
        bias (float, optional): Mesh bias in the z direction. Default is None.
        """
        from .pipeline import (
            run_pipeline, Target, OffsetStage, RotateStage, TranslateStage,
            StrikeNormalDirection, VerticalDirection, AllNodes, ExcludeNodes,
            MultiLayerMeshPolicy,
        )
        mid_h, mid_v, bot, rot, dx, dy = (perturbations[i] for i in range(6))
        node_sel = ExcludeNodes(fixed_nodes) if fixed_nodes else AllNodes()

        if average_direction is None and use_average_strike:
            average_direction = self.calculate_perturb_direction(
                self.geometry_ref.top_coords, use_average_strike=True)[0]
            direction_angle_unit = 'radians'
        else:
            direction_angle_unit = angle_unit

        direction = StrikeNormalDirection(
            use_average=use_average_strike,
            average_direction=average_direction,
            angle_unit=direction_angle_unit,
        )

        all_targets = [Target("top"), Target("bottom"), Target("layers")]

        stages = [
            OffsetStage(Target("bottom"), node_sel, direction, values=np.array([bot])),
            OffsetStage(Target("layers"), node_sel, direction, values=np.array([mid_h])),
            OffsetStage(Target("layers"), node_sel, VerticalDirection(), values=np.array([mid_v])),
            RotateStage(all_targets, angle=rot, pivot=pivot,
                        force_pivot_in_coords=force_pivot_in_coords),
            TranslateStage(all_targets, dx=dx, dy=dy),
        ]
        run_pipeline(self, stages,
                     mesh_policy=MultiLayerMeshPolicy(disct_z=disct_z, bias=bias),
                     angle_unit=angle_unit)
    #---------------------------------------------------------------------------------------------------------------#

    #-------------------------------Perturbing Bottom Coords by Translation and Rotation------------------------#
    @track_mesh_update(update_mesh=False, update_laplacian=False, expected_perturbations_count=3,
                       description="Rotate and Translate bottom coordinates.",
                       params_info={"perturbations": "[rot, dx, dy]", "pivot": "'start'/'end'/'midpoint'"})
    def perturb_BottomRotateTrans(self, perturbations, pivot='midpoint', is_utm=False,
                                  angle_unit='degrees', force_pivot_in_coords=False):
        """
        Modify the bottom coordinates by rotation and translation.

        Parameters:
        perturbations (list): A list containing the rotation angle and translation distance. Format: `[rotation, dx, dy]`.
        pivot (str or array-like): The pivot point for rotation. Default is 'midpoint'.
        is_utm (bool): Whether the coordinates are in UTM. Default is False.
        angle_unit (str): The unit of the rotation angle. Default is 'degrees'.
        force_pivot_in_coords (bool): Whether to force the pivot point to be within the coordinates. Default is False.
        """
        from .pipeline import (
            run_pipeline, Target, RotateStage, TranslateStage, NoMeshPolicy,
        )
        stages = [
            RotateStage([Target("bottom")], angle=perturbations[0],
                        pivot=pivot, pivot_source=Target("bottom"),
                        pivot_is_utm=is_utm, force_pivot_in_coords=force_pivot_in_coords),
            TranslateStage([Target("bottom")], dx=perturbations[1], dy=perturbations[2]),
        ]
        run_pipeline(self, stages, mesh_policy=NoMeshPolicy(), angle_unit=angle_unit)
        return self.bottom_coords

    @track_mesh_update(update_mesh=True, expected_perturbations_count=3,
                       description="Rotate and Translate bottom coords, then rebuild simple mesh.",
                       params_info={"perturbations": "[rot, dx, dy]", "pivot": "'start'/'end'/'midpoint'"})
    def perturb_BottomRotateTrans_simpleMesh(self, perturbations, pivot='midpoint', is_utm=False,
                                                        angle_unit='degrees', force_pivot_in_coords=False,
                                                        disct_z=None, bias=None, min_dz=None):
        from .pipeline import (
            run_pipeline, Target, RotateStage, TranslateStage, SimpleMeshPolicy,
        )
        stages = [
            RotateStage([Target("bottom")], angle=perturbations[0],
                        pivot=pivot, pivot_source=Target("bottom"),
                        pivot_is_utm=is_utm, force_pivot_in_coords=force_pivot_in_coords),
            TranslateStage([Target("bottom")], dx=perturbations[1], dy=perturbations[2]),
        ]
        run_pipeline(self, stages,
                     mesh_policy=SimpleMeshPolicy(disct_z=disct_z, bias=bias, min_dz=min_dz),
                     angle_unit=angle_unit)
    #------------------------------------------------------------------------------------------------------------#
