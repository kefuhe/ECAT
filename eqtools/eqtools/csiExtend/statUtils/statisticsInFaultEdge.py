# External libraries
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp2d, interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree
import sys
import os
import copy

# CSI routines
from csi import TriangularPatches
from csi import SourceInv as csiSourceInv
from ..geom_ops import calculate_average_direction


class StatisticsInFault(csiSourceInv):
    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None,
                 source=None, outdir='output_stats', source_type='tri',
                 top_tolerance=0.1, bottom_tolerance=0.1):
        """
        Initialize statistics in fault analysis
        Parameters:
        name : str
            Name of the source
        utmzone : str, optional
            UTM zone for the source
        ellps : str, optional
            Ellipsoid model for the source, default is 'WGS84'
        lon0 : float, optional
            Central meridian for the UTM zone, default is None
        lat0 : float, optional
            Central latitude for the UTM zone, default is None
        source : TriangularPatches, optional
            Fault geometry object containing vertices, faces and slip data
        outdir : str, optional
            Output directory for statistics results, default is 'output_stats'
        source_type : str, optional
            Type of the source, default is 'tri'
        top_tolerance : float, optional
            Tolerance for top edge vertices, default is 0.1 km
        bottom_tolerance : float, optional
            Tolerance for bottom edge vertices, default is 0.1 km

        Initializes the StatisticsInFault class, which extends csiSourceInv.
        """
        super().__init__(name, utmzone, ellps, lon0, lat0)

        if source is None:
            self.source = TriangularPatches('source', utmzone=utmzone, lon0=lon0, lat0=lat0)
        else:
            if source_type == 'tri':
                self.source = source
            elif source_type == 'rect':
                cosource = TriangularPatches(source.name, lon0=lon0, lat0=lat0)
                cosource.patches2triangles(source, numberOfTriangles=2)
                cosource.slip[:, 0] = np.repeat(source.slip[:, 0], 2)
                cosource.slip[:, 1] = np.repeat(source.slip[:, 1], 2)
                self.source = cosource

        self.outdir = outdir

        # Find fault edge triangles and vertices
        self.getfaultEdgeTriangles_and_EdgeLines(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, refind=True)
    
    def setSource(self, source):
        self.source = source 

    def setOutdir(self, outdir):
        self.outdir = outdir
    
    def getfaultEdgeTriangles_and_EdgeLines(self, top_tolerance=0.1, bottom_tolerance=0.1, refind=False):
        '''
        Left: In West, Right: In East
        Left: In North, Right: In South if the left/right can not be determined from west/east
        '''
        # find boundary and corner triangles indexes in fault.Faces
        fault = self.source
        if not hasattr(fault, 'edge_triangles_indices') or refind:
            fault.find_fault_edge_vertices(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, refind=True)
    
        if not hasattr(fault, 'edge_vertex_indices') or refind:
            fault.find_fault_fouredge_vertices(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, refind=True)

        self.edge_triangles_indices = fault.edge_triangles_indices
        self.edge_triangle_vertex_indices = fault.edge_triangle_vertex_indices
        self.edge_vertex_indices = fault.edge_vertex_indices
        self.edge_vertices = fault.edge_vertices
    
        def sort_edge_triangles(edge):
            edge_tri_sort = []
            for a_ind, b_ind in np.repeat(self.edge_vertex_indices[edge], 2)[1:-1].reshape(-1, 2):
                for tri_ind in self.edge_triangles_indices[edge]:
                    tri_vert_inds = fault.Faces[tri_ind]
                    if a_ind in tri_vert_inds and b_ind in tri_vert_inds:
                        edge_tri_sort.append(tri_ind)
                        break
            return np.array(edge_tri_sort, dtype=int)
    
        self.edge_triangles_indices['top'] = sort_edge_triangles('top')
        self.edge_triangles_indices['bottom'] = sort_edge_triangles('bottom')
    
        def sort_edge_by_center_z(edge):
            center_z = np.mean(fault.Vertices[fault.Faces[self.edge_triangles_indices[edge]], -1], axis=1)
            sort_order = np.argsort(center_z)
            self.edge_triangles_indices[edge] = self.edge_triangles_indices[edge][sort_order]
    
        sort_edge_by_center_z('left')
        sort_edge_by_center_z('right')

        # sort top and bottom from left to right
        if self.edge_vertex_indices['top'][0] not in self.edge_vertex_indices['left']:
            self.edge_vertex_indices['top'] = self.edge_vertex_indices['top'][::-1]
            self.edge_vertices['top'] = self.edge_vertices['top'][::-1]
            self.edge_triangles_indices['top'] = self.edge_triangles_indices['top'][::-1]
        
        if self.edge_vertex_indices['bottom'][0] not in self.edge_vertex_indices['left']:
            self.edge_vertex_indices['bottom'] = self.edge_vertex_indices['bottom'][::-1]
            self.edge_vertices['bottom'] = self.edge_vertices['bottom'][::-1]
            self.edge_triangles_indices['bottom'] = self.edge_triangles_indices['bottom'][::-1]
    
        # All Done
        return

    def generate_side_statistics(self, value=None, depth=None, side='right', step=0.2, yaxis_ticks=[0,10,20], xtick_angle_offset=0, 
                                 ytick_angle_offset=0, tick_scale=1.0, dip_direction_angle=None, xtick_scale=None, ytick_scale=None,
                                 xaxis_ticks=[1.0, 5.0], xaxis_zoffset=0.2, slip='total', bins=15, method='mean', 
                                 statkind='hist', zinterval=2.0, cutmethod='pdcut', outfile=None):
        """
        Generate side statistics and plot.

        Parameters:
        value (array): Value for the statistics.
        depth (array): Depth for the statistics.
        side (str): The side of the fault ('left' or 'right').
        step (float): Step size for the side edge line. Unit is km.
        yaxis_ticks (list): Y-axis ticks for the side edge line. y-axis is the depth axis.
        ytick_angle_offset (float): Angle offset for the y-ticks relative to the strike direction.
        xtick_angle_offset (float): Angle offset for the x-ticks relative to the dip angle.
        tick_scale (float): Scale for the x/y-ticks. unit is km.
        xtick_scale (float): Scale for the x-ticks. unit is km. if None it will be set to tick_scale.
        ytick_scale (float): Scale for the y-ticks. unit is km. if None it will be set to tick_scale.
        dip_direction_angle (float): Dip direction angle.
        slipinterval (float): Interval for slip.
        xaxis_ticks (list): X-axis ticks for the X-axis.
        xaxis_zoffset (float): Z-offset for the X-axis.
        slip (str): Slip type.
        bins (int): Number of bins for statistics.
        method (str): Method for statistics ('mean' or 'sum').
        statkind (str): Kind of statistics ('curve' or 'hist').
        zinterval (float): Interval for Z-axis.
        cutmethod (str): Method for cutting ('pdcut' or other).
        outfile (str): Output file name.
        """
        # Get side edge line
        lonlat = self.getSideEdgeLine(side=side, step=step, yaxis_ticks=yaxis_ticks, 
                                      ytick_angle_offset=ytick_angle_offset, tick_scale=tick_scale if ytick_scale is None else ytick_scale, 
                                      dip_direction_angle=dip_direction_angle)
        
        # Generate X-axis
        self.genXaxis(xaxis_ticks=xaxis_ticks, tick_scale=tick_scale if xtick_scale is None else xtick_scale, ytick_angle_offset=ytick_angle_offset, 
                      xtick_angle_offset=xtick_angle_offset, xaxis_zoffset=xaxis_zoffset, side=side)
        
        # Generate side statistics
        self.StatInSide(value=value, depth=depth, slip=slip, bins=bins, side=side, method=method, statkind=statkind, 
                        zinterval=zinterval, cutmethod=cutmethod, outfile=outfile)

    def getSideEdgeLine(self, side='right', dip_direction_angle=None, ytick_angle_offset=0, 
                        step=0, yaxis_ticks=[0, 10, 20], tick_scale=0.2,
                        write2file=True, outfile=None, plot=False):
        """
        Args:
            * dip_direction_angle: Default None, automatically calculate the average dip direction.
                ** Can manually input a primary dip direction.
            * ytick_angle_offset: Default 0, the offset angle of the tick direction relative to the strike direction.
            * step: Default 0, controls the offset distance of the y-axis from the fault, unit: km.
            * yaxis_ticks: Default [0, 10, 20], controls the positions of the y-axis ticks.
            * tick_scale: Default 0.2, controls the length of the tick lines, unit: km.
            * write2file: Default True, whether to write the results to a file.
            * outfile: Default None, the output file name.
        
        Kwargs:
            * step: + km depending on the dip direction.
            * axis_tick_dir: The angle of the x-axis in the rotated coordinate system along the dip direction.
        """
        # Directory where the statistical information will be saved
        outdir = self.outdir
        # Fault object
        fault = self.source
    
        # Define a function to check if an angle is between -pi/2 and pi/2
        @np.vectorize
        def check_angle(angle):
            return -np.pi/2.0 < angle <= np.pi/2.0
    
        def rotate_points(points, angle):
            return (points[:, 0] + points[:, 1]*1.j)*np.exp(1.j*angle/180*np.pi)
    
        def calculate_offset(x, y, step, sign, horz_angle, rotateAngle):
            xy_offset = np.ones_like(x)*step*sign
            step_x, step_y = xy_offset*np.cos(np.deg2rad(horz_angle)), xy_offset*np.sin(np.deg2rad(horz_angle))
            x += step_x
            y += step_y
            xy = (x + y*1.j)*np.exp(-1.j*rotateAngle/180*np.pi)
            return xy.real, xy.imag
    
        def write_to_file(lonlat, outdir, outfile):
            lonlat = pd.DataFrame(lonlat, columns=['lon', 'lat', 'depth'])
            lonlat.to_csv(os.path.join(outdir, outfile), sep=' ', float_format='%.6f', index=False, header=False)
    
        side_edge = self.edge_triangles_indices[side]
        edgeline_points = self.edge_vertices[side]
        # Determine the strike angle of the edge
        strike_rad = np.pi/2.0 - fault.getStrikes()[side_edge]
        tri_centerz = np.mean(fault.Vertices[fault.Faces[side_edge]], axis=1)[:, -1]
        # Check if the strike angle is between -pi/2 and pi/2
        tri_strike_flags = np.apply_along_axis(check_angle, 0, strike_rad)
        tri_strike_flags = tri_strike_flags[np.argsort(tri_centerz)]
        if np.sum(tri_strike_flags) < len(tri_strike_flags)/2.0:
            tri_strike_flags = ~tri_strike_flags
    
        # Order the edge points by depth
        pnt_verts = edgeline_points
        if dip_direction_angle is None:
            # Calculate the average dip direction of the edge
            diffy = pnt_verts[1:, 1] - pnt_verts[:-1, 1]
            diffx = pnt_verts[1:, 0] - pnt_verts[:-1, 0]
            DipDir_seq = np.rad2deg(np.arctan2(diffy, diffx))[tri_strike_flags]
            print(f'The sequence of dip direction of {side} edge is: ', DipDir_seq)
            if np.sum(DipDir_seq) == 0:
                if side == 'right':
                    top_side = self.edge_vertices['top'][-2:, :]
                else:
                    top_side = self.edge_vertices['top'][:2, :]
                strike_angle = np.rad2deg(np.arctan2(top_side[1, 1] - top_side[0, 1], top_side[1, 0] - top_side[0, 0]))
                dip_dir_angle = np.mean(strike_angle - 90)
            else:
                DipDir_vec = calculate_average_direction(pnt_verts)
                dip_dir_angle = np.rad2deg(np.arctan2(DipDir_vec[1], DipDir_vec[0]))
                strike_angle = dip_dir_angle + 90
            rotateAngle = -dip_dir_angle
            print('rotate angle is: ', rotateAngle)
        else:
            dip_dir_angle = dip_direction_angle
            strike_angle = dip_dir_angle + 90
            rotateAngle = -dip_dir_angle
            print('rotate angle is: ', rotateAngle)
        
        # Calculate the strike vector
        strike_vec = np.array([np.cos(np.deg2rad(strike_angle)), np.sin(np.deg2rad(strike_angle))])
        # Calculate the cross product to determine the direction
        dip_dir_vec = np.array([np.cos(np.deg2rad(dip_dir_angle)), np.sin(np.deg2rad(dip_dir_angle))])
        cross_product_z = np.cross(np.append(dip_dir_vec, 0), np.append(strike_vec, 0))[-1]
        if cross_product_z < 0:
            strike_vec *= -1
        # Calculate the angle between the dip direction vector and the strike vector
        ytick_angle = np.rad2deg(np.arccos(np.dot(dip_dir_vec, strike_vec) / (np.linalg.norm(dip_dir_vec) * np.linalg.norm(strike_vec))))
        ytick_angle_with_offset = ytick_angle + ytick_angle_offset
        print('Y-tick direction angle relative to dip is: {0:.2f}'.format(ytick_angle))
        print('Y-tick direction angle with offset is: {0:.2f}'.format(ytick_angle_with_offset))

        # Rotate the coordinates by the dip direction angle. In the case of a single dip angle, project the edge line to the horizontal, then rotate to the x-axis, with the vertex near the x-axis origin and the deep part projecting to the x-axis positive direction.
        coord_rot = rotate_points(pnt_verts, rotateAngle)
        x_trans, y_trans = coord_rot.real, coord_rot.imag
        if side == 'right':
            sign = 1 # np.sign(rotateAngle)
        elif side == 'left':
            sign = -1 # -np.sign(rotateAngle)
    
        # Offset along the horizontal angle. The horizontal angle here refers to the angle relative to the coordinate system after rotating by rotateAngle.
        x, y = calculate_offset(x_trans, y_trans, step, sign, ytick_angle_with_offset, rotateAngle)
        coord_rot1 = rotate_points(np.vstack((x, y)).T, rotateAngle)
        x1, y1 = coord_rot1.real, coord_rot1.imag
        z = pnt_verts[:, -1]
        lon, lat = fault.xy2ll(x, y)
        lonlat = np.vstack((lon, lat, z)).T
        
        self.sideedges = {}
        self.sideedges[side] = {
            'rotateAngle': rotateAngle, # The default value is the negative of the dip direction angle.
            'step'       : step, # Controls the offset distance of the y-axis from the fault.
            'sign'       : sign, # Controls the direction of the step.
            'x'          : x,
            'y'          : y,
            'x_trans'    : x1, # Coordinates after rotating by rotateAngle.
            'y_trans'    : y1, # Coordinates after rotating by rotateAngle.
            'depth'      : z,
            'llz'        : lonlat,
            'ytick_angle': ytick_angle,
            'ytick_angle_with_offset': ytick_angle_with_offset,
        }
        if plot:
            plt.plot(x-x[0], y-y[0], label='right')
            plt.plot(x1-x1[0], y1-y1[0], label='right_rot')
            plt.legend()
            plt.show()
    
        # Generate Y-axis tick lines
        z_tick = np.array(yaxis_ticks)
        fit = interp1d(z, x1, fill_value="extrapolate")
        x_pred = fit(z_tick)
        y_pred = np.ones_like(x_pred)*y1.mean()
        # Y-tick start
        xy_st = rotate_points(np.vstack((x_pred, y_pred)).T, -rotateAngle)
        lonlat_st = np.vstack(fault.xy2ll(xy_st.real, xy_st.imag) + (z_tick,)).T
        # Y-tick end
        xy_ed = calculate_offset(x_pred, y_pred, tick_scale, sign, ytick_angle_with_offset, rotateAngle)
        lonlat_ed = np.vstack(fault.xy2ll(xy_ed[0], xy_ed[1]) + (z_tick,)).T
    
        if outfile is None:
            outfile = 'fault_{}edge_{}.dat'.format(side, self.source.name)
            outfile_xaxis = 'fault_{}edge_{}_xaxis.dat'.format(side, self.source.name)
            outfile_yaxis = 'fault_{}edge_{}_yaxis.dat'.format(side, self.source.name)
        if write2file:
            write_to_file(lonlat, outdir, outfile)
            # Write Y-axis
            with open(os.path.join(outdir, outfile_yaxis), 'wt') as fout:
                for ist, ied in zip(lonlat_st, lonlat_ed):
                    print('>', file=fout)
                    print('{0:.3f} {1:.3f} {2:.1f}'.format(*ist), file=fout)
                    print('{0:.3f} {1:.3f} {2:.1f}'.format(*ied), file=fout)
    
        # All Done
        return lonlat

    def genXaxis(self, side='right', xaxis_ticks=[0.3, 4.3], xaxis_zoffset=0.5, 
                 ytick_angle_offset=0, xtick_angle_offset=0, tick_scale=0.2):
        """
        Generate the X-axis for the specified side edge.
    
        Args:
            * side (str): 'right' or 'left', specifies which side edge to use.
            * xaxis_ticks (list): List of tick positions along the X-axis (default is [0.3, 4.3]). unit is km.
                ** The first value is the starting position, and the second value is the ending position in km.
            * xaxis_scale (float): Scale factor for the X-axis (default is 1.0).
            * xaxis_zoffset (float): Offset for the Z-axis (default is 0.5).
            * ytick_angle_offset (float): Default 0, the offset angle of the tick direction relative to the strike direction.
            * xtick_angle_offset (float): Default 0, the offset angle of the tick direction relative to the dip angle.
            * tick_scale (float): Scale factor for the tick lines (default is 0.2).
    
        Returns:
            * None
    
        Notes:
            * The X-axis is generated based on the specified side edge and the provided parameters.
            * The generated X-axis and tick lines are saved to files.
        """
        outdir = self.outdir
        fault = self.source
    
        sideedges = self.sideedges[side]
        rotateAngle = sideedges['rotateAngle']
        z, x1 = sideedges['depth'], sideedges['x_trans']
        y1 = sideedges['y_trans']
        sign = sideedges['sign']
        ytick_angle = sideedges['ytick_angle']
    
        horz_angle = ytick_angle + ytick_angle_offset
    
        # Rotation axis_angle
        # Dip angle of the surface sub-patch
        dip_angle_0 = np.rad2deg(np.arctan2(z[1]-z[0], x1[1]-x1[0])) # range for arctan2 is [-pi, pi]
        print('Dip angle of the surface patch is: {0:.2f}'.format(dip_angle_0))
        xtick_angle_with_offset = xtick_angle_offset - 90 + dip_angle_0
        print('X-tick direction is: {0:.2f}'.format(xtick_angle_with_offset))
    
        # Offset 
        hdep = np.array([-xaxis_zoffset, -xaxis_zoffset])
    
        xy_offset = np.array(xaxis_ticks)*sign
        x_offset, y_offset = xy_offset*np.cos(np.deg2rad(horz_angle)), xy_offset*np.sin(np.deg2rad(horz_angle))
        x_st = x1[0] + x_offset
        y_st = y1.mean() + y_offset
        z_st = np.ones_like(y_st)*z[0]
        # Rotate across Y-axis
        zx2_vert = (z_st + x_st*1.j)*np.exp(1.j*np.deg2rad(xtick_angle_with_offset))
    
        x_st = zx2_vert.imag
        y_st = y_st
        z_st = hdep + zx2_vert.real
        z_ed = z_st + tick_scale*-1 # sign TODO: Check
    
        # Rotate back
        ## llz_st
        xy_st = (x_st + y_st*1.j)
        zx2_st = (z_st + xy_st.real*1.j)*np.exp(-1.j*np.deg2rad(xtick_angle_with_offset))
        xy2_st = (zx2_st.imag + xy_st.imag*1.j)*np.exp(-1.j*np.deg2rad(rotateAngle))
    
        xyz_st = np.vstack((xy2_st.real, xy2_st.imag, zx2_st.real)).T
        lon, lat = fault.xy2ll(xyz_st[:, 0], xyz_st[:, 1])
        llz_st = np.vstack((lon, lat, xyz_st[:, -1])).T
        ## llz_ed
        xy_ed = (x_st + y_st*1.j)
        zx2_ed = (z_ed + xy_ed.real*1.j)*np.exp(-1.j*np.deg2rad(xtick_angle_with_offset))
        xy2_ed = (zx2_ed.imag + xy_ed.imag*1.j)*np.exp(-1.j*np.deg2rad(rotateAngle))
    
        xyz_ed = np.vstack((xy2_ed.real, xy2_ed.imag, zx2_ed.real)).T
        lon, lat = fault.xy2ll(xyz_ed[:, 0], xyz_ed[:, 1])
        llz_ed = np.vstack((lon, lat, xyz_ed[:, -1])).T
    
        outfile_xaxis = 'fault_{}edge_{}_xaxis.dat'.format(side, self.source.name)
        with open(os.path.join(outdir, outfile_xaxis), 'wt') as fout:
            print('>', file=fout)
            for illz in llz_st:
                print('{0:.6f} {1:.6f} {2:.1f}'.format(*illz), file=fout)
            
            # ticks
            for ist, ied in zip(llz_st, llz_ed):
                print('>', file=fout)
                print('{0:.6f} {1:.6f} {2:.1f}'.format(*ist), file=fout)
                print('{0:.6f} {1:.6f} {2:.1f}'.format(*ied), file=fout)
        
    
        # Keep coordinate system
        sideedges['horz_angle'] = horz_angle
        sideedges['xtick_angle_with_offset'] = xtick_angle_with_offset
        sideedges['xaxis_zoffset'] = xaxis_zoffset
        sideedges['xaxis_ticks'] = xaxis_ticks
        sideedges['xaxis_xyz']  = xyz_st
        sideedges['xaxis_xyz1'] = np.vstack((x_st, y_st, z_st)).T
    
        # All Done
        return
    
    def StatInSide(self, value=None, depth=None, slip='total', bins=15, zinterval=2, cutmethod='pdcut',
                   side='right', method='mean', statkind='hist', outfile='stat_hist.gmt', output_2d=True, norm_ref=None):
    
        outdir = self.outdir
        fault = self.source
    
        sideedges = self.sideedges[side]
        rotateAngle = sideedges['rotateAngle']
        z, x1 = sideedges['depth'], sideedges['x_trans']
        y1 = sideedges['y_trans']
        sign = sideedges['sign']
        horz_angle = sideedges['horz_angle']
        step0, steped = sideedges['xaxis_ticks']
        # 此处，step指x轴单位长度的尺度信息
        step = steped - step0
    
        # Step 1: 先将数据在深度上进行分块
        if value is None or depth is None:
            value, depth, val_max = self._statinSide(value=value, depth=depth, slip=slip, side=side, method=method,
                            statkind=statkind, cutmethod=cutmethod, zinterval=zinterval, bins=bins, norm_ref=norm_ref)
    
        # Step 2.1: 对统计信息基底进行坐标旋转和平移操作
        fit = interp1d(z, x1, fill_value="extrapolate")
        z_pred = depth
        x_pred = fit(z_pred)
        y_pred = np.ones_like(x_pred)*y1.mean()
        xy_offset = step0*sign*np.exp(1.j*np.deg2rad(horz_angle))
        x_pred += xy_offset.real
        y_pred += xy_offset.imag
    
        xy2 = (x_pred + y_pred*1.j)*np.exp(-1.j*rotateAngle/180*np.pi)
        x2, y2, z2 = xy2.real, xy2.imag, z_pred
        lon2, lat2 = fault.xy2ll(x2, y2)
        lonlat2 = np.vstack((lon2, lat2, z2)).T
    
        # Step 2.2: 对统计信息顶点进行坐标旋转和平移操作
        xy_offset = step*value*sign*np.exp(1.j*np.deg2rad(horz_angle))
        x_pred += xy_offset.real
        y_pred += xy_offset.imag
        xy3 = (x_pred + y_pred*1.j)*np.exp(-1.j*rotateAngle/180*np.pi)
        x3, y3, z3 = xy3.real, xy3.imag, depth
        lon3, lat3 = fault.xy2ll(x3, y3)
        lonlat3 = np.vstack((lon3, lat3, z3)).T
    
        if statkind == 'hist':
            with open(os.path.join(outdir, outfile), 'wt') as fout:
                print('# Maximum value is: {0:.3f}'.format(val_max), file=fout)
                print('# Norm ref is: {0:.3f}'.format(norm_ref if norm_ref is not None else val_max), file=fout)
                for i in range(int(value.shape[0]/2.0)):
                    print('>', file=fout)
                    st1, st2 = lonlat2[2*i:2*i+2]
                    st4, st3 = lonlat3[2*i:2*i+2]
                    for st in [st1, st2, st3, st4]:
                        print('{0:.3f} {1:.3f} {2:.3f}'.format(*st), file=fout)
        elif statkind == 'curve':
            with open(os.path.join(outdir, outfile), 'wt') as fout:
                print('# Maximum value is: {0:.3f}'.format(val_max), file=fout)
                print('# Norm ref is: {0:.3f}'.format(norm_ref if norm_ref is not None else val_max), file=fout)
                print('>', file=fout)
                for i in range(value.shape[0]):
                    st = lonlat3[i]
                    # Check for NaN values before writing
                    if not np.isnan(st).any():
                        print('{0:.3f} {1:.3f} {2:.3f}'.format(*st), file=fout)
    
        # Output 2D format if requested
        if output_2d:
            base, ext = os.path.splitext(outfile)
            outfile_2d = base + '_2d' + ext
            with open(os.path.join(outdir, outfile_2d), 'wt') as fout:
                print('# Maximum value is: {0:.3f}'.format(val_max), file=fout)
                print('# Norm ref is: {0:.3f}'.format(norm_ref if norm_ref is not None else val_max), file=fout)
                if statkind == 'hist':
                    for i in range(int(value.shape[0]/2.0)):
                        print('>', file=fout)
                        print('{0:.3f} {1:.3f}'.format(depth[2*i], 0), file=fout)
                        print('{0:.3f} {1:.3f}'.format(depth[2*i+1], 0), file=fout)
                        print('{0:.3f} {1:.3f}'.format(depth[2*i+1], value[2*i+1]), file=fout)
                        print('{0:.3f} {1:.3f}'.format(depth[2*i], value[2*i]), file=fout)
                elif statkind == 'curve':
                    print('>', file=fout)
                    for i in range(value.shape[0]):
                        # Check for NaN values before writing
                        if not (np.isnan(depth[i]) or np.isnan(value[i])):
                            print('{0:.3f} {1:.3f}'.format(depth[i], value[i]), file=fout)
    
        # All Done
        return

    def _statinSide(self, value=None, depth=None, slip='total', side='right', method='mean',
                        statkind='hist', cutmethod='pdcut', zinterval=2, bins=10, norm_ref=None):
        '''
        statkind   :
            * hist
            * curve
            * bar
        cutmethod :
            * pdcut : pd.cut
            * hist  : np.histogram
        '''
        fault = self.source

        if depth is None:
            slip_samp = fault.slip
            if slip == 'total':
                slip_norm = np.linalg.norm(slip_samp, axis=1)
            elif slip == 'strikeslip':
                slip_norm = np.linalg.norm(slip_samp[:, 0][:, None], axis=1)
            elif slip == 'dipslip':
                slip_norm = np.linalg.norm(slip_samp[:, 1][:, None], axis=1)
            value = slip_norm
            depth = np.array(fault.getcenters())[:, -1]
        
        if cutmethod == 'hist': # Statistic number of the data
            acums, adists = np.histogram(depth, bins=bins)
            val_max = acums.max()
            val_norm = acums/val_max if norm_ref is None else acums/norm_ref
            # 将bin的左右和中点坐标均保留
            zt = adists[:-1]
            zb = adists[1:]
            zc = (zt + zb)/2.0
        elif cutmethod == 'pdcut':
            zcut = pd.cut(depth, np.arange(fault.top, fault.Vertices[:, 2].max()+0.1, zinterval))
            data = pd.DataFrame(value, columns=['value'])
            if method == 'mean':
                val_dep = data.groupby(zcut, observed=False).mean()
            elif method == 'sum':
                val_dep = data.groupby(zcut, observed=False).sum()
            val_max = val_dep.max().iloc[0]
            val_norm = val_dep/val_max if norm_ref is None else val_dep/norm_ref
            val_norm = val_norm.value.values
            ind = pd.IntervalIndex(val_dep.index)
            zc = ind.mid.values
            zt = ind.left.values
            zb = ind.right.values

        if statkind == 'hist':
            value = np.repeat(val_norm, 2)
            depth = np.vstack((zt, zb)).T.flatten()
        else:
            value = val_norm
            depth = zc

        print('Maximum value is: {0:.3f}'.format(val_max))
        print('Norm ref is: {0:.3f}'.format(norm_ref if norm_ref is not None else val_max))

        # All Done
        return value, depth, val_max
    
    def _statinTop(self, value=None, lonlat=None, hinterval=2.0, slip='total_top', statkind='curve', 
                   height_scale=1.0, doStat=True, bins=15, method='mean', cutmethod='pdcut', 
                   discretizeInterval=0.2, depth_eps=0.25, xtick_angle_offset=0, KeepOriginCoord=True,
                   output_2d=True, outdir='.', outfile='stat_top.gmt', norm_ref=None):
        """
        Perform statistical analysis along the top edge of the fault.

        Parameters:
        * value: Optional value for the analysis.
        * lonlat: Optional longitude and latitude for the analysis.
        * discretizeInterval: Interval for discretizing the surface trace (default is 0.2).
        * slip (str): Type of slip to analyze (default is 'total_top'). 
            Optional values are 'total_top', 'strikeslip_top', 'dipslip_top', 'total_all', 'strikeslip_all', 'dipslip_all'.
        * statkind (str): Type of statistical analysis to perform ('curve', 'hist', 'bar', default is 'curve').
            * 'curve': Output the curve of the maximum value position.
                - Parameters:
                    * xtick_angle_offset: Offset angle for the x-ticks relative to the dip angle.
                    * method: Method for the analysis.
            * 'bar': Output vertical lines.
                - Parameters:
                    * height_scale: Scale for the height.
                    * zoffset: Offset for the z-axis.
            * 'hist': Output histogram mode.
                - Parameters:
                    * bins: Number of bins for the histogram.
                    * depth_eps: Depth epsilon for the analysis.
                    * hinterval (float): Horizontal interval for the analysis (default is 2.0).
        * height_scale (float): Scale for the height (default is 1.0).
        * doStat (bool): Whether to perform statistical analysis (default is True).
        * bins (int): Number of bins for the histogram (default is 15).
        * method (str): Method for the analysis (default is 'mean').
        * cutmethod (str): Method for cutting the data ('pdcut', 'hist', 'None', default is 'pdcut').
            * 'pdcut': Use hinterval to set the interval.
            * 'hist': Use bins to set the horizontal interval.
            * 'None': Do not perform statistical analysis, directly output the original coordinates (only for 'bar' and 'curve').
        * discretizeInterval (float): Interval for discretizing the data (default is 0.2).
        * depth_eps (float): Depth epsilon for the analysis (default is 0.25).
        * KeepOriginCoord (bool): Whether to keep the original coordinates (default is True).
        * norm_ref (float): Reference value for normalization (default is None).
            If None, the maximum value of the data will be used for normalization.

        Returns:
        * top_x: Transformed x-coordinates.
        * top_z: Transformed z-coordinates.
        * verts: Vertices of the fault.
        * trans_verts: Transformed vertices.
        * strikes: Strike angles.
        * dips: Dip angles.
        """
        fault = self.source
        side = 'top'

        top_edge_triangles = self.edge_triangles_indices[side]
        top_vert_inds = self.edge_vertex_indices[side]

        top_edge_tri_sort = []
        for a_ind, b_ind in np.repeat(top_vert_inds,2)[1:-1].reshape(-1,2):
            for tri_ind in top_edge_triangles:
                tri_vert_inds = fault.Faces[tri_ind]
                if a_ind in tri_vert_inds and b_ind in tri_vert_inds:
                    top_edge_tri_sort.append(tri_ind)
                    break
        top_edge_tri_sort = np.array(top_edge_tri_sort, dtype=int)

        # Get the information of the top edge of the fault and sort it
        centers = np.asarray(fault.getcenters())
        top_edge_centers = centers[top_edge_tri_sort, :]
        strikes = fault.getStrikes()[top_edge_tri_sort]
        dips = fault.getDips()[top_edge_tri_sort]
        # Rotation axis_angle
        dips = np.deg2rad(xtick_angle_offset) + dips

        if lonlat is None:
            if 'all' in slip:
                ind_samp = np.arange(0, fault.slip.shape[0])
            elif 'top' in slip:
                ind_samp = top_edge_tri_sort
            slip_samp = fault.slip[ind_samp, :]
            if 'total' in slip:
                slip_norm = np.linalg.norm(slip_samp, axis=1)
            elif 'strikeslip' in slip:
                slip_norm = np.linalg.norm(slip_samp[:, 0][:, None], axis=1)
            elif 'dipslip' in slip:
                slip_norm = np.linalg.norm(slip_samp[:, 1][:, None], axis=1)
            value = slip_norm
            x_samp, y_samp = centers[ind_samp, 0], centers[ind_samp, 1]
            lon_samp, lat_samp = fault.xy2ll(x_samp, y_samp)
            lonlat = np.vstack((lon_samp, lat_samp)).T


        # 断层迹线离散化，目的为了寻找临近点，将数据坐标投影到走向曲线上
        xf, yf = self.edge_vertices[side][:, 0], self.edge_vertices[side][:, 1]
        fault.trace(xf, yf, utm=True) # Keep the trace direction is from left to right
        fault.discretize_trace(every=discretizeInterval, threshold=discretizeInterval/4.0)
        # 构建KDTree用于最近邻搜索
        xy_trace = np.vstack((fault.xi, fault.yi)).T
        tree = KDTree(xy_trace)
        # 将输入坐标投影到断层迹线上
        xorg, yorg = fault.ll2xy(lonlat[:, 0], lonlat[:, 1])
        xy = np.vstack((xorg, yorg)).T
        distances, ind_dist = tree.query(xy)
        # Input (lonlat, value) is transferred to (distc, value)
        dis_trace = fault.cumdistance(discretized=True)
        distc = dis_trace[ind_dist]

        if value is None:
            value = np.ones_like(lonlat[:, 0])

        # 不做统计，直接按原始数据输出
        if not doStat:
            if KeepOriginCoord:
                xi, yi = xorg, yorg
            else:
                xi, yi = fault.xi[ind_dist], fault.yi[ind_dist]
            zi = np.ones_like(xi)*fault.top
            # 用于提取走向角和倾角的判断坐标，即点中心坐标
            x_dist = np.repeat(distc, 3)
            val_max = value.max()
            value = value/val_max if norm_ref is None else value/norm_ref
            print('Maximum value is: {0:.3f}'.format(val_max))
            print('The norm ref value of the original data is: ', val_max if norm_ref is None else norm_ref)
            x_angle = np.repeat(xi, 3)
            value = np.repeat(value, 3)
            xi = np.repeat(xi, 3)
            yi = np.repeat(yi, 3)
            zi = np.repeat(zi, 3)
        # 做统计，cutmehotd == 'pdcut' or 'hist'
        else:
            # Bug: hist只进行计数统计
            if cutmethod == 'hist':
                acums, adists = np.histogram(distc, bins=bins)
                val_max = np.nanmax(acums) 
                val_norm = acums / val_max if norm_ref is None else acums / norm_ref
                print('Maximum value is: {0:.3f}'.format(val_max))
                print('The norm ref value of the histogram is: ', val_max if norm_ref is None else norm_ref)
                # 将bin的左右和中点坐标均保留
                xl = adists[:-1]
                xr = adists[1:]
                xc = (xl + xr) / 2.0
            elif cutmethod == 'pdcut':
                zcut = pd.cut(distc, np.arange(distc.min(), distc.max() + 0.1, hinterval))  # ind.mid ind.left ind.right
                data = pd.DataFrame(value, columns=['value'])
                if method == 'mean':
                    val_stk = data.groupby(zcut, observed=False).mean()
                elif method == 'sum':
                    val_stk = data.groupby(zcut, observed=False).sum()
                val_max = val_stk.max().iloc[0]
                val_norm = val_stk / val_max if norm_ref is None else val_stk / norm_ref
                print('Maximum value is: {0:.3f}'.format(val_max))
                print('The norm ref value of the pdcut is: ', val_max if norm_ref is None else norm_ref)
                val_norm = val_norm.value.values
                # 坐标提取
                ind = pd.IntervalIndex(val_stk.index)
                xl, xc, xr = ind.left.values, ind.mid.values, ind.right.values
            # 提取左、中、右坐标和中心点
            # print(fault.xi.shape, fault.yi.shape)
            x_dist = np.vstack((xl, xc, xr)).T.flatten()
            # print(x_dist, dis_trace)
            inds = np.searchsorted(dis_trace, x_dist)
            inds = np.clip(inds, 0, len(fault.xi) - 1)
            xi, yi = fault.xi[inds], fault.yi[inds]
            zi = np.ones_like(xi) * fault.top
            x_angle = np.repeat(xi[1::3], 3)
            value = np.repeat(val_norm, 3)
        
        if output_2d:
            base, ext = os.path.splitext(outfile)
            outfile_2d = base + '_2d' + ext
            with open(os.path.join(outdir, outfile_2d), 'wt') as fout:
                print('# Maximum value is: {0:.3f}'.format(val_max), file=fout)
                print('# Norm ref value is: {0:.3f}'.format(norm_ref if norm_ref is not None else val_max), file=fout)
                if statkind == 'hist':
                    for i in range(int(value.shape[0]/3.0)):
                        print('>', file=fout)
                        print('{0:.3f} {1:.3f}'.format(x_dist[3*i], 0), file=fout)
                        print('{0:.3f} {1:.3f}'.format(x_dist[3*i+2], 0), file=fout)
                        print('{0:.3f} {1:.3f}'.format(x_dist[3*i+2], value[3*i+2]), file=fout)
                        print('{0:.3f} {1:.3f}'.format(x_dist[3*i], value[3*i]), file=fout)
                elif statkind == 'curve':
                    print('>', file=fout)
                    for i in range(int(value.shape[0]/3.0)):
                        print('{0:.3f} {1:.3f}'.format(x_dist[3*i+1], value[3*i+1]), file=fout)
                elif not doStat:
                    print('>', file=fout)
                    for i in range(int(x_angle.shape[0]/3.0)):
                        print('{0:.3f} {1:.3f}'.format(x_dist[3*i+1], value[3*i+1]), file=fout)
            
        if statkind == 'curve' or statkind == 'bar':
            x_angle = x_angle[1::3]
            xi = xi[1::3]
            yi = yi[1::3]
            zi = zi[1::3]
            value = value[1::3]
        elif statkind == 'hist':
            x_angle = np.vstack((x_angle[0::3], x_angle[2::3])).T.flatten()
            xi = np.vstack((xi[0::3], xi[2::3])).T.flatten()
            yi = np.vstack((yi[0::3], yi[2::3])).T.flatten()
            zi = np.vstack((zi[0::3], zi[2::3])).T.flatten()
            value = np.vstack((value[0::3], value[2::3])).T.flatten()
        
        # 获得对应走向角和倾角
        ind_angle = np.searchsorted(top_edge_centers[:, 0], x_angle)
        ind_angle = np.clip(ind_angle, 0, len(top_edge_centers) - 1)
        strikes = strikes[ind_angle]
        dips = dips[ind_angle]
        
        top_x, top_z = value * np.cos(dips) * height_scale, value * np.sin(dips) * height_scale
        verts = np.vstack((xi, yi, zi)).T
        trans_verts = (verts[:, 0] + verts[:, 1] * 1.j) * np.exp(1.j * strikes)
        
        # All Done
        return top_x, top_z, verts, trans_verts, strikes, dips

    def StatinTop(self, value=None, lonlat=None, hinterval=2.0, slip='total_top', bins=15, depth_eps=0.25, doStat=True,
                      zoffset=0.2, height_scale=1.0, xtick_angle_offset=0, method='mean', cutmethod='pdcut', statkind='hist',
                      outfile='stat_histInTop.gmt', discretizeInterval=0.2, output_2d=True, norm_ref=None):
        '''
        statkind :
            * hist
            * bar
            * curve

        Bug: 如果断层反倾得话，xtick_angle_offset会导致想不同方向偏转
        '''
        outdir = self.outdir
        fault = self.source
        side = 'top'

        top_x, top_z, verts, trans_verts, strikes, dips = self._statinTop(value=value, lonlat=lonlat, hinterval=hinterval, 
                    slip=slip, statkind=statkind, height_scale=height_scale, bins=bins, method=method, cutmethod=cutmethod, 
                    discretizeInterval=discretizeInterval, depth_eps=depth_eps, xtick_angle_offset=xtick_angle_offset, doStat=doStat, 
                    output_2d=output_2d, outdir=outdir, outfile=outfile, norm_ref=norm_ref)

        # 起始位置偏移项
        step0 = zoffset
        step0_x, step0_z = step0*np.cos(dips), step0*np.sin(dips)

        # 底部顶点
        trans_verts -= step0_x
        z = verts[:, -1] - step0_z
        xy = trans_verts*np.exp(-1.j*strikes)
        x0, y0 = xy.real, xy.imag
        lon, lat = fault.xy2ll(x0, y0)
        lonlatz1 = np.vstack((lon, lat, z)).T

        # 顶部顶点
        trans_verts -= top_x
        z -= top_z
        xy = trans_verts*np.exp(-1.j*strikes)
        x0, y0 = xy.real, xy.imag
        lon, lat = fault.xy2ll(x0, y0)
        lonlatz2 = np.vstack((lon, lat, z)).T

        if statkind == 'bar':
            with open(os.path.join(outdir, outfile), 'wt') as fout:
                for i in range(lonlatz1.shape[0]):
                    print('>', file=fout)
                    st1 = lonlatz1[i, :]
                    st3 = lonlatz2[i, :]
                    for st in [st1, st3]:
                        print('{0:.3f} {1:.3f} {2:.3f}'.format(*st), file=fout)
        elif statkind == 'curve':
            with open(os.path.join(outdir, outfile), 'wt') as fout:
                print('>', file=fout)
                for i in range(lonlatz2.shape[0]):
                    st = lonlatz2[i, :]
                    print('{0:.3f} {1:.3f} {2:.3f}'.format(*st), file=fout)
        elif statkind == 'hist':
            with open(os.path.join(outdir, outfile), 'wt') as fout:
                for i in range(int(lon.shape[0]/2.0)):
                    print('>', file=fout)
                    st1, st2 = lonlatz1[2*i:2*i+2]
                    st4, st3 = lonlatz2[2*i:2*i+2]
                    for st in [st1, st2, st3, st4]:
                        print('{0:.3f} {1:.3f} {2:.3f}'.format(*st), file=fout)
        
        # All Done 
        return

    def StatinDepth(self, slip='total', bins=None, depth_interval=None, 
                    depth_min=None, depth_max=None, depth_edges=None, method='mean', 
                    normalize=False, norm_method='max', normalize_stats=False,
                    auto_depth_layers=False, layer_tolerance=0.5, min_layer_thickness=1.0,
                    clustering_method='simple_clustering', max_layers=20,
                    plot=True, plot_curve=True, plot_patch_slip=True, outfile=None, 
                    outfile_curve=None, figsize=(4, 5), figname=None):
        """
        Convenience function for depth statistics analysis with auto layer detection
        
        Parameters:
        ----------
        slip : str or array
            Slip type ('total', 'strike', 'dip') or slip array
        bins : int, optional
            Number of depth bins (ignored if depth_interval or depth_edges is specified)
        depth_interval : float, optional
            Depth interval size in km (ignored if depth_edges is specified)
        depth_min, depth_max : float, optional
            Depth range in km, auto-calculated if None (ignored if depth_edges is specified)
        depth_edges : array-like, optional
            Custom depth bin edges in km, e.g., [0, 2, 5, 9, 13, 18, 25]
            Takes priority over all other binning parameters
        method : str
            Statistical method ('mean', 'max', 'median')
        normalize : bool, optional
            Whether to normalize the slip values before statistics (default: False)
        norm_method : str or float, optional
            Normalization method ('max', 'sum', 'l2', 'percentile', or custom value)
        normalize_stats : bool, optional
            Whether to normalize the statistical results after computation (default: False)
        auto_depth_layers : bool, optional
            Whether to automatically detect depth layers from vertices (default: False)
        layer_tolerance : float, optional
            Tolerance for grouping vertices into layers in km (default: 0.5)
        min_layer_thickness : float, optional
            Minimum thickness between layers in km (default: 1.0)
        clustering_method : str, optional
            Method for clustering depths ('histogram', 'kmeans', 'density') (default: 'histogram')
        max_layers : int, optional
            Maximum number of layers to detect (default: 20)
        plot : bool, optional
            Whether to plot the results (default: True)
        plot_curve : bool, optional
            Whether to plot statistical curves (default: True)
        plot_patch_slip : bool, optional
            Whether to plot individual patch slips (default: True)
        outfile : str, optional
            Output file for histogram data (default: None)
        outfile_curve : str, optional
            Output file for curve data (default: None)
        figsize : tuple, optional
            Figure size (default: (4, 5))
        figname : str, optional
            Figure name for saving plots (default: None)
    
        Returns:
        --------
        depth_centers, stat_values, std_values : arrays
            Statistical results
        """
        from .stat_utils import DepthStatistics
        source = self.source
        analyzer = DepthStatistics(source)
        
        # Compute statistics with all parameters including auto layer detection
        results = analyzer.compute_statistics(
            slip=slip, 
            bins=bins, 
            depth_interval=depth_interval,
            depth_min=depth_min, 
            depth_max=depth_max, 
            depth_edges=depth_edges, 
            method=method,
            normalize=normalize, 
            norm_method=norm_method,
            normalize_stats=normalize_stats,
            auto_depth_layers=auto_depth_layers,
            layer_tolerance=layer_tolerance,
            min_layer_thickness=min_layer_thickness,
            clustering_method=clustering_method,
            max_layers=max_layers
        )
        
        # Print auto-detection results if used
        if auto_depth_layers and results['params']['auto_depth_layers']:
            print(f"Auto-detected {results['params']['bins']} depth layers using {clustering_method} method")
            if 'custom_depth_edges' in results['params']:
                print(f"Detected layer boundaries: {results['params']['custom_depth_edges']}")
                print(f"Layer thicknesses: {results['params']['bin_widths']}")
        
        if plot:
            fig, axes = analyzer.plot_depth_histogram(results, plot_curve=plot_curve, 
                                                      plot_patch_slip=plot_patch_slip, 
                                                      figsize=figsize)
            
            # Generate appropriate filename
            filename_parts = [method, source.name]
            if auto_depth_layers:
                filename_parts.append(f'auto_{clustering_method}')
            if normalize_stats:
                filename_parts.append('norm')
            
            figname = figname or f'slip_depth_histogram_{"_".join(filename_parts)}.png'
            
            plt.savefig(figname, dpi=300, bbox_inches='tight')
            plt.show()
        
        # Save outputs with informative filenames if auto-detection was used
        if auto_depth_layers and (outfile is None or outfile_curve is None):
            base_name = f'{source.name}_{method}'
            if auto_depth_layers:
                base_name += f'_auto_{clustering_method}'
            
            if outfile is None:
                outfile = f'{base_name}_depth_hist.gmt'
            if outfile_curve is None:
                outfile_curve = f'{base_name}_depth_curve.dat'
        
        analyzer.save_outputs(results, outfile=outfile, outfile_curve=outfile_curve)
        analyzer.print_summary(results)
        
        return results['depth_centers'], results['stat_values'], results['std_values']

    def calculate_cumulative_distance(self, points):
        """
        Calculate the cumulative distance along the points.

        Parameters:
        points (numpy.ndarray): An array of shape (N, 2) representing the points.

        Returns:
        cumulative_distance (numpy.ndarray): An array of shape (N,) representing the cumulative distance.
        """
        distances = np.sqrt(np.diff(points[:, 0])**2 + np.diff(points[:, 1])**2)
        cumulative_distance = np.insert(np.cumsum(distances), 0, 0)
        return cumulative_distance

    def smooth_points(self, points, sigma=1):
        """
        Smooth the points using Gaussian filter based on cumulative distance.

        Parameters:
        points (numpy.ndarray): An array of shape (N, 2) representing the points.
        sigma (float): The standard deviation for Gaussian kernel.

        Returns:
        smoothed_points (numpy.ndarray): An array of shape (N, 2) representing the smoothed points.
        """
        cumulative_distance = self.calculate_cumulative_distance(points)
        smoothed_x = gaussian_filter1d(points[:, 0], sigma=sigma, mode='nearest')
        smoothed_y = gaussian_filter1d(points[:, 1], sigma=sigma, mode='nearest')
        smoothed_points = np.column_stack((smoothed_x, smoothed_y))
        return smoothed_points

    def calculate_gradient_and_curvature(self, points, smooth=False, sigma=1):
        """
        Calculate the gradient and curvature of a set of points.

        Parameters:
        points (numpy.ndarray): An array of shape (N, 2) representing the points.
        smooth (bool): Whether to smooth the points before calculation.
        sigma (float): The standard deviation for Gaussian kernel if smoothing is applied.

        Returns:
        gradients (numpy.ndarray): An array of shape (N-1,) representing the gradients.
        curvatures (numpy.ndarray): An array of shape (N-2,) representing the curvatures.
        """
        if smooth:
            points = self.smooth_points(points, sigma=sigma)
        
        # Calculate differences between consecutive points
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])
        
        # Calculate gradients
        gradients = dy / dx
        
        # Calculate second differences
        ddx = np.diff(dx)
        ddy = np.diff(dy)
        
        # Calculate curvatures
        curvatures = (ddy * dx[:-1] - ddx * dy[:-1]) / (dx[:-1]**2 + dy[:-1]**2)**1.5 # np.abs
        
        return gradients, curvatures

    def normalize(self, values):
        """
        Normalize the values to the range [0, 1].

        Parameters:
        values (numpy.ndarray): An array of values to be normalized.

        Returns:
        normalized_values (numpy.ndarray): An array of normalized values.
        """
        return (values - np.min(values)) / (np.max(values) - np.min(values))

    def generate_gradient_and_curvature_statistics(self, points='top', smooth=False, sigma=1, 
                                                   output_gradient=False, output_curvature=True, 
                                                   gradient_outfile='gradient.gmt', 
                                                   curvature_outfile='curvature.gmt',
                                                   slip='total_all', bins=15, hinterval=6.0, depth_eps=0.25, 
                                                   xtick_angle_offset=0, zoffset=0.2, height_scale=8.0, 
                                                   method='mean', cutmethod='pdcut', statkind='curve'):
        """
        Generate gradient and curvature statistics and plot.
    
        Parameters:
        * points (numpy.ndarray or str): An array of shape (N, 2) representing the points, or 'top' or 'bottom'.
        * smooth (bool): Whether to smooth the points before calculation.
        * sigma (float): The standard deviation for Gaussian kernel if smoothing is applied.
        * output_gradient (bool): Whether to output gradient statistics (default is False).
        * output_curvature (bool): Whether to output curvature statistics (default is True).
        * gradient_outfile (str): Output file name for gradient statistics (default is 'gradient.gmt').
        * curvature_outfile (str): Output file name for curvature statistics (default is 'curvature.gmt').
        * slip (str): Slip type for statistics (default is 'total_all').
        * bins (int): Number of bins for statistics (default is 15).
        * hinterval (float): Horizontal interval for statistics (default is 6.0).
        * depth_eps (float): Depth epsilon for statistics (default is 0.25).
        * xtick_angle_offset (float): Angle offset for the x-ticks relative to the dip angle (default is 0).
        * zoffset (float): Offset for the z-axis (default is 0.2).
        * height_scale (float): Scale for the height (default is 8.0).
        * method (str): Method for statistics ('mean' or 'sum', default is 'mean').
        * cutmethod (str): Method for cutting ('pdcut' or other, default is 'pdcut').
        * statkind (str): Kind of statistics ('curve' or 'hist', default is 'curve').
        """
        self.getfaultEdgeTriangles_and_EdgeLines()
        
        # Plotting
        if not isinstance(points, np.ndarray):
            if points == 'top':
                points = self.edge_vertices['top']
            elif points == 'bottom':
                points = self.edge_vertices['bottom']
        gradients, curvatures = self.calculate_gradient_and_curvature(points, smooth=smooth, sigma=sigma)
        
        # Normalize gradients and curvatures
        # normalized_gradients = self.normalize(gradients)
        # normalized_curvatures = self.normalize(curvatures)
        
        # Use appropriate coordinates for gradients and curvatures
        x_coords_gradients = (points[:-1, 0] + points[1:, 0]) / 2  # Use midpoints for gradients
        y_coords_gradients = (points[:-1, 1] + points[1:, 1]) / 2  # Use midpoints for gradients
        lon_gradients, lat_gradients = self.source.xy2ll(x_coords_gradients, y_coords_gradients)
        lonlat_gradients = np.column_stack((lon_gradients, lat_gradients))
        
        x_coords_curvatures = (points[:-2, 0] + points[2:, 0]) / 2  # Use midpoints for curvatures
        y_coords_curvatures = (points[:-2, 1] + points[2:, 1]) / 2  # Use midpoints for curvatures
        lon_curvatures, lat_curvatures = self.source.xy2ll(x_coords_curvatures, y_coords_curvatures)
        lonlat_curvatures = np.column_stack((lon_curvatures, lat_curvatures))
    
        # Output gradient statistics if requested
        if output_gradient:
            self.StatinTop(value=gradients, lonlat=lonlat_gradients, slip=slip, bins=bins, hinterval=hinterval, depth_eps=depth_eps, 
                           xtick_angle_offset=xtick_angle_offset, zoffset=zoffset, height_scale=height_scale, 
                           outfile=gradient_outfile, method=method, cutmethod=cutmethod, doStat=False, statkind=statkind)
    
        # Output curvature statistics if requested
        if output_curvature:
            self.StatinTop(value=np.abs(curvatures), lonlat=lonlat_curvatures, slip=slip, bins=bins, hinterval=hinterval, depth_eps=depth_eps, 
                           xtick_angle_offset=xtick_angle_offset, zoffset=zoffset, height_scale=height_scale, 
                           outfile=curvature_outfile, method=method, cutmethod=cutmethod, doStat=False, statkind=statkind)
            basename = os.path.splitext(curvature_outfile)[0]
            curvature_value_outfile = basename + '_value.gmt'
            with open(os.path.join(self.outdir, curvature_value_outfile), 'wt') as fout:
                for i in range(len(curvatures)):
                    print('{0:.5g}'.format(curvatures[i]), file=fout)


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    from collections import OrderedDict

    # -----------------------------------Proj Information-------------------------------------#
    # center for local coordinates--M7.1 epicenter 
    lon0, lat0 = 101.31, 37.80

    # -------------------------Generate Triangular Fault Object-------------------------------#
    # Case 2:
    source = TriangularPatches('Menyuan_main', lon0=lon0, lat0=lat0)
    slipfile = r'slip_total_0.gmt'
    source.readPatchesFromFile(slipfile)
    # 设置地表迹线
    trace = os.path.join('..', 'Fault_Trace_Menyuan_Yanghongfeng_scale.txt')
    trace = pd.read_csv(trace, names=['lon', 'lat'], sep=r'\s+', comment='#')
    source.trace(trace.lon.values, trace.lat.values)

    statobj = StatisticsInFault('statinfault', lon0=lon0, lat0=lat0, source=source)
    statobj.getfaultEdgeTriangles_and_EdgeLines()

    lonlat = statobj.getSideEdgeLine(plot=False, step=0.5, yaxis_ticks=[0, 5, 10, 15], horz_angle=120, tick_scale=0.5)
    # 为了让统计信息沿断层面，xtick_angle_offset通常不要设置，即不旋转倾向角
    statobj.genXaxis(xaxis_ticks=[0.8, 4.2], tick_scale=1.0, horz_angle=120, xtick_angle_offset=0, xaxis_zoffset=0.7)

    data = pd.read_csv(r'd:\2022Menyuan\2022Menyuan\RelocatedAftershocks\FanLiping\Proj2Fault\seis_reloc_proj.gmt', sep=r'\s+')
    statobj.StatInSide(depth=data.dep.values, bins=15, side='right', statkind='hist', cutmethod='hist', outfile='stat_hist.gmt')

    statobj.StatInSide(slip='strikeslip', bins=15, side='right', method='mean', statkind='curve', zinterval=1.5,
                       cutmethod='pdcut', outfile='curve_statInSide.gmt')

    # Statistics in Top
    statobj.StatinTop(slip='strikeslip_top', bins=15, hinterval=1.0, depth_eps=0.25, xtick_angle_offset=0,
                      zoffset=0.2, height_scale=4.0, outfile='stat_histInTop.gmt', method='sum', cutmethod='pdcut')

    # 现场勘查
    slp_file = r'c:\Users\kfhe\Desktop\Menyuan_Surface\surfslip_panjiawei.csv'
    slp_pan = pd.read_csv(slp_file, sep=r'\s+')
    statobj.StatinTop(slp_pan.slip.values, slp_pan.iloc[:, 2:].values, method='mean', depth_eps=0.25, cutmethod=None, doStat=False,
                      zoffset=0.2, height_scale=4.0, xtick_angle_offset=0, outfile='bar_statInTop.gmt', statkind='bar')