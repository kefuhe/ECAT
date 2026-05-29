import numpy as np
from eqtools.csiExtend.multifaultsolve_boundLSE import multifaultsolve_boundLSE as multifaultsolve
from csi import (gps, gpstimeseries, 
                 insar, insartimeseries)
import os
import sys
# using the C++ backend
os.environ['CUTDE_USE_BACKEND'] = 'cpp' # cuda, cpp, or opencl

import pandas as pd
import matplotlib.pyplot as plt

def computeSlipDirection(self, scale=1.0, factor=1.0, ellipse=False,nsigma=1.):
    '''
    Computes the segment indicating the slip direction.

    Kwargs:
        * scale     : can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
        * factor    : Multiply by a factor
        * ellipse   : Compute the ellipse
        * nsigma    : How many times sigma for the ellipse

    Returns:
        * None
    '''

    # Create the array
    self.slipdirection = []

    # Check Cm if ellipse
    if ellipse:
        self.ellipse = []
        assert(self.Cm!=None), 'Provide Cm values'

    # Loop over the patches
    if self.N_slip == None:
        self.N_slip = len(self.patch)
    for p in range(self.N_slip):
        # Get some geometry
        xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)
        strike = strike - np.pi if strike > np.pi else strike
        # Get the slip vector
        slip = self.slip[p,:]
        rake = np.arctan2(slip[1],slip[0])

        # Compute the vector
        #x = np.sin(strike)*np.cos(rake) + np.sin(strike)*np.cos(dip)*np.sin(rake)
        #y = np.cos(strike)*np.cos(rake) - np.cos(strike)*np.cos(dip)*np.sin(rake)
        #z = -1.0*np.sin(dip)*np.sin(rake)
        x = (np.sin(strike)*np.cos(rake) - np.cos(strike)*np.cos(dip)*np.sin(rake))
        y = (np.cos(strike)*np.cos(rake) + np.sin(strike)*np.cos(dip)*np.sin(rake))
        z =  1.0*np.sin(dip)*np.sin(rake)

        # Scale these
        if scale.__class__ is float:
            sca = scale
        elif scale.__class__ is str:
            if scale in ('total'):
                sca = np.sqrt(slip[0]**2 + slip[1]**2 + slip[2]**2)*factor
            elif scale in ('strikeslip'):
                sca = slip[0]*factor
            elif scale in ('dipslip'):
                sca = slip[1]*factor
            elif scale in ('tensile'):
                sca = slip[2]*factor
            else:
                print('Unknown Slip Direction in computeSlipDirection')
                sys.exit(1)
        x *= sca
        y *= sca
        z *= sca

        # update point
        xe = xc + x
        ye = yc + y
        ze = zc + z

        # Append ellipse
        if ellipse:
            self.ellipse.append(self.getEllipse(p,ellipseCenter=[xe, ye, ze],factor=factor,nsigma=nsigma))

        # Append slip direction
        self.slipdirection.append([[xc, yc, zc],[xe, ye, ze]])

    # All done
    return


def writeSlipDirection2File(self, filename, scale=1.0, factor=1.0,
                            neg_depth=False, ellipse=False,nsigma=1.):
    '''
    Write a psxyz compatible file to draw lines starting from the center 
    of each patch, indicating the direction of slip. Scale can be a real 
    number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'

    Args:
        * filename      : Name of the output file

    Kwargs:
        * scale         : Scale of the line
        * factor        : Multiply slip by a factor
        * neg_depth     : if True, depth is a negative nmber
        * ellipse       : Write the ellipse
        * nsigma        : Nxsigma for the ellipse

    Returns:
        * None
    '''

    # Copmute the slip direction
    computeSlipDirection(self, scale=scale, factor=factor, ellipse=ellipse,nsigma=nsigma)

    # Write something
    print('Writing slip direction to file {}'.format(filename))

    # Open the file
    fout = open(filename, 'w')

    # Loop over the patches
    for p in self.slipdirection:

        # Write the > sign to the file
        fout.write('> \n')

        # Get the center of the patch
        xc, yc, zc = p[0]
        lonc, latc = self.xy2ll(xc, yc)
        if neg_depth:
            zc = -1.0*zc
        fout.write('{} {} {} \n'.format(lonc, latc, zc))

        # Get the end of the vector
        xc, yc, zc = p[1]
        lonc, latc = self.xy2ll(xc, yc)
        if neg_depth:
            zc = -1.0*zc
        fout.write('{} {} {} \n'.format(lonc, latc, zc))

    # Close file
    fout.close()

    if ellipse:
        # Open the file
        fout = open('ellipse_'+filename, 'w')

        # Loop over the patches
        for e in self.ellipse:

            # Get ellipse points
            ex, ey, ez = e[:,0],e[:,1],e[:,2]

            # Depth
            if neg_depth:
                ez = -1.0 * ez

            # Conversion to geographical coordinates
            lone,late = self.putm(ex*1000.,ey*1000.,inverse=True)

            # Write the > sign to the file
            fout.write('> \n')

            for lon,lat,z in zip(lone,late,ez):
                fout.write('{} {} {} \n'.format(lon, lat, -1.*z))
        # Close file
        fout.close()

    # All done
    return


class FullTimeSeriesInversion(multifaultsolve):
    def __init__(self, name, faults, tau, decay_function=None, reference_time=None, insar_offset_correction=None, geodata=None, verticals=None):
        super(FullTimeSeriesInversion, self).__init__(name, faults)
        self.green_functions = []  # 存储添加的格林函数
        self.geodata = geodata or []
        self.verticals = verticals or []
        self.tau = tau
        self.set_decay_function(decay_function)
        self.initial_model = None
        self.reference_time = reference_time
        self.insar_offset_correction = insar_offset_correction or []
        self.accumulated_offset_correction_size = 0
        self.update_properties()

    def update_properties(self):
        self.offset_correction_sizes = [len(data.lon) for data in self.geodata if data.name in self.insar_offset_correction]
        self.offset_correction_positions = self.calculate_offset_correction_positions()
        self.total_offset_correction_size = sum(self.offset_correction_sizes)

    def calculate_offset_correction_positions(self):
        positions = {}
        start_position = 0
        for ifault in self.faults:
            start_position += len(ifault.slipdir) * ifault.Faces.shape[0]
            start_position += np.sum([npoly for npoly in ifault.poly.values() if npoly is not None], dtype=int)
        for data in self.geodata:
            if data.name in self.insar_offset_correction and data.dtype == 'insar':
                end_position = start_position + len(data.lon)
                positions[data.name] = (start_position, end_position)
                start_position = end_position
        return positions

    def set_decay_function(self, decay_function):
        self.decay_function = decay_function if decay_function else self._default_decay_function

    def set_initial_model(self, initial_model):
        self.initial_model = initial_model
    
    def get_green_function_positions(self):
        if not hasattr(self, 'green_function_positions'):
            self.green_function_positions = {}
            start_row_position = 0
            for data in self.geodata:
                if isinstance(data, gps) and hasattr(data, 'timeseries') and data.timeseries:
                    for station in data.station:
                        len_data = len(data.timeseries[station].east.value) * data.obs_per_station
                elif isinstance(data, insartimeseries) and 'timeseries' in data.dtype:
                    len_data = sum([len(isar.vel_los) for isar in data.timeseries])
                else:
                    len_data = self.faults[0].d[data.name].shape[0]  # 获取当前数据集的行数
                start_col_position = 0  # 重置列的起始位置
                for fault in self.faults:
                    len_cols = fault.Gassembled.shape[1]  # 获取当前数据集在当前fault对应的格林函数的列数
                    self.green_function_positions[(data.name, fault.name)] = (start_row_position, start_row_position + len_data, start_col_position, start_col_position + len_cols)
                    start_col_position += len_cols  # 更新列的起始位置
                start_row_position += len_data

        return self.green_function_positions

    def calculate_slip_and_poly_positions(self):
        self.slip_positions = {}
        self.poly_positions = {}
        start_position = 0
        for fault in self.faults:
            npatches = fault.Faces.shape[0]
            num_slip_samples = len(fault.slipdir)*npatches
            num_poly_samples = np.sum([npoly for npoly in fault.poly.values() if npoly is not None], dtype=int)
            self.slip_positions[fault.name] = (start_position, start_position + num_slip_samples)
            self.poly_positions[fault.name] = (start_position + num_slip_samples, start_position + num_slip_samples + num_poly_samples)
            start_position += num_slip_samples + num_poly_samples

    def calculate_sample_slip_only_positions(self):
        slip_only_positions = []
        for fault in self.faults:
            fault_name = fault.name
            slip_start, slip_end = self.slip_positions[fault_name]
            slip_only_positions.extend(list(range(slip_start, slip_end)))
        slip_only_positions = np.array(slip_only_positions)
        self.sample_slip_only_positions = slip_only_positions
        return slip_only_positions

    def print_parameter_positions(self):
        """打印每个参数的位置"""
        print("Parameter positions:")
        for fault in self.faults:
            print(f"  Slip positions: {self.slip_positions[fault.name]}")
            print(f"  Poly positions: {self.poly_positions[fault.name]}")

    def _default_decay_function(self, t, tau):
        return 1 - np.exp(-t/tau)

    def assemble_data(self):
        assembled_deformation = []
        assembled_time = []
        for data, vertical in zip(self.geodata, self.verticals):
            if isinstance(data, gps) and hasattr(data, 'timeseries') and data.timeseries:
                for station in data.station:
                    igpstimeseries = data.timeseries[station]
                    assembled_deformation.extend([igpstimeseries.east.value, igpstimeseries.north.value])
                    if vertical:
                        assembled_deformation.append(igpstimeseries.up.value)
                    assembled_time.extend([igpstimeseries.time]*data.obs_per_station)
            elif isinstance(data, insartimeseries) and 'timeseries' in data.dtype:
                for i, isar in enumerate(data.timeseries):
                    assembled_deformation.append(isar.vel_los)
                    assembled_time.append(np.full(isar.vel_los.shape, data.time[i]))
            else:
                # 如果数据没有时序特性，直接从self.faults[0].d中使用data.name来索引
                assembled_deformation.append(self.faults[0].d[data.name])
        self.d = np.hstack(assembled_deformation)
        self.time = np.hstack(assembled_time)

    def assemble_Cd(self):
        assembled_Cd = []
        for data, vertical in zip(self.geodata, self.verticals):
            if isinstance(data, gps) and hasattr(data, 'timeseries') and data.timeseries:
                for station in data.station:
                    igpstimeseries = data.timeseries[station]
                    assembled_Cd.extend([igpstimeseries.east.error**2, igpstimeseries.north.error**2])
                    if vertical:
                        assembled_Cd.append(igpstimeseries.up.error**2)
            elif isinstance(data, insartimeseries) and 'timeseries' in data.dtype:
                for isar in data.timeseries:
                    assembled_Cd.append(isar.err**2)
            else:
                # 如果数据没有时序特性，直接从self.faults[0].Cd中使用data.name来索引
                assembled_Cd.append(self.faults[0].Cd[data.name])
        self.Cd = np.diag(np.hstack(assembled_Cd))

    def assemble_G(self):
        if self.reference_time is None:
            raise ValueError("Reference time is not set, please set it first.")
        Gassembled = []
        st_row = 0
        ed_row = 0
        # self.data_indices = {}  # 创建一个新的字典来保存每个数据集在self.G中的行的起止索引
        for i, data in enumerate(self.geodata):
            if isinstance(data, gps) and hasattr(data, 'timeseries') and data.timeseries:
                ed_row += self.faults[0].d[data.name].shape[0]
                if self.green_functions is not None and i < len(self.green_functions):
                    Gi = self.green_functions[i]
                else:
                    Gi = self.G[st_row:ed_row, :]
                st_row = ed_row
                # 提取与站点三个方向分量对应的行
                if data.obs_per_station == 3:
                    Ge = Gi[:Gi.shape[0]//3, :]
                    Gn = Gi[Gi.shape[0]//3:2*Gi.shape[0]//3, :]
                    Gu = Gi[2*Gi.shape[0]//3:, :]
                else:
                    Ge = Gi[:Gi.shape[0]//2, :]
                    Gn = Gi[Gi.shape[0]//2:, :]
                for irow, station in enumerate(data.station):
                    igpsts = data.timeseries[station]
                    # 预先计算衰减函数
                    decay_func = np.array([self.decay_function((t - self.reference_time).days, self.tau) for t in igpsts.time])
                    # 按照时间序列的顺序，为每个时间点装配对应的格林函数
                    Ge_temp = Ge[irow, :] * decay_func[:, np.newaxis]
                    Gn_temp = Gn[irow, :] * decay_func[:, np.newaxis]
                    # 如果gps对象有垂直方向的数据，也处理垂直方向的数据
                    if data.obs_per_station == 3:
                        Gu_temp = Gu[irow, :] * decay_func[:, np.newaxis]
                    # 如果有InSAR数据需要添加初始偏移矫正，为GPS数据添加相应的零列
                    if self.insar_offset_correction:
                        zeros = np.zeros((Ge_temp.shape[0], self.total_offset_correction_size))
                        Ge_temp = np.hstack([Ge_temp, zeros])
                        Gn_temp = np.hstack([Gn_temp, zeros])
                        if data.obs_per_station == 3:
                            Gu_temp = np.hstack([Gu_temp, zeros])
                    Gassembled.extend([Ge_temp, Gn_temp])
                    if data.obs_per_station == 3:
                        Gassembled.extend([Gu_temp])
            elif isinstance(data, insartimeseries):
                ed_row += self.faults[0].d[data.name].shape[0]
                if self.green_functions is not None and i < len(self.green_functions):
                    Gi = self.green_functions[i]
                else:
                    Gi = self.G[st_row:ed_row, :]
                st_row = ed_row
                for isar in data.timeseries:
                    decay_func = np.array([self.decay_function((t - self.reference_time).days, self.tau) for t in data.time])
                    G_temp = Gi * decay_func[:, np.newaxis]
                    # 如果当前的InSAR数据需要添加初始偏移矫正，添加一个单位阵作为新的列
                    if data.name in self.insar_offset_correction:
                        zeros_left = np.zeros((G_temp.shape[0], self.accumulated_offset_correction_size))
                        zeros_right = np.zeros((G_temp.shape[0], self.total_offset_correction_size - self.accumulated_offset_correction_size - len(data.lon)))
                        G_temp = np.hstack([zeros_left, np.eye(G_temp.shape[0]), zeros_right])
                        self.accumulated_offset_correction_size += len(data.lon)
                    Gassembled.extend([G_temp])
            else:
                ed_row += self.faults[0].d[data.name].shape[0]
                # 如果数据没有时序属性，直接将Gi添加到Gassembled
                if self.green_functions is not None and i < len(self.green_functions):
                    Gi = self.green_functions[i]
                else:
                    Gi = self.G[st_row:ed_row, :]
                st_row = ed_row
                # 在最右侧添加与insartimeseries相关的偏移部分的零矩阵
                if self.insar_offset_correction:
                    zeros = np.zeros((Gi.shape[0], self.total_offset_correction_size))
                    Gi = np.hstack([Gi, zeros])
                Gassembled.append(Gi)
        self.G = np.vstack(Gassembled)  # self.G用于存储装配后的格林函数

    def set_green_function_to_zero(self, data_name, fault_names):
        """
        设置某些断层对应的格林函数为0。

        参数:
        data_name: str, 数据的名称
        fault_names: list of str, 需要设置为0的断层的名字列表
        """
        # 检查green_function_positions是否存在
        if not hasattr(self, 'green_function_positions'):
            raise AttributeError("The object does not have attribute 'green_function_positions'. Please call 'get_green_function_positions' first.")

        # 设置指定断层的格林函数为0
        for fault_name in fault_names:
            st_row, ed_row, st_col, ed_col = self.green_function_positions[(data_name, fault_name)]
            print(f"Setting green function to zero for data {data_name} and fault {fault_name} at rows {st_row}-{ed_row} and columns {st_col}-{ed_col}")
            self.G[st_row:ed_row, st_col:ed_col] = 0

    def forward_model(self):
        # 正演生成合成数据
        synth = np.dot(self.G, self.mpost)

        # 分配合成数据到每个时序数据的synth属性
        index = 0
        for data in self.geodata:
            if isinstance(data, gps):
                if hasattr(data, 'timeseries') and data.timeseries:
                    for station in data.station:
                        igpstimeseries = data.timeseries[station]
                        n = len(igpstimeseries.east.value)
                        directions = [igpstimeseries.east, igpstimeseries.north, igpstimeseries.up]
                        for i in range(data.obs_per_station):
                            directions[i].synth = synth[index:index+n]
                            index += n
                        if data.obs_per_station < 3:
                            directions[2].synth = np.zeros_like(igpstimeseries.east.synth)
                else:
                    nsta = len(data.lon)
                    data.synth = np.zeros((nsta, 3))
                    for i in range(data.obs_per_station):
                        data.synth[:, i] = synth[index:index+nsta]
                        index += nsta
            elif isinstance(data, insartimeseries):
                for isar in data.timeseries:
                    n = len(isar.vel_los)
                    isar.synth = synth[index:index+n]
                    index += n
            elif isinstance(data, insar):
                n = len(data.vel_los)
                data.synth = synth[index:index+n]
                index += n

    def compute_residuals(self):
        for data in self.geodata:
            if isinstance(data, gps):
                if hasattr(data, 'timeseries') and data.timeseries:
                    for station in data.station:
                        igpstimeseries = data.timeseries[station]
                        directions = [igpstimeseries.east, igpstimeseries.north, igpstimeseries.up]
                        for i in range(data.obs_per_station):
                            if not hasattr(directions[i], 'synth'):
                                self.forward_model()
                            directions[i].res = directions[i].value - directions[i].synth
                        if data.obs_per_station < 3:
                            directions[2].res = np.zeros_like(igpstimeseries.east.res)
                else:
                    data.res = np.zeros_like(data.vel_enu)
                    for i in range(data.obs_per_station):
                        data.res[:, i] = data.value[:, i] - data.synth[:, i]
            elif isinstance(data, insartimeseries):
                for isar in data.timeseries:
                    if not hasattr(isar, 'synth'):
                        self.forward_model()
                    isar.res = isar.vel_los - isar.synth
            elif isinstance(data, insar):
                if not hasattr(data, 'synth'):
                    self.forward_model()
                data.res = data.vel_los - data.synth

    def compute_squares(self):
        self.compute_residuals()  # 确保残差已经计算

        squares = []  # 用于存储所有残差的平方
        inv_cov_diag = np.diag(np.linalg.inv(self.Cd))  # 计算协方差矩阵的逆的对角线部分
        start = 0  # 用于记录当前数据的开始位置

        for data in self.geodata:
            if isinstance(data, gps):
                if hasattr(data, 'timeseries') and data.timeseries:
                    for station in data.station:
                        igpstimeseries = data.timeseries[station]
                        directions = [igpstimeseries.east, igpstimeseries.north, igpstimeseries.up]
                        for i in range(data.obs_per_station):
                            length = len(directions[i].res)  # 计算当前数据的长度
                            res = directions[i].res
                            squares.append(res * res * inv_cov_diag[start:start+length])
                            start += length  # 更新开始位置
                else:
                    for i in range(data.obs_per_station):
                        length = len(data.res[:, i])  # 计算当前数据的长度
                        res = data.res[:, i]
                        squares.append(res * res * inv_cov_diag[start:start+length])
                        start += length  # 更新开始位置
            elif isinstance(data, insartimeseries):
                for isar in data.timeseries:
                    length = len(isar.res)  # 计算当前数据的长度
                    res = isar.res
                    squares.append(res * res * inv_cov_diag[start:start+length])
                    start += length  # 更新开始位置
            elif isinstance(data, insar):
                length = len(data.res)  # 计算当前数据的长度
                res = data.res
                squares.append(res * res * inv_cov_diag[start:start+length])
                start += length  # 更新开始位置

        return squares

    def compute_total_rmse(self):
        squares = self.compute_squares()
        total_rmse = np.sqrt(np.mean(np.concatenate(squares)))  # 计算总的RMSE
        return total_rmse

    def compute_chi_square(self):
        squares = self.compute_squares()
        chi_square = np.sum(np.concatenate(squares))  # 计算卡方统计量
        return chi_square
    

if __name__ == '__main__':
    import os
    from collections import OrderedDict
    from csi import TriangularPatches
    # define gps
    stafile = r'cor_cGPS_bak.txt'
    gpsdata = gps('post', lon0=98.25, lat0=34.5)
    gpsdata.setStatFromFile(stafile, header=1, initVel=True)
    gpsdata.initializeTimeSeries(stationfile=True, suffix='_obs-cv')

    geodata = [gpsdata]

    # -------------------------Generate Triangular Fault Object-------------------------------#
    faultdir = r'e:\Maduo_Postseismic\Postseismic_Inversion'
    faultnode = os.path.join(faultdir, 'mesh', 'fault_nodes.inp')
    fault_dict = {
                    'Maduo_Main': os.path.join(faultdir, 'mesh', 'fault1.tri'),
                    'Maduo_Tip': os.path.join(faultdir, 'mesh', 'fault2.tri'),
                    }
    trifaults = OrderedDict()
    projstr_trelis = '+proj=utm +lon_0={0} +lat_0={1}'.format(98.25, 34.5) # '+proj=utm +zone={}'.format(utmzone)
    for faultname in fault_dict:
        trifault = TriangularPatches(faultname, lon0=98.25, lat0=34.5, ellps='WGS84', verbose=True)
        # read the patch from Abaqus file
        trifault.readPatchesFromAbaqus(faultnode, fault_dict[faultname], lon0=98.25, lat0=34.5,
                                       readpatchindex=True, projstr=projstr_trelis)
        #trifault.setTrace(0.01)
        trifaults[faultname] = trifault
    trifaults_list = [trifaults[faultname] for faultname in trifaults]
    nfaults = len(trifaults)

    for trifault in trifaults_list:
        trifault.setTrace(0.01)
        for obsdata in geodata:
            trifault.buildGFs(obsdata, vertical=False, slipdir='sd', method='cutde', verbose=True)

    # # ----------------------Assemble data and GreenFns for Inversion------------------------#
    for faultname in trifaults:
        trifault = trifaults[faultname]
        # assemble data
        trifault.assembled(geodata)

        # assemble GreensFns
        if faultname == 'Maduo_Tip':
            trifault.assembleGFs(geodata, polys=[None], slipdir='sd', verbose=True, custom=False)
        else:
            trifault.assembleGFs(geodata, polys=[None], slipdir='sd', verbose=True, custom=False)

    for gpsith in geodata:
        gpsith.buildCd(direction='en')
    # assemble data Covariance materices, You should assemble the Green's function matrix first
    for faultname in trifaults:
        trifault = trifaults[faultname]
        # 貌似verbose会导致打印信息格式不匹配问题
        trifault.assembleCd(geodata, verbose=False, add_prediction=None)

        # compute the areas of the tri-patches
        trifault.computeArea()

    # define multifault
    import pandas as pd
    eqtime = pd.Timestamp('2021-05-21T06:00:00')
    solver = FullTimeSeriesInversion('post', trifaults_list, tau=65.0, decay_function=None, 
                                     reference_time=eqtime, insar_offset_correction=None, geodata=geodata, verticals=[False])
    solver.assembleGFs()
    # assemble timeseries
    solver.assemble_data()
    solver.assemble_Cd()
    solver.Cd = np.eye(solver.d.shape[0])
    solver.assemble_G()

    # ----------------------Set the initial model------------------------#
    # # 设置rake角约束 
    solver.setrakebound([[-30.0, 30.0]])

    solver_info = []
    pws = [0.0001, 0.001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 5.0, 10.0, 50., 100., 1000.]
    for pw in [5.0]:
        # solving the m parameters for faults
        solver.ConstrainedLeastSquareSoln(penWeight=pw, method='Mudpy', lap_bounds=('locked', 'free', 'free', 'free'))
        solver_info.append([pw, solver.mpost.copy()])
        # distributes the m parameters to the faults
        solver.distributem(verbose=True)
        solver.forward_model()
        print('Total RMSE: {} and chi: {}'.format(solver.compute_total_rmse(), solver.compute_chi_square()))

    if len(trifaults) == 1:
        trifaults['Maduo_Main'].plot(drawCoastlines=False, plot_on_2d=False)
    else:
        # Compute the moment Magnitude of the faults
        trifault_combine = trifaults['Maduo_Main'].duplicateFault()
        for patch,slip in zip(trifaults['Maduo_Tip'].patch, trifaults['Maduo_Tip'].slip):
            trifault_combine.N_slip = trifault_combine.slip.shape[0] + 1
            trifault_combine.addpatch(patch, slip)
        trifault_combine.plot(drawCoastlines=False, plot_on_2d=False)
    
    solver.compute_residuals()
    gpsdata.writeTimeSeries(data_type='value', outdir='data')
    gpsdata.writeTimeSeries(data_type='synth', outdir='synth')
    gpsdata.writeTimeSeries(data_type='res', outdir='residuals')

    for sta in gpsdata.station:
        gpsdata.plot_gpstimeseries_at_site(sta, solver.reference_time, direction='EN', timeunit='D', figsize=(7.0, 1.8))
        plt.savefig(os.path.join('figs', '{0}.png'.format(sta)), dpi=300)
    plt.show()
    # ----------------------Output the results------------------------#
    slipoutdir = os.path.join('output_gpsts_data', 'output_30km')
    for i, trifault in enumerate(trifaults_list):
        trifault.slip /= 100.0
        trifault.writePatches2File(os.path.join(slipoutdir, 'slip_total_{0}_mudpy_30km.gmt'.format(i)), add_slip='total')
        trifault.writeSlipDirection2File(os.path.join(slipoutdir, 'slip_dir_{0}_mudpy_30km.gmt'.format(i)), neg_depth=False, scale='total', factor=10.0)
        # 倾向改变时，考虑用这个命令替代自带的输出命令
        writeSlipDirection2File(trifault, os.path.join(slipoutdir, 'slip_dir_{0}_mudpy_30km.gmt'.format(i)), neg_depth=False, scale='total', factor=10.0)