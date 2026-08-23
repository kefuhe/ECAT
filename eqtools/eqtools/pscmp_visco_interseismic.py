'''
Simple calculating codes about Earthquake cycle

Written by Kefeng He, January 2023
'''

import pandas as pd
from pandas import IndexSlice as idx
import numpy as np
import os
from glob import glob
import re

# 导入库 eqtools
from .pcmptools import gentsseq_pscmp, fmt_pscmp

# ----------------------Read PSCMP Timeseries-------------------------------------------#

def data4pscmpzG(geodata, outfile='sites4pscmp.dat'):
    '''
    用于提取pscmp特定站点信息
    
    Args    :
        * geodata    : csi库中数据类型列表， e.g., [gps1, gps2, insar1, multigps1]
    
    Kwargs  :
        * outfile    : 保存坐标信息的文本输出文件
    
    Comment :
        * np.loadtxt可用于加载数据， np.loadtxt(outfile).reshape()
    '''
    with open(outfile, 'wt') as fout:
        obslons, obslats = np.empty((0,)), np.empty((0,))
        for obsdata in geodata:
            print('#', obsdata.dtype, obsdata.name, obsdata.obs_per_station, obsdata.lon.shape[0])
            print('#', obsdata.dtype, obsdata.name, obsdata.obs_per_station, obsdata.lon.shape[0], file=fout)
            obslons = np.hstack((obslons, obsdata.lon))
            obslats = np.hstack((obslats, obsdata.lat))
        obslonlats = np.vstack((obslons, obslats)).T
        print('#', obslonlats.shape, file=fout)
        print('# lon lat', file=fout)
        obslonlats.tofile(fout, sep=' ', format='%.6f')
    
    # All Done
    return
    

# 计算 UTM 区域号
def calculate_utm_zone(lon):
    import math
    return math.floor((lon + 180) / 6) + 1


def ll2xy(lon, lat, utmzone=None, lon0=None, lat0=None, ellps='WGS84', returnProj=False):
    from pyproj import Proj, CRS
    import pyproj

    # transform to np.array
    lon, lat = np.array(lon), np.array(lat)
    
    def get_utm_zone(lon):
        return int((lon + 180) / 6) + 1

    if utmzone is None and lon0 is not None:
        utmzone = get_utm_zone(lon0)

    if pyproj.__version__[0] == '2':
        if utmzone is not None:
            putm = Proj(proj='utm', zone=utmzone, ellps=ellps)
        else:
            lon0 = lon0 if lon0 is not None else lon.mean()
            lat0 = lat0 if lat0 is not None else lat.mean()
            string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(lat0, lon0, ellps)
            putm = Proj(string)
    else:
        if utmzone is not None:
            epsg_code = 32600 + utmzone if lat.mean() > 0 else 32700 + utmzone
            putm = Proj(CRS.from_epsg(epsg_code))
        else:
            lon0 = lon0 if lon0 is not None else lon.mean()
            utmzone = get_utm_zone(lon0)
            epsg_code = 32600 + utmzone if lat.mean() > 0 else 32700 + utmzone
            putm = Proj(CRS.from_epsg(epsg_code))
    
    x, y = putm(lon, lat)

    # All done
    if returnProj:
        return x, y, putm
    else:
        return x, y


def caldistmat(coord1, coord2, inputCoordinates='lonlat', sort=True, axis=1, utmzone=None, lon0=None, lat0=None, ellps='WGS84'):
    '''
    计算两个坐标序列之间的距离，目的是为了排序提取对应站点
    Input   :
        * coord1              : 坐标序列(n1, m)，m为坐标维数
        * coord2              : 坐标序列(n2, m), coord1和coord2的维数需要一致, 2/3
        * inputCoordinates    : lonlat or utm
        * sort                : 按照距离大小重新排序，并范围对应原来索引号，否则仅返回distmat
        * axis                : 排序沿着的轴
    
    Return  :
        * distmat             : 当sort == False时
        * distmat, sort_index : 当sort == True时
    '''
    coord1 = np.asarray(coord1)
    coord2 = np.asarray(coord2)
    assert coord1.shape[1] == coord2.shape[1], 'The dimension of the input coordinates need to be consistent.'
    if inputCoordinates == 'lonlat':
        if (utmzone is None) and ((lon0 is None) and (lat0 is None)):
            lon0, lat0 = coord2[:, :2].mean(axis=0)

        x1, y1, putm = ll2xy(coord1[:, 0], coord1[:, 1], utmzone=utmzone, lon0=lon0, lat0=lat0, returnProj=True)
        x2, y2 = putm(coord2[:, 0], coord2[:, 1])
        xyz1 = np.vstack((x1, y1, np.zeros_like(x1))).T
        xyz2 = np.vstack((x2, y2, np.zeros_like(x2))).T
        if coord1.shape[1] == 3:
            xyz1[:, 2] = coord1[:, 2]
            xyz2[:, 2] = coord2[:, 2]
        
        distmat = np.linalg.norm(xyz1[:, np.newaxis, :] - xyz2[np.newaxis, :, :], axis=2)
    else:
        distmat = np.linalg.norm(coord1[:, np.newaxis, :] - coord2[np.newaxis, :, :], axis=2)

    if sort:
        arg_sort = np.argsort(distmat, axis=axis)
        return distmat, arg_sort
    else:
        # All Done
        return distmat


def extract_dispfromPscmpV2(filename, unit='m', allsites=True, index=None, 
                            sitecoords=None, interpMethod='nearest', returnIndinPscmp=False):
    '''
    Extract disp at ENU coordinate system from PSCMP output.
    
    Args     :
        * filename       : pscmp输出的特定时刻的结果文件
    
    Kwargs   :
        * unit           : 提取后的形变的保存单位，默认：m
        * allsites       : True/False, default: True, 提取所有站点
        * index          : None:则保留所有数据，list/array：则返回序列对应索引值
        * sitecoords     : None/list/array，欲匹配提取的站点坐标位置，按照最近距离来提取数据
                           先匹配index，再匹配sitecoords
        * interpMethod   : 用于scipy插值函数的method参数

    Return   :
        * DataFrame      : columns: E(+) N(+) Up(+) +表示正方向
    
    Comment  :
        * Built by kfhe at 01/23/2023 
    '''
    from scipy.interpolate import griddata

    columns = 'Lon[deg] Lat[deg] Ux Uy Uz'.split()
    data = pd.read_csv(filename, sep=r'\s+', usecols=columns)
    if not allsites:
        assert (index is not None) or (sitecoords is not None), 'One of index and sitecoords need to be not None.'
        if index is not None:
            data = data.loc[index, :]
        else:
            sitecoords = np.asarray(sitecoords)
            coord1 = sitecoords
            coord2 = data[['Lon[deg]', 'Lat[deg]']].values
            # 投影
            lon0, lat0 = coord2.mean(axis=0)
            x1, y1 = ll2xy(coord1[:, 0], coord1[:, 1], lon0=lon0, lat0=lat0)
            x2, y2 = ll2xy(coord2[:, 0], coord2[:, 1], lon0=lon0, lat0=lat0)
            xy1 = np.vstack((x1, y1)).T
            xy2 = np.vstack((x2, y2)).T
            # 插值
            grid_vals = griddata(xy2, data[['Ux', 'Uy', 'Uz']].values, xy1, method=interpMethod)
            data = pd.DataFrame(np.hstack((coord1, grid_vals)), columns=columns)
    scale = 1.0
    if unit == 'm':
        scale = 1.0
    elif unit == 'mm':
        scale = 1000.0
    elif unit == 'cm':
        scale = 100.
    elif unit == 'dm':
        scale = 10.
    data['Uz'] = -data.Uz*scale
    data['Ux'] = data.Ux*scale
    data['Uy'] = data.Uy*scale
    newname = ['{0}({1})'.format(i, unit) for i in 'N E U'.split()]
    namedict = dict(list(zip('Ux Uy Uz'.split(), newname)))
    data.rename(namedict, axis=1, inplace=True)
    for key in ['se', 'sn', 'su', 'sen', 'seu', 'snu']:
        data[key] = 0.0

    # 重新设置索引为顺序索引，避免读取ts时序时重排顺序
    data.reset_index(inplace=True)
    data.rename({'index': 'IndinPscmp'}, axis=1, inplace=True)

    # All Done
    if (not allsites) and (interpMethod == 'nearest') and returnIndinPscmp:
        _, ind_sort = caldistmat(coord1, coord2, inputCoordinates='lonlat', sort=True, axis=1)
        index = ind_sort[:, 0]
        return data, index
    else:
        return data


def extract_dispfromPscmp(filename, unit='m', allsites=True, index=None, sitecoords=None):
    '''
    Extract disp at ENU coordinate system from PSCMP output.
    
    Args     :
        * filename       : pscmp输出的特定时刻的结果文件
    
    Kwargs   :
        * unit           : 提取后的形变的保存单位，默认：m
        * allsites       : True/False, default: True, 提取所有站点
        * index          : None:则保留所有数据，list/array：则返回序列对应索引值
        * sitecoords     : None/list/array，欲匹配提取的站点坐标位置，按照最近距离来提取数据
                           先匹配index，再匹配sitecoords
        * interpMethod   : 用于scipy插值函数的method参数

    Return   :
        * DataFrame      : columns: E(+) N(+) Up(+) +表示正方向
    
    Comment  :
        * Built by kfhe at 01/23/2023 
    '''

    data = pd.read_csv(filename, sep=r'\s+', usecols='Lat[deg] Lon[deg] Ux Uy Uz'.split())
    if not allsites:
        assert (index is not None) or (sitecoords is not None), 'One of index and sitecoords need to be not None.'
        if index is not None:
            data = data.loc[index, :]
        else:
            sitecoords = np.asarray(sitecoords)
            coord1 = sitecoords
            coord2 = data[['Lon[deg]', 'Lat[deg]']].values
            lon0, lat0 = coord2.mean(axis=0)
            x1, y1 = ll2xy(coord1[:, 0], coord1[:, 1], lon0=lon0, lat0=lat0)
            x2, y2 = ll2xy(coord2[:, 0], coord2[:, 1], lon0=lon0, lat0=lat0)
            # coord1 = np.vstack((x1, y1)).T
            # coord2 = np.vstack((x2, y2)).T
            _, ind_sort = caldistmat(coord1, coord2, inputCoordinates='lonlat', sort=True, axis=1)
            index = ind_sort[:, 0]
            data = data.loc[index, :]
    scale = 1.0
    if unit == 'm':
        scale = 1.0
    elif unit == 'mm':
        scale = 1000.0
    elif unit == 'cm':
        scale = 100.
    elif unit == 'dm':
        scale = 10.
    data['Uz'] = -data.Uz*scale
    data['Ux'] = data.Ux*scale
    data['Uy'] = data.Uy*scale
    newname = ['{0}({1})'.format(i, unit) for i in 'N E U'.split()]
    namedict = dict(list(zip('Ux Uy Uz'.split(), newname)))
    data.rename(namedict, axis=1, inplace=True)
    for key in ['se', 'sn', 'su', 'sen', 'seu', 'snu']:
        data[key] = 0.0

    # 重新设置索引为顺序索引，避免读取ts时序时重排顺序
    data.reset_index(inplace=True)
    data.rename({'index': 'IndinPscmp'}, axis=1, inplace=True)

    # All Done
    if not allsites:
        return data, index
    else:
        return data


def extract_pscmp_ts(dirname, filenames=None, unit='m', tunit='Y', allsites=True, index=None, sitecoords=None, interpMethod='nearest'):
    '''
    从pscmp输出目录中提取对应的时序文件下的形变，组成时序形变数据
    e.g., eqdate = pd.Timestamp('2021-05-21')

    Input    :
        * dirname        : pscmp结果文件目录
        * unit           : 用于extract_dispfromPscmp控制提取形变存储的单位
        * tunit          : Y/D. 用于pscmpts输出时的Index索引单位，为Y时，则转为yr的浮点数，否则转为：Y, D代表的字母
        * filenames      : 读取所有pscmp结果文件还是特定文件，默认None表示读取所有结果文件
        * allsites       : True/False, default: True, 提取所有站点
        * index          : None:则保留所有数据，list/array：则返回序列对应索引值
        * sitecoords     : None/list/array，欲匹配提取的站点坐标位置，按照最近距离来提取数据
                           先匹配index，再匹配sitecoords
    
    Return   :
        * pscmpts        : 
    
    Comment  :
        * 为了避免震间情况，dt = 10000s yr，超出timedelta的范围，因此我们这里将时间排序以year的浮点数进行，同时保存对应时间字符串
    '''
    if filenames is None:
        template = 'snapshot_*.dat'
        files = glob(os.path.join(dirname, template))
    else:
        files = [os.path.join(dirname, ifile) for ifile in filenames]
    filedict = {}

    pattern = r'(?:snapshot_)([0-9._]+)_(day|week|month|year)(?:\.dat)'
    indexdict = {
        'day': 'D',
        'month': 'M',
        'week': 'W',
        'year': 'Y'
    }
    for i, ifile in enumerate(files):
        basename = os.path.basename(ifile)
        # print(ifile, basename)
        if 'coseism' in basename:
            dt = pd.Timedelta('0D')
            dtstr = '0' + 'D'
            dt = 0.0
        else:
            res = re.match(pattern, basename)
            # print(basename, res)
            tval, intunit = res.group(1), indexdict[res.group(2)]
            tval = tval.replace('_', '.')
            # Bug: M好像在pd.Timedelta中代表min，这里需要更正后续
            if intunit in ('D', 'M', 'W'):
                dt = pd.Timedelta(tval + intunit)
                dtstr = tval + intunit
                # 转换为年的浮点数
                dt = (dt.days + dt.seconds/24.0/3600.0)/365.2425 if tunit == 'Y' else (dt.days + dt.seconds/24.0/3600.0)
            else:
                dt = float(tval) if tunit == 'Y' else float(tval)*365.2425
                dtstr = tval + intunit
        if not allsites:
            filedict[dt] = extract_dispfromPscmpV2(ifile, unit=unit, allsites=allsites, index=index, sitecoords=sitecoords, interpMethod=interpMethod)
        else:
            filedict[dt] = extract_dispfromPscmpV2(ifile, unit=unit)
        filedict[dt]['date'] = dt
        filedict[dt]['datestr'] = dtstr
        filedict[dt]['tunit'] = tunit
    # 按时间和站点输出顺序排列
    pscmpts = pd.concat(filedict.values()).reset_index().set_index(['date', 'index']).sort_index(level=0)
    return pscmpts
# ----------------------------------------------------------------------------------------#

# ----------------------------------------------------------------------------------------#
# @jit(nopython=True, cache=True, nogil=True)
def intp_disp(t, tobs, disp):
    '''
    根据黏弹性时序格林函数，插值出待求时刻的形变
    '''
    return np.interp(t, tobs, disp)

# @jit(nopython=True, cache=True, nogil=True)
def intp_disp_ENU(t, tobs, disp):
    '''
    根据黏弹性时序格林函数，插值出待求时刻的形变
    '''
    dE, dN, dU = disp[:, 0], disp[:, 1], disp[:, 2]
    intpE, intpN, intpU = intp_disp(t, tobs, dE), intp_disp(t, tobs, dN), intp_disp(t, tobs, dU)
    visco = np.vstack((intpE, intpN, intpU)).T
    return visco

# @jit(nopython=True, cache=True, nogil=True)
def Gvisco_parallel(t, tobs, disp, k):
    return k, intp_disp_ENU(t, tobs, disp)


def interp_pscmp_ts(pscmpts, obsdate, unit='m', intp_tunit='D', mcpu=4, eqdate=None, removeCo=True, outfile=None):
    '''
    pscmpts在时间上插值
    Args      :
        * pscmpts           : 由extract_pscmp_ts提取的时间序列数据，相似格式的数据也可
        * obsdate           : 需要插值的时间节点位置， 默认为pd.Timestamp形式的时间序列/pd.DatetimeIndex
                              也可是pd.DatetimeIndex的参数

    Kwargs    :
        * outfile           : pickle.dump输出目录；默认None，表示不输出
        * eqdate            : 起始时间，如果为None,则起始时间默认为obsdate的第一个值
        * unit              : extract_dispfromPscmp提取的形变的单位，默认为m
        * removeCo          : 是否移除同震形变，默认True,即移除
    
    Return    :
        * intp_pscmpts      : 时间上插值后的形变时间序列
    
    Comment   :
        * 为了运行并行需要在__name__ == '__main__'中，且需要有以下声明在main中最前面
         __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
        freeze_support() # For Windows support
    '''
    import multiprocessing as mp
    from tqdm import tqdm
    idx = pd.IndexSlice
    # from multiprocessing import Pool, freeze_support, RLock

    disp_cols = ['{0}({1})'.format(idir, unit) for idir in 'E N U'.split()]
    dispENU = pscmpts[disp_cols]
    if removeCo:
        dispENU[disp_cols] -= dispENU.loc[idx[0.0, :], disp_cols].droplevel(0)
    date_index = dispENU.index.get_level_values(0).unique()
    intunit = pscmpts.iloc[0].tunit
    caldt_intp = (date_index*pd.Timedelta(f'1{intunit}')/pd.Timedelta(f'1{intp_tunit}')).to_numpy()
    site_index = dispENU.index.get_level_values(1).unique()

    obsdate = pd.DatetimeIndex(obsdate)
    obsdt = pd.TimedeltaIndex([iobs-eqdate for iobs in obsdate])
    obsdt_intp = np.array([(dt.days + dt.seconds/24.0/3600.0)/365.2425 for dt in obsdt])
    obsdt_intp = obsdt_intp*pd.Timedelta(f'1Y')/pd.Timedelta(f'1{intp_tunit}')

    pool = mp.Pool(mcpu) if mcpu is not None else mp.cpu_count()
    rows, cols = obsdt.shape[0], site_index.shape[0]
    pscmp_obsts = np.empty((rows, cols, 3), dtype=np.float_)
    pscmp_obsts[:, :, :] = np.nan
    # break
    # Join the parallel Pool
    jobs = [pool.apply_async(Gvisco_parallel, args=(obsdt_intp, caldt_intp, dispENU.loc[idx[:, site_index[k]], disp_cols].values, k))
                            for k in range(cols)]
        
    pool.close()
    # result_list_tqdm = []
    for job in tqdm(jobs):
        # result_list_tqdm.append(job.get())
        k, visco = job.get()
        pscmp_obsts[:, k, :] = visco
    pool.join()
    
    # 输出索引index
    obsindex = pd.MultiIndex.from_product((obsdt_intp, site_index), names=['date', 'index'])
    obsdata = pd.DataFrame(pscmp_obsts.reshape((-1, 3)), index=obsindex, columns=disp_cols)
    lonlat = pscmpts.loc[idx[0.0, :], ['Lon[deg]', 'Lat[deg]']]
    lonlat.index = site_index
    obsdata = pd.merge(lonlat, obsdata, left_index=True, right_on=obsdata.index.get_level_values(1))
    obsdata.drop('key_0', axis=1, inplace=True)
    obsdata['tunit'] = intp_tunit
    if outfile is not None:
        obsdata.to_pickle(outfile)

    # All Done
    return obsdata
# ----------------------------------------------------------------------------------------#

# ----------------------------------------------------------------------------------------#
def calviscoGfromPscmp(pscmpts, T=None, diffint=None, unit='m'):
    '''
    Calculation of viscoelastic interseismic effects from the pscmp results file
    
    Args     :
        * pscmpts       : DataFrame, index: [Date, index], 从pscmp结果文件中读取的时序结果
            * columns   : ['IndinPscmp', 'Lat[deg]', 'Lon[deg]', 'N(m)', 'E(m)', 'U(m)', 'se', 'sn', 'su', 'sen', 'seu', 'snu', 'datestr']
            * index     : ['date', 'index']

    Kwargs   :
        * diffint       : pscmp为计算黏弹性震间影响，计算形变差值的间隔，default: None，表示直接从DataFrame索引中计算出来
        * unit          : 用于extract_dispfromPscmp控制提取形变存储的单位
        * T             : 地震周期, default: None，即直接从DataFrame索引中计算出来

    Return   :
        * viscoG        : 单位震间位错对应的黏弹性松弛格林函数，速度方向为构造速度方向为正，和构造速度方向一致
                          The viscoelastic relaxation Green function corresponding to the unit interseismic dislocation, 
                          the velocity direction is positive and consistent with the tectonic velocity direction

    Comment  :
        * built by kfhe at 01/24/2023
        * Represents the viscoelastic relaxation Green's function corresponding to unit velocity
    '''

    disp_cols = ['{0}({1})'.format(idir, unit) for idir in 'E N U'.split()]
    dispENU = pscmpts[disp_cols]
    date_index = dispENU.index.get_level_values(0).unique()
    site_index = dispENU.index.get_level_values(1).unique()

    diffint = date_index[2] - date_index[1] if diffint is None else diffint
    before_dates = date_index[1::3]
    norm_dates = date_index[2::3]
    after_dates = date_index[3::3]
    T = norm_dates[1] - norm_dates[0] if T is None else T
    obst = norm_dates[0]

    # 1. 计算黏弹性影响的第一部分
    ## before disp
    bdisp = dispENU.loc[idx[before_dates, :], :]
    bdisp = bdisp.rename(dict(zip(before_dates, norm_dates)), level=0, axis=0)
    ## norm_disp
    ndisp = dispENU.loc[idx[norm_dates, :], :]
    ## after disp
    adisp = dispENU.loc[idx[after_dates, :], :]
    adisp = adisp.rename(dict(zip(after_dates, norm_dates)), level=0, axis=0)
    # 计算差值的平均
    diff1 = (ndisp - bdisp)/diffint
    diff2 = (adisp - ndisp)/diffint
    diff = (diff1 + diff2)/2.0

    sumdiff = diff.sum(axis='index', level=1)*T # *T

    # 2. 提取最后一个累积形变/T
    t_st = date_index[0]
    t_ed = norm_dates[-1]
    disp_ed = dispENU.loc[idx[t_ed, :], :]
    disp_ed = disp_ed.droplevel(0, axis=0)

    disp_st = dispENU.loc[idx[t_st, :], :]
    disp_st = disp_st.droplevel(0, axis=0)
    final_disp = (disp_ed - disp_st) # /T
    
    # 纯黏弹性影响部分, from Diao et al., 2022
    viscoG = sumdiff - final_disp

    # All Done
    return  viscoG

#---------------------------------------------------------------------------------#


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    # 1.生成待计算时间序列
    nts, nobs = gentsseq_pscmp([95], [100], 2, cycles=9)
    # 2. 生成以上时间序列在pscmp结果文件中对应文件名
    filenames = fmt_pscmp(nobs, truncation=0, screen=False, onlyfilename=True)
    # 3. 用于csi中fault反演的数据集坐标
    sitecoords = np.loadtxt(r'd:\2022Menyuan\Interseismic_InSAR\3DDisp_External\3DDisp\3DDisp2InterInv\InvCode\sites4pscmp.dat').reshape(-1, 2)
    # 4. 提取pscmp生成的时间序列
    data = extract_dispfromPscmpV2(r'e:\Haiyuan_ViscoInterseismic\PSGRN_PSCMP\Maxwell_20km\pscmp_gps_lc1_0e19_regular_LLL\snapshot_10090_year.dat', allsites=False, sitecoords=sitecoords)
    # dirname = r'e:\Haiyuan_ViscoInterseismic\PSGRN_PSCMP\Maxwell_25km\compar_analysis\pscmp_gps_lc1e19u_regular_T1000'
    # pscmpts = extract_pscmp_ts(dirname, filenames=filenames, allsites=False, sitecoords=sitecoords)
    # # 5. 计算对应于每个站点的格林函数
    # vscale = 4e-2 # m/yr
    # viscoG = calviscoGfromPscmp(pscmpts, T=100)/4.0

    dirname = r'e:\Maduo_psgrn_pscmp\Maxwell_32km_crust_mantle\pscmp_gps_lc4_0e17_regular'
    from glob import glob
    filenames = glob(os.path.join(dirname, 'snapshot_*.dat'))
    filenames = [os.path.basename(ifile) for ifile in filenames]
    pscmpts = extract_pscmp_ts(dirname, filenames=filenames, allsites=False, sitecoords=sitecoords)
    eqdate = pd.Timestamp('2021-05-21')
    obsdate = pd.DatetimeIndex(['2021-05-26', '2021-06-19', '2021-07-13', '2021-07-25', '2021-08-06', '2021-08-12', '2021-08-18', '2021-08-24'])

    obsdata = interp_pscmp_ts(pscmpts, obsdate, eqdate=eqdate, mcpu=6)
