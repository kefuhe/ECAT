'''
Written by Kefeng He, January 2023
'''

import numpy as np
import pandas as pd


def gents_pscmp(tobs, T, diffint=5, cycles=10):
    '''
    根据输入来输出pscmp中需要的时间输出格式
    用于计算地震周期形变用
    Input   :
        tobs    : 观测时刻到最近一个地震的时间
        diffint : 用于计算速度的时间期的间隔值，生成 tobs + nT +/- diffint 以及 tobs + nT本身
        cycles  : 生成的周期数，n in (0, cycles), default = 10
        T       : 地震周期
    Output  :
        nts     : 时间点总数
        tobsseq : 用于输入到pscmp中的时间点位置
        (nts, tobsseq)
    '''
    tobs_icycle = [tobs + T*icyc for icyc in range(cycles+1)]
    pretobs_icycle = [itobs-diffint for itobs in tobs_icycle]
    posttobs_icycle = [itobs+diffint for itobs in tobs_icycle]
    tobsseq = tobs_icycle + pretobs_icycle + posttobs_icycle
    tobsseq.insert(0, 0)
    tobsseq.sort()
    nts = len(tobsseq)
    return nts, tobsseq


def gentsseq_pscmp(tobsseq, Tseq, diffint=5, cycles=10):
    '''
    Object   :
        将时序观测和时序周期打印到pscmp格式
    Input    :
        tobsseq   : list/np.ndarray
        Tseq      : list/np.ndarray
        diffint   : default = 5, float/int/list/np.ndarray
        cycles    : default= 10
    '''
    assert type(tobsseq) in (list, np.ndarray), 'tobsseq must be list/np.ndarray'
    assert type(Tseq) in (list, np.ndarray), 'Tseq must be list/np.ndarray'
    ntobsseq = np.asarray(tobsseq)
    nTseq = np.asarray(Tseq)
    if type(diffint) in (int, float):
        ndiffint = np.ones_like(nTseq) * diffint
    else:
        ndiffint = np.asarray(diffint)
    assert ((ntobsseq.shape[0] == nTseq.shape[0]) and (nTseq.shape[0] == ndiffint.shape[0])), 'The size must be equal'
    nts, outtobs = 0, []
    for itobs, iT, idiff in zip(ntobsseq, nTseq, ndiffint):
        ints, itobsseq = gents_pscmp(itobs, iT, idiff, cycles=cycles)
        nts += ints
        outtobs.extend(itobsseq)

    outtobs = np.asarray(outtobs)
    outtobs = np.unique(outtobs)
    nts = outtobs.shape[0]
    # All Done
    return nts, outtobs


def fmt_pscmp(tobsseq, tunit='year', truncation=0, outfile=None, onlyfilename= False, screen=True):
    '''
    Object  : 
        根据要打印的时间节点来自动输出打印时间序列字符串到pscmp
    Args   :
        * tobsseq       : 采样的时间节点序列
    
    Kwargs :
        * tunit         : 输出文件的单位
        * outfile       : None 打印到列表返回，否则打印到相应文件中
        * onlyfilename  :仅打印文件名，而不是pscmp整行字符串
        * truncation    : 保存的浮点数，在文件名中用'_'代替'.'
        * screen        : 是否打印到屏幕
    
    Return :
        * None if outfile is not None else fmt_list
    '''
    import re

    fmt = "    {0:.2f}  'snapshot_{1:." + f'{truncation}' + "f}_{2}.dat'        |{1:.2f} {2}"
    if tunit == 'year':
        scale = 365.2425
    elif tunit == 'day':
        scale = 1.0
    elif tunit == 'week':
        scale = 7.0
    elif tunit == 'month':
        scale = 30.0
    
    fmt_list = [fmt.format(scale*itobs, itobs, tunit) for itobs in tobsseq]

    if truncation != 0:
        pattern = r'(?<=snapshot_)([0-9]+)(\.)(?=[0-9]+' + f'_{tunit}.dat)'
        # pattern = r'(snapshot_[0-9]+)(\.)'
        # print(pattern)
        fmt_list = [re.sub(pattern, r'\1_', ifmt) for ifmt in fmt_list]
    if onlyfilename:
        pattern = '(?:.+)(snapshot.+\.dat)(?:.+)'
        fmt_list = [re.match(pattern, ifmt).group(1) for ifmt in fmt_list]

    if outfile is not None:
        with open(outfile, 'wt') as fout:
            for ifmt in fmt_list:
                print(ifmt, file=fout)

    if screen:
        for ifmt in fmt_list:
            print(ifmt)

    # All done
    if outfile is None:
        return fmt_list
    else:
        return None


def pscmp_irregularpos(obspnts, pntperline=3, outfile='pscmp_irregularpos.dat'):
    '''
    Object  :
        将不规则观测点规整到pscmp的输入格式
    Input   :
        obspnts : 输入的观测点位坐标数据
            pandas : 如果格式为DataFrame，则必须包含lon, lat对应字段
            numpy  : 如果格式为numpy/list，则必须为n*2模式，且第一列为经度，第二列为纬度
        outfile : 规整后符合pscmp格式的坐标点位文件
    '''
    if type(obspnts) is pd.DataFrame:
        obspnts = obspnts[['lon', 'lat']].values
    elif type(obspnts) in (list, np.ndarray):
        obspnts = np.asarray(obspnts)
    size = obspnts.shape[0]
    lon, lat = obspnts[:, 0], obspnts[:, 1]
    with open(outfile, 'wt') as fout:
        print('  {0:d}'.format(size), file=fout)
        for i in range(size):
            if i == 0:
                print('   ', end='', file=fout)
            if (i+1)%pntperline == 0:
                print('({0:6.3f},{1:7.3f})\n'.format(lat[i], lon[i]), end='   ', file=fout)
            else:
                print('({0:6.3f},{1:7.3f}), '.format(lat[i], lon[i]), end='', file=fout) 
    
    # All done
    return


if __name__ == '__main__':
    nts, nobs = gentsseq_pscmp([100, 900], [1000, 1500], 5)
    fmt_pscmp(nobs, truncation=2, screen=True, onlyfilename=True)