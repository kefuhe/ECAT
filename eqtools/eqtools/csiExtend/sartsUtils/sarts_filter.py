import h5py
import numpy as np
import pandas as pd
import pickle
from math import floor, ceil
from numpy import log
import multiprocessing as mp
import os
from tqdm import tqdm
from numba import complex128,float64,jit, njit
from scipy.optimize import curve_fit


def todotyear(date):
    
    dotyear = date.year + ((date.microsecond*1.e-6 + date.second + 
                date.hour*3600 + date.minute*60)/3600./24. + date.dayofyear)/365.2425
    return dotyear


@jit(nopython=True, cache=True, nogil=True)
def Logs(t, tau, amp, a):
    '''
    定义对数衰减项
    tau: 单位天
    '''
    if tau < 0:
        raise ValueError("'tau' must be a larger num than 0.")
    elif tau == 0:
        raise ZeroDivisionError("'tau' equal 0, 'tau' should be larger than 0")
    y = np.zeros_like(t)
    y = a+ amp*log(1 + t*365./tau)
    return y


@jit(nopython=True, cache=True)
def Logs_parallel(t, tau, amp, a, i, k):
    return i, k, Logs(t, tau, amp, a)


def curve_fit_parallel(t, data, bounds, i, k, Logs=Logs):
    return i, k, curve_fit(Logs, t, data, bounds=bounds)[0]


@jit(nopython=True,parallel=True)
def Forward_numba(out, fitparas, ts, inds):
    '''
    numba： parallel=True时，里面不支持广播；
    如果一定要广播，下面一条jit可能有用
    @njit(parallel={"inplace_binop":False})
    '''
    for (i, k), iparas in zip(inds, fitparas):
        # for it in range(ts.shape[0]):
        #     itime = ts[it]
        itau, iamp, ia = iparas
        dlos = Logs(ts, itau, iamp, ia)
        out[:, i, k] = dlos

    # All Done
    return


class Inverse(object):
    def __init__(self, ncpu=mp.cpu_count(), sarts=None, time=None, bound=None):
        self.ncpu = ncpu
        self.sarts = sarts if sarts is not None else None
        self.time = time if time is not None else None
        self.bound = bound if bound is not None else None

    
    def write2pickle(self, outfile='popts.pkl'):
        popts = self.popts
        with open(outfile, 'wb') as fout:
            pickle.dump(popts, fout, protocol=None)
        
        # All Done
        return
    
    def setTime(self, time):
        self.time = time
    
    def setSarTS(self, sarts):
        self.sarts = sarts
    
    def setBound(self, bound):
        self.bound = bound
    
    def loopParallel(self, ncpu=None, bound=None, mask=None):
        if ncpu is not None:
            self.ncpu = ncpu
        
        if bound is not None:
            self.setBound(bound)
        
        pool = mp.Pool(self.ncpu)
        t = self.time

        tsts = self.sarts
        # Join the parapllel pool
        endsar_mask = tsts[-1, :, :].copy()
        rows, cols = endsar_mask.shape
        if mask is not None:
            endsar_mask[mask == False] = np.nan

        
        jobs = [pool.apply_async(curve_fit_parallel, args=(t, tsts[:, i, k], self.bound, i, k)) 
                for i in range(rows) for k in range(cols) if not np.isnan(endsar_mask[i, k])]

        pool.close()
        
        popts = np.empty(tsts[0].shape, dtype=object)
        for job in tqdm(jobs):
            i, k, popt = job.get() 
            popts[i, k] = popt
        pool.join()
    
        self.popts = popts

        # All Done
        return
    

class Forward(object):
    def __init__(self, dateseries=None, cotime=None, popts=None):
        self.dateseries = dateseries if dateseries is not None else None
        self.cotime = cotime if cotime is not None else None
        self.popts = popts if popts is not None else None
    
    def setDateseries(self, dateseries):
        self.dateseries = dateseries
    
    def setDateseries(self, cotime):
        self.cotime = cotime
    
    def setPopts(self, popts):
        self.popts = popts
    
    def setPoptsFromfile(self, poptsfile='popts.pkl'):
        with open(poptsfile, 'rb') as fin:
            popts = pickle.load(fin)
        
        self.popts = popts

        # All Done
        return
    
    def union_dateseries(self, other_dateseries, inplace=True):

        dateseries = pd.DatetimeIndex(pd.to_datetime(self.dateseries))
        other_dateseries = pd.DatetimeIndex(pd.to_datetime(other_dateseries))
        union_dateseries = dateseries.union(other_dateseries)
        if inplace:
            self.dateseries = union_dateseries
        return union_dateseries
    
    def calts(self):
        dateseries = self.dateseries
        cotime = self.cotime

        ts = todotyear(dateseries) - todotyear(cotime)
        ts = ts.to_numpy()
        self.ts = ts 

        # All Done
        return
    
    def loopNumba(self):
        '''
        正演计算numba加速比multiprocessing并行快~8-9倍
        TODO:
            * 测试GPU加速方式
        '''
        ntime = self.dateseries.shape[0]
        rows, cols = self.popts.shape
        popts = self.popts
        sar_ts = np.empty((ntime, rows, cols), dtype=np.float_)
        sar_ts[:, :, :] = np.nan
        if hasattr(self, 'ts'):
            ts = self.ts
        else:
            self.calts()
            ts = self.ts
        
        # not None
        inds = [[i, k] for i in range(rows) for k in range(cols) if popts[i, k] is not None]
        inds = np.array(inds, dtype=np.int_)

        fitparas_notnone = popts[inds[:, 0], inds[:, 1]]
        fitparas_asarray = np.array([fitparas_notnone[i].tolist() for i in range(fitparas_notnone.shape[0])])

        Forward_numba(sar_ts, fitparas_asarray, ts, inds)

        self.sar_ts = sar_ts
    
        # All Done
        return

    def loopParallel(self, ncpu=mp.cpu_count()):
        pool = mp.Pool(ncpu)
        dateseries = self.dateseries
        rows, cols = self.popts.shape
        fitparas = self.popts
        sar_ts = np.empty((dateseries.shape[0], rows, cols), dtype=object)
        sar_ts[:, :, :] = np.nan
        if hasattr(self, 'ts'):
            ts = self.ts
        else:
            self.calts()
            ts = self.ts
        
        # Join the parallel Pool
        jobs = [pool.apply_async(Logs_parallel, args=(ts, *fitparas[i, k], i, k)) 
                            for i in range(rows) for k in range(cols) if fitparas[i, k] is not None]
            
        pool.close()

        for job in tqdm(jobs):
            i, k, dlosts = job.get() 
            sar_ts[:, i, k] = dlosts
        pool.join()

        self.sar_ts = sar_ts

        # All Done
        return
    
    def write2H5file(self, outfile='sar_ts_filter.h5'):
        '''
        '''

        dateseries = self.dateseries
        sar_ts = self.sar_ts
        # 如果你要在根group下创建dataset
        dsar_h5 = h5py.File(outfile, 'w')
        # create_dataset只接受ascii格式，所以需要编码
        timesD = np.array(['{0:%Y%m%d}'.format(t).encode() for t in dateseries])
        timesD = timesD.reshape(dateseries.shape[0], 1, 1)
        timesD = dsar_h5.create_dataset('time', data=timesD)
        dlos = dsar_h5.create_dataset('sar_ts', data=sar_ts.astype(np.float_))
        dsar_h5.close()

        # All Done
        return


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    #--------手动设置参数-----------------#
    ## 地震时刻
    cotime = pd.Timestamp('2022-01-08')
    
    #-------------------------------------#
    
    #---------设置InSAR数据文件---------#
    dirname = os.path.join('..', '..', 'geo')
    filenames = {
        'avgSpatialCoh': 'geo_avgSpatialCoh.h5',
        'geoRadar': 'geo_geometryRadar.h5',
        'maskTempCoh': 'geo_maskTempCoh.h5',
        'temporalCoherence': 'geo_temporalCoherence.h5',
        'velocity': 'geo_velocity.h5',
        'ts_gacos_demerr': 'geo_timeseries_demErr.h5'
    }
    #-----------------------------------#

    #---------读取InSAR数据和相关信息---#
    # Extract SAR time series
    # keys: ['bperp', 'date', 'timeseries']
    with h5py.File(os.path.join(dirname, filenames['ts_gacos_demerr']), 'r+') as ts:
        # timeseries = ts['timeseries'][:]
        dateseries = pd.DatetimeIndex(pd.to_datetime(ts['date'][:], format='%Y%m%d'))
    #-----------------------------------#

    forwardobj = Forward(dateseries, cotime)
    forwardobj.setPoptsFromfile('popts.pkl')
    # forwardobj.loopParallel()
    forwardobj.loopNumba()
    forwardobj.write2H5file()
