'''
A class assemble timeseries disp object to a single inverse problem
Written by kefeng He, July 2023.

'''


import copy
import numpy as np
import pyproj as pp
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl 
import seaborn as sns
import scienceplots
from scipy.spatial import KDTree


class combinedmodelsolve(object):
    '''
    A class assemble timeseries disp object to a single inverse problem.

    Args:
        * name          : Name of the project.
        * faults        : List of faults from verticalfault or pressure .

    '''
    VALID_TIMEUNITS = ('Y', 'y', 'year', 'M', 'm', 'month', 'W', 'w', 'week', 'D', 'd', 'day', 'S', 's', 'sec', 'second')

    def __init__(self, name, eqdate, geotsdatalist=None, verbose=True):

        self.verbose = verbose
        if self.verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initializing solver object")

        # Ready to compute?
        self.ready = False
        self.eqdate = eqdate
        self.figurePath = './'

        from collections import OrderedDict
        self.geotsdatadict = OrderedDict()
        if geotsdatalist is not None:
            for its in geotsdatalist:
                self.geotsdatadict[its.name] = its

        dispts0 = geotsdatalist[0]
        # Store things into self
        self.name = name

        # check the utm zone
        self.utmzone = dispts0.utmzone
        self.lon0 = dispts0.lon0
        self.lat0 = dispts0.lat0
        self.xy2ll = dispts0.xy2ll
        self.ll2xy = dispts0.ll2xy



        self.type = 'CombinedModel'

        # All done
        return

    def getcoviscogrid(self, coviscogrid):
        '''
        viscoelastic relaxation timeseries data due to coseismic slip distribution from pylith/pscmp.
        '''
        self.coviscogrid = coviscogrid 

        # All Done
        return
    
    def getasviscogrid(self, asviscogrid):
        '''
        Viscoelastic relaxation timeseries data due to afterslip distribution from pylith/pscmp.
        '''
        self.asviscogrid = asviscogrid

        # All Done
        return
    
    #-------------------Caluculate cumulative afterslip------------------------------------------#
    def calAS(self, tau, alpha=1.0, timeunit='D', dispunit='cm'):
        '''
        Calculate cumulative afterslip
        '''
        for ikey, itsdata in tqdm(self.geotsdatadict.items()):
            if itsdata.dtype in ('gpstimeseries',):
                coord = np.array([[itsdata.lon, itsdata.lat]])

                time = itsdata.time
                intpts = self.calASatcoord(coord, time, tau=tau, alpha=alpha, dispunit=dispunit, timeunit=timeunit)

                itsdata.east.cumas = intpts.dispts[:, 0, 0]
                itsdata.north.cumas = intpts.dispts[:, 0, 1]
                itsdata.up.cumas = intpts.dispts[:, 0, 2]
            elif itsdata.dtype in ('insartimeseries',):
                coord = np.vstack((itsdata.lon, itsdata.lat)).T
                los = itsdata.timeseries[0].los
                intpts = self.calASatcoord(coord, itsdata.time, tau=tau, alpha=alpha, dispunit=dispunit, timeunit=timeunit)
                for it, saratitime in enumerate(itsdata.timeseries):
                    saratitime.cumas = np.sum(intpts.dispts[it, :, :]*los, axis=1)
                
        # All Done
        return
    
    def calASatsite(self, gpsts, tau, alpha=1.0, dispunit='cm', timeunit='D'):
        '''
        tau unit is uniform with timeunit
        '''
        coord = np.array([gpsts.lon, gpsts.lat])
        timeseries = gpsts.time
        cum_afterslip = self.calASatcoord(coord, timeseries, tau=tau, alpha=alpha, dispunit=dispunit, timeunit=timeunit)
        # All Done
        return cum_afterslip # eqtools.DispBase对象
    
    def calASatcoord(self, coord, timeseries, tau, alpha=1.0, dispunit='cm', timeunit='D'):
        '''
        '''
        from ..pscmp_visco_postseismic import as_cum
        griddata = self.asviscogrid
        coord = np.array(coord)
        if coord.ndim == 1:
            coord = coord.reshape(-1, 2)
        timeseries = pd.DatetimeIndex(timeseries)
        tsfloat = self._todotyr(timeseries)*self._timescale(distunit=timeunit)
        # 空间插值
        intp_data = griddata.extractDisp(sitecoords=coord, dispunit=dispunit, timeunit=timeunit, dispInTimes=None, interpTimeMethod='linear', refTime=None)
        disp0 = intp_data.dispts[0, :, :]
        # 时间上累积
        cum_afterlisp = as_cum(tsfloat[:, None].repeat(3, axis=1)[:, None, :], disp0[None, :, :], tau_as=tau, alpha=alpha)

        # 转换为DispBase格式
        from .DispBase import DispBase
        disp = DispBase('afterslip', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0)
        disp.timeseries = tsfloat
        disp.timeunit = timeunit
        disp.dispunit = dispunit
        z = np.zeros((coord.shape[0], ))
        disp.llz = np.hstack((coord, z[:, None]))
        x, y = disp.ll2xy(coord[:, 0], coord[:, 1])
        disp.xyz = np.vstack((x, y, z)).T 
        disp.dispts = cum_afterlisp

        # All Done
        return disp # eqtools.DispBase对象
    #-------------------------------------------------------------------- -----------------------#
    
    #-------------------Caluculate viscoelastic relaxation due to co-slip------------------------#
    def calCV(self, timeunit='D', dispunit='cm'):
        '''
        Calculate viscoelastic relaxation due to coseismic slip distribution
        '''
        for ikey, itsdata in tqdm(self.geotsdatadict.items()):
            if itsdata.dtype in ('gpstimeseries',):
                coord = np.array([[itsdata.lon, itsdata.lat]])
                
                time = itsdata.time
                intpts = self.calCVatcoord(coord, time, timeunit=timeunit, dispunit=dispunit)

                itsdata.east.cv = intpts.dispts[:, 0, 0]
                itsdata.north.cv = intpts.dispts[:, 0, 1]
                itsdata.up.cv = intpts.dispts[:, 0, 2]
            elif itsdata.dtype in ('insartimeseries',):
                coord = np.vstack((itsdata.lon, itsdata.lat)).T
                los = itsdata.timeseries[0].los
                intpts = self.calCVatcoord(coord, itsdata.time, timeunit=timeunit, dispunit=dispunit)
                for it, saratitime in enumerate(itsdata.timeseries):
                    saratitime.cv = np.sum(intpts.dispts[it, :, :]*los, axis=1)
                
        # All Done
        return
    
    def calCVatcoord(self, coord, timeseries, timeunit='D', dispunit='cm'):
        '''
        coord: 1D or 2D [lon, lat] or [[lon, lat], [lon2, lat2]]
        timeseries : list/pd.DatatimeIndex

        return : eqtools.DispBase对象
        '''
        griddata = self.coviscogrid
        coord = np.array(coord)
        if coord.ndim == 1:
            coord = coord.reshape(-1, 2)
        time = pd.DatetimeIndex(timeseries)
        tsfloat = self._todotyr(time)
        intpdata = griddata.extractDisp(sitecoords=coord, dispunit=dispunit, timeunit=timeunit, dispInTimes=tsfloat, interpTimeMethod='linear', refTime=0)
        # All Done
        return intpdata # eqtools.DispBase对象

    def calCVatsite(self, gpsts, timeunit='D', dispunit='cm'):
        '''
        coord: (lon, lat)
        timeseries : list/pd.DatatimeIndex
        '''
        coord = np.array([gpsts.lon, gpsts.lat])
        timeseries = gpsts.time
        covisco = self.calCVatcoord(coord, timeseries, timeunit=timeunit, dispunit=dispunit)
        # All Done
        return covisco # eqtools.DispBase对象
    #-------------------------------------------------------------------- -----------------------#

    #----------------------Caluculate viscoelastic relaxation due to afterslip-------------------#
    def calAV(self, tau, alpha=1.0, timeunit='D', dispunit='cm', onlyAV=True, mcpu=4):
        '''
        Calculate viscoelastic relaxation due to coseismic slip distribution
        '''
        for ikey, itsdata in tqdm(self.geotsdatadict.items()):
            if itsdata.dtype in ('gpstimeseries',):
                coord = np.array([[itsdata.lon, itsdata.lat]])
                time = itsdata.time
                intpts = self.calAVatcoord(coord, time, tau, alpha=alpha, timeunit=timeunit, dispunit=dispunit, onlyAV=onlyAV, mcpu=mcpu)

                if onlyAV:
                    itsdata.east.av = intpts.dispts[:, 0, 0]
                    itsdata.north.av = intpts.dispts[:, 0, 1]
                    itsdata.up.av = intpts.dispts[:, 0, 2]
                else:
                    itsdata.east.as_av = intpts.dispts[:, 0, 0]
                    itsdata.north.as_av = intpts.dispts[:, 0, 1]
                    itsdata.up.as_av = intpts.dispts[:, 0, 2]
            elif itsdata.dtype in ('insartimeseries',):
                coord = np.vstack((itsdata.lon, itsdata.lat)).T
                los = itsdata.timeseries[0].los
                intpts = self.calAVatcoord(coord, itsdata.time, tau, alpha=alpha, timeunit=timeunit, dispunit=dispunit, onlyAV=onlyAV, mcpu=mcpu)
                for it, saratitime in enumerate(itsdata.timeseries):
                    if onlyAV:
                        saratitime.av = np.sum(intpts.dispts[it, :, :]*los, axis=1)
                    else:
                        saratitime.as_av = np.sum(intpts.dispts[it, :, :]*los, axis=1)
                
        # All Done
        return

    def calAVatsite(self, gpsts, tau, alpha=1.0, timeunit='D', dispunit='cm', onlyAV=True, mcpu=4):
        '''
        coord: (lon, lat)
        timeseries : list/pd.DatatimeIndex
        '''
        coord = np.array([gpsts.lon, gpsts.lat])
        timeseries = gpsts.time
        as_av = self.calAVatcoord(coord, timeseries, tau, alpha=alpha, timeunit=timeunit, dispunit=dispunit, onlyAV=onlyAV, mcpu=mcpu)
        # All Done
        return as_av # eqtools.DispBase对象
    
    def calAVatcoord(self, coord, timeseries, tau, alpha=1.0, timeunit='D', dispunit='cm', onlyAV=True, mcpu=4):
        '''
        coord: 1D or 2D [lon, lat] or [[lon, lat], [lon2, lat2]]
        timeseries : list/pd.DatatimeIndex

        return : eqtools.DispBase对象
        '''
        from ..pscmp_visco_postseismic import calAS_AV
        griddata = self.asviscogrid
        coord = np.array(coord)
        if coord.ndim == 1:
            coord = coord.reshape(-1, 2)
        time = pd.DatetimeIndex(timeseries)
        tsfloat = self._todotyr(time)*self._timescale(distunit=timeunit)
        if onlyAV:
            reftime = 0
        else:
            reftime = None
        intpdata = griddata.extractDisp(sitecoords=coord, dispunit=dispunit, timeunit=timeunit, dispInTimes=None, interpTimeMethod='linear', refTime=reftime)
        intpdata_pd = intpdata.to_DataFrame(dispunit=dispunit, tunit=timeunit, dt2date=False, origintime=self.eqdate)
        as_av = calAS_AV(intpdata_pd, time, eqdate=self.eqdate, alpha=alpha, tau=tau, unit=dispunit, intp_tunit=timeunit, onlyAV=onlyAV, mcpu=mcpu)

        # 需要保证排序正确
        disp_col = 'E({0}) N({0}) U({0})'.format(dispunit).split() 
        dispts = as_av.loc[:, disp_col].sort_index(axis=0, level=0).values

        # 转换为DispBase格式
        from .DispBase import DispBase
        disp = DispBase('as_av', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0)
        disp.timeseries = tsfloat
        disp.timeunit = timeunit
        disp.dispunit = dispunit
        z = np.zeros((coord.shape[0], ))
        disp.llz = np.hstack((coord, z[:, None]))
        x, y = disp.ll2xy(coord[:, 0], coord[:, 1])
        disp.xyz = np.vstack((x, y, z)).T 
        disp.dispts = dispts.reshape((-1, coord.shape[0], 3))

        # All Done
        return disp # eqtools.DispBase对象
    #--------------------------------------------------------------------------------------------#

    #-----------------------------Calculate cumulative displacement------------------------------#
    def calcumdispatdate2gps(self, coords, date, tau, alpha, stations=None, timeunit='D', dispunit='cm', 
                             mcpu=4, ref2time=False, reftime=None):
        '''
        
        '''
        coords = np.array(coords)
        if coords.ndim == 1:
            coords = coords.reshape(-1, 2)
        time = pd.DatetimeIndex([date])
        # calculate cv, av and as
        cv = self.calCVatcoord(coords, time, timeunit=timeunit, dispunit=dispunit)
        av_as = self.calAVatcoord(coords, time, tau, alpha, timeunit=timeunit, dispunit=dispunit, onlyAV=False, mcpu=mcpu)
        cumdisp = cv.dispts[0, :, :] + av_as.dispts[0, :, :]

        cumgps = gps('{0:%Y-%m-%d}'.format(time[0]), lon0=self.lon0, lat0=self.lat0, utmzone=self.utmzone)
        cumgps.lon = coords[:, 0]
        cumgps.lat = coords[:, 1]
        if stations is not None:
            cumgps.station = stations
        else:
            cumgps.station = None
        cumgps.vel_enu = cumdisp
        cumgps.err_enu = np.ones_like(cumgps.vel_enu)

        # All Done
        return cumgps 

    def caldispAtGPS(self, gps, timeseries, tau, alpha, timeunit='D', dispunit='cm', 
                             mcpu=4, ref2time=False, reftime=None):
        '''
        timeseries : list of time str or pd.DatetimeIndex or time str or pd.Timestamp
        '''
        coords = np.vstack((gps.lon, gps.lat)).T
        if type(timeseries) in (str,):
            time = pd.DatetimeIndex([timeseries])
        elif hasattr(timeseries, '__len__'):
            time = pd.DatetimeIndex(timeseries)
        else:
            time = pd.DatetimeIndex([timeseries])

        # calculate cv, av and as
        cv = self.calCVatcoord(coords, time, timeunit=timeunit, dispunit=dispunit)
        av_as = self.calAVatcoord(coords, time, tau, alpha, timeunit=timeunit, dispunit=dispunit, onlyAV=False, mcpu=mcpu)
        dispts = cv.dispts + av_as.dispts

        out = copy.deepcopy(gps)
        out.name = '{0:%Y-%m-%d}'.format(time[0])
        out.initializeTimeSeries(time=time, los=True, verbose=False)

        for idx, istat in enumerate(out.station):
            out.timeseries[istat].east.value = dispts[:, idx, 0]
            out.timeseries[istat].north.value = dispts[:, idx, 1]
            out.timeseries[istat].up.value = dispts[:, idx, 2]

        # All Done
        return out 
    
    def extractAroundGPSfromsarts(self, sarts, gpsnet, distance, doprojection=True, reference=False):
        '''
        Returns a gps object with values projected along the LOS around the 
        gps stations included in gps. In addition, it projects the gps displacements 
        along the LOS

        Args:
            * gps           : gps object
            * distance      : distance to consider around the stations

        Kwargs:
            * doprojection  : Projects the gps enu disp into the los as well
            * reference     : if True, removes to the InSAR the average gps displacemnt in the LOS for the points overlapping in time.
            * verbose       : Talk to me

        Returns:
            * gps with vel_los, err_los, los 
        '''
        out = sarts.extractAroundGPS(gpsnet, distance, doprojection, reference)
        # All Done
        return out

    def calcumdispatdate2gmt(self, coords, date, tau, alpha, stations=None, timeunit='D', dispunit='cm', 
                             mcpu=4, ref2time=False, reftime=None, gmtfile='cumdisp.gmt'):
        '''
        
        '''
        cumgps = self.calcumdispatdate2gps(coords, date, tau, alpha, stations, timeunit, dispunit, mcpu, ref2time, reftime)
        coords = np.vstack((cumgps.lon, cumgps.lat, np.zeros_like(cumgps.lat))).T 
        vel = cumgps.vel_enu 
        err = cumgps.err_enu 
        renu = np.zeros_like(err)
        gmtinfo = np.hstack((coords, vel, err, renu))
        pdgps = pd.DataFrame(gmtinfo, columns='Lon Lat dep(m) de({0}) dn({0}) du({0}) sde({0}) sdn({0}) sdu({0}) ren reu rnu'.format(dispunit).split())
        if cumgps.station is not None:
            pdgps['Sta'] = cumgps.station
        
        pdgps.to_csv(gmtfile, sep=' ', index=False, float_format='%.6f')

        # All Done
        return

    #--------------------------------------------------------------------------------------------#

    #---------------------------------------Get profile-------------------------------------------#
    def getprofile(self, tau, alpha, profname, loncenter, latcenter, length, azimuth, width, data='data', 
                timeseries=None, timeunit='D', dispunit='cm', mcpu=4):
        '''
        获取跨断层的剖面数据。

        参数:
        tau: 时间常数，用于计算剖面上各点的位移。
        alpha: 空间常数，用于计算剖面上各点的位移。
        profname: 剖面的名称。
        loncenter, latcenter: 剖面的中心坐标（经度和纬度）。
        length: 剖面的长度。
        azimuth: 剖面的方位角。
        width: 剖面的宽度。
        data: 使用的数据类型，默认为 'data'。
        timeseries: 时间序列，如果为 None，则使用默认的时间序列。
        timeunit: 时间单位，默认为 'D'（天）。
        dispunit: 位移单位，默认为 'cm'（厘米）。
        mcpu: 用于计算的 CPU 数量，默认为 4。

        返回值:
        无。结果将保存在 self.profiles[profname] 中。
        '''

        cogrid = self.coviscogrid
        # asgrid = self.asviscogrid  

        if timeseries is None:
            timedt = cogrid.timeseries.copy()
            cogrid_timeunit = cogrid.timeunit
            timedt *= self._timescale(cogrid_timeunit, timeunit)
            timeseries = pd.TimedeltaIndex(timedt, unit=timeunit) + self.eqdate

        cogrid.getprofile(profname, loncenter, latcenter, length, azimuth, width, data)
        # self.profiles = cogrid.profiles  
        if not hasattr(self, 'profiles'):
            self.profiles = {}
        self.profiles[profname] = cogrid.profiles[profname]
        prof = self.profiles[profname]
        Bol = prof['Station Index']

        # Get the coordinates of the colocalized points.
        colon = cogrid.lon[Bol]
        colat = cogrid.lat[Bol]
        # cox = cogrid.x[Bol]
        # coy = cogrid.y[Bol]

        coord = np.vstack((colon, colat)).T
        cvatcocoord = self.calCVatcoord(coord, timeseries, timeunit=timeunit, dispunit=dispunit)
        as_avatcoord = self.calAVatcoord(coord, timeseries, tau, alpha, timeunit=timeunit, dispunit=dispunit, onlyAV=False, mcpu=mcpu)

        prof['timeseries'] = timeseries
        prof['cv_deformation'] = cvatcocoord.dispts
        prof['as_av_deformation'] = as_avatcoord.dispts
        prof['total_deformation'] = cvatcocoord.dispts + as_avatcoord.dispts
        prof['Parallel Velocity'] = np.sum(prof['Vectors'][0].reshape(1, 1, 2)*prof['total_deformation'][:, :, :-1], axis=2)
        prof['Normal Velocity'] = np.sum(prof['Vectors'][1].reshape(1, 1, 2)*prof['total_deformation'][:, :, :-1], axis=2)
        
        # All done
        return

    def get_profile_with_coords(self, coords, tau, alpha, profname, loncenter, latcenter, 
                                length, azimuth, width, data='data', 
                                timeseries=None, timeunit='D', dispunit='cm', mcpu=4):
        '''
        根据输入的共址坐标数组获取跨断层的剖面数据。

        参数:
        coords: 共址坐标数组。
        其他参数与 getprofile 方法相同。

        返回值:
        无。结果将保存在 self.profiles[profname] 中。
        '''

        cogrid = self.coviscogrid

        if timeseries is None:
            timedt = cogrid.timeseries.copy()
            cogrid_timeunit = cogrid.timeunit
            timedt *= self._timescale(cogrid_timeunit, timeunit)
            timeseries = pd.TimedeltaIndex(timedt, unit=timeunit) + self.eqdate

        cogrid.getprofile(profname, loncenter, latcenter, length, azimuth, width, data)

        if not hasattr(self, 'profiles'):
            self.profiles = {}
        self.profiles[profname] = cogrid.profiles[profname]
        prof = self.profiles[profname]

        # Find the nearest stations in Bol for the input coordinates.
        Bol = prof['Station Index']
        tree = KDTree(np.vstack((cogrid.lon[Bol], cogrid.lat[Bol])).T)
        _, Bol = tree.query(coords, k=1)
        prof['Station Index'] = prof['Station Index'][Bol]
        # Update Distance
        prof['Distance'] = prof['Distance'][Bol]
        prof['Normal Distance'] = prof['Normal Distance'][Bol]

        # Get the coordinates of the colocalized points.
        # colon = cogrid.lon[Bol]
        # colat = cogrid.lat[Bol]
        # Note: The coordinates of the colocalized points are not necessarily the same as the Bol coordinates.
        colon = coords[:, 0]
        colat = coords[:, 1]

        coord = np.vstack((colon, colat)).T
        cvatcocoord = self.calCVatcoord(coord, timeseries, timeunit=timeunit, dispunit=dispunit)
        as_avatcoord = self.calAVatcoord(coord, timeseries, tau, alpha, timeunit=timeunit, dispunit=dispunit, onlyAV=False, mcpu=mcpu)

        prof['timeseries'] = timeseries
        prof['cv_deformation'] = cvatcocoord.dispts
        prof['as_av_deformation'] = as_avatcoord.dispts
        prof['total_deformation'] = cvatcocoord.dispts + as_avatcoord.dispts
        prof['Parallel Velocity'] = np.sum(prof['Vectors'][0].reshape(1, 1, 2)*prof['total_deformation'][:, :, :-1], axis=2)
        prof['Normal Velocity'] = np.sum(prof['Vectors'][1].reshape(1, 1, 2)*prof['total_deformation'][:, :, :-1], axis=2)

        # All done
        return

    def intersectProfileFault(self, name, fault):
        '''
        Gets the distance between the fault/profile intersection and the profile center.

        Args:
            * name      : name of the profile.
            * fault     : fault object from verticalfault.

        Returns:
            * distance  : float
        '''

        # Grab the fault trace
        xf = fault.xf
        yf = fault.yf

        # Grab the profile
        prof = self.profiles[name]

        # import shapely
        import shapely.geometry as geom
        
        # Build a linestring with the profile center
        Lp = geom.LineString(prof['EndPoints'])

        # Build a linestring with the fault
        ff = []
        for i in range(len(xf)):
            ff.append([xf[i], yf[i]])
        Lf = geom.LineString(ff)

        # Get the intersection
        if Lp.crosses(Lf):
            Pi = Lp.intersection(Lf)
            p = Pi.coords[0]
        else:
            return None

        # Get the center
        lonc, latc = prof['Center']
        xc, yc = self.ll2xy(lonc, latc)

        # Get the sign 
        xa,ya = prof['EndPoints'][0]
        vec1 = [xa-xc, ya-yc]
        vec2 = [p[0]-xc, p[1]-yc]
        sign = np.sign(np.dot(vec1, vec2))

        # Compute the distance to the center
        d = np.sqrt( (xc-p[0])**2 + (yc-p[1])**2)*sign

        # All done
        return d

    def plotprofile(self, profname, direction='parallel', dispunit='cm', figsize=None, 
                    style=['science'], fontsize=None, legend_frame=True, 
                    time_index=-1, time=None, offset=0, csigpsobj=None, ylim=None, 
                    use_mathtext=True, usetex=True, plot_values=['total', 'cv', 'as_av']):
        '''
        Plot the profile.
        '''
        import scienceplots
        plt.style.use(style)
        if legend_frame:
            plt.rc('legend', frameon=True, framealpha=0.7,
                fancybox=True, numpoints=1)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.formatter.use_mathtext'] = use_mathtext
        plt.rcParams['text.usetex'] = usetex
        plt.rcParams['mathtext.fontset'] = 'dejavusans'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica','DejaVu Sans', 'Bitstream Vera Sans', 
                                        'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 
                                        'Lucid', 'Avant Garde', 'sans-serif']
        if figsize is not None:
            plt.rcParams['figure.figsize'] = figsize
        if fontsize is not None:
            plt.rcParams['axes.labelsize'] = fontsize
            plt.rcParams['axes.labelsize'] = fontsize
            plt.rcParams['xtick.labelsize'] = fontsize
            plt.rcParams['ytick.labelsize'] = fontsize
            plt.rcParams['legend.fontsize'] = fontsize
            plt.rcParams['font.size'] = fontsize
        
        prof = self.profiles[profname]
        timeseries = prof['timeseries']

        # If time is provided, find the closest index in timeseries
        if time is not None:
            time_index = (np.abs(timeseries - time)).argmin()

        cv_deformation = self.profiles[profname]['cv_deformation'][time_index, :, :]
        as_av_deformation = self.profiles[profname]['as_av_deformation'][time_index, :, :]
        total_deformation = self.profiles[profname]['total_deformation'][time_index, :, :]

        # 根据用户输入的参数来设置perd_vert，ylim和输出文件名
        if direction == 'normal':
            projection_vector = self.profiles[profname]['Vectors'][1]
            filename = 'Profile_acrossfault_perpendicular'
            direction = 'perpendicular'
        elif direction == 'parallel':
            projection_vector = self.profiles[profname]['Vectors'][0]
            filename = 'Profile_acrossfault_parallel'
            direction = 'parallel'
        elif direction == 'east':
            projection_vector = np.array([1, 0])
            filename = 'Profile_acrossfault_east'
            direction = 'east'
        elif direction == 'north':
            projection_vector = np.array([0, 1])
            filename = 'Profile_acrossfault_north'
            direction = 'north'
        else:
            assert False, 'direction must be one of "parallel", "normal", "east" or "north".'

        cv_projection = np.dot(cv_deformation[:, :-1], projection_vector)
        as_av_projection = np.dot(as_av_deformation[:, :-1], projection_vector)
        total_projection = np.dot(total_deformation[:, :-1], projection_vector)
        if 'total' in plot_values:
            plt.scatter(self.profiles[profname]['Distance'] - offset, total_projection, label='CV + AS + AV')
        if 'cv' in plot_values:
            plt.scatter(self.profiles[profname]['Distance'] - offset, cv_projection, label='CV')
        if 'as_av' in plot_values:
            plt.scatter(self.profiles[profname]['Distance'] - offset, as_av_projection, label='AS + AV')

        if csigpsobj is not None:
            # Plot GPS Observations
            x = csigpsobj.profiles[profname]['Distance'] - offset
            # 获取需要的站点名称
            required_stations = csigpsobj.profiles[profname]['Stations']
            # 找到这些站点在gps_enu.station中的索引
            indices = [np.where(csigpsobj.station == station)[0][0] for station in required_stations]
            # 使用这些索引来提取vel_enu和err_enu
            vel_enu_selected = csigpsobj.vel_enu[indices, :-1]
            err_enu_selected = csigpsobj.err_enu[indices, :-1]
            # 计算投影
            y = np.dot(vel_enu_selected, projection_vector)
            yerr = np.sqrt(np.sum((err_enu_selected * projection_vector)**2, axis=1))

            plt.errorbar(x, y, yerr=yerr*2, fmt='o', color='#ff2c00', capsize=3, ecolor='#828282', 
                        mec='#ff2c00', label='GPS')
        plt.axhline(y=0, xmin=0, xmax=1) #, label='Zero dsip.'
        plt.axvline(x=0, ymin=0, ymax=1, c='r')
        # ax = plt.gca()
        # ax.annotate(
        #     "KPJF",
        #     xy=(0, 4.0),
        #     xytext=(30, 2.0),
        #     # bbox={"boxstyle": "round", "fc": "none", "ec": "g"},
        #     arrowprops={"arrowstyle": "->"},
        # )
        plt.legend()
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel('Distance from fault (km)')
        plt.ylabel(f'Fault-{direction} disp. ({dispunit})')
        plt.savefig(f'{filename}.tif', dpi=300)
        plt.savefig(f'{filename}.pdf')
        
        return plt.gcf()

    #--------------------------------------------------------------------------------------------#

    #---------------------------------------Plot image-------------------------------------------#
    def plottsatsite(self, geosite, tau=None, alpha=None, direction='EN', timeunit='D', 
                     figsize=(7.0, 1.8), dispunit='cm', timeinterval=1, ref2start=False, 
                     style=['science', 'nature'], fontsize=None, legend_frame=True):
        '''
        geosite: csi gps timeseries object (8.0, 3.0)
        return : fig obj 单栏：3.3，最大3.5, 双栏：7.0

        style='seaborn-ticks' ['science', 'nature']
        
        print(plt.style.available)打印可用的style
        '''

        ncols = len(direction)
        direction = direction.upper()

        # Set default properties for plotting
        import scienceplots
        plt.style.use(style)
        if legend_frame:
            plt.rc('legend', frameon=True, framealpha=0.7,
                fancybox=True, numpoints=1)
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans', 'Arial', 'Lucida Grande', 'Verdana', 
                                        'Geneva, Lucid', 'Avant Garde', 'sans-serif']

        plt.rcParams['figure.figsize'] = figsize

        if fontsize is not None:
            plt.rcParams['axes.labelsize'] = fontsize
            plt.rcParams['axes.labelsize'] = fontsize
            plt.rcParams['xtick.labelsize'] = fontsize
            plt.rcParams['ytick.labelsize'] = fontsize
            plt.rcParams['legend.fontsize'] = fontsize
            plt.rcParams['font.size'] = fontsize

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
        if ncols == 1:
            axs = [axs]
            direction = [direction]
        obst = self._todotyr(geosite.time)*self._timescale(distunit=timeunit)
        dirdict = {
            'E': 'east',
            'N': 'north',
            'U': 'up'
        }
        if hasattr(geosite.east, 'cv'):
            for idir, iax in zip(direction, axs):
                idata = getattr(geosite, dirdict[idir])
                obs = idata.value
                err = idata.error
                cv = idata.cv
                if hasattr(idata, 'cumas'):
                    cumas = idata.cumas
                    av = idata.av
                    sim = cumas + av + cv
                else:
                    as_av = idata.as_av
                    sim = cv + as_av
                
                # Plot
                if ref2start:
                    offset = np.interp([obst[0]], obst, sim)
                    iobs = obs - obs[0] + offset
                else:
                    iobs = obs
                iax.errorbar(obst, iobs, err, fmt='o', ms=3, errorevery=1, 
                             ecolor='#d1d1d1', label='Obs.', zorder=0)
                iax.plot(obst, cv, label='CV')
                if hasattr(idata, 'cumas'):
                    iax.plot(obst, cumas, label='AS')
                    iax.plot(obst, av, label='AV')
                else:
                    iax.plot(obst, as_av, label='AS+AV')
                iax.plot(obst, sim, label='CV+AS+AV', c='k')
                iax.set_xlabel('Lapsed time ({0})'.format(timeunit))
                iax.set_ylabel('{0} displacement ({1})'.format(dirdict[idir].capitalize(), dispunit))
        else:
            assert tau is not None and alpha is not None, 'Tau and Alpha must be inputed in this case.'
            # 单独计算, timeinterval单位为timeunit
            simt = np.arange(0, obst.max()+timeinterval, timeinterval)
            time = self.eqdate + pd.TimedeltaIndex(simt, unit=timeunit)
            coord = np.array([geosite.lon, geosite.lat])

            cv = self.calCVatcoord(coord, time, timeunit=timeunit, dispunit=dispunit)
            cumas = self.calASatcoord(coord, time, tau, alpha, dispunit=dispunit, timeunit=timeunit)
            av = self.calAVatcoord(coord, time, tau, alpha, timeunit=timeunit, dispunit=dispunit, onlyAV=True, mcpu=4)

            for idir, iax in zip(direction, axs):
                idata = getattr(geosite, dirdict[idir])
                obs = idata.value
                err = idata.error
                if idir == 'E':
                    icumas = cumas.dispts[:, 0, 0]
                    iav = av.dispts[:, 0, 0]
                    icv = cv.dispts[:, 0, 0]
                elif idir == 'N':
                    icumas = cumas.dispts[:, 0, 1]
                    iav = av.dispts[:, 0, 1]
                    icv = cv.dispts[:, 0, 1]
                else:
                    icumas = cumas.dispts[:, 0, 2]
                    iav = av.dispts[:, 0, 2]
                    icv = cv.dispts[:, 0, 2]
                
                if ref2start:
                    sim = icumas + iav + icv
                    offset = np.interp([obst[0]], simt, sim)
                    iobs = obs - obs[0] + offset
                else:
                    iobs = obs
                iax.errorbar(obst, iobs, err, fmt='o', ms=3, errorevery=1, 
                             ecolor='#d1d1d1', label='{0} Obs.'.format(geosite.name), zorder=0)

                # Plot
                iax.plot(simt, icv, label='CV')
                iax.plot(simt, icumas, label='AS')
                iax.plot(simt, iav, label='AV')
                iax.plot(simt, icv + iav + icumas, label='CV+AS+AV', c='k')
                iax.set_xlabel('Lapsed time ({0})'.format(timeunit.lower()))
                iax.set_ylabel('{0} displacement ({1})'.format(dirdict[idir].capitalize(), dispunit))
        for iax in axs:
            # iax.tick_params(axis='y', left=True, right=True)
            iax.set_xlim(0, np.ceil(obst.max()))
        axs[0].legend()
        # plt.tight_layout()
        # All Done
        return fig 

    #--------------------------------------------------------------------------------------------#

    #-----------------------------------Calculate Misfit-----------------------------------------#

    def calMisfit(self, alpharange, taurange, sarweight=1, visco=1.0e18, orbit=False,
                  ref2start=True, timeunit='D', dispunit='cm', mcpu=4):
        '''
        sarweight: default gpsweight = 1
        e.g., alpharange = np.arange(0.001, 0.3, 0.01); taurange = np.arange(5, 75, 5)
        '''

        if not hasattr(self, 'gpsmisfitdict'):
            self.gpsmisfitdict = {}
            self.sarmisfitdict = {}
        for ialpha in alpharange:
            for itau in taurange:
                self.gpsmisfitdict[(visco, itau, ialpha)] = 0.0
                self.sarmisfitdict[(visco, itau, ialpha)] = 0.0

        
        self.calCV(timeunit=timeunit, dispunit=dispunit)
        for itau in taurange:
            self.calAV(itau, alpha=1.0, timeunit=timeunit, dispunit=dispunit, onlyAV=False, mcpu=mcpu)
            for ialpha in alpharange:
                for igeotsname, igeots in self.geotsdatadict.items():
                    if igeots.dtype in ('gpstimeseries', ):
                        iobs = np.vstack((igeots.east.value, igeots.north.value)).T
                        ierr = np.hstack((igeots.east.error, igeots.north.error))
                        icv = np.vstack((igeots.east.cv, igeots.north.cv)).T
                        ias_av = np.vstack((igeots.east.as_av, igeots.north.as_av)).T
                        if ref2start:
                            # obs
                            iobs -= iobs[0, :][np.newaxis, :]
                            # sim
                            icv -= icv[0, :][np.newaxis, :]
                            ias_av -= ias_av[0, :][np.newaxis, :]
                        iobs = iobs.flatten(order='F')
                        icv = icv.flatten(order='F')
                        ias_av = ias_av.flatten(order='F')

                        # for ialpha in alpharange:
                        self.gpsmisfitdict[(visco, itau, ialpha)] += np.sum((iobs - icv - ialpha*ias_av)**2/ierr**2)
                        # print(igeots.name, np.sum((iobs - icv - ialpha*ias_av)**2/ierr**2))
                    elif igeots.dtype in ('insartimeseries',):
                        if ref2start:
                            sar0 = igeots.timeseries[0]
                            for isar in igeots.timeseries[1:]:
                                iobs = isar.vel - sar0.vel
                                ierr = isar.err
                                icv = isar.cv - sar0.cv
                                ias_av = isar.as_av - sar0.as_av
                                # Considering the effect from orbit bias
                                if orbit:
                                    isim = icv + ialpha*ias_av + isar.orbit
                                else: 
                                    isim = icv + ialpha*ias_av
                                # for ialpha in alpharange:
                                self.sarmisfitdict[(visco, itau, ialpha)] += np.sum((iobs - isim)**2/ierr**2)
                        else:
                            for isar in igeots.timeseries:
                                iobs = isar.vel
                                ierr = isar.err
                                icv = isar.cv
                                ias_av = isar.as_av
                                # for ialpha in alpharange:
                                self.sarmisfitdict[(visco, itau, ialpha)] += np.sum((iobs - icv - ialpha*ias_av)**2/ierr**2)

        # All Done
        return
    
    def savemisfit2pickle(self, dirname='.', filename_suffix='_misfit.pkl'):
        from pickle import dump 
        gpsmisfit_filename = os.path.join(dirname, 'gpsts'+filename_suffix)
        with open(gpsmisfit_filename, 'wb') as fout:
            dump(self.gpsmisfitdict, fout)
        
        sarmisfit_filename = os.path.join(dirname, 'sarts'+filename_suffix)
        with open(sarmisfit_filename, 'wb') as fout:
            dump(self.sarmisfitdict, fout)
        
        # All Done
        return

    #--------------------------------------------------------------------------------------------#
    @staticmethod
    def convert_timeunit(unit):
        if unit in ('Y', 'y', 'year'):
            return 365.2425 * 24 * 60 * 60  # a year is approximately 365.2425 days
        elif unit in ('M', 'm', 'month'):
            return 30.4369 * 24 * 60 * 60  # a month is approximately 30.44 days
        elif unit in ('W', 'w', 'week'):
            return 7 * 24 * 60 * 60  # a week is 7 days
        elif unit in ('D', 'd', 'day'):
            return 24 * 60 * 60  # a day is 24 hours
        else:  # 'S', 's', 'sec', 'second'
            return 1  # a second is the base unit

    def _todotyr(self, timeseries):
        if not hasattr(self, 'eqdate'):
            raise AttributeError("The 'eqdate' attribute is not set. Please set it before calling this method.")
        dtinpdtime = pd.DatetimeIndex(timeseries) - self.eqdate
        dtfloat = dtinpdtime.total_seconds().values / self.convert_timeunit('Y')  # unit in year 
        return dtfloat

    def _timescale(self, orgunit='Y', distunit='D'):
        if orgunit not in self.VALID_TIMEUNITS:
            raise ValueError(f"Invalid original unit: {orgunit}. Valid units are: {self.VALID_TIMEUNITS}")
        if distunit not in self.VALID_TIMEUNITS:
            raise ValueError(f"Invalid destination unit: {distunit}. Valid units are: {self.VALID_TIMEUNITS}")

        orgunit_in_seconds = self.convert_timeunit(orgunit)
        distunit_in_seconds = self.convert_timeunit(distunit)

        return pd.Timedelta(orgunit_in_seconds, 's') / pd.Timedelta(distunit_in_seconds, 's')


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    
    from collections import OrderedDict
    from csi.gps import gps
    from csi.insar import insar
    from csi.gpstimeseries import gpstimeseries
    from csi.insartimeseries import insartimeseries
    from eqtools.dispExtract import PscmpDisp, PylithDisp
    from pandas import IndexSlice as idx
    from glob import glob
    import re
    # -----------------------------------Proj Information-------------------------------------#
    from global_share import *
    import sys
    tsdate = '2021-08-24'

    lon0 = 98.25
    lat0 = 34.5
    # ---------------------------------Generate GPS Object------------------------------------#
    gpstsdata = []
    sitepattern = r'*_obs_csi.csv' # r'*_obs_csi.csv'
    gpstsdir = r'd:\Maduo_Postseismic\Postseismic_Inversion\GPSData\posfile\img_xiong'
    gpstspathdir = r'd:\Maduo_Postseismic\Postseismic_Inversion\GPSData\posfile'
    sitefiles = glob(os.path.join(gpstsdir, sitepattern))
    siteinfo = pd.read_csv(os.path.join(gpstspathdir, 'cor_cGPS.txt'), sep='\s+', index_col=2)
    station = []
    for isitefile in sitefiles:
        pattern = r'(?<=\\)(\w{4})(?=_obs_csi)'
        isitename = re.search(pattern, isitefile).group(0)
        igpsts = gpstimeseries(isitename, lon0=lon0, lat0=lat0)
        igpsts.read_from_file(isitefile)
        igpsts.lon = siteinfo.loc[isitename, 'Lon']
        igpsts.lat = siteinfo.loc[isitename, 'Lat']
        igpsts.lonlat2xy()
        gpstsdata.append(igpsts)
        station.append(isitename)

    # ----------------------------------Generate SAR Object-----------------------------------#
    # TODO : 将orbit信息在这里保存到sar时间序列中，用于计算残差
    asarts_t099a = insartimeseries('T099A', utmzone=utmzone, ellps='WGS84')
    dsarts_t106d = insartimeseries('T106D', utmzone=utmzone, ellps='WGS84')

    asarbasedir = r'd:\Maduo_Postseismic\T099A\tsfit_fliter'
    dsarbasedir = r'd:\Maduo_Postseismic\T106D\tsfit_fliter'
    # sar形变保存在cm级
    asarts_t099a.read_from_sarfiles(obsdate[1:], asarbasedir, r'S1T099A{0:%Y-%m-%d}_ifg', factor=1, factor_obj='vel')
    dsarts_t106d.read_from_sarfiles(obsdate[1:], dsarbasedir, r'S1T106D{0:%Y-%m-%d}_ifg', factor=1, factor_obj='vel')

    sartsdata = [asarts_t099a, dsarts_t106d]
    # 组合时序观测数据
    geotsdata = gpstsdata # + sartsdata
    #--------------------------------------------CSI Part--------------------------------------#

    # 计算映射函数对象，类似于断层中的格林函数
    ## cogrid
    ivisco= 0.5 # 2.0
    ivisco = ivisco*ebase
    iname = f'{ivisco:.1e}'.replace('.', '_').replace('+', '')
    dirname = os.path.join('..', f'output{iname}')
    filename = os.path.join(dirname, 'covisco-groundsurf.h5')
    projstr = '+proj=utm +lon_0=98.25 +lat_0=34.5'
    cogrid = PylithDisp('pylith', filename=filename, lon0=lon0, lat0=lat0)
    cogrid.readdispts(filename, projInPylith=projstr)

    ## asgrid pylith
    dirname = os.path.join(r'f:\maduo_shearzone\multifault\burgers1e19_as\shearzone_w40km\maduo_shear_multifault_t10km', f'output{iname}')
    filename = os.path.join(dirname, 'covisco-groundsurf.h5')
    projstr = '+proj=utm +lon_0=98.25 +lat_0=34.5'
    asgrid = PylithDisp('pylith', filename=filename, lon0=lon0, lat0=lat0)
    asgrid.readdispts(filename, projInPylith=projstr)

    # asgrid pscmp
    dirname = os.path.join(r'e:\Maduo_psgrn_pscmp\Maxwell_32km_crust_mantle', 
                            'afterslip_driven_visco', 'pscmp1_0', f'pscmp_gps_lc{iname}_regular')
    filename = None # 使用默认模板匹配方法：snapshot_*.dat
    asgrid = PscmpDisp('pscmp', filename=filename, lon0=lon0, lat0=lat0)
    asgrid.readdispts(filename, dirname=dirname, projInPscmp=None)

    # 解算
    alpha = 0.071
    tau = 65

    msolve = combinedmodelsolve('msolve', geotsdatalist=geotsdata, eqdate=eqdate)
    msolve.getcoviscogrid(cogrid)
    msolve.getasviscogrid(asgrid)
    # msolve.calCV()
    # msolve.calAS(tau=tau, alpha=alpha)
    # msolve.calAV(tau)

    # 绘制gps拟合图
    for isitename, icsisite in msolve.geotsdatadict.items():
        if icsisite.dtype in ('gpstimeseries',):
            fig = msolve.plottsatsite(msolve.geotsdatadict[isitename], tau=tau, alpha=alpha, direction='EN', timeunit='D', dispunit='cm', 
                                      timeinterval=2, ref2start=True, style=['science', 'nature'], figsize=(7, 2.5), fontsize=None) # 7, 2.625
            fig.savefig(os.path.join('Modeling_gpsts', '{0}_EN.pdf'.format(isitename)), dpi=300) # _pylith
            plt.close(fig)
    

    # 计算GPS点位的时序拟合，由于这个时序是时间点固定的，通常可用作InSAR采样点用
    timedt = np.arange(0, 172, 1)
    timeseries = pd.TimedeltaIndex(timedt, unit='D') + msolve.eqdate
    gpsnet = gps('Maduo', lon0=lon0, lat0=lat0)
    gpsnet.setStat(station, siteinfo.loc[station, 'Lon'].values, siteinfo.loc[station, 'Lat'].values)
    gpsout = msolve.caldispAtGPS(gpsnet, timeseries, tau, alpha, timeunit='D', dispunit='cm', 
                             mcpu=4, ref2time=False, reftime=None)

    out = msolve.extractAroundGPSfromsarts(asarts_t099a, gpsout, 3, doprojection=True, reference=False)