from csi import SourceInv
from copy import deepcopy
from scipy.interpolate import griddata
import numpy as np
import pandas as pd 
from csi import csiutils as utils
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class DispBase(ABC, SourceInv):
    VALID_TIMEUNITS = ('Y', 'y', 'year', 'M', 'm', 'month', 'W', 'w', 'week', 'D', 'd', 'day', 'S', 's', 'sec', 'second')
    VALID_DISPUNITS = ('m', 'dm', 'cm', 'mm', 'km')

    def __init__(self, name, program=None, filename=None, 
                 utmzone=None, ellps='WGS84', lon0=None, lat0=None):
        super(DispBase, self).__init__(name, utmzone, ellps, lon0, lat0)
        self.filename = filename
        self.program = name if program is None else program
        self.xyz = None
        self.llz = None
        self.timeseries = None
        self.dispts = None # (nt, np, nd)
        self.dispunit = 'm'
        self.timeunit = 'Y' # use to pd.Timedelta()
        self.origintime = None
    
    @abstractmethod
    def readdispts(self, filename=None, **kwargs):
        pass
    
    def findNearestNeighborIndex(self, xyz_sample, xyz_training, utm=True):
        '''
        Discover the index position of the training set in the sample set
        
        Args   :
            * xyz_sample   : sample set 
            * xyz_training : training set 
            
        Kwargs :
            * utm          : default True; Whether the input coordinates are projection coordinates
        
        Return :
            * d            : distance
            * x            : index
        '''
        from scipy.spatial import cKDTree # call the nearest index
        
        ndim = xyz_sample.shape[1]
        assert ndim == xyz_training.shape[1], 'The spatial dimension must be equal between sample and training coordinates'
        
        if utm:
            xyz1 = xyz_sample
            xyz2 = xyz_training
        else:
            x1, y1 = self.ll2xy(xyz_sample[:, 0], xyz_sample[:, 1])
            x2, y2 = self.ll2xy(xyz_training[:, 0], xyz_training[:, 1])
            if ndim == 2:
                z1 = np.zeros_like(x1)
                z2 = np.zeros_like(x2)
            else:
                z1 = xyz_sample[:, 2]
                z2 = xyz_training[:, 2]
            xyz1 = np.vstack((x1, y1, z1)).T 
            xyz2 = np.vstatck((x2, y2, z2)).T
            
        # Find the index of the nearest point
        kd = cKDTree(data=xyz1, leafsize=1000)
        # d: distance, x: index
        dist, indx = kd.query(xyz2)
        
        # All Done
        return dist, indx
    
    def extractDisp(self, sitecoords=None, interpDispMethod='nearest', utm=False,
                          dispunit='m', timeunit='Y', dispInTimes=None, interpTimeMethod='nearest', refTime=None):
        '''
        Extract disp timeseries in site coords
            * If utm is False, format is Lon Lat

            * If utm is True, format is X Y (in km)

        Args:
            * sitecoords : Coordinates of stations to be extracted.

        Kwargs:
            * utm : Specify nature of coordinates

            * interpMethod : Interpolating method used in griddata
            * dispInTimes  : list/ndarray/None; unit in 'Y'
            * refTime      : None/0/other

        Returns:
            * dispInVerts, dist, indx
        '''
        from scipy.interpolate import griddata
        from scipy.interpolate import interp1d
        
        xyz = self.xyz
        if sitecoords is None:
            # Extract all stations
            dist = np.zeros_like(xyz[:, 0])
            indx = np.arange(xyz.shape[0])
            grid_valsinspace = self.dispts
        else:
            if utm:
                xi, yi = sitecoords[:, 0], sitecoords[:, 1]
            else:
                xi, yi = self.ll2xy(sitecoords[:, 0], sitecoords[:, 1])
            zi = np.zeros_like(xi)
            xyzi = np.vstack((xi, yi, zi)).T
        
            # Find the index of the nearest point
            dist, indx = self.findNearestNeighborIndex(xyz, xyzi, utm=True)

            # （nt, nc, nDim） to (nc, nt, nDim)
            value = np.swapaxes(self.dispts, 0, 1)
            grid_valsinspace = griddata(xyz, value, xyzi, method=interpDispMethod)
            grid_valsinspace = np.swapaxes(grid_valsinspace, 0, 1)
            
        # Refer to a specificed time
        if refTime is not None:
            indt = np.searchsorted(self.timeseries, refTime)
            grid_valsinspace = grid_valsinspace - grid_valsinspace[indt, :, :][None, :, :]
        dispInVerts = grid_valsinspace
        
        # interp the time series
        if dispInTimes is not None:
            grid_valsintime = griddata(self.timeseries, grid_valsinspace, dispInTimes, method=interpTimeMethod)
            dispInVerts = grid_valsintime
        
        intpverts = DispBase('intpdata', self.program, self.filename, self.utmzone, self.ellps, self.lon0, self.lat0)
        intpverts.dispts = dispInVerts
        intpverts.index = indx
        intpverts.dist = dist
        intpverts.xyz = self.xyz[indx]
        intpverts.llz = self.llz[indx]
        intpverts.timeseries = deepcopy(np.asarray(dispInTimes)) if dispInTimes is not None else deepcopy(self.timeseries)
        intpverts.transferdispunit(dispunit)
        intpverts.dispunit = dispunit
        intpverts.transfertimeunit(timeunit)
        intpverts.timeunit = timeunit
        # All Done.
        return intpverts
    
    def ExtractDispInIndex(self, index, dispunit='m', dispInTimes=None, interpTimeMethod='nearest', refTime=None):
        '''
        Keep stations with index and remove others
        '''
        grid_valsinspace = self.dispts[:, index, :]*self._dispscale(dispunit)
        
        # Refer to a specificed time
        if refTime is not None:
            indt = np.searchsorted(self.timeseries, refTime)
            grid_valsinspace = grid_valsinspace - grid_valsinspace[indt, :, :][None, :, :]
        dispInVerts = grid_valsinspace
        
        # interp the time series
        if dispInTimes is not None:
            grid_valsintime = griddata(self.timeseries, grid_valsinspace, dispInTimes, method=interpTimeMethod)
            dispInVerts = grid_valsintime
        
        indexverts = DispBase('indexdata', self.program, self.filename, self.utmzone, self.ellps, self.lon0, self.lat0)
        indexverts.xyz = self.xyz[index]
        indexverts.llz = self.llz[index]
        indexverts.dispts = self.dispts[:, index, :]
        indexverts.index = index 
        indexverts.timeseries = deepcopy(np.asarray(dispInTimes)) if dispInTimes is not None else self.timeseries
        self.dispunit = dispunit
        self.timeunit = 'Y'
        # All Done
        return indexverts
    
    def to_DataFrame(self, dispunit='m', tunit='Y', dt2date=False, origintime=None):
        '''
        tunit    : Y/D/W
        dispunit : m/mm/cm/dm/km
        dispts   : (nt, nc, nDim)
        '''
        dispts = self.dispts*self._dispscale(dispunit)
        if dt2date:
            assert origintime is not None or self.origintime is not None, 'Origintime must be set'
            starttime = origintime if origintime is not None else self.origintime
            self.origintime = starttime
            date = starttime + pd.TimedeltaIndex(['{0:.10f}{1}'.format(idt, self.timeunit) for idt in self.timeseries])
        else:
            date = self.timeseries*self._timescale(tunit) # dt_float in tunit
        index = np.arange(dispts.shape[1])
        llz = self.llz

        nt = self.timeseries.shape[0]
        rllz = np.repeat(llz[None, :, :], nt, axis=0)
        rllz = rllz.reshape((-1, 3))
        rdispts = dispts.reshape((-1, 3))
        data = np.hstack((rllz, rdispts))
        mindex = pd.MultiIndex.from_product([date, index], names=['date', 'index'])
        columns = 'Lon[deg] Lat[deg] z E({0}) N({0}) U({0})'.format(dispunit).split()

        pddata = pd.DataFrame(data, index=mindex, columns=columns)
        pddata['tunit'] = tunit
        pddata.sort_index(level=0, inplace=True)
        # All Done
        return pddata 

    def getprofile(self, name, loncenter, latcenter, length, azimuth, width, data='data'):
        '''
        Project the GPS velocities onto a profile. 
        Works on the lat/lon coordinates system.

        Args:
            * name              : Name of the profile.
            * loncenter         : Profile origin along longitude.
            * latcenter         : Profile origin along latitude.
            * length            : Length of profile.
            * azimuth           : Azimuth in degrees.
            * width             : Width of the profile.

        Kwargs:
            * data              : Do the profile through the 'data' or the 'synth'etics.

        Returns:
            * None: Profiles are stored in self.profiles
        '''

        # the profiles are in a dictionary
        if not hasattr(self, 'profiles'):
            self.profiles = {}
        self.profiles[name] = {}

        # What data do we want
        if data is 'data':
            values = self.dispts
            self.profiles[name]['data type'] = 'data'
        elif data is 'synth':
            values = self.synth
            self.profiles[name]['data type'] = 'synth'
        elif data is 'res':
            values = self.dispts - self.synth
            self.profiles[name]['data_type'] = 'res'

        # Convert the lat/lon of the center into UTM.
        xc, yc = self.ll2xy(loncenter, latcenter)

        # Get the profile
        self.x = self.xyz[:, 0]
        self.y = self.xyz[:, 1]
        self.lon = self.llz[:, 0]
        self.lat = self.llz[:, 1]
        Dalong, Dacros, Bol, boxll, box, xe1, ye1, xe2, ye2, lon, lat = \
                utils.coord2prof(self, xc, yc, length, azimuth, width, minNum=1)

        # 4. Get these GPS
        vel = values[:,Bol,:]

        # Create the lists that will hold these values
        Vacros = []; Valong = []; Vup = []

        # Get some numbers
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]

        # Create vectors
        vec1 = np.array([x2-x1, y2-y1]) # xe1 to x1
        vec1 = vec1/np.sqrt( vec1[0]**2 + vec1[1]**2 )
        vec2 = np.array([x4-x1, y4-y1])
        vec2 = vec2/np.sqrt( vec2[0]**2 + vec2[1]**2 )

        # Loop on the timeseries
        for it in range(vel.shape[0]):
            vel_it = vel[it, :, :]
            Vacros_it = []
            Valong_it = []
            Vup_it = []
            # Loop on the stations
            for p in range(vel.shape[1]):
                # Project velocities
                Vacros_it.append(np.dot(vec2,vel_it[p,0:2]))
                Valong_it.append(np.dot(vec1,vel_it[p,0:2]))
                # Up direction
                Vup_it.append(vel_it[p,2])
            # Velocities perpendiculare to the fault
            Vacros.append(Vacros_it)
            # Velocities parallel to the fault
            Valong.append(Valong_it)
            # Vertical velocities
            Vup.append(Vup_it)
            
        # Store it in the profile list
        dic = self.profiles[name] 
        dic['Center'] = [loncenter, latcenter]
        dic['Length'] = length
        dic['Width'] = width
        dic['Box'] = np.array(boxll)
        dic['Normal Velocity'] = np.array(Vacros)
        dic['Parallel Velocity'] = np.array(Valong)
        dic['Vertical Velocity'] = np.array(Vup)
        dic['Distance'] = np.array(Dalong)
        dic['Normal Distance'] = np.array(Dacros)
        dic['Station Index'] = np.array(Bol)
        dic['EndPoints'] = [[xe1, ye1], [xe2, ye2]]
        lone1, late1 = self.xy2ll(xe1, ye1)
        lone2, late2 = self.xy2ll(xe2, ye2)
        dic['EndPointsLL'] = [[lone1, late1],
                              [lone2, late2]]
        dic['Vectors'] = [vec1, vec2]
    
        # all done
        return

    def writeProfile2File(self, name, filename, fault=None):
        '''
        Writes the profile named 'name' to the ascii file filename.

        Args:
            * name      : Name of the profile to write out
            * filename  : Name of the output file

        Kwargs:
            * fault     : Add the location of a fault (uses the fault trace)

        Returns:
            * None
        '''

        # open a file
        fout = open(filename, 'w')

        # Get the dictionary
        dic = self.profiles[name]

        # Write the header
        fout.write('#---------------------------------------------------\n')
        fout.write('# Profile Generated with StaticInv\n')
        fout.write('# Center: {} {} \n'.format(dic['Center'][0], dic['Center'][1]))
        fout.write('# Endpoints: \n')
        fout.write('#           {} {} \n'.format(dic['EndPointsLL'][0][0], dic['EndPointsLL'][0][1]))
        fout.write('#           {} {} \n'.format(dic['EndPointsLL'][1][0], dic['EndPointsLL'][1][1]))
        fout.write('# Box Points: \n')
        fout.write('#           {} {} \n'.format(dic['Box'][0][0],dic['Box'][0][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][1][0],dic['Box'][1][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][2][0],dic['Box'][2][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][3][0],dic['Box'][3][1]))
        
        # Place faults in the header                                                     
        if fault is not None:
            if fault.__class__ is not list:                                             
                fault = [fault]
            fout.write('# Fault Positions: \n')                                          
            for f in fault:
                d = self.intersectProfileFault(name, f)
                fout.write('# {}          {} \n'.format(f.name, d))
        
        fout.write('#---------------------------------------------------\n')

        # Write the values
        for i in range(len(dic['Distance'])):
            d = dic['Distance'][i]
            Vp = dic['Parallel Velocity'][i]
            Ep = dic['Parallel Error'][i]
            Vn = dic['Normal Velocity'][i]
            En = dic['Normal Error'][i]
            Vu = dic['Vertical Velocity'][i]
            Eu = dic['Vertical Error'][i]
            fout.write('{} {} {} {} {} {} {} \n'.format(d, Vp, Ep, Vn, En, Vu, Eu))

        # Close the file
        fout.close()

        # all done
        return

    def plotprofile(self, name, legendscale=10., fault=None, data=['parallel', 'normal', 'vertical'], show=True):
        '''
        Plot profile.

        Args:
            * name      : Name of the profile.

        Kwargs:
            * legendscale   : Length of the legend arrow.
            * fault         : Add a fault on the plot
            * data          : list of type of data to use
            * show          : Show me

        Returns:
            * None
        '''

        if type(data) is str:
            data = [data]

        # Plo the map
        if 'vertical' in data:
            vertical=True
        else:
            vertical=False
        self.plot(faults=fault, figure=None, show=False, legendscale=legendscale, vertical=vertical)

        # plot the box on the map
        b = self.profiles[name]['Box']
        bb = np.zeros((5, 2))
        for i in range(4):
            x, y = b[i,:]
            if x<0.:
                x += 360.
            bb[i,0] = x
            bb[i,1] = y
        bb[4,0] = bb[0,0]
        bb[4,1] = bb[0,1]
        self.fig.carte.plot(bb[:,0], bb[:,1], '.k', zorder=0)
        self.fig.carte.plot(bb[:,0], bb[:,1], '-k', zorder=0)

        # open a figure
        fig = plt.figure()
        prof = fig.add_subplot(111)

        # plot the profile
        if 'parallel' in data:
            x = self.profiles[name]['Distance']
            y = self.profiles[name]['Parallel Velocity']
            ey = self.profiles[name]['Parallel Error']
            p = prof.errorbar(x, y, yerr=ey, 
                              label='Profile Parallel', marker='.', linestyle='')
        if 'normal' in data:
            x = self.profiles[name]['Distance']
            y = self.profiles[name]['Normal Velocity']
            ey = self.profiles[name]['Normal Error']
            q = prof.errorbar(x, y, yerr=ey, 
                              label='Profile Normal', marker='.', linestyle='')
        if 'vertical' in data:
            x = self.profiles[name]['Distance']
            y = self.profiles[name]['Vertical Velocity']
            ey = self.profiles[name]['Vertical Error']
            r = prof.errorbar(x, y, yerr=ey,
                              label='Vertical', marker='.', linestyle='')

        # If a fault is here, plot it
        if fault is not None:
            # If there is only one fault
            if fault.__class__ is not list:
                fault = [fault]
            # Loop on the faults
            for f in fault:
                # Get the distance
                d = self.intersectProfileFault(name, f)
                if d is not None:
                    ymin, ymax = prof.get_ylim()
                    prof.plot([d, d], [ymin, ymax], '--', label=f.name)

        # plot the legend
        prof.legend()

        # Show to screen 
        if show:
            self.fig.show(showFig=['map'])

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
    
    def setorigintime(self, time):
        '''
        time       : pd.Timestamp
        '''
        self.origintime = time 
        # All Done
        return
    
    def removetimeseries(self, timeseries, rst=1e-5):
        '''
        timeseries     : sequence of float
        '''
        distmat = np.abs(self.timeseries[None, :] - np.array(timeseries)[:, None])
        indx = np.argmin(distmat, axis=-1)
        dist = np.min(distmat, axis=-1)
        for i, ind in enumerate(indx):
            if dist[i] < rst:
                self.removetimeindex(ind)
            else:
                print('{0:.5f} over the threshold'.format(timeseries[i]))
        # All Done
        return
    
    def removetimeindex(self, timeindex):
        tslist = self.timeseries.tolist()
        tslist.pop(timeindex)
        self.timeseries = np.array(tslist)
        index = np.arange(self.dispts.shape[0]).tolist()
        index.pop(timeindex)
        self.dispts = self.dispts[:, index, :]
        # All Done
        return
    
    def transferdispunit(self, dispunit):
        '''
        dispunit   : m/mm/cm/dm/km
        '''
        self.dispts *= self._dispscale(dispunit)
        self.dispunit = dispunit
        # All Done
        return
    
    def transfertimeunit(self, timeunit):
        '''
        timeunit  : Y/D/W, Whatever units for pd.Timedelta is fine
        '''
        self.timeseries*= self._timescale(timeunit)
        self.timeunit = timeunit
        # All Done
        return

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
        
    def _timescale(self, unit='Y'):
        if unit not in self.VALID_TIMEUNITS:
            raise ValueError(f"Invalid original unit: {unit}. Valid units are: {self.VALID_TIMEUNITS}")
        if self.timeunit not in self.VALID_TIMEUNITS:
            raise ValueError(f"Invalid time unit: {self.timeunit}. Valid units are: {self.VALID_TIMEUNITS}")

        unit_in_seconds = self.convert_timeunit(unit)
        timeunit_in_seconds = self.convert_timeunit(self.timeunit)
        # All done
        return pd.Timedelta(timeunit_in_seconds, 's') / pd.Timedelta(unit_in_seconds, 's')
    
    @staticmethod
    def convert_dispunit(unit):
        if unit == 'm':
            return 1.0
        elif unit == 'dm':
            return 10.0
        elif unit == 'cm':
            return 100.0
        elif unit == 'mm':
            return 1000.0
        else:  # 'km'
            return 1e-3

    def _dispscale(self, unit='m'):
        return self._dispunitscale(unit)/self._dispunitscale(self.dispunit)

    def _dispunitscale(self, unit='m'):
        if unit not in self.VALID_DISPUNITS:
            raise ValueError(f"Invalid unit: {unit}. Valid units are: {self.VALID_DISPUNITS}")
        scale = self.convert_dispunit(unit)
        # All Done
        return scale