import numpy as np
import pandas as pd
import os
from .DispBase import DispBase


class PscmpDisp(DispBase):
    def __init__(self, name, program='pscmp', filename=None, dirname=None,
                 utmzone=None, ellps='WGS84', lon0=None, lat0=None):
        super(PscmpDisp, self).__init__(name, program, filename, utmzone, ellps, lon0, lat0)
        self.dirname = dirname
        self.filename = r'snapshot_*.dat' if filename is None else filename
    
    def readSnapshot(self, filename, projInPscmp=None, factor=1.0):
        '''
        Extract deformation from a single deformation snapshot file
        
        '''

        columns = 'Lon[deg] Lat[deg] Ux Uy Uz'.split()
        data = pd.read_csv(filename, sep=r'\s+', usecols=columns)
        
        coord = data[['Lon[deg]', 'Lat[deg]']].values
        z = np.zeros_like(coord[:, 0])
        llz = np.hstack((coord, z[:, None]))
        
        data['Uz'] = -data.Uz*factor
        data['Ux'] = data.Ux*factor
        data['Uy'] = data.Uy*factor
        newname = ['{0}'.format(i) for i in 'N E U'.split()]
        namedict = dict(list(zip('Ux Uy Uz'.split(), newname)))
        data.rename(namedict, axis=1, inplace=True)
        disp_snap = data[['E', 'N', 'U']].values

        # All Done
        return llz, disp_snap
    
    def readdispts(self, filename=None, dirname=None, factor=1.0):
        '''
        Read all output time node deformations and form a time series
        
        Kwargs    :
            * filename   : None;glob template; list of filename
        '''
        from glob import glob 
        import re
        
        if dirname is None:
            dirname = self.dirname
        else:
            self.dirname = dirname
        if not os.path.isdir(dirname):
            raise ValueError('Directory {0} does not exist!'.format(dirname))

        if filename is None:
            template = self.filename
            files = glob(os.path.join(dirname, template))
        else:
            files = [os.path.join(dirname, ifile) for ifile in filename]
        self.files = files
        
        pattern = r'(?:snapshot_)([0-9._]+)_(day|week|month|year)(?:\.dat)'
        indexdict = {
            'day': [1.0, 'D'],
            'month': [30.0, 'D'],
            'week': [1.0, 'W'],
            'year': [1.0, 'Y']
        }
        
        timeseries = []
        dispts = []
        for i, ifile in enumerate(files):
            basename = os.path.basename(ifile)
            if 'coseism' in basename:
                dt = 0.0
            else:
                res = re.match(pattern, basename)
                tval, intunit = res.group(1), indexdict[res.group(2)][1]
                tval = tval.replace('_', '.')
                tval = str(float(tval)*indexdict[res.group(2)][0])

                if intunit in ('D', 'W'):
                    dt = pd.Timedelta(tval + intunit)
                    # 转换为年的浮点数
                    dt = (dt.days + dt.seconds/24.0/3600.0)/365.2425
                else:
                    dt = float(tval)
            timeseries.append(dt)
            llz, disp_snap = self.readSnapshot(ifile, factor=factor)
            dispts.append(disp_snap)
        
        # sort timeseries
        timeseries = np.array(timeseries)
        dispts = np.array(dispts)
        ind = np.argsort(timeseries)
        timeseries = timeseries[ind]
        dispts = dispts[ind, :, :]
        
        self.llz = llz
        x, y = self.ll2xy(llz[:, 0], llz[:, 1])
        self.xyz = np.vstack((x, y, llz[:, -1])).T
        self.timeseries = np.array(timeseries)
        self.dispts = dispts
        self.timeunit = 'Y'
        self.dispunit = 'm'
        # All Done
        return


if __name__ == '__main__':
    dirname = r'e:\Maduo_psgrn_pscmp\Maxwell_25km\pscmp_gps_lc1_0e18_regular'
    filename = None
    data = PscmpDisp('pscmp', dirname=dirname, lon0=100, lat0=34)
    data.readdispts()

    aind = np.random.randint(0, 100, (100,))
    intpdata = data.extractDisp(data.xyz[aind,:], utm=True, dispunit='mm')