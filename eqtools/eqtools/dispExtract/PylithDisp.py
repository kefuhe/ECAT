import numpy as np
import pandas as pd
from pyproj import Proj
from h5py import File
from .DispBase import DispBase
import math

# 计算 UTM 区域号
def calculate_utm_zone(lon):
    return math.floor((lon + 180) / 6) + 1

# 获取投影字符串中的经度
def get_longitude(proj_string):
    lon_start = proj_string.find('+lon_0=') + len('+lon_0=')
    lon_end = proj_string.find(' ', lon_start)
    return float(proj_string[lon_start:lon_end])


class PylithDisp(DispBase):
    def __init__(self, name, program='pylith', filename=None, 
                 utmzone=None, ellps='WGS84', lon0=None, lat0=None):
        super(PylithDisp, self).__init__(name, program, filename, utmzone, ellps, lon0, lat0)
    
    def readdispts(self, filename=None, projInPylith=None, factor=1.0):
        '''
        
        Kwargs    :
            * filename     : default None; HDF5 displacement output from pylith
            * projInPylith : Projection parameters of coordinates in pylith
            * factor       : The scale factor that transfers deformation unit in pylith to meter
        '''
        import pyproj

        filename = filename if filename is not None else self.filename
        assert filename is not None, "filename must be not None."

        with File(filename, 'r') as fin:
            xyzInPylith = fin['geometry/vertices'][:]
            timeseries = fin['time'][:].squeeze()*(pd.Timedelta('1S')/pd.Timedelta('365.2425D'))
            dispts = fin['vertex_fields']['displacement'][:]
        
        if pyproj.__version__[0] == '2':
            self.pInpylith = Proj(projInPylith)
            lon, lat = self.pInpylith(xyzInPylith[:, 0], xyzInPylith[:, 1], inverse=True)
        else:
            from pyproj import CRS, Transformer
            # 获取投影字符串
            proj_string = projInPylith
            if '+zone=' not in proj_string and 'utm' in proj_string.lower():
                # 计算 UTM 区域号
                utm_zone = calculate_utm_zone(get_longitude(proj_string))
                # 添加 UTM 区域号到投影字符串
                proj_string += ' +zone=' + str(utm_zone)
            # 创建 CRS 对象
            self.pInpylith = CRS.from_string(proj_string)
            # 创建 Transformer 对象；EPSG:4326 是一个坐标参考系统的标识符，它代表了 WGS 84 地理坐标系统
            transformer = Transformer.from_crs(proj_string, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(xyzInPylith[:, 0], xyzInPylith[:, 1])
        
        self.xyzInPylith = xyzInPylith
        z = xyzInPylith[:, 2]

        x, y = self.ll2xy(lon, lat)
        self.llz = np.vstack((lon, lat, z)).T
        self.xyz = np.vstack((x, y, z)).T
        self.timeseries = timeseries
        self.dispts = dispts*factor
        self.timeunit = 'Y'
        self.dispunit = 'm'
        
        # All Done
        return


if __name__ == '__main__':
    filename = 'f:\maduo_shearzone\multifault\shearzone_w20km_bak\maduo_shear_multifault_t20km\output1_0e18\covisco-groundsurf.h5'
    projstr = '+proj=utm +lon_0=98.25 +lat_0=34.5'
    data = PylithDisp('pylith', filename=filename, lon0=100, lat0=34)
    data.readdispts(projInPylith=projstr)

    aind = np.random.randint(0, 100, (100,))
    intpdata = data.extractDisp(data.xyz[aind,:], dispInTimes=[0.5, 1.0], interpTimeMethod='linear', utm=True, dispunit='mm')