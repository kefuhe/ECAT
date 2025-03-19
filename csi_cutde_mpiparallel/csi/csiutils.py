import numpy as np
import scipy.interpolate as sciint
try:
    from netCDF4 import Dataset as netcdf
except:
    from scipy.io.netcdf import netcdf_file as netcdf

#----------------------------------------------------------------
#----------------------------------------------------------------
# A Dictionary with the months

months = {'JAN': 1,
          'FEB': 2,
          'MAR': 3, 
          'APR': 4, 
          'MAY': 5,
          'JUN': 6,
          'JUL': 7,
          'AUG': 8,
          'SEP': 9, 
          'OCT': 10,
          'NOV': 11,
          'DEC': 12}

#----------------------------------------------------------------
#----------------------------------------------------------------
# A routine to write netcdf files

def write2netCDF(filename, lon, lat, z, increments=None, nSamples=None, 
        title='CSI product', name='z', scale=1.0, offset=0.0, mask=None,
        xyunits=['Lon', 'Lat'], units='None', interpolation=True, verbose=True, 
        noValues=np.nan):
    '''
    Creates a netCDF file  with the arrays in Z. 
    Z can be list of array or an array, the size of lon.
                
    .. Args:
        
        * filename -> Output file name
        * lon      -> 1D Array of lon values
        * lat      -> 1D Array of lat values
        * z        -> 2D slice to be saved
        * mask     -> if not None, must be a 2d-array of a polynome to mask 
                      what is outside of it. This option is really long, so I don't 
                      use it...
   
    .. Kwargs:
               
        * title    -> Title for the grd file
        * name     -> Name of the field in the grd file
        * scale    -> Scale value in the grd file
        * offset   -> Offset value in the grd file
                
    .. Returns:
          
        * None
    '''

    if interpolation:

        # Check
        if nSamples is not None:
            if type(nSamples) is int:
                nSamples = [nSamples, nSamples]
            dlon = (lon.max()-lon.min())/nSamples[0]
            dlat = (lat.max()-lat.min())/nSamples[1]
        if increments is not None:
            dlon, dlat = increments

        # Resample on a regular grid
        olon, olat = np.meshgrid(np.arange(lon.min(), lon.max(), dlon),
                                 np.arange(lat.min(), lat.max(), dlat))
    else:
        # Get lon lat
        olon = lon
        olat = lat
        if increments is not None:
            dlon, dlat = increments
        else:
            dlon = olon[0,1]-olon[0,0]
            dlat = olat[1,0]-olat[0,0]

    # Create a file
    fid = netcdf(filename,'w')

    # Create a dimension variable
    fid.createDimension('side',2)
    if verbose:
        print('Create dimension xysize with size {}'.format(np.prod(olon.shape)))
    fid.createDimension('xysize', np.prod(olon.shape))

    # Range variables
    fid.createVariable('x_range','d',('side',))
    fid.variables['x_range'].units = xyunits[0]

    fid.createVariable('y_range','d',('side',))
    fid.variables['y_range'].units = xyunits[1]
    
    # Spacing
    fid.createVariable('spacing','d',('side',))
    fid.createVariable('dimension','i4',('side',))

    # Informations
    if title is not None:
        fid.title = title
    fid.source = 'CSI.utils.write2netCDF'

    # Filing rnage and spacing
    if verbose:
        print('x_range from {} to {} with spacing {}'.format(olon[0,0], olon[0,-1], dlon))
    fid.variables['x_range'][0] = olon[0,0]
    fid.variables['x_range'][1] = olon[0,-1]
    fid.variables['spacing'][0] = dlon

    if verbose:
        print('y_range from {} to {} with spacing {}'.format(olat[0,0], olat[-1,0], dlat))
    fid.variables['y_range'][0] = olat[0,0]
    fid.variables['y_range'][1] = olat[-1,0]
    fid.variables['spacing'][1] = dlat
    
    if interpolation:
        # Interpolate
        interpZ = sciint.LinearNDInterpolator(np.vstack((lon, lat)).T, z, fill_value=noValues)
        oZ = interpZ(olon, olat)
    else:
        # Get values
        oZ = z

    # Masking?
    if mask is not None:
        
        # Import matplotlib.path
        import matplotlib.path as path

        # Create the path
        poly = path.Path([[lo, la] for lo, la in zip(mask[:,0], mask[:,1])], 
                closed=False)

        # Create the list of points
        xy = np.vstack((olon.flatten(), olat.flatten())).T

        # Findthose outside
        bol = poly.contains_points(xy)

        # Mask those out
        oZ = oZ.flatten()
        oZ[bol] = np.nan
        oZ = oZ.reshape(olon.shape)

    # Range
    zmin = np.nanmin(oZ)
    zmax = np.nanmax(oZ)
    fid.createVariable('{}_range'.format(name),'d',('side',))
    fid.variables['{}_range'.format(name)].units = units
    fid.variables['{}_range'.format(name)][0] = zmin
    fid.variables['{}_range'.format(name)][1] = zmax

    # Create Variable
    fid.createVariable(name,'d',('xysize',))
    fid.variables[name].long_name = name
    fid.variables[name].scale_factor = scale
    fid.variables[name].add_offset = offset
    fid.variables[name].node_offset=0

    # Fill it
    fid.variables[name][:] = np.flipud(oZ).flatten()

    # Set dimension
    fid.variables['dimension'][:] = oZ.shape[::-1]

    # Synchronize and close
    fid.sync()
    fid.close()

    # All done
    return

#----------------------------------------------------------------
#----------------------------------------------------------------
# A routine to extract a profile

def coord2prof(csiobject, xc, yc, length, azimuth, width, minNum=5, ref_to_start=False):
    '''
    Routine returning the profile

    Args:
        * csiobject         : An instance of a csi class that has
                              x and y attributes
        * xc                : X pos of center
        * yc                : Y pos of center
        * length            : length of the profile.
        * azimuth           : azimuth of the profile.
        * width             : width of the profile.
        * minNum            : minimum number of points to consider the profile
        * ref_to_start      : If True, the distance is computed from the start
                              of the profile

    Returns:
        dis                 : Distance from the center
        norm                : distance perpendicular to profile
        ind                 : indexes of the points
        boxll               : lon lat coordinates of the profile box used
        xe1, ye1            : coordinates (UTM) of the profile endpoint
        xe2, ye2            : coordinates (UTM) of the profile endpoint
                 (xe1, ye1)
            1   +----*--->+ 2
               /  | /vec1/
              /azi|/    /
             /    /    /  (1,2,3,4) is the point order of the box/boxll
      vec2 |/_   /    /  vec1/vec2 is the unit vector parallel/normal to fault
         4 +-----*---+ 3
            (xe2, ye2)
        * azi is the azimuth normal to fault
        * vacross is the velocity normal to fault
        * valong is the velocity parallel to fault
    '''

    # Azimuth into radians
    alpha = azimuth*np.pi/180.

    # Copmute the across points of the profile
    xa1 = xc - (width/2.)*np.cos(alpha)
    ya1 = yc + (width/2.)*np.sin(alpha)
    xa2 = xc + (width/2.)*np.cos(alpha)
    ya2 = yc - (width/2.)*np.sin(alpha)

    # Compute the endpoints of the profile
    xe1 = xc + (length/2.)*np.sin(alpha)
    ye1 = yc + (length/2.)*np.cos(alpha)
    xe2 = xc - (length/2.)*np.sin(alpha)
    ye2 = yc - (length/2.)*np.cos(alpha)

    # Convert the endpoints
    elon1, elat1 = csiobject.xy2ll(xe1, ye1)
    elon2, elat2 = csiobject.xy2ll(xe2, ye2)

    # Design a box in the UTM coordinate system.
    x1 = xe1 - (width/2.)*np.cos(alpha)
    y1 = ye1 + (width/2.)*np.sin(alpha)
    x2 = xe1 + (width/2.)*np.cos(alpha)
    y2 = ye1 - (width/2.)*np.sin(alpha)
    x3 = xe2 + (width/2.)*np.cos(alpha)
    y3 = ye2 - (width/2.)*np.sin(alpha)
    x4 = xe2 - (width/2.)*np.cos(alpha)
    y4 = ye2 + (width/2.)*np.sin(alpha)

    # Convert the box into lon/lat for further things
    lon1, lat1 = csiobject.xy2ll(x1, y1)
    lon2, lat2 = csiobject.xy2ll(x2, y2)
    lon3, lat3 = csiobject.xy2ll(x3, y3)
    lon4, lat4 = csiobject.xy2ll(x4, y4)

    # make the box
    box = []
    box.append([x1, y1])
    box.append([x2, y2])
    box.append([x3, y3])
    box.append([x4, y4])

    # make latlon box
    boxll = []
    boxll.append([lon1, lat1])
    boxll.append([lon2, lat2])
    boxll.append([lon3, lat3])
    boxll.append([lon4, lat4])

    # Get the points in this box.
    # 1. import shapely and nxutils
    import matplotlib.path as path
    import shapely.geometry as geom

    # 2. Create an array with the positions
    XY = np.vstack((csiobject.x, csiobject.y)).T

    # 3. Create a box
    rect = path.Path(box, closed=False)

    # 4. Find those who are inside
    Bol = rect.contains_points(XY)

    # 4. Get these values
    xg = csiobject.x[Bol]
    yg = csiobject.y[Bol]
    lon = csiobject.lon[Bol]
    lat = csiobject.lat[Bol]

    # Check if lengths are ok
    assert len(xg)>minNum, \
            'Not enough points to make a worthy profile: {}'.format(len(xg))

    # 5. Get the sign of the scalar product between the line and the point
    vec = np.array([xe1-xc, ye1-yc])
    xy = np.vstack((xg-xc, yg-yc)).T
    sign = np.sign(np.dot(xy, vec))

    # 6. Compute the distance (along, across profile) and get the velocity
    # Create the list that will hold these values
    Dacros = []; Dalong = []
    # Build lines of the profile
    Lalong = geom.LineString([[xe1, ye1], [xe2, ye2]])
    Lacros = geom.LineString([[xa1, ya1], [xa2, ya2]])
    # Build a multipoint
    PP = geom.MultiPoint(np.vstack((xg,yg)).T.tolist())
    # Loop on the points
    for p in range(len(PP.geoms)):
        Dalong.append(Lacros.distance(PP.geoms[p])*sign[p])
        Dacros.append(Lalong.distance(PP.geoms[p]))

    if ref_to_start:
        # Get the distance to the start
        Dalong = np.array(Dalong)
        PP = geom.MultiPoint(np.array([[xe2, ye2]]).tolist())
        offset = Lacros.distance(PP.geoms[0])*-1
        Dalong = Dalong - offset

    Dalong = np.array(Dalong)
    Dacros = np.array(Dacros)

    # All done
    return Dalong, Dacros, Bol, boxll, box, xe1, ye1, xe2, ye2, lon, lat

#----------------------------------------------------------------
#----------------------------------------------------------------
# get intersection between profile and a fault trace

def intersectProfileFault(xe1, ye1, xe2, ye2, xc, yc, fault):
    '''
    Gets the distance between the fault/profile intersection and the profile center.
    Args:
        * xe1, ye1  : X and Y coordinates of one endpoint of the profile
        * xe2, ye2  : X and Y coordinates of the other endpoint of the profile
        * xc, yc    : X and Y coordinates of the centre of the profile
        * fault     : CSI fault object that has a trace.
    '''

    # Import shapely
    import shapely.geometry as geom

    # Grab the fault trace
    xf = fault.xf
    yf = fault.yf

    # Build a linestring with the profile center
    Lp = geom.LineString([[xe1, ye1],[xe2, ye2]])

    # Build a linestring with the fault
    ff = []
    for i in range(len(xf)):
        ff.append([xf[i], yf[i]])
    Lf = geom.LineString(ff)

    # Get the intersection
    if Lp.crosses(Lf):
        Pi = Lp.intersection(Lf)
        if type(Pi) is geom.point.Point:
            p = Pi.coords[0]
        else:
            return None
    else:
        return None

    # Get the sign
    vec1 = [xe1-xc, ye1-yc]
    vec2 = [p[0]-xc, p[1]-yc]
    sign = np.sign(np.dot(vec1, vec2))

    # Compute the distance to the center
    d = np.sqrt( (xc-p[0])**2 + (yc-p[1])**2)*sign

    # All done
    return d

# List splitter
def _split_seq(seq, size):
    newseq = []
    splitsize = 1.0/size*len(seq)
    for i in range(size):
            newseq.append(seq[int(round(i*splitsize)):int(round((i+1)*splitsize))])
    return newseq

# Check if points are colocated
def colocated(point1,point2,eps=0.):
    '''
    Check if point 1 and 2 are colocated
    Args:
        * point1: x,y,z coordinates of point 1
        * point2: x,y,z coordinates of point 2
        * eps   : tolerance value 
    '''
    if np.linalg.norm(point1-point2)<=eps:
        return True
    return False


#------------------------------------------------------------------------------------------------#
#---------------------------------- Utilities were added by Kefeng He----------------------------#
#------------------------------------------------------------------------------------------------#
def utm_zone_epsg(longitude, latitude, verbose=True):
    '''
    Input:
        * longitude     :  (-180, 180) or (0, 360)
        * latitude      :  (-90, 90)
        * verbose       : Print the info
    Output:
        * zone          : The number of the UTM Zone
        * EPSG          : The number corresponging to the EPSG
    Example:
        lon = 201.
        lat = 55.
        zone, epsg = utm_zone_epsg(lon, lat)
        print(zone, epsg)
    '''
    if longitude > 180:
        longitude = longitude - 360.0
    zone = int(np.round((183+longitude)/6,0))

    EPSG=32700-np.round((45+latitude)/90,0)*100+np.round((183+longitude)/6,0)
    EPSG = int(EPSG)
    if verbose:
        print("Zone is", zone)
        print("EPSG is",EPSG)
    return zone, EPSG


# # -------------------------------- Read Tiff Images -----------------------------------#
# def read_tiff(unwfile):
#     '''
#     Warning : insar(csi库文件)导入需要在gdal前面可能存在cartopy，shapely库冲突放在其后
#     Input:
#         * unwfile     : tiff格式影像文件
#     Output:
#         * im_data     : 数据矩阵
#         * im_geotrans : 横纵坐标起始点和步长
#         * im_proj     : 投影参数
#         * im_width    : 像元的行数
#         * im_height   : 像元的列数
#     Example:
#         # 数据存储信息
#         im_data, im_geotrans, im_proj, im_width, im_height = read_tiff(phase_file)
#         # im_data = im_data*0.056/(-4*np.pi) # 存储为相位则需要这一项，如果存储为直接形变则不用
#         los = im_data
#         # 读取元数据
#         x_lon, y_step, x_lat, y_step, mesh_lon, mesh_lat = read_tiff_info(phase_file, im_width, im_height)
#         x_lowerright = np.nanmin(x_lon)
#         x_upperleft = np.nanmax(x_lon)
#         y_lowerright = np.nanmin(x_lat)
#         y_upperleft = np.nanmax(x_lat)

#         coordrange = [x_lowerright, x_upperleft, y_lowerright, y_upperleft]

#         plt.imshow(los, interpolation='none', # cmap=cmap, norm=norm,
#                 origin='upper', extent=[np.nanmin(x_lon), np.nanmax(x_lon), np.nanmin(x_lat), np.nanmax(x_lat)], aspect='auto', vmax=1, vmin=-1)

#         # 读取方位角和入射角文件信息
#         # 方位角信息
#         azi, _, _, _, _ = read_tiff(azi_file)
#         # 入射角信息
#         inc, _, _, _, _ = read_tiff(inc_file)

#         # Input to CSI InSAR
#         sar.read_from_binary(im_data, lon=mesh_lon.flatten(), lat=mesh_lat.flatten(), azimuth=azi.flatten(), incidence=inc.flatten(), downsample=1)
#         sar.checkNaNs()
#     '''
#     from osgeo import gdal
#     from pyproj import Proj

#     dataset = gdal.Open(unwfile, gdal.GA_ReadOnly)  # 打开文件
#     im_width = dataset.RasterXSize  # 栅格矩阵的列数
#     im_height = dataset.RasterYSize  # 栅格矩阵的行数
#     im_bands = dataset.RasterCount  # 波段数
#     im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率/步长
#     im_proj = dataset.GetProjection()  # 地图投影信息，字符串表示
#     im_band = dataset.GetRasterBand(1)
#     # im_data = np.asarray(imread(filename))
#     # 下面这个一直进行数据读取有问题
#     im_data = im_band.ReadAsArray(0, 0, im_width, im_height) # 将栅格图像值存为数据矩阵
#     del dataset
#     return im_data, im_geotrans, im_proj, im_width, im_height


# def read_tiff_info(unwfile, im_width, im_height, meshout=True):
#     '''
#     Object:
#         * 获取tiff文件坐标系统和横轴、纵轴范围
#     Input:
#         * tiff文件文件名
#     Output:
#         * x_lon, y_step, x_lat, y_step
#     '''
#     from osgeo import gdal
#     from pyproj import Proj

#     metadata = gdal.Info(unwfile, format='json', deserialize=True)
#     upperLeft = metadata['cornerCoordinates']['upperLeft']
#     lowerRight = metadata['cornerCoordinates']['lowerRight']
#     x_upperleft, y_upperleft = upperLeft
#     x_lowerright, y_lowerright = lowerRight
#     x_step = (x_upperleft - x_lowerright)/im_width
#     y_step = -(y_upperleft - y_lowerright)/im_height
#     x_lon = np.linspace(x_upperleft, x_lowerright, im_width)
#     x_lat = np.linspace(y_upperleft, y_lowerright, im_height)

#     if meshout:
#         mesh_lon, mesh_lat = np.meshgrid(x_lon, x_lat)
#         return x_lon, y_step, x_lat, y_step, mesh_lon, mesh_lat
#     else:
#         return x_lon, x_step, x_lat, y_step


# def write_tiff(filename, im_proj, im_geotrans, im_data=None):
#     # 判断栅格数据的数据类型
#     from osgeo import gdal
#     from pyproj import Proj

#     if 'int8' in im_data.dtype.name:
#         datatype = gdal.GDT_Byte
#     elif 'int16' in im_data.dtype.name:
#         datatype = gdal.GDT_UInt16
#     else:
#         datatype = gdal.GDT_Float32
        
#     # 判读数组维数
#     # im_data.shape可能有三个元素，对应多维矩阵，即多个波段；
#     # 也可能有两个元素，对应二维矩阵，即一个波段
#     if len(im_data.shape) == 3:
#         im_bands, im_height, im_width = im_data.shape
#     else:
#         im_bands = 1
#         (im_height, im_width) = im_data.shape
# #---------------------------------------------------------------------------------#

# #---------------------------------3D Projection-----------------------------------#

# # 获取等值线，以及在其中插值Z，可以根据需求调换XYZ的顺序，以保证投影面积最大
# def plot_contourinsurf(x, y, z, t, levels=3, dtype='tri', topo=None, N=None,
#                         intptype='linear', intpidx=None, XYZinds=[0, 1, 2]):
#     '''
#     Input :
#         * x           :
#         * y           :
#         * z           : z值得是和节点大小相等，tricontour需要信息在节点上
#         * t           :
#         * levels      :
#         * dtype       :
#         * topo        :
#         * N           :
#         * inteptype   :
#         * intpidx     :
#         * XYZinds     :
    
#     Output :
#         * line3d      :
#         * lec         :
#     '''
#     from scipy.interpolate import griddata
#     import numpy as np
#     from pyproj import Proj
#     import matplotlib.pyplot as plt
    
#     xyz = np.hstack((x.flatten()[:, None], y.flatten()[:, None], z.flatten()[:, None]))
#     if dtype == 'rect':
#         conts = plt.contour(x, y, t,levels=levels, alpha=0.)
#         plt.close()
#     else:
#         # 下面用的量值列表取代了通常用的levels的分级信息，重新设置函数完善相应设置
#         conts = plt.tricontour(x, y, topo, t, [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], alpha=0.) # levels=levels
#         plt.close() 
    
#     if intpidx is None:
#         intpidx = np.arange(xyz.shape[0], dtype=np.int_)
        

#     # 对应不同level的线段列表
#     allsegs = conts.allsegs
#     levels = conts.levels
#     intpsegs = []
#     for i in range(levels.size):
#         intp_segis = []
#         n = len(allsegs[i])
#         if n>0:
#             segis = allsegs[i]
#             # ct = levels[i]
#             for segi in segis:
#                 cx = segi[:, 0]
#                 cy = segi[:, 1]
#                 cz = griddata(xyz[intpidx, :2], xyz[intpidx, 2], (cx, cy), method=intptype)
#                 cxyz = np.hstack((cx[:, None], cy[:, None], cz[:, None]))
#                 intp_segis.append(cxyz)
#             intpsegs.append(intp_segis)
#         else:
#             intpsegs.append([])

#     # 获取同震的等值线，以及对应的形变量值
#     line3d = []
#     lec = []
#     for i in range(levels.size):
#         segis = intpsegs[i]
#         if segis:
#             for line in segis:
#                 line3d.append(line[:, XYZinds])
#                 lec.append(levels[i])
#     return line3d, lec
   


# # 1. 进行三角元中心位错插值到三角点上的操作
# # 2. 根据需求提取部分三角元用于计算插值和计算位错
# def extract_cline(XYZ, cotopo, invdist, mag_slip, triinds, XYZinds=[2, 0, 1], levels=3, srange=[-7, 7.], cmap=None):
#     '''
#     Object : 将单元三角位错插值到节点上，然后利用plot_contourinsurf函数求取对应的三维等值曲线集
#     Input  :
#         * XYZ           : Nnodes*3, 三角元坐标顶点
#         * cotopo        : Ntri*3, 三角元个数和topology组合
#         * invdist       : 
#     '''
#     invdist = invdist[:, triinds]
#     cotopo = cotopo[triinds, :]
#     mag_slip = mag_slip[triinds]
#     cotri = []
#     magtri = []
#     for i in range(XYZ.shape[0]):
#         cotmp = []
#         for k, topo in enumerate(cotopo):
#             if i in topo:
#                 cotmp.append(k)
#         magtmp = np.dot(invdist[i, cotmp], mag_slip.flatten()[cotmp])/np.sum(np.asarray(invdist[i, cotmp]))
#         magtri.append(magtmp)
#         cotri.append(cotmp)

#     magtri = np.array(magtri)
#     # idx表示
#     idxs = np.where(np.isnan(magtri), False, True)

#     # YZX ,插值X
#     indx, indy, indz = XYZinds
#     XYZinds = np.argsort(XYZinds)
#     line3d, lec = plot_contourinsurf(XYZ[:, indx], XYZ[:, indy], 
#                                          XYZ[:, indz], magtri[:], 
#                                          dtype='tri', topo=cotopo, 
#                                          intptype='linear', levels=levels, 
#                                          XYZinds=XYZinds, intpidx=idxs)

#     # 获取对应的RGBA颜色列表
#     # smin, smax = srange
#     # norm = mpl.colors.Normalize(smin, smax)
#     # m = cm.ScalarMappable(norm=norm, cmap=cm.Blues)
#     # ec = m.to_rgba(lec)
#     return line3d, lec


# def calc_crossprod(x, y, z):
#     '''
#     计算三点的叉积
#     '''
#     vec1 = np.array([x[1] - x[0],
#                     y[1] - y[0],
#                     z[1] - z[0]]
#                     )
#     vec2 = np.array([x[2] - x[0],
#                     y[2] - y[0],
#                     z[2] - z[0]]
#                     )
#     return np.cross(vec1, vec2)


# def rotation(x, y, angle, angtype='Angle'):
#     coords = x + 1.j*y
#     angrad = np.deg2rad(angle)
#     res = coords*np.exp(1.j*angrad)
#     return res.real, res.imag
# #---------------------------------------------------------------------------------#



