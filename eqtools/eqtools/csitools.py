'''
Written by Kefeng He, January 2023
'''

import numpy as np
import pyproj as pp
from csi import TriangularPatches as csitrifault
from csi import RectangularPatches as csirectfault
from csi import SourceInv
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from csi import TriangularPatches as csitri
from csi import TriangularTents as csitent


# 输入到pscmp的子断层块格式
# n   O_lat   O_lon   O_depth length  width strike dip   np_st np_di start_time
# [-] [deg]   [deg]   [km]    [km]     [km] [deg]  [deg] [-]   [-]   [day]
#     pos_s   pos_d   slp_stk slp_ddip open
#     [km]    [km]    [m]     [m]      [m]
pscmp_slppatch_fmt = '{0:4d} {1:9.4f} {2:9.4f} {3:6.1f} {4:6.1f} {5:6.1f} {6:6.1f} \
        {7:6.1f} {8:3d} {9:3d} {10:6.1f}\n    {11:9.4f} {12:9.4f} {13:9.3f} {14:9.3f} {15:9.3f}'
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def csipatch2pscmp(csifault, ipatch, center=False, start_time=0, slip=True):
    '''
    Input   :
        * csifault  : TriangularPatches or RectangularPatches
        * ipatch    : [0, npatches)
        * center    : False，参考点为左上点；True:参考点为中心点
        * start_time: 起始时间
        * slip      : True,用csifault.slip的值，否则直接用slip作为islip
    Output  :
        * islppatch_str : 适合pscmp的单个滑动子块字符串
    '''
    if slip is True:
        islip = csifault.slip[ipatch, :]
    else:
        islip = np.asarray(slip)

    x, y, z, width, length, strikerad, diprad = csifault.getpatchgeometry(ipatch, center=center)
    strikedeg = np.rad2deg(strikerad)
    dipdeg = np.rad2deg(diprad)
    if not center:
        pos_s = length/2.0
        pos_d = width/2.0
    else:
        pos_s, pos_d = 0.0, 0.0
    O_lon, O_lat = csifault.xy2ll(x, y)
    O_depth = z
    islppatch_str = pscmp_slppatch_fmt.format(ipatch+1, O_lat, O_lon, O_depth, length, width, strikedeg, 
                                                dipdeg, 1, 1, start_time, pos_s, pos_d, *islip)

    # All Done
    return islppatch_str
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def csi2pscmp(csifault, center=False, start_time=0, slip=True, outfile='pscmp_slppatches.dat'):
    '''
    Input   :
        * csifault  :
        * outfile   :
        * start_time:
        * slip      :
        * center    :
    Comment :
        * pscmp     : 左旋为正，正断为正
        * csi       : csifault默认逆时针存储节点序号，for rect,默认排列顺序如下：
                p2     #--------# p1
                      /        /
                     /        /
                p3  #        #    p4
        * built by kfhe at 01/19/2023
    '''
    if slip is True:
        slip = csifault.slip
    else:
        slip = np.asarray(slip).reshape(-1, 3)
    if slip.shape[0] == 1:
        slip = np.ones_like(csifault.slip)*slip
    
    slppatches = []
    for ipatch in range(csifault.slip.shape[0]):
        islip = slip[ipatch, :]
        islppatch_str = csipatch2pscmp(csifault, ipatch, center=center, start_time=start_time, slip=islip)
        slppatches.append(islppatch_str)  
    if outfile.__class__ in (str, ):
        with open(outfile, 'wt') as fout:
            for islppatch_str in slppatches:
                print(islppatch_str, file=fout)
    else:
        for islppatch_str in slppatches:
            print(islppatch_str)

    # All Done
    return
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
def csi2Tent(trifault, method='linear', nvec=[0, 0, 1], 
                     scale=300.0, tometer=True, filename='tri2tentSlip.dat',
                     trunc=6):
    '''
    将csi中fault对象的滑动投影到Pylith Tritent形式
    nvec: 将断层端点坐标沿nvec正反方向分别平移scale长度，然后，将这些放在一起，进行最临近插值
    tometer：将xyz坐标转为单位m？默认：yes
    
    example:
        lon0 = 98.25
        lat0 = 34.5
        faults = csitri('maduo', lon0=lon0, lat0=lat0)

        # main slip
        faults.readPatchesFromFile('Maduo_Main_Coulomb_MPa.gmt')
        intpTrislip2Tent(faults, filename='tri2tentSlip_main_as.dat')
        # Tip slip
        faults = csitri('maduo', lon0=lon0, lat0=lat0)
        faults.readPatchesFromFile('Maduo_Tip_Coulomb_MPa.gmt')
        intpTrislip2Tent(faults, filename='tri2tentSlip_sec_as.dat')
    '''
    slip = trifault.slip
    xyz_km = np.array(trifault.getcenters())
    xyz_km[:, -1] *= -1
    upvec = np.array(nvec)*scale if nvec[-1] > 0 else -np.array(nvec)*scale
    downvec = -upvec
    xyz_up = xyz_km + upvec[np.newaxis, :]
    xyz_down = xyz_km + upvec[np.newaxis, :]
    xyz_km_ud = np.vstack([xyz_up, xyz_km, xyz_down])

    slip_ud = np.vstack((slip, slip, slip))

    verts = trifault.Vertices
    verts[:, -1] *= -1

    fault_slip = griddata(xyz_km_ud, slip_ud, verts, fill_value=0., method=method)
    
    plt.scatter(verts[:, 0], verts[:, -1], cmap=cm.jet, c=fault_slip[:, 0], vmin=0, vmax=5.)
    plt.colorbar()
    plt.show()

    xyzslip = pd.DataFrame(np.hstack((verts, fault_slip)), columns=['x', 'y', 'z', 'ss', 'ds', 'open'])
    if tometer:
        xyzslip['x'] = xyzslip['x']*1000.0
        xyzslip['y'] = xyzslip['y']*1000.0
        xyzslip['z'] = xyzslip['z']*1000.0
    #
    xyzslip.to_csv(filename, index=False, sep=' ', columns=['x', 'y', 'z', 'ss', 'ds', 'open'], 
                  float_format=f'%.{trunc}f')

    # All Done
    return
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
class Pscmp2gmt(csirectfault):

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, 
                        verbose=False):
        '''
        Args:
            * name      : Name of the object.
            * utmzone   : UTM zone  (optional, default=None)
            * ellps     : ellipsoid (optional, default='WGS84')
        '''
        
        super(Pscmp2gmt,self).__init__(name,
                                          utmzone=utmzone,
                                          ellps=ellps, 
                                          lon0=lon0,
                                          lat0=lat0, 
                                          verbose=verbose)
        
        # All done
        return


    def readPatchesFromPscmp(self, pscmpfile, skipheader=0, skipfooter=0, donotreadslip=False):
        '''
        Reads patches from a PSCMP formatted file.

        Args     :
            * pscmpfile      : pscmp.inp
        
        Kwargs   :
            * skipheader     : 需要跳过的头部非patch部分
            * skipfooter     : 需要跳过的尾部非patch部分
            * donotreadslip  ：是否读pscmp中的滑动值， 默认读取

        Comment  :
            * Time     : The function is built at 01/19/2023
        '''
        import pandas as pd
        # create the lists
        self.patch = []
        self.patchll = []
        self.Cm   = []
        if not donotreadslip:
            Slip = []

        # read all lines
        with open(pscmpfile, 'rt') as fin:
            A = fin.readlines()
        # remove header
        A = A[skipheader:-skipfooter]

        # depth
        D = 0.0

        def faultcoord2en(faultcoord, diprad, strrad):
            '''
            Input   :
                * faultcoord      : 断层坐标系，沿走向为正，沿倾向向下为正
                * diprad          : 断层倾角弧度
                * strrad          : 断层走向角弧度
            '''
            faultcoord = np.asarray(faultcoord).reshape(-1, 2)
            x, y = faultcoord[:, 0], -faultcoord[:, 1]
            midx, midy = x, y*np.cos(diprad)
            finalxy = (midx + midy*1.j) * np.exp(1.j*(np.pi/2.0 - strrad))
            en = np.vstack((finalxy.real, finalxy.imag)).T
            # All done
            return en.flatten()
        
        # Loop over the file
        i = 0
        # npatch = 0
        while i<len(A):
            
            # n lat lon dep leng wid strike dip np_st np_di st
            columns = 'n lat lon dep leng wid strike dip np_st np_di st'.split()
            dtype = (int,) + (float,)*7 + (int,)*2 + (float,)
            AA = [dtype[ind](istr.strip()) for ind, istr in enumerate(A[i].strip().split())]
            ifault = pd.DataFrame([AA], columns=columns)
            ifault['ip'] = ifault['n'].values[0]
            # npatches in ifault
            npatches = int(ifault.np_st.values[0]*ifault.np_di.values[0])
            columns_patch = 'pos_s pos_d slp_stk slp_ddip open'.split()
            dtype = (float,)*5
            ipatch = [[dtype[ind](istr.strip()) for ind, istr in enumerate(iA.strip().split())] for iA in A[i+1:i+npatches+1]]
            ipatch = pd.DataFrame(ipatch, columns=columns_patch)
            ipatch['ip'] = ifault['n'].values[0]

            # concerate the ifault and ipatch
            ifault = pd.merge(ifault, ipatch, left_on='ip', right_on='ip', how='inner')

            # Loop npatch in ifault
            for k in range(ifault.shape[0]):
                ipat = ifault.iloc[k]

                # dx, dy, dz
                diprad = np.deg2rad(ipat.dip)
                strrad = np.deg2rad(ipat.strike)
                swid = ipat.wid/2.0/ipat.np_di
                sleng = ipat.leng/2.0/ipat.np_st
                sdep = swid*np.sin(diprad)
                cx, cy = ipat.pos_s, ipat.pos_d
                cz = cy*np.sin(diprad)
                dx1, dy1, dz1 = cx + sleng, cy - swid, cz - sdep
                dx2, dy2, dz2 = cx - sleng, cy - swid, cz - sdep
                dx3, dy3, dz3 = cx - sleng, cy + swid, cz + sdep
                dx4, dy4, dz4 = cx + sleng, cy + swid, cz + sdep

                # proj the reference point to xyz
                xkm, ykm, zkm = *self.ll2xy(ipat.lon, ipat.lat), ipat.dep

                # get the values
                z1, z2, z3, z4 = zkm + dz1, zkm + dz2, zkm +dz3, zkm + dz4
                xy1 = np.array([xkm, ykm]) + faultcoord2en([dx1, dy1], diprad, strrad)
                xy2 = np.array([xkm, ykm]) + faultcoord2en([dx2, dy2], diprad, strrad)
                xy3 = np.array([xkm, ykm]) + faultcoord2en([dx3, dy3], diprad, strrad)
                xy4 = np.array([xkm, ykm]) + faultcoord2en([dx4, dy4], diprad, strrad)
                # translate to utm
                lon1, lat1 = self.xy2ll(*xy1)
                lon2, lat2 = self.xy2ll(*xy2)
                lon3, lat3 = self.xy2ll(*xy3)
                lon4, lat4 = self.xy2ll(*xy4)

                # Depth
                mm = min([float(z1), float(z2), float(z3), float(z4)])
                if D<mm:
                    D=mm
                # Set points
                p1 = [*xy1, z1]; p1ll = [lon1, lat1, z1]
                p2 = [*xy2, z2]; p2ll = [lon2, lat2, z2]
                p3 = [*xy3, z3]; p3ll = [lon3, lat3, z3]
                p4 = [*xy4, z4]; p4ll = [lon4, lat4, z4]


                # Store these
                p = [p1, p2, p3, p4]
                pll = [p1ll, p2ll, p3ll, p4ll]
                p = np.array(p)
                pll = np.array(pll)
                # Store these in the lists
                self.patch.append(p)
                self.patchll.append(pll)
            
            # Get the slip value
            if not donotreadslip:
                slip = ipatch[['slp_stk', 'slp_ddip', 'open']].values.tolist()
                Slip.extend(slip)
            # Change +/- to csi format
            Slip = np.array(Slip)
            Slip[:, 1] *= -1

            # increase i
            i += int(1+ifault.shape[0])

        # depth
        self.depth = D
        self.z_patches = np.linspace(0,D,5)

        # Translate slip to np.array
        if not donotreadslip:
            self.initializeslip(values=np.array(Slip))
        else:
            self.initializeslip()

        # Compute equivalent patches
        self.computeEquivRectangle()

        # All Done
        return 
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------


if __name__ == '__main__':
    from csi import RectangularPatches, faultwithdip, TriangularPatches
    import os
    from collections import OrderedDict
    trifaults = OrderedDict()

    utmzone = None
    lon0, lat0 = 101.31, 37.80

    trifault = faultwithdip('LLLBend', utmzone=None, ellps='WGS84', verbose=True, lon0=lon0, lat0=lat0)
    faultdir = r'd:\2022Menyuan\Interseismic_InSAR\3DDisp_External\3DDisp\3DDisp2InterInv\InvCode\mesh'
    trifault.file2trace(os.path.join(faultdir, 'fault_geometry_for_Haiyuan_3DVisco_extend1000km.txt'))
    trifault.top = 0. 
    trifault.width = 20.
    trifault.numz = 1
    trifault.buildPatchesNoDisc(89.5, 195, 100)
    nfaults = len(trifaults)
    # trifault.deletepatches([nfaults-1,0])
    trifault.setTrace(0.1)
    # trifault.writePatches2File(os.path.join('output', 'slip_total_shallow20km.gmt'), add_slip='total')
    trifault.plot(drawCoastlines=False, plot_on_2d=False)
    csi2pscmp(trifault, slip=[1, 0, 0], center=True)

    pscmp = Pscmp2gmt('test', lon0=106, lat0=33)
    # filename = r'd:\WC_benchmark\wan_2017\pscmpWan_benchmark.inp'
    # pscmp.readPatchesFromPscmp(filename, skipheader=241, skipfooter=1)

    filename = r'd:\psgrn_benchmark\fomosto-psgrn-pscmp-master\examples\pscmp08-wenchuan.inp'
    pscmp.readPatchesFromPscmp(filename, skipheader=254, skipfooter=1)
    pscmp.plot(drawCoastlines=False)