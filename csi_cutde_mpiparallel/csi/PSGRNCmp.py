import os
import pickle
import numpy as np 
import pandas as pd
from glob import glob


# -----------------------------------------------------------------------------------------
def pscmpslip2dis(data, p, slip, BIN_PSCMP='PSCMP_BIN', psgrndir='psgrnfcts', pscmpinp='./pscmp_user.inp'):
    '''
    rectfault, 
    ss,ds,op will all have shape (Nd,3) for 3 components

    x, y, depth, width, length, strike, dip = rectfault.getpatchgeometry(p)
    lon, lat = rectfault.xy2ll(x, y)
    ss, ds, ts = pscmpslip2dis(data, p, slip=SLP)
    # pscmp left-lateral is postive, and normal is postive
    '''

    # Get executables
    # Environment variables need to be set in advance
    BIN_PSCMP = os.environ[BIN_PSCMP]
    # pscmp_template = os.path.join(BIN_PSCMP, 'pscmp_template.inp')
    ds_pscmpinp = pscmpinp[:-4] + '_ds.inp'
    ss_pscmpinp = pscmpinp[:-4] + '_ss.inp'

    # 输入到pscmp的格式
    optformat = '{0:4d} {1:9.4f} {2:9.4f} {3:6.1f} {4:6.1f} {5:6.1f} {6:6.1f} \
        {7:6.1f} {8:3d} {9:3d} {10:6.1f}\\n    {11:9.4f} {12:9.4f} {13:9.3f} {14:9.3f} {15:9.3f}'

    lon, lat, depth, width, length, strike, dip = p
    # strike, dip = strike*180./np.pi, dip*180./np.pi
    if slip[1] == 1:
        source_ds = optformat.format(1, lat, lon, depth, length, width, strike, dip, 1, 1, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0)

        filename = 'snapshot_coseism_ds{0:d}.dat'.format(1)
        # psgrnfcts_template
        os.system("sed -r 's/psgrnfcts_template/" + psgrndir + "/' {0} > {1}".format(pscmpinp, ds_pscmpinp))
        os.system("sed -r 's/snapshot_coseism.dat/" + filename + "/' {0} > tmp; mv tmp {0}".format(ds_pscmpinp))
        os.system("sed -r '/^   1/{N;s/.+/" + source_ds + "/}}' {0} > tmp; mv tmp {0}".format(ds_pscmpinp))
        os.system('"{0}/fomosto_pscmp2008a" {1} > /dev/null'.format(BIN_PSCMP, ds_pscmpinp))
        

    if slip[0] == 1:
        source_ss = optformat.format(1, lat, lon, depth, length, width, strike, dip, 1, 1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        filename = 'snapshot_coseism_ss{0:d}.dat'.format(1)
        os.system("sed -r 's/psgrnfcts_template/" + psgrndir + "/' {0} > {1}".format(pscmpinp, ss_pscmpinp))
        os.system("sed -r 's/snapshot_coseism.dat/" + filename + "/' {0} > tmp; mv tmp {0}".format(ss_pscmpinp))
        os.system("sed -r '/^   1/{N;s/.+/" + source_ss + "/}}' {0} > tmp; mv tmp {0}".format(ss_pscmpinp))
        os.system('"{0}/fomosto_pscmp2008a" {1} > /dev/null'.format(BIN_PSCMP, ss_pscmpinp))


    # 从文件中读取格林函数进来
    ds_file = os.path.join('pscmp_crust1', 'snapshot_coseism_ds1.dat')
    data = pd.read_csv(ds_file, sep=r'\s+')
    ds = data['Ux Uy Uz'.split()]
    ds.rename({'Uy': 'dx', 'Ux': 'dy', 'Uz': 'dz'}, inplace=True, axis=1)
    ds.loc[:, 'dz'] = ds.loc[:, 'dz']*-1
    ds = ds[['dx', 'dy', 'dz']].values

    ss_file = os.path.join('pscmp_crust1', 'snapshot_coseism_ss1.dat')
    data = pd.read_csv(ss_file, sep=r'\s+')
    ss = data['Ux Uy Uz'.split()]
    ss.rename({'Uy': 'dx', 'Ux': 'dy', 'Uz': 'dz'}, inplace=True, axis=1)
    ss.loc[:, 'dz'] = ss.loc[:, 'dz']*-1
    ss = ss[['dx', 'dy', 'dz']].values

    ts = np.zeros_like(ss)

    return ss, ds, ts
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
def changePoints2pscmpfile(data, BIN_PSCMP='PSCMP_BIN', pscmpinp='./pscmp_user.inp'):
    '''
    用于更新pscmp.inp文件中需求的观测点位信息
    '''
    # Get executables
    # Environment variables need to be set in advance
    BIN_PSCMP = os.environ[BIN_PSCMP]
    pscmp_template = os.path.join(BIN_PSCMP, 'pscmp_template.inp')

    import math
    lon, lat = data.lon, data.lat
    npnts = data.lon.shape[0]
    output_format = '({0:.3f},{1:.3f})'
    pntlines = ''
    for i, (ilat, ilon) in enumerate(zip(lat, lon)):
        if i%3 == 0:
            pntlines += '   ' + output_format.format(ilat, ilon)
        elif i%3 == 1:
            pntlines += ', ' + output_format.format(ilat, ilon)
        else:
            pntlines += ', ' + output_format.format(ilat, ilon) + '\n'
    if i%3 != 2:
        pntlines += '\n'
    
    assert os.path.exists(pscmp_template), 'Not find the pscmp input file: {}'.format(pscmp_template)
    os.system('cp "{0}" {1}'.format(pscmp_template, pscmpinp))
    with open(pscmpinp, 'rt') as fin:
        lines = fin.readlines()
        nlines = int(lines[71].strip())
        lines[71] = '  {0:d}\n'.format(npnts)
        lines[72] = pntlines
    
    nlines = math.ceil(nlines/3.)
    lines = lines[:73] + lines[72+nlines:]
    
    with open(pscmpinp, 'wt') as fout:
        for line in lines:
            print(line, end='', file=fout)

    return
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------