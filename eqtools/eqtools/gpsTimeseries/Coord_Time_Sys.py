#！/usr/bin/env python
import numpy as np
# import matplotlib.pyplot as plt
from numpy import sin ,cos, tan, pi, arctan2, arctan, arccos, arcsin, newaxis
from numpy import deg2rad, rad2deg 
import copy
'''
convert coordinate to different coordinate 
convert data to different form 
form  WQ && ShenZK to PSGRN/PSCMP
'''

### modified by hkf, 24/8/2017

###-------------------------------------------------------------------###
###                     Ellipsoid Constance                           ###
###-------------------------------------------------------------------###
WGS84 = dict(a = 6378137.0, b = 6356752.3142,
        c = 6399593.6258, e_2 = 0.00669437999013,
        e1_2 = 0.00673949674227, alpha = 1/298.257223563)

CGCS2000 = {'a': 6378137.0, 'b': 6356752.3141, 
            'c': 6399593.6259, 'alpha': 1/298.257222101,
            'e_2': 0.00669438002290, 'e1_2': 0.00673949677548}

###-------------------------------------------------------------------###
###                       auxiliary tools                             ###
###-------------------------------------------------------------------###
def Getatan(Y, X, *, ArgType='Degree', Rotmode='+X2+Y'):
    '''
    get atan2(y,x) , range is [-pi, pi]
    ArgType = Degree | Rad
    Rotmode = +X2+Y | +X2-Y
    '''
    if Rotmode == '+X2+Y':
        return arctan2(Y, X) if ArgType == 'Degree' else arctan2(Y, X)*180/pi
    else:
        return arctan2(-Y, X) if ArgType=='Degree' else arctan2(-Y, X)*180/pi
    
###-------------------------------------------------------------------###
###                   Coordinate Transform                            ###
###-------------------------------------------------------------------###
def BLH2XYZ(BLH, *, ArgType='Degree', Ellipsoid=WGS84):
    '''
    convert geodetic coordinates to space rectangular coordinates
    ArgType  =  'Degree' or 'Rad'
    BLH is a array-like seq 
    '''
    
    # extract Ellipsoid parameters to general api
    a, b, c, alpha, e_2, e1_2 = (Ellipsoid['a'], Ellipsoid['b'], 
                                 Ellipsoid['c'], Ellipsoid['alpha'], 
                                 Ellipsoid['e_2'], Ellipsoid['e1_2'])
    
    # convert BLH to a 2-ndim array 
    BLH=np.asarray(BLH)
    if BLH.ndim ==1:
        BLH.resize((1,3))
        
    # INIT XYZ && get the size of BLH's row
    XYZ,size=np.zeros_like(BLH,dtype=float),BLH.shape[0]
    
    # assign BLH to B, L, H
    B,L,H=BLH[:,0],BLH[:,1],BLH[:,2]
    
    if ArgType == 'Degree':
        B,L=np.deg2rad(B),np.deg2rad(L)
    W=np.sqrt(1-e_2*sin(B)**2)
    N=a/W
    XYZ[:,0] = (N+H)*cos(B)*cos(L)
    XYZ[:,1] = (N+H)*cos(B)*sin(L)
    XYZ[:,2] = (N*(1-e_2)+H)*sin(B)
    return XYZ

    
def XYZ2BLH(XYZ, *, ArgType='Degree', Ellipsoid=WGS84):
    '''
    convert space rectangular coordinates to geodetic coordinates
    ArgType  =  'Degree' or 'Rad'
    BLH is a array-like seq  
    default ouput: BLH by (np.ndarray) ,of which B,L is degree 
                   you can set ArgType to 'Rad' to change B,L by rad
    Input: XYZ array-like(list,tuple,array)
    '''
    a, b, c, alpha, e_2, e1_2 = (Ellipsoid['a'], Ellipsoid['b'], 
                                 Ellipsoid['c'], Ellipsoid['alpha'], 
                                 Ellipsoid['e_2'], Ellipsoid['e1_2'])
    XYZ=np.asarray(XYZ)
    if XYZ.ndim ==1:
        XYZ.resize((1,3))
    BLH,size=np.zeros_like(XYZ,dtype=float),XYZ.shape[0]
    X,Y,Z=XYZ[:,0],XYZ[:,1],XYZ[:,2]
    BLH[:,1] = Getatan(Y, X)
    k=1+e1_2
    t0=Z/np.sqrt(X**2+Y**2)
    p=c*e_2/np.sqrt(X**2+Y**2)
    t1=copy.copy(t0)
    for i in range(size):
        delt=1.0
        while delt>1e-15:
            t2=t0[i]+p[i]*t1[i]/np.sqrt(k+t1[i]**2)
            delt=np.fabs(t2-t1[i])
            t1[i]=t2
    BLH[:,0]=np.arctan(t1)
    BLH[:,2]=np.sqrt(X**2+Y**2)/cos(BLH[:,0])-a/np.sqrt(1-e_2*sin(BLH[:,0])**2)
    if ArgType == 'Degree':
        BLH[:,0],BLH[:,1]=np.rad2deg(BLH[:,0]),np.rad2deg(BLH[:,1])
    return BLH


def L2T(dxyz, Borg, Lorg, XYZorg, *, ArgType='Degree', Ellipsoid=WGS84):
    '''
    Input:
    Borg,Lorg is array-like  ,and default degree
    XYZorg is Reference point coordinates ,is array-like 
    dxyz is diff ,is array-like
    Output:
    XYZ
    '''
    a, b, c, alpha, e_2, e1_2 = (Ellipsoid['a'], Ellipsoid['b'], 
                                 Ellipsoid['c'], Ellipsoid['alpha'], 
                                 Ellipsoid['e_2'], Ellipsoid['e1_2'])
    Borg,Lorg=np.asarray(Borg),np.asarray(Lorg)
    if ArgType == 'Degree':
        Borg,Lorg=np.deg2rad(Borg),np.deg2rad(Lorg)
    dxyz,XYZorg=np.asarray(dxyz),np.asarray(XYZorg)
    if XYZorg.ndim == 1:
        dxyz.resize((1,3))
        XYZorg.resize((1,3))
    XYZ,size=np.zeros_like(dxyz,dtype=float),dxyz.shape[0]
    transmat=np.zeros((3,3))
    for i in range(size):
        B=Borg[i]
        L=Lorg[i]
        transmat[0,:]=[-sin(B)*cos(L),-sin(L),cos(B)*cos(L)]
        transmat[1,:]=[-sin(B)*sin(L),cos(L),cos(B)*sin(L)]
        transmat[2,:]=[cos(B),0,sin(B)]
        XYZ[i,:]=(XYZorg[i,:].transpose()+np.dot(transmat,dxyz[i])).transpose()
    return XYZ


def T2L(XYZorg, Borg, Lorg, XYZend, *, ArgType='Degree', Ellipsoid=WGS84):
    '''
    '''
    a, b, c, alpha, e_2, e1_2 = (Ellipsoid['a'], Ellipsoid['b'], 
                                 Ellipsoid['c'], Ellipsoid['alpha'], 
                                 Ellipsoid['e_2'], Ellipsoid['e1_2'])
    Borg,Lorg=np.asarray(Borg),np.asarray(Lorg)
    XYZorg,XYZend=np.asarray(XYZorg),np.asarray(XYZend)
    if ArgType == 'Degree':
        Borg,Lorg=np.deg2rad(Borg),np.deg2rad(Lorg)
    if XYZorg.ndim ==1:
        XYZorg.resize((1,3))
        XYZend.resize((1,3))
    dxyz,size=np.zeros_like(XYZorg,dtype=float),XYZorg.shape[0]
    XYZdiff=XYZend-XYZorg
    transmat=np.zeros((3,3))
    for i in range(size):
        B0=Borg[i]
        L0=Lorg[i]
        transmat[0,:]=-sin(B0)*cos(L0),-sin(L0),cos(B0)*cos(L0)
        transmat[1,:]=-sin(B0)*sin(L0),cos(L0),cos(B0)*sin(L0)
        transmat[2,:]=cos(B0),0,sin(B0)
        dxyz[i,:]=(np.dot(transmat.transpose(),XYZdiff[i])).transpose()
    return dxyz


###-------------------------------------------------------------------###
###                    Coordinate System Transform                    ###
###-------------------------------------------------------------------###
def transform(theta, x, y):
    '''
    坐标转换
    Input:
        theta：转换角度，坐标系顺时针转，theta为正，否则为负
        x, y: 原坐标系中坐标值（右手系），可以为单值/np.array
    Output:
        xnew, ynew
    '''
    coord = x + y*1.0j
    transcoord = coord*np.exp(theta*1.0j)
    return transcoord.real, transcoord.imag


###-------------------------------------------------------------------###
###                    uxiliary tools for Time                        ###
###-------------------------------------------------------------------###
def IsLeap(Year):
    if (Year%4==0 and Year%100!=0) or Year%400==0:
        return True
    else:
        return False

def HMS2DotD(HMS):
    if isinstance(HMS,float) or isinstance(HMS,int):HMS = [HMS]
    HMS = np.asarray(HMS).reshape(-1,1)
    return (HMS[:,0]+HMS[:,1]/60.0+HMS[:,2]/3600.0)/24.0
    
def DotD2HMS(DotD):
    if isinstance(DotD, float): DotD = [DotD]
    DotD = np.asarray(DotD)
    HMS = np.zeros((DotD.shape[0],3))
    HMS[:,0] = np.floor(DotD*24.0)
    HMS[:,1] = np.floor((DotD*24.0-HMS[:,0])*60)
    HMS[:,2] = DotD*24.0*3600 - HMS[:,0]*3600 -HMS[:,1]*60
    return HMS 
###-------------------------------------------------------------------###
###                        Time convince                              ###
###-------------------------------------------------------------------###
def YMD2MJD(YMD, hms=[11,59,0.0]):
    '''
    YMD is seq : (yyyy,mm,dd)
    hh,min,sec : can be a array-like
    '''
    YMD = np.asarray(YMD).reshape(-1,3)
    hms = np.asarray(hms).reshape(-1,3)
    
    hh, min, sec = hms[:,0],hms[:,1], hms[:,2]
    Year, Mon, Day = YMD[:,0], YMD[:,1], YMD[:,2]
    size = YMD.shape[0]
    for i in range(size):
        if Mon[i] <= 2:
            Year[i] -= 1
            Mon[i] += 12
    JD = np.floor(365.25*Year) + np.floor(30.6001*(Mon+1)) + Day + (hh+min/60.0+sec/3600.0)/24.0 + 1720981.5
    MJD = JD - 2400000.5
    return MJD

def MJD2YMD(MJD):
    '''
    MJD is a float or seq 
    '''
    if isinstance(MJD,float):
        MJD = np.array([MJD])
    else:
        MJD = np.asarray(MJD)
    MJD =np.asarray(MJD)
    JD = MJD + 2400000.5
    a = np.floor(JD+0.5)
    b = a + 1537
    c = np.floor((b-122.1)/365.25)
    d = np.floor(365.25*c)
    e = np.floor((b-d)/30.600)
    D = b-d-np.floor(30.6001*e)+(JD+0.5-a)
    M = e-1-12*np.floor(e/14.0)
    Y = c-4715-np.floor((7.0+M)/10.0)
    return np.hstack((Y.reshape(-1,1),M.reshape(-1,1),D.reshape(-1,1)))
    

days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def YMD2DOY(YMD):
    '''
    YMD is seq : (yyyy,mm,dd)
    hh,min,sec : can be a array-like
    Output: array[days,seconds of the day] 
    '''
    YMD = np.asarray(YMD).reshape(-1,3)
    Doy = np.zeros((YMD.shape[0],2))
    
    for i in range(YMD.shape[0]):
        Doy[i,0] += days_in_month[i]
        if i == 1 and IsLeap(YMD[i,0]):
            Doy[i,0] += 1
    
    Doy[:,0] += YMD[:,2]
    return Doy 

def DOY2YMD(Year,Doy):
    '''
    Year && Doy is integear type or seq, don't think about the seconds of the day 
    '''
    seq = isinstance(Year, int)
    if seq: 
        Year, Doy = np.array([Year]), np.array([Doy])
    else:
        Year, Doy = np.asarray(Year), np.asarray(Doy)
    day, month = Doy, np.zeros_like(Doy)
    for size in range(Doy.shape[0]):
        for i in range(12):
            dim = days_in_month[i]
            if IsLeap(Year[i]) and i==1: dim += dim
            if day[size] <= dim: break
            day[size] -= dim 
        month[size] = i+1
    return np.hstack((Year.reshape(-1,1),month.reshape(-1,1),day.reshape(-1,1)))

def YMD2GPST(YMD,HMS=[11,59,00]):
    '''
    返回的是周内秒和周数
    '''
    MJD0 = YMD2MJD([1980,1,6],hms=[0,0,0.0])
    MJD = YMD2MJD(YMD,HMS)
    Diff = MJD - MJD0 
    GPST = np.zeros((MJD.shape[0],2))
    GPST[:,0] = np.floor(Diff/7.0)
    GPST[:,1] = (Diff -GPST[:,0]*7)*24*3600
    return GPST

def MJD2DotY(MJD):
    YMD = MJD2YMD(MJD)
    print(YMD[0,0])
    return YMD[0,0] + (MJD - YMD2MJD([YMD[0,0], 1, 1], [0, 0, 0]))/365
    
def YMD2DotY(YMD, hms=[11, 59, 0]):
    YMD = np.asarray(YMD).reshape(-1,3)
    hms = np.asarray(hms).reshape(-1,3)
    MJD = YMD2MJD(YMD, hms)
    YMDAN = YMD.copy()
    YMDAN[:,1:] = 1
    return YMD[:,0] + (MJD - YMD2MJD(YMDAN,[0, 0, 0]))/365
    
def DotY2YMD(): pass 
#########################################################################
###                                                                   ###
###                         calculate the distance                    ###
###                                                                   ###
#########################################################################

def sphdistance(pointA, pointB):
    '''
    计算两点经纬度之间的距离point: (Lon, Lat)
    '''
    ra=6378.140 #赤道半径
    rb=6356.755 #极半径 （km）
    flatten=(ra-rb)/ra  #地球偏率
    pointA, pointB = deg2rad(pointA), deg2rad(pointB)
    pA=arctan(rb/ra*tan(pointA[1]))
    pB=arctan(rb/ra*tan(pointB[1]))
    xx=arccos(sin(pA)*sin(pB)+cos(pA)*cos(pB)*cos(pointA[0]-pointB[0]))
    c1=(sin(xx)-xx)*(sin(pA)+sin(pB))**2/cos(xx/2)**2
    c2=(sin(xx)+xx)*(sin(pA)-sin(pB))**2/sin(xx/2)**2
    dr=flatten/8*(c1-c2)
    distance=ra*(xx+dr)
    return distance
