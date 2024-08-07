'''
Written by Kefeng He, Mar. 2023
'''

# import libs
import pandas as pd
import numpy as np
from csi import gps as csigps
from csi import multigps as csimgps
import matplotlib.pyplot as plt
import os

def genViscoAb(lon, lat, strikes_rad, datainblk1, datainblk2, extzeros=[0, 0], npatches=0, paranums=0):
    '''
    Object:
        * 指定点位坐标生成和块体相关的旋转参数，并投影到断层走向方向作为走滑分量约束的条件
    Input:
        * lon              : 边界待求点近似经度 (n, )
        * lat              : 边界待求点近似纬度 (n, )
        * strikes_rad      : 与lon,lat为长度一致的一维向量 (n, )
        * datainblk1       : 子断层相关的两个块体之一中的数据集 （上面板），参与块体旋转计算
        * datainblk2       : 子断层相关的两个块体之二的数据集（下面板/参考面板），参与块体旋转计算，作为被减去的部分
        * extzeros         : 需要在滑动和欧拉参数之间插入的零矩阵的列数 [X_ss X_ds X_euler_1 X_strain_1 X_euler_2 X_euler_2, ...]；
                             上面向量为未知参数排序，extzeros则表示分别对应与传入第一个和第二个数据对象对应的欧拉参数从X_ds之后的顺序位置
        * npatches         : 待求断层子块个数，包括所有断层分段的子断层个数和（用于计算X_ss + X_ds个数）
        * paranums         : 矩阵列数, 待求参数 m （可直接由装配格林函数直接计算得到, trifault.Gassembled.shape[1]）
    Output:
        * A          : Design matrix with unequal constraints
        * b          : Constant Vector with unequal constraints

    The line corresponding to the patch/location is the slip the upper-plate relative to the lower-plate.
    np.dot(V_visco[None, :], A[0, :])*x = V_visco
    V_visco + G*x = V_obs
    
    Here, only Strike-slip components are constrained
    Added by kfhe, at 10/26/2021
    '''
    size = lon.shape[0]
    A = np.zeros((size, paranums)) # (n, m)
    b = None
    strikes = np.asarray(strikes_rad)
    vec_str = np.vstack((np.cos(np.pi/2. - strikes), np.sin(np.pi/2. - strikes))).T
    lonc, latc = lon, lat
    xc, yc = datainblk1.ll2xy(lonc, latc)
    EulerMat1 = datainblk1.genEulerMatrix(lonc, latc) #  (n, 2) if component = 2, default 2
    EulerMat2 = datainblk2.genEulerMatrix(lonc, latc) #  (n, 2) if component = 2, default 2
    # 由于待求点为同一个点，且欧拉设计矩阵只和站点坐标相关，和块体无关；
    # 所以EulerMat1_strike == EulerMat2_strike
    EulerMat1_strike = EulerMat1[:size, :] * vec_str[:, 0][:, None] + EulerMat1[size:, :] * vec_str[:, 1][:, None]
    EulerMat2_strike = EulerMat2[:size, :] * vec_str[:, 0][:, None] + EulerMat2[size:, :] * vec_str[:, 1][:, None]

    nslip = int(npatches*2)
    Atmp = np.zeros((1, nslip))
    A[:, :nslip] = Atmp
    A[:, nslip+extzeros[0]: nslip+extzeros[0]+3] = EulerMat1_strike
    A[:, nslip+extzeros[1]: nslip+extzeros[1]+3] = -EulerMat2_strike
    b = np.zeros((A.shape[0]))
    return A, b


def genExtAb(fault, index, datainblk1, datainblk2, extzeros=[0, 0], paranums=0):
    '''
    Object:
        * 指定断层和需要约束的索引位置，生成和块体相关的旋转参数，并投影到断层走向方向作为走滑分量约束的条件
    Input:
        * fault            : 断层对象
        * index            : 子断层的顺序索引号
        * datainblk1       : 子断层相关的两个块体之一中的数据集，参与块体旋转计算
        * datainblk2       : 子断层相关的两个块体之二的数据集，参与块体旋转计算，作为被减去的部分
        * extzeros         : 需要在滑动和欧拉参数之间插入的零矩阵的列数
        * paranums         : 矩阵列数，默认直接取自断层对象？
    Output:
        * A          : Design matrix with unequal constraints
        * b          : Constant Vector with unequal constraints
    
    Here, only Strike-slip components are constrained
    Added by kfhe, at 10/26/2021
    --------
    
    Modified by kfhe, at 06/17/2022
    '''
    size = len(index)
    A = np.zeros((size, paranums))
    b = None
    centers = np.array(fault.getcenters())
    strikes = np.asarray(fault.getStrikes())[index]
    vec_str = np.vstack((np.cos(np.pi/2. - strikes), np.sin(np.pi/2. - strikes))).T
    xc, yc = centers[index, 0], centers[index, 1]
    lonc, latc = fault.xy2ll(xc, yc)
    EulerMat1 = datainblk1.genEulerMatrix(lonc, latc)
    EulerMat2 = datainblk2.genEulerMatrix(lonc, latc)
    # 由于待求点为同一个点，且欧拉设计矩阵只和站点坐标相关，和块体无关；
    # 所以EulerMat1_strike == EulerMat2_strike
    EulerMat1_strike = EulerMat1[:size, :] * vec_str[:, 0][:, None] + EulerMat1[size:, :] * vec_str[:, 1][:, None]
    EulerMat2_strike = EulerMat2[:size, :] * vec_str[:, 0][:, None] + EulerMat2[size:, :] * vec_str[:, 1][:, None]

    Atmp = np.eye(len(fault.patch))
    Atmp = Atmp[index, :]
    Atmp = np.hstack((Atmp, np.zeros_like(Atmp)))
    nslip = Atmp.shape[1]
    A[:, :nslip] = Atmp
    A[:, nslip+extzeros[0]: nslip+extzeros[0]+3] = EulerMat1_strike
    A[:, nslip+extzeros[1]: nslip+extzeros[1]+3] = -EulerMat2_strike
    b = np.zeros((A.shape[0]))
    return -A, b


def selectPatches(fault, tvert1, tvert2, mindep, maxdep):
    '''
    Object:
        * 根据迹线上采的任意两个点的投影XY坐标和所选深度范围，选择相应块体
    '''
    pselect = []
    strike, dip = np.mean(fault.getStrikes()), np.mean(fault.getDips())
    ddep = (maxdep - mindep)/np.tan(dip)
    dx, dy = np.cos(-strike)*ddep, np.sin(-strike)*ddep
    x1, y1 = tvert1[0], tvert1[1]
    x2, y2 = tvert2[0], tvert2[1]
    xmin, xmax = np.sort([x1, x2, x1+dx, x2+dx])[[0, -1]]
    ymin, ymax = np.sort([y1, y2, y1+dy, y2+dy])[[0, -1]]
    for p in range(len(fault.patch)):
        x1, x2, x3, width, length, strike, dip = fault.getpatchgeometry(p)
        if x1> xmin and x1 < xmax and x2>ymin and x2<ymax and x3>mindep and x3<maxdep:
            pselect.append(p)
    return pselect



def selectPatches_trans(fault, tvert1, tvert2, mindep, maxdep, tol=0.2):
    '''
    进行坐标转换后，然后再进行子块选取
    '''
    pselect = []
    tx1, ty1 = tvert1[0], tvert1[1]
    tx2, ty2 = tvert2[0], tvert2[1]
    slope = np.arctan2(ty2-ty1, tx2-tx1)
    slp_len = np.sqrt((ty2-ty1)**2 + (tx2-tx1)**2)
    for p in range(len(fault.patch)):
        x1, x2, x3, width, length, strike, dip = fault.getpatchgeometry(p)
        xy_trans = ((x1-tx1) + (x2-ty1)*1j)*np.exp(-slope*1j)
        x_trans = xy_trans.real
        if x_trans >=-tol and x_trans<slp_len+tol and x3>mindep and x3<maxdep:
            pselect.append(p)
    return pselect