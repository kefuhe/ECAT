# -*- coding: utf-8 -*-
#！/usr/bin/env python
import pandas as pd
import numpy as np
from . import Coord_Time_Sys as cts
import yaml
from yaml import Loader
import re
from numpy import sin, cos, log, exp, pi
from scipy.optimize import curve_fit
from PyAstronomy import pyasl
import matplotlib.pyplot as plt
from copy import deepcopy


class ConfigDict(dict):
    '''
    配置信息类型，继承自dict
    '''
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self):
        self['Linear'] = True
        self['PeriodCycle'] = False
        self['Breaks'] = dict()
        self['EQinfos'] = dict()
        self['TimeRange'] = [pd.Timestamp('1900-01-01'), None]
        self['Velinfos'] = dict()

    def readcmd(self, cmdfilename):
        '''
        读取配置数据拟合配置文件信息，格式为YAML格式，如： cmdprecamp.yaml
        PeriodCycle: false
        Linear: true
        TimeRange: [1999-03-16t00:00:00, null]
        VelVal:
            H035: [40.24, 13.26]
            #H010: [37.25, 14.25, 8.35]
        Velfile:
            - precampfit.vel
            #- precamp.greenspline
        EQfile: '/home/kfhe/汶川震后数据/chuandian/pycode/dataprocessingcfgfiles/eq_rename.rename'
        '''
        with open(cmdfilename, encoding='utf-8') as fincong:
            cmdparams = yaml.load(fincong, Loader=Loader)
        # 更新fitconfigparams
        if cmdparams.get('PeriodCycle') is not None:
            self['PeriodCycle'] = cmdparams['PeriodCycle']
        if cmdparams.get('Linear') is not None:
            self['Linear'] = cmdparams['Linear']
        if cmdparams.get('TimeRange') is not None:
            self['TimeRange'][0] = pd.to_datetime(cmdparams['TimeRange'][0]) if cmdparams['TimeRange'][0] is not None else None
            self['TimeRange'][1] = pd.to_datetime(cmdparams['TimeRange'][1]) if cmdparams['TimeRange'][1] is not None else None
        # 读取速度文件, 列表可为2，也可为3，根据长度判断
        if cmdparams.get('Velfile') is not None:
            for filename in cmdparams['Velfile']:
                self['Velinfos'].update(self.readvel(filename))
        # 读取速度值，此更新会覆盖文件中的值和预设值
        if cmdparams.get('VelVal') is not None:
            self['Velinfos'].update(cmdparams['VelVal'])
        if cmdparams.get('EQfile') is not None:
            # print(cmdparams['EQfile'])
            eqinfos, brkinfos = self.ReadEQ(cmdparams['EQfile'])
            # print(eqinfos, brkinfos)
            self['EQinfos'].update(eqinfos)
            self['Breaks'].update(brkinfos)
    
    def readvel(self, filename):
        '''
        读入速度文件，文件格式为：Name Ve Vn Vu, ...包含头行
        velocity: 字典，Name: Ve, Vn, Vu
        '''
        velocity = pd.read_csv(filename, sep='\s+', index_col=0, comment='#')
        velocity = velocity.T
        velocity = {idx: [velocity[idx].get('Ve'), velocity[idx].get('Vn'), velocity[idx].get('Vu')] for idx in velocity}
        return velocity

    def ReadEQ(self, filename):
        '''
        读取地震文件，其中和tsfit地震文件的主要不同就只是tau值是否为-1
        '''
        eqinfos, brkinfos = dict(), dict()
        try:
            with open(filename, 'rt') as fin:
                for Line in fin:
                    # 空行跳过
                    if re.match(r'^\s*$', Line):
                        continue
                    # 用#注释行跳过
                    if Line[0] == '#':
                        continue
                    # 用*注释行跳过
                    if Line[0] == '*':
                        continue
                    # 如果第一列为空格，则继续
                    if Line[0] == ' ':
                        # 移除行中#注释
                        Line = re.sub(r'\#.*$', '', Line)
                        # 如果发现地震定义eq_def
                        if Line.find('eq_def') != -1:
                            Linelist = Line.split()
                            name = Linelist[1]
                            LonLatDep = [float(Linelist[3]), float(Linelist[2]), float(Linelist[5])]
                            radrange = float(Linelist[4])
                            decayflag = 0
                            # print(LonLatDep, name, radrange)
                            cotime = pd.to_datetime(' '.join(Linelist[6:]), format='%Y %m %d %H %M')
                            eqinfos.update({name: EQinfo(name, LonLatDep, cotime, radrange, decayflag)})
                        if Line.find('eq_log') != -1:
                            Linelist = Line.split()
                            # 存储为以地震名为键，以衰减时间为值的字典
                            eqinfos[Linelist[1]].decayflag = float(Linelist[2])
                        # 如果存在break项
                        if Line.find('break') != -1:
                            Linelist = Line.split()
                            # 如果brks_all中之前没有存储该站点， 则先赋予一个空列表
                            if brkinfos.get(Linelist[1]) is None:
                                brkinfos[Linelist[1]] = []
                            # 填写的时间为pd.Timestamp
                            brkinfos[Linelist[1]].append(pd.to_datetime(' '.join(Linelist[2:]), format='%Y %m %d %H %M'))
            return eqinfos, brkinfos
        except IOError as ioerr:
            print('Reading EQfile error: '+str(ioerr))


class GPSTimeSeriesFitInfo(object):
    '''
    时间范围，如果不输入，则使用默认的时间范围
    '''
    def __init__(self, posdata, fitconfigparams, timerange=None):
        # 保存数据
        self.name = posdata.name
        self.fitconfigparams = fitconfigparams
        self.timerange = self.get_timerange(posdata, timerange)
        self.posdata = self.get_posdata(posdata)
        self.t = np.array([datetime2decimalyear(date) for date in self.posdata['Date']])
        self.Linear = self.fitconfigparams['Linear'] 
        self.PeriodCycle = self.fitconfigparams['PeriodCycle']
        self.set_velocity(posdata.name)
        self.set_earthquake_info(posdata.BLH[-2::-1])
        self.set_breaks(posdata.name)

    def get_timerange(self, posdata, timerange):
        if timerange is None:
            timerange = self.fitconfigparams['TimeRange']
        starttime = max(timerange[0], posdata.data['Date'].min())
        endtime = timerange[1] if timerange[1] is not None and timerange[1] < posdata.data['Date'].max() else posdata.data['Date'].max()
        return [starttime, endtime]

    def get_posdata(self, posdata):
        return posdata.data.query('Date >= @self.timerange[0] & Date <= @self.timerange[1]')

    def set_velocity(self, sitename):
        velocity = self.fitconfigparams['Velinfos'].get(sitename)
        self.Ve, self.Vn, self.Vu = velocity if velocity is not None else (None, None, None)

    def set_earthquake_info(self, sitecoord):
        self.EQs, self.coflag, self.decayflag = dict(), dict(), dict()
        for key in self.fitconfigparams['EQinfos']:
            eq = self.fitconfigparams['EQinfos'][key]
            if eq.inrange(sitecoord):
                if eq.cotime < self.timerange[1]:
                    self.EQs[key] = eq
                    # 0表示不列方程，1表示列方程
                    if eq.cotime < self.timerange[0]:
                        self.coflag[key] = 0
                    else:
                        self.coflag[key] = 1
                    # 0表示不列方程，1表示列方程
                    if eq.decayflag == 0:
                        self.decayflag[key] = 0
                    else:
                        self.decayflag[key] = 1

    def set_breaks(self, sitename):
        self.brks = self.fitconfigparams['Breaks'].get(sitename)
        if self.brks is not None:
            self.brks = deepcopy(self.brks)
            for idx, brk in enumerate(self.brks):
                if brk <= self.starttime or brk >= self.endtime:
                    self.brks.pop(idx)

    @property
    def starttime(self):
        return self.timerange[0]
    
    @property
    def endtime(self):
        return self.timerange[1]


class Posdata(object):

    def __init__(self, data=None, name=None, BLH=None):
        self._data = data
        self.name = name
        self.BLH = BLH

    @classmethod
    def read(cls, filename):
        """从文件中读取数据，并创建一个新的Posdata对象"""
        try:
            with open(filename, 'rt') as fin:
                filehead, name, BLH = cls._parse_file_head(fin)
                obsdata = pd.read_csv(filename, sep='\s+', skiprows=36, parse_dates={'Date':['YYYYMMDD', 'HHMMSS']}, escapechar='*', comment='#')
                data = obsdata.loc[:, 'Date dN dE dU Sn Se Su'.split()]
                data.loc[:, 'dN dE dU Sn Se Su'.split()] *= 1000.0
                return cls(data, name, np.asarray(BLH))
        except Exception as e:
            raise ValueError(f"无法从文件{filename}中读取数据!") from e

    @classmethod
    def _parse_file_head(cls, fin):
        """解析文件头，获取文件头信息，站点名称和BLH坐标"""
        endfield = 'End Field Description'
        filehead = ''
        while True:
            line = fin.readline()
            filehead += line
            if "NEU Reference position" in line:
                BLH = [float(s) for s in line.strip().split()[4:7]]
            elif "4-character ID" in line:
                name = line.strip().split()[-1]
            if line.startswith(endfield):
                break
        return filehead, name, BLH

    def update_data(self, new_data):
        """更新数据"""
        self._data = new_data

    def head(self, *args, **kwargs):
        """返回数据的头部信息"""
        return self._data.head(*args, **kwargs)

    @property
    def data(self):
        """获取数据"""
        return self._data

    @property
    def loc(self):
        """获取数据的loc属性"""
        return self._data.loc

    @property
    def iloc(self):
        """获取数据的iloc属性"""
        return self._data.iloc

    def __getitem__(self, index):
        """获取指定索引的数据"""
        return self._data[index]


class EQinfo(object):

    def __init__(self, eqname, focus, cotime, radiationrange, decayflag):
        '''
        eqname: 地震简称，两字母大写，如：'WC'
        focus: 震源位置， 类数组[Lon(deg), Lat(deg), Dep(km)]
        cotime: 发震时刻，pd.Timestamp('2008-05-12 06:28:00')
        radiationrange: 地震同震辐射范围， 如：800, 单位：km
        decayflag: 是否估计震后影响，0：不估计，val>0， 估计，且tau已知
        '''
        self._eqname = eqname
        self._focus = np.asarray(focus)
        self._cotime = cotime
        self._radiationrange = radiationrange
        self._decayflag = decayflag
    
    def inrange(self, sitecoord):
        if cts.sphdistance(self.focus[:2], sitecoord) < self.radiationrange:
            return True
        return False
    
    def __str__(self):
        return self._eqname + ': \n'\
            + '    focus: ' +str(self._focus)\
            + ' cotime: ' + str(self._cotime)\
            + ' radiationrange: ' + str(self._radiationrange)\
            + ' decayflag: ' + str(self._decayflag)
    
    @property
    def eqname(self):
        return self._eqname
    
    @eqname.setter
    def eqname(self, eqname):
        assert len(eqname) == 2, "地震名称必须为2字符缩写"
        self._eqname = eqname.upper()
    
    @property
    def focus(self):
        return self._focus
    
    @focus.setter
    def focus(self, focus):
        assert len(focus) == 3, "必须以类数组形式提供[经度(deg)，纬度(deg)、深度(km)]" 
        self._focus = np.asarray(focus)
    
    @property
    def cotime(self):
        return self._cotime
    
    @cotime.setter
    def cotime(self, cotime):
        assert isinstance(cotime, pd.Timestamp), "输入时间类型不是pd.Timestamp"
        self._cotime = cotime
    
    @property
    def radiationrange(self):
        return self._radiationrange
    
    @radiationrange.setter
    def radiationrange(self, radiationrange):
        self._radiationrange = radiationrange
    
    @property
    def decayflag(self):
        return self._decayflag
    
    @decayflag.setter
    def decayflag(self, decayflag):
        assert decayflag == -1 or decayflag >=0, "输入衰减标志码错误"
        self._decayflag = decayflag
 
#########################################################################
###                            定义拟合中间函数                       ###
#########################################################################


def datetime2decimalyear(datetime):
    '''
    Input：pydatetime/pd.datetime/np.datetime
    e.g. Timestamp('2009-08-12 14:22:32.000456')
    Output: float, e.g. 2019.639972095528
    '''
    return pyasl.decimalYear(datetime)


def decimalyear2datetime(decimalyear):
    '''
    Input: decimal year, e.g. 2019.639972095528
    Output: Timestamp('2009-08-12 14:22:32.000456')
    '''
    pydatetime = pyasl.decimalYearGregorianDate(decimalyear)
    return pd.to_datetime(pydatetime)


def Break(t, t0, param):
    '''
    定义break函数，输入t0时间，输出break阶跃函数
    '''
    return param * np.heaviside(t - t0, 0)


def Line(t, t0, a, b):
    '''
    定义线性项函数
    '''
    return a + b*(t - t0)


def Cycle(t, c, d, e, f):
    '''
    定义周期项
    '''
    return (c*sin(2*pi*t) + d*cos(2*pi*t) +
            e*sin(4*pi*t) + f*cos(4*pi*t))


def Exps(t, teq, tau, g):
    '''
    定义指数衰减项
    '''
    if tau < 0:
        raise ValueError("'tau' must be a larger num than 0.")
    elif tau == 0:
        raise ZeroDivisionError("'tau' equal 0, 'tau' should be larger than 0")
    return g * np.heaviside(t - teq, 0)*(1 - exp(-(t-teq)*365.0/tau))


def Logs(t, teq, tau, g):
    '''
    定义对数衰减项
    '''
    if tau < 0:
        raise ValueError("'tau' must be a larger num than 0.")
    elif tau == 0:
        raise ZeroDivisionError("'tau' equal 0, 'tau' should be larger than 0")
    temp = np.zeros_like(t)
    temp[t > teq] = g*log(1 + (t[t > teq] - teq)*365.0/tau)
    return temp


class FitModel(object):

    def __init__(self, site, PostFunc=Exps, LineFunc=Line, CycleFunc=Cycle, BreakFunc=Break):
        '''
        初始化：选择拟合函数
        '''
        self.DecayFunc = PostFunc
        self.LinearFunc = LineFunc
        self.BreakFunc = BreakFunc
        self.CycleFunc = CycleFunc

        self.initialize(site)
        self.configure()
    
    def configure(self):
        '''
        配置信息：配置参量的初始化信息，和范围信息
        将输入配置选项翻译
        '''
        # 速度目前支持可选择性提供
        from collections import OrderedDict
        # 待求参数的name
        self.coefE_name = []
        # 除了拟合函数的自变量：时间，包含对应于函数的其他参数，待求参数按顺序设置为序列整数值
        self.coefE_value = OrderedDict()
        self.coefN_name = []
        self.coefN_value = OrderedDict()
        self.coefU_name = []
        self.coefU_value = OrderedDict()

        # 一定会估计一个常数项
        argidxE, argidxN, argidxU = 0, 0, 0
        self.coefE_name.append('CONSTANT')
        self.coefN_name.append('CONSTANT')
        self.coefU_name.append('CONSTANT')

        # 这里设置t0项，设置为观测值的平均情况
        self.coefE_value['Linear'] = [self.t.mean()]
        self.coefN_value['Linear'] = [self.t.mean()]
        self.coefU_value['Linear'] = [self.t.mean()]

        # self.coefE_value['Linear'] = [argidxE]
        self.coefE_value['Linear'].append(argidxE)
        self.coefN_value['Linear'].append(argidxN)
        self.coefU_value['Linear'].append(argidxU)
        argidxE += 1
        argidxN += 1
        argidxU += 1
        # 如果线性估计项为0，则表示，只估计常数项，这里将速度直接设置为0
        if not self.Linear:
            self.Ve = self.Vn = self.Vu = 0.0
            self.coefE_value['Linear'].append(0.0)
            self.coefN_value['Linear'].append(0.0)
            self.coefU_value['Linear'].append(0.0)
        else:
            if self.Ve is None:
                self.coefE_name.append('VELOCITY')
                self.coefE_value['Linear'].append(argidxE)
                argidxE += 1
            else:
                self.coefE_value['Linear'].append(self.Ve)
            if self.Vn is None:
                self.coefN_name.append('VELOCITY')
                self.coefN_value['Linear'].append(argidxN)
                argidxN += 1
            else:
                self.coefN_value['Linear'].append(self.Vn)
            if self.Vu is None:
                self.coefU_name.append('VELOCITY')
                self.coefU_value['Linear'].append(argidxU)
                argidxU += 1
            else:
                self.coefU_value['Linear'].append(self.Vu)
        if self.PeriodCycle:
            self.coefE_name.extend(['ASIN', 'ACOS', 'SASIN', 'SACOS'])
            self.coefN_name.extend(['ASIN', 'ACOS', 'SASIN', 'SACOS'])
            self.coefU_name.extend(['ASIN', 'ACOS', 'SASIN', 'SACOS'])
            self.coefE_value['PeriodCycle'] = list(range(argidxE, argidxE+4))
            self.coefN_value['PeriodCycle'] = list(range(argidxN, argidxN+4))
            self.coefU_value['PeriodCycle'] = list(range(argidxU, argidxU+4))
            argidxE += 4
            argidxN += 4
            argidxU += 4
        if self.brks is not None and self.brks != []:
            for idx, brk in enumerate(self.brks):
                self.coefE_name.append('Break_' + str(idx))
                self.coefN_name.append('Break_' + str(idx))
                self.coefU_name.append('Break_' + str(idx))
                # 阶跃项在这里转化为了小数年
                brk = datetime2decimalyear(brk)
                self.coefE_value['Break_' + str(idx)] = [brk, argidxE]
                self.coefN_value['Break_' + str(idx)] = [brk, argidxN]
                self.coefU_value['Break_' + str(idx)] = [brk, argidxU]
                argidxE += 1
                argidxN += 1
                argidxU += 1
        for eq in self.EQs:
            if self.coflag[eq] == 1:
                self.coefE_name.append('CoBreak_' + eq)
                self.coefN_name.append('CoBreak_' + eq)
                self.coefU_name.append('CoBreak_' + eq)
                cotime = datetime2decimalyear(self.EQs[eq].cotime)
                self.coefE_value['CoBreak_' + eq] = [cotime, argidxE]
                self.coefN_value['CoBreak_' + eq] = [cotime, argidxN]
                self.coefU_value['CoBreak_' + eq] = [cotime, argidxU]
                argidxE += 1
                argidxN += 1
                argidxU += 1
            if self.decayflag[eq] == 1:
                self.coefE_name.append('Decay_' + eq)
                self.coefN_name.append('Decay_' + eq)
                self.coefU_name.append('Decay_' + eq)
                # cotime在这里转化为了小数年
                cotime = datetime2decimalyear(self.EQs[eq].cotime)
                self.coefE_value['Decay_' + eq] = [cotime, self.EQs[eq].decayflag, argidxE]
                self.coefN_value['Decay_' + eq] = [cotime, self.EQs[eq].decayflag, argidxN]
                self.coefU_value['Decay_' + eq] = [cotime, self.EQs[eq].decayflag, argidxU]
                argidxE += 1
                argidxN += 1
                argidxU += 1
        # 存储拟合返回的欲求参数值
        self.fitE_popt = None
        self.fitN_popt = None
        self.fitU_popt = None
        self.fitE_pcov = None
        self.fitN_pcov = None
        self.fitU_pcov = None
    
    def initialize(self, gpstimeseriesfitinfo):
        '''
        这里获得单站相关信息，并转接到该拟合对象中
        '''
        self.siteinfo = gpstimeseriesfitinfo
        self.brks = gpstimeseriesfitinfo.brks
        self.coflag = gpstimeseriesfitinfo.coflag
        self.decayflag = gpstimeseriesfitinfo.decayflag
        self.starttime = gpstimeseriesfitinfo.timerange[0]
        self.endtime = gpstimeseriesfitinfo.timerange[1]
        self.EQs = gpstimeseriesfitinfo.EQs
        self.Ve = gpstimeseriesfitinfo.Ve
        self.Vn = gpstimeseriesfitinfo.Vn
        self.Vu = gpstimeseriesfitinfo.Vu
        self.PeriodCycle = gpstimeseriesfitinfo.PeriodCycle
        self.Linear = gpstimeseriesfitinfo.Linear
        self.posdata = gpstimeseriesfitinfo.posdata
        # 这里时间已经转化为小数年的np.array数组
        self.t = gpstimeseriesfitinfo.t

    def full_filter(self, t, dir='E'):
        # 这里设置t0为一致的，均为中间值，可以考虑改成参数项
        # t0 = np.mean(t)
        coef_value = getattr(self, 'coef' + dir + '_value')
        coef_name = getattr(self, 'coef' + dir + '_name')
        def pos_filter(t, *p):
            '''
            观测值单方向的待拟合函数
            '''
            y = 0
            for flag in coef_value:
                if 'Linear' in flag:
                    args = deepcopy(coef_value['Linear'])
                    args[1] = p[args[1]]
                    if 'VELOCITY' in coef_name:
                        args[2] = p[args[2]]
                    #else:
                        #args[1] = getattr(self, 'V' + dir.lower())
                    y += self.LinearFunc(t, *args)
                if 'PeriodCycle' in flag:
                    args = [p[i] for i in coef_value['PeriodCycle']]
                    y += self.CycleFunc(t, *args)
                if 'Break' in flag or 'CoBreak' in flag:
                    args = [coef_value[flag][0], p[coef_value[flag][1]]]
                    y += self.BreakFunc(t, *args)
                if 'Decay' in flag or 'DECAY' in flag:
                    y += self.DecayFunc(t, *(coef_value[flag][0:2]), p[coef_value[flag][2]])
            return y

        return pos_filter

    def setpoptinit(self, dir='E'):
        coef_name = getattr(self, 'coef' + dir + '_name')
        length = len(coef_name)
        pinit = [1.0,]*length
        return pinit
    
    def setpoptbound(self, dir='E'):
        coef_name = getattr(self, 'coef' + dir + '_name')
        length = len(coef_name)
        plower = tuple([-np.inf,]*length)
        pupper = tuple([np.inf, ]*length)
        return [plower, pupper]
    
    def fit(self, taurange=None, eqname=None, dir='E'):
        '''
        这里开始拟合，得道拟合参数：self.coef_
        '''
        t = self.t
        # 没有进行深拷贝，因此更改会同步到self.coefE_value
        coef_value = getattr(self, 'coef' + dir + '_value')
        minperr = 0.0
        mintau = 0.0
        minres = np.inf
        minfit_func = None
        minpopt, minpcov = None, None
        # 获得拟合参数初始值
        pinit = self.setpoptinit(dir)
        # 获得拟合参数范围
        pbound = self.setpoptbound(dir)
        if eqname is not None and taurange is not None:
            for tau in range(*taurange):
                coef_value['Decay_' + eqname][1] = tau
                fit_func = self.full_filter(t, dir)
                popt, pcov = curve_fit(fit_func, t, self.posdata['d' + dir], p0=pinit, bounds=pbound, maxfev=5000, sigma=self.posdata['S' + dir.lower()])
                perr = np.sqrt(np.diag(pcov))
                sim = fit_func(t, *popt)
                # res是否考虑误差项，需要考虑
                res = np.sqrt(np.sum(((sim - self.posdata['d' + dir])/(self.posdata['S' + dir.lower()]))**2))
                
                if res < minres:
                    minperr = perr
                    mintau = tau
                    minpopt = popt
                    minres = res
            print(mintau)
            coef_value['Decay_' + eqname][1] = mintau
        else:
            minfit_func = self.full_filter(t, dir)
            minpopt, minpcov = curve_fit(minfit_func, t, self.posdata['d' + dir], p0=pinit, bounds=pbound, maxfev=5000, sigma=self.posdata['S' + dir.lower()])
            minperr = np.sqrt(np.diag(minpcov))
        setattr(self, 'fit' + dir + '_func', minfit_func)
        setattr(self, 'fit' + dir + '_popt', minpopt)
        setattr(self, 'fit' + dir + '_pcov', minperr)

    def predict(self, time=None, dir='E', fitflag='11111', eqname={'WC': [0, 1]}):
        '''
        timearray：时间数组，用于预测/模拟观测量
        如果同震或震后为0，则看后面中相应选项，其它全为0；否则为1，则全1
        fitflag和eqname两项用于控制提取拟合信息
        '''
        coef_value = deepcopy(getattr(self, 'coef' + dir + '_value'))
        coef_value_reverse = deepcopy(coef_value)
        coef_fullfilter = deepcopy(coef_value)
        coef_name = getattr(self, 'coef' + dir + '_name')
        fit_popt = getattr(self, 'fit' + dir + '_popt')
        lastflag = None
        for idx, flag in enumerate(coef_name):
            if 'CONSTANT' in flag:
                coef_value['Linear'][1] = fit_popt[idx] * int(fitflag[0])
                coef_fullfilter['Linear'][1] = fit_popt[idx]
                coef_value_reverse['Linear'][1] = fit_popt[idx] * (1 - int(fitflag[0]))
            if 'VELOCITY' in flag:
                coef_value['Linear'][2] = fit_popt[idx] * int(fitflag[0])
                coef_fullfilter['Linear'][2] = fit_popt[idx]
                coef_value_reverse['Linear'][2] = fit_popt[idx] * (1 - int(fitflag[0]))
            elif lastflag == 'CONSTANT':
                coef_value['Linear'][2] *= int(fitflag[0])
                coef_value_reverse['Linear'][2] *= (1 - int(fitflag[0]))
            if 'ASIN' == flag:
                coef_value['PeriodCycle'] = [arg * int(fitflag[1]) for arg in fit_popt[idx: idx+4]]
                coef_fullfilter['PeriodCycle'] = [arg for arg in fit_popt[idx: idx+4]]
                coef_value_reverse['PeriodCycle'] = [arg * (1 - int(fitflag[1])) for arg in fit_popt[idx: idx+4]]
            if 'Break' in flag:
                coef_value[flag][1] = fit_popt[idx] * int(fitflag[2])
                coef_fullfilter[flag][1] = fit_popt[idx]
                coef_value_reverse[flag][1] = fit_popt[idx] * (1 - int(fitflag[2]))
            if 'CoBreak' in flag:
                if fitflag[3] == '1':
                    coef_value[flag][1] = fit_popt[idx]
                    coef_fullfilter[flag][1] = fit_popt[idx]
                    coef_value_reverse[flag][1] = 0.0
                else:
                    if flag[-2:] in eqname:
                        coef_value[flag][1] = fit_popt[idx] * eqname[flag[-2:]][0]
                        coef_fullfilter[flag][1] = fit_popt[idx]
                        coef_value_reverse[flag][1] = fit_popt[idx] * (1 - eqname[flag[-2:]][0])
                    else:
                        coef_value[flag][1] = fit_popt[idx] * 0.0
                        coef_fullfilter[flag][1] = fit_popt[idx]
                        coef_value_reverse[flag][1] = fit_popt[idx] * 1.0
            if 'Decay' in flag:
                if fitflag[4] == '1':
                    coef_value[flag][-1] = fit_popt[idx]
                    coef_fullfilter[flag][-1] = fit_popt[idx]
                    coef_value_reverse[flag][-1] = 0.0
                else:
                    if flag[-2:] in eqname:
                        coef_value[flag][-1] = fit_popt[idx] * eqname[flag[-2:]][1]
                        coef_fullfilter[flag][-1] = fit_popt[idx]
                        coef_value_reverse[flag][-1] = fit_popt[idx] * (1 - eqname[flag[-2:]][1])
                    else:
                        coef_value[flag][-1] = fit_popt[idx] * 0.0
                        coef_fullfilter[flag][-1] = fit_popt[idx]
                        coef_value_reverse[flag][-1] = fit_popt[idx] * 1.0
            lastflag = flag
        y = self._predict(time, coef_value)
        obs = self.posdata['d' + dir] - self._predict(self.posdata['Date'], coef_value_reverse)
        # 设置一些self参数
        # _predict：为根据fitflag设置，使相应项为0的按需拟合参数
        setattr(self, 'coef' + dir + '_predict', coef_value)
        # _fullfilter：为拟合原始数据的所有参数值，包含时间项的完整参数
        setattr(self, 'coef' + dir + '_fullfilter', coef_fullfilter)
        setattr(self, 'coef' + dir + '_predict_reverse', coef_value_reverse)
        return y, obs

    def _predict(self, t, coefs):
        y = 0
        t = np.array([datetime2decimalyear(date) for date in t])
        for flag in coefs:
            if 'Linear' in flag:
                y += self.LinearFunc(t, *coefs[flag])
            if 'PeriodCycle' in flag:
                y += self.CycleFunc(t, *coefs[flag])
            if 'Break' in flag or 'CoBreak' in flag:
                y += self.BreakFunc(t, *coefs[flag])
            if 'Decay' in flag or 'DECAY' in flag:
                y += self.DecayFunc(t, *coefs[flag])
        return y


def readposlist(listfile='poslist'):
    '''
    读取pos文件的文件名列表文件
    '''
    poslist = np.genfromtxt(listfile, dtype=str, comments='#')
    return poslist


from .detect_outliers import detect_outliers, detect_outliers_with_isolation_forest
try:
    from pandas.plotting import (register_matplotlib_converters as register)
except ImportError:
    from pandas.tseries.converter import register
register()

def plotdisplacement(sim, obs, coef_predict, sitename):
        '''
        sim: simt simE simN simU
        obs: obst obsE obsN obsU
        coef_predict
        sitename: 站点名称
        '''
        # 11/25/2019, modified by kfh
        obs = obs.copy()
        # inds = detect_outliers(obs, 0, ['obsE', 'obsN', 'obsU', 'Se', 'Sn', 'Su'])
        inds = detect_outliers_with_isolation_forest(obs, ['obsE', 'obsN', 'obsU', 'Se', 'Sn', 'Su'])
        obs.drop(index=inds, inplace=True)
        
        
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
        
        axes[0].plot(sim['simt'], sim['simE'], color='r', label='fitted curve')
        #axes[0].scatter(obs['obst'], obs['obsE'], s=1, label='data')
        axes[0].errorbar(obs['obst'], obs['obsE'], yerr=2*obs['Se'], ms=2, fmt="o",color="blue",ecolor='grey',elinewidth=2,capsize=4)
        # 画出有影响地震时刻的位置
        for flag in coef_predict:
            if 'CoBreak' in flag:
                teq = decimalyear2datetime(coef_predict[flag][0])
                axes[0].axvline(teq, 0, 1, color = "r", linestyle = "solid")
                axes[1].axvline(teq, 0, 1, color = "r", linestyle = "solid")
                axes[2].axvline(teq, 0, 1, color = "r", linestyle = "solid")
            if 'Break' in flag and not flag.startswith('Co'):
                tbrk = decimalyear2datetime(coef_predict[flag][0])
                axes[0].axvline(tbrk, 0, 1, color = "b", linestyle = "solid")
                axes[1].axvline(tbrk, 0, 1, color = "b", linestyle = "solid")
                axes[2].axvline(tbrk, 0, 1, color = "b", linestyle = "solid")
        axes[1].plot(sim['simt'], sim['simN'], color='r', label='fitted curve')
        # axes[1].scatter(obs['obst'], obs['obsN'], s=1, label='data')
        axes[1].errorbar(obs['obst'], obs['obsN'], yerr=2*obs['Sn'], ms=2, fmt="o",color="blue",ecolor='grey',elinewidth=2,capsize=4)
        axes[2].plot(sim['simt'], sim['simU'], color='r', label='fitted curve')
        # axes[2].scatter(obs['obst'], obs['obsU'], s=1, label='data')
        axes[2].errorbar(obs['obst'], obs['obsU'], yerr=2*obs['Su'], ms=2, fmt="o",color="blue",ecolor='grey',elinewidth=2,capsize=4)
        # 画出地震时刻的位置
        for ax in axes:
            ax.set_xlabel('T/yr')
        axes[0].set_ylabel('E/mm')
        axes[1].set_ylabel('N/mm')
        axes[2].set_ylabel('U/mm')
        axes[0].legend(framealpha=1, shadow=True, loc='best')
        axes[1].legend(framealpha=1, shadow=True, loc='best')
        axes[2].legend(framealpha=1, shadow=True, loc='best')

        fig.suptitle(sitename, fontsize='large', y=0.98)
        return fig

def plot_gpstimeseries(sim, obs, coef_predict, sitename, 
                       direction='EN', figsize=(7.0, 1.8), 
                       style=['science', 'nature'], fontsize=8, 
                       legend_frame=True, xlim=None, errorevery=1, 
                       x_as_date=True, ref_date=None, unit='D'):
    '''
    return : fig obj 单栏：3.3，最大3.5, 双栏：7.0

    style='seaborn-ticks' ['science', 'nature']
    
    print(plt.style.available)打印可用的style
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # 设置默认的绘图属性
    import scienceplots as sp
    plt.style.use(style)
    plt.rcParams['text.usetex'] = False
    if legend_frame:
        plt.rc('legend', frameon=True, framealpha=0.7,
            fancybox=True, numpoints=1)
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans', 'Arial', 'Lucida Grande', 'Verdana', 
                                        'Geneva, Lucid', 'Avant Garde', 'sans-serif']
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['font.size'] = fontsize

    obs = obs.copy()
    inds = detect_outliers_with_isolation_forest(obs, ['obsE', 'obsN', 'obsU', 'Se', 'Sn', 'Su'])
    obs.drop(index=inds, inplace=True)

    if not x_as_date:
        if ref_date is None:
            raise ValueError("If x_as_date is False, ref_date must be provided.")
        ref_date = pd.to_datetime(ref_date)
        obs.loc[:, 'obst'] = (obs['obst'] - ref_date).dt.total_seconds() / (60*60*24)  # convert to days
        sim.loc[:, 'simt'] = (sim['simt'] - ref_date).dt.total_seconds() / (60*60*24)  # convert to days
        if unit == 'Y':
            obs.loc[:, 'obst'] /= 365.2425  # convert to years
            sim.loc[:, 'simt'] /= 365.2425  # convert to years

    fig, axes = plt.subplots(nrows=1, ncols=len(direction), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
        direction = [direction]
    direction_map = {'E': 'East', 'N': 'North', 'U': 'Up'}
    for ax, dir in zip(axes, direction):
        ax.errorbar(obs['obst'], obs['obs'+dir], yerr=2*obs['S'+dir.lower()], ms=3, 
                    fmt="o", ecolor='#d1d1d1', errorevery=errorevery, 
                    label='{} Obs.'.format(sitename), zorder=0) # ,elinewidth=2,capsize=4
        ax.plot(sim['simt'], sim['sim'+dir], label='Synth')
        ax.set_xlabel('Time (yr)' if unit == 'Y' else 'Time (D)')
        ax.set_ylabel(f'{direction_map[dir]} Disp. (mm)')

        for flag in coef_predict:
            if 'CoBreak' in flag:
                teq = decimalyear2datetime(coef_predict[flag][0])
                if not x_as_date:
                    teq = (teq - ref_date).total_seconds() / (60*60*24)  # convert to days
                    if unit == 'Y':
                        teq /= 365.25  # convert to years
                ax.axvline(teq, 0, 1, color = "r", linestyle = "solid")
            if 'Break' in flag and not flag.startswith('Co'):
                tbrk = decimalyear2datetime(coef_predict[flag][0])
                if not x_as_date:
                    tbrk = (tbrk - ref_date).total_seconds() / (60*60*24)  # convert to days
                    if unit == 'Y':
                        tbrk /= 365.25  # convert to years
                ax.axvline(tbrk, 0, 1, color = "b", linestyle = "solid")
            
        if xlim is not None:
            ax.set_xlim(xlim)
    axes[0].legend() # framealpha=legend_frame, loc='best'

    plt.tight_layout()
    if x_as_date:
        fig.autofmt_xdate()  # 自动调整日期标签，防止重叠
    return fig


def main(posdata, savedir, taurange=None, fitflag='00011', eqname={'MD': [0, 1]}, siteinfo=None, velinfo=None):
    # 创建 ConfigDict 对象并初始化
    fitconfigparams = ConfigDict()
    # 读取配置文件
    fitconfigparams.readcmd('cmdprecamp.yaml')
    if isinstance(siteinfo, dict):
        siteinfo.update({posdata.name: posdata.BLH[:2]})
    # 生成特定站点的拟合信息文件
    site = GPSTimeSeriesFitInfo(posdata, fitconfigparams)
    # site.brks = None
    # 改变单站信息
    # site.Vn = None
    # site.Ve = None
    print(site.Ve, site.Vn)
    # 配置拟合模型
    fitmodel = FitModel(site, PostFunc=Logs) # ptx.Exps for image ; ptx.Logs for image_xiong
    # 观测时间序列
    obst = fitmodel.posdata['Date']
    # time: 生成待预测时间序列
    simt = np.arange(2021.3, 2022.9, 0.01) #(2021.3, 2021.9, 0.01)
    simt = obst
    # 转成Timestamp的类型进行拟合
    # simt = np.array([ptc.decimalyear2datetime(t) for t in simt])
    # 拟合E方向，生成拟合参数
    if taurange is not None:
        fitmodel.fit(dir='E', taurange=taurange, eqname='MD') # 'RC'
        fitmodel.fit(dir='N', taurange=taurange, eqname='MD') # None eqname设为None时，便不搜索tau值，直接用默认的tau值
        fitmodel.fit(dir='U', taurange=taurange, eqname='MD')
    else:
        fitmodel.fit(dir='E')
        fitmodel.fit(dir='N')
        fitmodel.fit(dir='U')
    # 根据时间序列，预测形变，需要时间是Timestamp的序列
    simE, obsE = fitmodel.predict(simt, fitflag=fitflag, eqname=eqname, dir='E')
    simN, obsN = fitmodel.predict(simt, fitflag=fitflag, eqname=eqname, dir='N')
    simU, obsU = fitmodel.predict(simt, fitflag=fitflag, eqname=eqname, dir='U')

    # sim obs coef_predit
    sim = pd.DataFrame({'simt': simt, 'simE': simE, 'simN': simN, 'simU': simU})
    obs = pd.DataFrame({'obst': obst, 'obsE': obsE, 'obsN': obsN, 'obsU': obsU})
    obs = pd.concat([obs, posdata.data[['Se', 'Sn', 'Su']]], axis=1)
    # print(obs.columns)
    coef_predict = fitmodel.coefE_predict
    sim.to_csv(savedir + posdata.name + '_sim.csv', sep=' ', 
               index=False, float_format='%.6f', date_format='%Y-%m-%dT%H:%M:%S')
    obs.loc[obs.obst > pd.Timestamp('2021-05-21')].to_csv(savedir + posdata.name + '_obs.csv', sep=' ', 
                                                          index=False, float_format='%.6f', date_format='%Y-%m-%dT%H:%M:%S')
    if isinstance(velinfo, dict):
        velE = fitmodel.coefE_fullfilter['Linear'][2]
        velN = fitmodel.coefN_fullfilter['Linear'][2]
        velU = fitmodel.coefU_fullfilter['Linear'][2]
        velinfo.update({site.name: np.array([velE, velN, velU])})
    print(site.name)
    # 完全拟合数据的所有参数值
    print(fitmodel.coefE_fullfilter)
    # 根据fitflag和eqname拟合参数的参数，部分参数被设置为0
    print(fitmodel.coefU_name, fitmodel.coefU_predict, sep='\n')
    # return sim, obs
    # 画图，显示或者存储
    # 11/26/2019 modified by kfh
    coef = {}
    for flag in coef_predict:
        if 'Break' in flag:
            teq = decimalyear2datetime(coef_predict[flag][0])
            if teq < pd.Timestamp('2021-05-21'):
                # _ = coef_predict.pop(flag)
                pass
            else:
                coef[flag] = coef_predict[flag]
        else:
            coef[flag] = coef_predict[flag]
    fig = plotdisplacement(sim.query('simt > "2021-05-21"'), obs.query('obst > "2021-05-21"'), coef, site.name) # coef_predict
    # fig.show()
    fig.savefig(savedir + posdata.name + '.jpg') # ,format='pdf'
    # plt.close(fig)
    # fig.show()


if __name__ == '__main__':
    '''
    运行多文件处理
    '''
    savedir = r'g:\Ridgecrest GPS\gps_timeseries\posfile\img\\'
    poslistfile = r'g:\Ridgecrest GPS\gps_timeseries\pytimeseries\posfilelist'
    
    poslist = readposlist(poslistfile)
    siteinfo = dict()
    velinfo = dict()
    for posfile in poslist[:4]:
        main(posfile, savedir, fitflag='00000', eqname={'RC': [0, 1]}, siteinfo=siteinfo, velinfo=velinfo)
    siteinfo = pd.DataFrame(siteinfo, index=('Lat', 'Lon')).T
    velinfo = pd.DataFrame(velinfo, index=('velE', 'velN', 'velU')).T
    # 用于储存站点相关的坐标和名称
    siteinfo.to_csv(savedir + 'siteinfo' + '.csv', sep=',', index=True)
    velinfo.to_csv(savedir + 'velinfo' + '.csv', sep=',', index=True)
    tmp = input()
