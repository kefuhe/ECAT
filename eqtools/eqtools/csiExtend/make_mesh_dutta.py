import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.optimize import least_squares as lsq
import math

def compute_transform(p1, p2):
    p1_complex = p1[0] + 1.j*p1[1]
    p2_complex = p2[0] + 1.j*p2[1]
    scale = 2 / np.abs(p2_complex - p1_complex)
    angrot_r = np.angle(p2_complex - p1_complex)
    euler = scale * np.exp(-1j * angrot_r)
    trans = euler * p1_complex
    return euler, trans

def findpoint(euler, trans, somep):
    original_shape = somep.shape
    if len(original_shape) == 1:
        somep = somep.reshape((1, 2))
    somep_complex = somep[:, 0] + 1.j*somep[:, 1]
    p = euler * somep_complex - trans
    result = np.array([p.real, p.imag]).T
    return result.reshape(original_shape)

def reversefindpoint(euler, trans, somep):
    original_shape = somep.shape
    if len(original_shape) == 1:
        somep = somep.reshape((1, 2))
    somep_complex = somep[:, 0] + 1.j*somep[:, 1]
    p = (somep_complex + trans) / euler
    result = np.array([p.real, p.imag]).T
    return result.reshape(original_shape)

def split_coords_into_segments(top_coords, layers_coords, bottom_coords, num_segments, bias=1.0):
    """
    将折线分割成指定数量的段，并根据给定的偏差来调整每个段的长度。

    参数:
    top_coords (numpy.ndarray): 顶部的坐标，形状为 (n, d)，其中 n 是点的数量，d 是维度。
    layers_coords (list of numpy.ndarray): 中间层的坐标，每个元素的形状为 (n, d)。
    bottom_coords (numpy.ndarray): 底部的坐标，形状为 (n, d)。
    num_segments (int): 要将折线分割成的段的数量。
    bias (float): 用于调整每个段长度的偏差, default=1.0。

    返回:
    numpy.ndarray: 分割后的坐标，形状为 (num_segments, n, d)。
    """
    n, d = top_coords.shape
    result = np.zeros((num_segments, n, d))

    all_coords = np.stack([top_coords] + layers_coords + [bottom_coords])
    diffs = np.diff(all_coords, axis=0)
    lengths = np.linalg.norm(diffs, axis=2)
    total_length = np.sum(lengths, axis=0)
    arc_lengths = np.cumsum(np.vstack([np.zeros((1, n)), lengths]), axis=0)

    for i in range(n):
        f = interp1d(arc_lengths[:, i], all_coords[:, i, :], axis=0)
        # 对bias=1的情况进行处理
        if bias == 1:
            segment_arc_lengths = np.linspace(0, total_length[i], num_segments)
        else:
            # Sn = min_dx * (bias**n - 1)/(bias - 1)
            # 计算min_dz
            min_dz = total_length[i] / (bias**(num_segments-1) - 1) * (bias - 1)
            # 根据深度层的数量和位置，使用偏差公式来更新每一层的坐标
            segment_arc_lengths = min_dz * (bias**np.arange(num_segments) - 1) / (bias - 1)
            segment_arc_lengths[-1] = total_length[i]
        result[:, i, :] = f(segment_arc_lengths)

    return result

def discrectize_dipcurve(maxdep, a, b, surftrace, disct_z=None, bias=None, min_dx=None, dip_dir=None):
    """
    对给定的地表曲线进行离散化，生成一个三维的地质断层模型。

    参数:
    maxdep -- 最大深度
    a, b -- 二次曲线的参数, z = a*x^2 + b*x + c, c = xi1 - a*zi1^2 - b*zi1
    surftrace -- 地表曲线的坐标
    disct_z -- 沿着深度方向的离散化级数
    bias -- 离散化级数的偏差
    min_dx -- 最小的离散化步长
    dip_dir -- 断层的倾向方向

    返回值:
    xfault, yfault, zfault -- 三维的断层模型的坐标
    """

    # 提取地表曲线的坐标
    xdata = surftrace[:, 0]
    ydata = surftrace[:, 1]
    zsurf = surftrace[:, 2]

    num_x = xdata.shape[0]

    if dip_dir is None:
        # 计算 strike
        dy = np.diff(ydata)
        dx = np.diff(xdata)
        strike = np.arctan2(dy, dx)
        # 计算 strike 的平均值
        strike_avg = (strike[:-1] + strike[1:]) / 2
        # 使用 np.pad 代替 np.concatenate
        strike = np.pad(strike_avg, (1, 1), 'edge')
        dip_dir_avg = np.mean(strike) - np.pi/2
    else:
        dip_dir_avg = np.radians(dip_dir)

    # euler rotation
    xy_new = (xdata + 1j*ydata)*np.exp(-1j*dip_dir_avg)

    # 定义一些共享的变量和函数
    xi1 = 0.0
    zi1 = zsurf[0]
    zi2 = -maxdep

    # 当 a == 0 时，曲线是一条直线
    if a == 0:
        c = xi1 - b*zi1
        # 定义一次多项式
        # y = b*x + c
        pol = np.poly1d([b, c])
        # 求一阶导数 y' = b
        pol_diff = np.polyder(pol, 1)
        # 弧长函数：S = ∫√(1+(f'(z))^2)dz
        integrad = np.sqrt((pol_diff[0])**2+1)
        fulllen = integrad*zi2 - integrad*zi1
        length = lambda z: integrad*z
    # 当 a != 0 时，曲线是一个二次曲线
    else:
        c = xi1 - a*zi1**2 - b*zi1
        # 定义二次多项式
        pol = np.poly1d([a, b, c])
        # 求导
        pol_diff = np.polyder(pol, 1)
        # 弧长函数：S = ∫√(1+(f'(z))^2)dz
        integrad = lambda z: np.sqrt((pol_diff[0]*z+pol_diff[1])**2+1)
        fulllen = integrate.quad(integrad, zi1, zi2)[0]
        # 由于z小于0，所以积分的上限和下限是反的，所以length <= 0
        length = lambda z: integrate.quad(integrad, 0, z)[0]

    # 如果 disct_z 为 None，则使用 min_dx 和 bias 来计算 disct_z
    if disct_z is None:
        # 检查 min_dx 和 bias 是否都被提供
        if min_dx is None or bias is None:
            raise ValueError("If disct_z is None, both min_dx and bias must be provided.")
        
        # 检查 bias 和 min_dx 是否满足条件
        if np.abs(fulllen)*(bias-1) < -min_dx:
            raise ValueError("The bias or min_dx need to be reset.")
        
        # 计算 disct_z 和 seglen
        # Sn = min_dx * (bias**n - 1)/(bias - 1)
        disct_z = int(np.log(1+np.abs(fulllen)/min_dx*(bias-1))/np.log(bias))
        seglen = min_dx
    else:
        # 如果 disct_z 不为 None，那么设置 bias 为 1.0
        bias = 1.0
        seglen = abs(fulllen/disct_z)
    # 初始化存储 z 值的数组
    zinc = np.zeros(disct_z+1)
    zinc[0] = zi1

    # 计算每个分段的 z 值
    for segment in range(disct_z):
        # 计算目标长度
        target_length = length(zinc[segment]) - seglen
        # 定义目标函数
        target_func = lambda z: length(z) - target_length
        # 用最小二乘法求解
        result = lsq(target_func, -10, method='lm')['x']
        if isinstance(result, np.ndarray) and result.size == 1:
            result = result.item()
        zinc[segment+1] = result

        # 在每次迭代后增加 seglen
        seglen *= bias

    # 计算每个分段的 x 值
    xinc = np.polyval(np.array([a, b, c]), zinc)
    xy_new_msh = xinc[:, np.newaxis] + xy_new[np.newaxis, :]
    # euler rotation back to original coordinate
    xyfault = xy_new_msh * np.exp(1j*dip_dir_avg)
    xfault, yfault = xyfault.real, xyfault.imag
    zfault = np.tile(zinc, (num_x, 1)).T

    return xfault, yfault, zfault

def makemesh(maxdep,model,surftrace,disct_z=None, bias=None, min_dx=None):
    """
    生成离散的断层块，适用于表面痕迹为任意曲线的情况。

    参数:
    maxdep: 断层的最大深度（公里），正数
    model: [D1 D2 S1 S2]，模型参数
    surftrace: 地表迹线，N*3(x,y,z)的数组
    disct_z: 沿倾向的断层子元离散个数
    bias: 沿倾向方向，每个分段的长度增加的比例
    min_dx: 每个分段的最小（初始）长度

    返回值:
    p, q, r, trired, ang, xfault, yfault, zfault
    Comment:
    Change from Dutta et al., 2021
    """
    # 检查输入参数
    if maxdep <= 0:
        raise ValueError("maxdep must be a positive number.")
    if len(model) != 4:
        raise ValueError("model must be a list with 4 elements.")
    if surftrace.shape[1] != 3:
        raise ValueError("surftrace must be a N*3 array.")
    if disct_z is not None and disct_z <= 0:
        raise ValueError("disct_z must be a positive number.")

    a = model[0] / 100
    b = model[1] / 100
    amod1 = model[2]
    amod2 = model[3] / 10
    # Step 1: Discretize the dip curve, build the fault surface without bottom edge correction
    xfault, yfault, zfault = discrectize_dipcurve(maxdep, a, b, surftrace, disct_z, bias, min_dx)
    disct_x = xfault.shape[1] - 1
    disct_z = xfault.shape[0] - 1

    # Step 2: Correct the bottom edge of the fault surface
    # 用底部两端的端点来计算 p1 和 p2
    p1, p2 = np.array([xfault[-1, 0], yfault[-1, 0]]), np.array([xfault[-1,-1], yfault[-1,-1]])
    euler, trans = compute_transform(p1, p2)

    # 初始化存储 x、y 值的数组
    diffx, diffy, botfaultx, botfaulty = [np.zeros(disct_x) for _ in range(4)]

    # ---------------------这种是假设第二步是按照平均走向法线偏移---------------------#
    # 计算每个分段的新的 x、y 坐标
    bottom = np.array([xfault[-1, 1:-1], yfault[-1, 1:-1]]).T
    bottomref = findpoint(euler, trans, bottom)
    
    xfault_newref = bottomref[:, 0]
    yfault_newref = amod2 * (xfault_newref) * (xfault_newref - amod1) * (xfault_newref - 2)
    
    xyfault_newref = np.array([xfault_newref, yfault_newref]).T
    fault_new = reversefindpoint(euler, trans, xyfault_newref)
    
    botfaultx[1:] = fault_new[:, 0]
    botfaulty[1:] = fault_new[:, 1]
    diffx[1:], diffy[1:] = bottom[:, 0] - botfaultx[1:], bottom[:, 1] - botfaulty[1:]
    #---------------------------------------------------------------------------------#

    #---------------------这种是假设第二步是按照走向法线，由于不同偏移方向所以需要求解交点---------------------#
    # # 计算每个分段的新的 x、y 坐标
    # for i in range(1, disct_x):
    #     top = np.array([xfault[0, i], yfault[0, i]])
    #     bottom = np.array([xfault[-1, i], yfault[-1, i]])

    #     topref, bottomref = findpoint(euler, trans, top), findpoint(euler, trans, bottom)

        # # 计算地表点i到最深部对应点i的斜率和截距
        # if np.isclose(topref[0], bottomref[0]):
        #     Apol = float('inf')
        # else:
        #     Apol = (topref[1] - bottomref[1]) / (topref[0] - bottomref[0])
        # Bpol = topref[1] - Apol * topref[0]

        # # 计算新的 x、y 坐标
        # ## 断层倾角90度的情况?
        # if Apol == float('inf') or Bpol == float('inf') or Apol > 1e10 or Bpol > 1e10:
        #     xfault_newref = topref[0]
        #     yfault_newref = amod2 * (xfault_newref) * (xfault_newref - amod1) * (xfault_newref - 2)
        # else:
        #     # 计算多项式的根: amod2 * x * (x - amod1) * (x - 2) - Apol * x - Bpol = 0
        #     ## 即水平面上顶点-底点的直线与多项式的交点
        #     # 创建 polsolve 数组
        #     polsolve = np.array([amod2, -(2 + amod1) * amod2, amod1 * amod2 * 2 - Apol, -Bpol])
        #     solx = np.roots(polsolve)
        #     indreal = np.where((solx > 0) & (solx < 2) & (solx.imag == 0))[0].astype('int')
        #     xfault_newref = min(solx[indreal].real)
        #     yfault_newref = Apol * xfault_newref + Bpol

        # xyfault_newref = np.array([xfault_newref, yfault_newref])
        # fault_new = reversefindpoint(euler, trans, xyfault_newref)

        # botfaultx[i] = fault_new[0]
        # botfaulty[i] = fault_new[1]
        # diffx[i], diffy[i] = bottom[0] - botfaultx[i], bottom[1] - botfaulty[i]
    #------------------------------------------------------------------------------------------------------------#
    
    # 更新 xfault 和 yfault 的值
    for i in range(1, disct_x): # +1
        if i == disct_x:
            for j in range(1, disct_z + 1):
                factor = 0.8 if j == disct_z else 1
                xfault[j, i] -= factor * diffx[i - 1] * (j - 1) / (disct_z - 1)
                yfault[j, i] -= factor * diffy[i - 1] * (j - 1) / (disct_z - 1)
        else:
            xfault[-1, i], yfault[-1, i] = botfaultx[i], botfaulty[i]
            for j in range(1, disct_z):
                xfault[j, i] -= diffx[i] * j / disct_z
                yfault[j, i] -= diffy[i] * j / disct_z

    # 计算角度 ang
    ang1 = np.zeros(disct_z - 1)
    for i in range(1, disct_z):
        zdiff, xdiff = abs(np.mean(zfault[i + 1, :]) - np.mean(zfault[i, :])), abs(np.mean(xfault[i + 1, :]) - np.mean(xfault[i, :]))
        theta = np.arctan(zdiff / xdiff) * 180 / math.pi
        diffall = np.sqrt(diffx ** 2 + diffy ** 2)
        ang1[i - 1] = 0 if (theta > 40) or (max(diffall) > 50) else 1
    ang = 0 if np.any(ang1 == 0) else 1

    # 准备返回的数据
    p, q, r = (xfault.T).reshape(-1, 1), (yfault.T).reshape(-1, 1), (zfault.T).reshape(-1, 1)
    rowp, colp, numpatch = xfault.shape[0], yfault.shape[1], 2 * ((xfault.shape[0] - 1) * (xfault.shape[1] - 1))
    trired = np.zeros((numpatch, 3), dtype=int)
    # 初始化计数器
    triangle_index = 0
    # 遍历每一列和每一行
    for column in range(0, colp - 1):
        for row in range(0, rowp - 1):
            # 计算索引
            index = column * rowp + row
            # 填充 trired 数组
            trired[triangle_index, :] = np.array([index, index + 1, index + rowp + 1])
            trired[triangle_index + 1, :] = np.array([index, index + rowp + 1, index + rowp])
            # 更新计数器
            triangle_index += 2

    return p, q, r, trired, ang, xfault, yfault, zfault

def test_makemesh():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    maxdep = 30
    model = [6, 6, 0.5, 3]
    # 定义起点和终点
    start = np.array([0, 0, 0])
    end = np.array([50, 50, 0])

    # 生成线段
    surftrace = np.array([np.linspace(start[i], end[i], 15) for i in range(3)]).T
    disct_z = 20

    p, q, r, trired, ang, xfault, yfault, zfault = makemesh(maxdep, model, surftrace, min_dx=2.0, bias=1.1)

    # 创建一个 3D 图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三角形网格
    ax.plot_trisurf(p.ravel(), q.ravel(), r.ravel(), triangles=trired)

    plt.show()
    # print(zfault)

if __name__ == '__main__':
    test_makemesh()