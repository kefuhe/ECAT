'''
Written by kfhe, at 12/19/2022
'''

import numpy as np

def get_discretized_points(num_points, start, end, sample_ratio):
    '''
    Object : 获取水平面非均匀采样情况下，点位的分散距离
    Input  :
        num_points  : the discreted number in the given horizontal range
        start       : the start point of the horizontal range
        end         : the end point of the horizontal range
        sample_ratio: the ratio of the maximum interval and the minmum interval
    Output :
        sample_points: the discreted points
    '''

    # Check if the number of points is greater than 1
    if num_points <= 1:
        raise ValueError("The number of points must be greater than 1.")

    # Check if the sample ratio is greater than or equal to 1
    if sample_ratio < 1:
        raise ValueError("The sample ratio must be greater than or equal to 1.")

    # Calculate the base interval
    base_interval = 2 * (end - start) / (num_points - 1) / (1 + sample_ratio)

    # Generate the actual intervals
    intervals = base_interval * (1 + (sample_ratio - 1) * np.arange(num_points - 1) / (num_points - 2))

    # Generate the sample points
    sample_points = np.concatenate(([start], start + np.cumsum(intervals)))

    return sample_points


def pnt_dist(nr, r1, r2, samplratio):
    '''
    Object : 获取水平面非均匀采样情况下，点位的分散距离
    Input  :
        nr         : the discreted number in the given horizontal range
        r1         : the start point of the horizontal range
        r2         : the end point of the horizontal range
        samplratio : the ratio of the maximum interval and the minmum interval
    Output :
        samppnts    : the discreted points
    '''
    dr = 2*(r2-r1)/(nr-1.0)/(1.0 + samplratio)
    samppnts = []
    samppnts.append(r1)
    for i in range(2, nr+1):
        dract = dr*(1.0 + (samplratio-1.0)*(i-2.0)/(nr-2.0))
        samppnts.append(samppnts[-1] + dract)
    samppnts = np.array(samppnts)
    return samppnts


if __name__ == '__main__':
    samppnts = pnt_dist(201, 0, 1500, 12.0)
    print(samppnts)