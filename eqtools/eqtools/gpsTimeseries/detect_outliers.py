import numpy as np
from collections import Counter

# Outlier detection
def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # quartile spacing (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than n outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


from sklearn.ensemble import IsolationForest

def detect_outliers_with_isolation_forest(df, features):
    # fit the model
    clf = IsolationForest(contamination=0.01)
    clf.fit(df[features])
    
    # predictions
    y_pred_outliers = clf.predict(df[features])
    
    # Outliers are marked with -1, inliers are marked with 1.
    # We create a mask for outliers, then use it to get the indices of the outliers.
    outliers_mask = y_pred_outliers == -1
    outliers_indices = df[outliers_mask].index.tolist()
    
    return outliers_indices

from statsmodels.tsa.arima.model import ARIMA

def detect_outliers_with_arima(series, order=(1, 1, 1), threshold=3):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    residuals = pd.DataFrame(model_fit.resid)
    outliers_mask = np.abs(residuals) > threshold * residuals.std()
    return np.where(outliers_mask)

def detect_outliers_with_rolling_statistics(series, window_size, threshold=3):
    rolling_mean = series.rolling(window=window_size).mean()
    rolling_std = series.rolling(window=window_size).std()
    outliers_mask = np.abs(series - rolling_mean) > threshold * rolling_std
    return np.where(outliers_mask)

from scipy import stats

def detect_outliers_with_zscore(series, threshold=3):
    z_scores = stats.zscore(series)
    return np.where(np.abs(z_scores) > threshold)

def detect_outliers_with_iqr(series, threshold=1.5):
    q25, q75 = np.percentile(series, 25), np.percentile(series, 75)
    iqr = q75 - q25
    cut_off = iqr * threshold
    lower, upper = q25 - cut_off, q75 + cut_off
    return np.where((series > upper) | (series < lower))

def detect_outliers_with_mad(series, threshold=3.5):
    median = np.median(series)
    mad = np.median([np.abs(y - median) for y in series])
    modified_z_scores = 0.6745 * (series - median) / mad
    return np.where(np.abs(modified_z_scores) > threshold)

def detect_outliers_with_moving_average(series, window_size, threshold=3):
    moving_avg = series.rolling(window=window_size).mean()
    residuals = series - moving_avg
    z_scores = stats.zscore(residuals)
    return np.where(np.abs(z_scores) > threshold)

def detect_outliers_with_hampel_filter(series, window_size, threshold=3):
    k = 1.4826 # scale factor for Gaussian distribution
    new_series = series.copy()
    n = len(series)
    indices = []
    for i in range(window_size, n - window_size):
        x0 = np.median(series[i - window_size:i + window_size])
        S0 = k * np.median(np.abs(series[i - window_size:i + window_size] - x0))
        if np.abs(series[i] - x0) > threshold * S0:
            indices.append(i)
            new_series[i] = x0
    return indices

from collections import Counter
from scipy import stats
from sklearn.ensemble import IsolationForest

def detect_outliers(df, features, z_threshold=3, contamination=0.01):
    # Z-Score method
    z_scores = stats.zscore(df[features])
    z_outliers = np.where(np.abs(z_scores) > z_threshold)

    # Isolation Forest method
    clf = IsolationForest(contamination=contamination)
    clf.fit(df[features])
    y_pred_outliers = clf.predict(df[features])
    if_outliers = df[y_pred_outliers == -1].index.tolist()

    # Combine the results
    combined_outliers = np.concatenate((z_outliers, if_outliers), axis=None)
    outlier_counts = Counter(combined_outliers)

    # Consider a data point to be an outlier if it is identified by both methods
    final_outliers = [index for index, count in outlier_counts.items() if count > 1]

    return final_outliers

import pandas as pd


# 特征清洗：异常值清理用用箱图；
# 分为两步走，一步是单列异常值处理，
# 第二步是多列分组异常值处理 11/25/2019
def remove_filers_with_boxplot(data, columns=None):
    if columns is not None:
        p = data.boxplot(column=columns, return_type='dict')
    else:
        p = data.boxplot(return_type='dict')
        columns = data.columns
    for index,value in enumerate(columns):
        # 获取异常值
        # print(index, value)
        fliers_value_list = p['fliers'][index].get_ydata()
        # 删除异常值
        for flier in fliers_value_list:
            data = data[data.loc[:,value] != flier]
    return data


# filename = r'..\posfile\P597.cwu.final_igs14.pos'
# data = pd.read_csv(filename, sep='\s+', skiprows=36, escapechar='*', parse_dates={'Date':['YYYYMMDD', 'HHMMSS']})

# mark = data.query('Date > "2019-07-07"').copy()