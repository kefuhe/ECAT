# Standard library imports
import sys
from typing import List, Union

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.interpolate import interp1d
from sklearn.linear_model import (
    Lasso, Ridge, ElasticNet, QuantileRegressor, LinearRegression,
    TheilSenRegressor, RANSACRegressor, HuberRegressor, LassoCV
)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from ..plottools import DegreeFormatter


class InterpolationModel:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        if len(x.shape) == 2:
            x = x[:, 0]
        return self.model(x)

class PolynomialModel:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        if len(x.shape) == 2:
            x = x[:, 0]
        return self.model(x)

class SVRModel:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        return self.model.predict(x)

class DecisionTreeModel:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        return self.model.predict(x)

class RegressionFitter:
    def __init__(self, x_values, y_values, degree=3):
        self.x_values = x_values
        self.y_values = y_values
        self.degree = degree
        self.fit_methods = {
            'ols': self.fit_ols,
            'theil_sen': self.fit_theil_sen,
            'ransac': self.fit_ransac,
            'huber': self.fit_huber,
            'lasso': self.fit_lasso,
            'ridge': self.fit_ridge,
            'elasticnet': self.fit_elasticnet,
            'quantile': self.fit_quantile,
            'groupby_interpolation': self.fit_groupby_interpolation,
            'polynomial': self.fit_polynomial,
            'svr': self.fit_svr,
            'decision_tree': self.fit_decision_tree,
        }
        self.color_dict = {
            'ols': '#1f77b4', 
            'theil_sen': '#ff7f0e', 
            'ransac': '#2ca02c', 
            'huber': '#d62728', 
            'lasso': '#9467bd', 
            'ridge': '#8c564b',
            'elasticnet': '#e377c2',
            'quantile': '#7f7f7f',
            'groupby_interpolation': '#bcbd22', 
            'polynomial': '#17becf'
        }
        self.line_style_dict = {
            'ols': '-', 
            'theil_sen': '--', 
            'ransac': '-.', 
            'huber': ':', 
            'lasso': '-', 
            'ridge': '--',
            'elasticnet': '-.',
            'quantile': ':',
            'groupby_interpolation': '--', 
            'polynomial': '-.'
        }

    def fit_model(self, method, **kwargs):
        model = self.fit_methods.get(method)
        if model is None:
            raise ValueError(f"Unknown method '{method}'.")
        model = model(**kwargs)
        y_pred = model.predict(self.x_values[:, np.newaxis])
        valid_indices = ~np.isnan(y_pred)
        mse = mean_squared_error(self.y_values[valid_indices], y_pred[valid_indices])
        return model, mse

    def fit_ols(self):
        model = make_pipeline(PolynomialFeatures(self.degree), LinearRegression())
        model.fit(self.x_values[:, np.newaxis], self.y_values)
        return model

    def fit_theil_sen(self):
        model = make_pipeline(PolynomialFeatures(self.degree), TheilSenRegressor(max_iter=1000, random_state=42))
        model.fit(self.x_values[:, np.newaxis], self.y_values)
        return model

    def fit_ransac(self):
        model = make_pipeline(PolynomialFeatures(self.degree), RANSACRegressor(random_state=42))
        model.fit(self.x_values[:, np.newaxis], self.y_values)
        return model

    def fit_lasso(self):
        lasso_pipeline = make_pipeline(StandardScaler(), LassoCV(cv=10, verbose=0, eps=0.001, n_alphas=100, tol=0.0001, max_iter=50000))
        model = make_pipeline(PolynomialFeatures(self.degree), lasso_pipeline)
        model.fit(self.x_values[:, np.newaxis], self.y_values)
        return model

    def fit_huber(self, epsilon=1.35):
        model = make_pipeline(PolynomialFeatures(self.degree), HuberRegressor(epsilon=epsilon))
        model.fit(self.x_values[:, np.newaxis], self.y_values)
        return model

    def fit_ridge(self, alpha=1.0):
        model = make_pipeline(PolynomialFeatures(self.degree), StandardScaler(), Ridge(alpha=alpha))
        model.fit(self.x_values[:, np.newaxis], self.y_values)
        return model

    def fit_elasticnet(self, alpha=1.0, l1_ratio=0.5):
        model = make_pipeline(PolynomialFeatures(self.degree), StandardScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio))
        model.fit(self.x_values[:, np.newaxis], self.y_values)
        return model

    def fit_quantile(self, quantile=0.5):
        model = make_pipeline(PolynomialFeatures(self.degree), StandardScaler(), QuantileRegressor(quantile=quantile, solver='highs'))
        model.fit(self.x_values[:, np.newaxis], self.y_values)
        return model

    def fit_svr(self, C=1.0, epsilon=0.2):
        model = make_pipeline(PolynomialFeatures(self.degree), StandardScaler(), SVR(C=C, epsilon=epsilon))
        model.fit(self.x_values[:, np.newaxis], self.y_values)
        return SVRModel(model)

    def fit_decision_tree(self, max_depth=None):
        model = make_pipeline(PolynomialFeatures(self.degree), DecisionTreeRegressor(max_depth=max_depth))
        model.fit(self.x_values[:, np.newaxis], self.y_values)
        return DecisionTreeModel(model)

    def fit_groupby_interpolation(self):
        x_values, y_values = self.x_values, self.y_values
        data_frame = pd.DataFrame(np.vstack((x_values, y_values)).T, columns=['x', 'y'])
        x_bins = pd.cut(data_frame.x, np.arange(x_values.min(), x_values.max()+0.1, 4.0))
        mean_values = data_frame.groupby(x_bins, observed=True).mean()
        mean_values.dropna(axis=0, inplace=True)
        cubic_interp_model = interp1d(mean_values.x.values, mean_values.y.values, kind = "cubic", bounds_error=False)
        return InterpolationModel(cubic_interp_model)

    def fit_polynomial(self):
        # 拟合多项式模型
        poly_model = np.polyfit(self.x_values, self.y_values, self.degree)
        poly_model_fn = np.poly1d(poly_model)
        return PolynomialModel(poly_model_fn)

    def plot_fit(self, models, mses=None, x_fit=None, show=False, 
                save_fig=None, dpi=600, style=['science'], ax=None, swap_axes=False, 
                show_error_in_legend=True, use_lon_lat=False, fault=None,
                fontsize=None, figsize=None, scatter_props=None):
        """
        绘制拟合曲线。

        参数:
        models: dict。键是方法名，值是对应的模型。
        mses: dict。键是方法名，值是对应的mse值。默认为None。
        x_fit: numpy array。表示x坐标。
        show: bool。表示是否显示图像。默认为False。
        save_fig: str。表示保存图像的文件名。如果为None，则不保存图像。默认为None。
        dpi: int。表示保存图像的分辨率。默认为600。
        style: str。表示样式。默认为'science'。
        ax: matplotlib.axes.Axes。表示绘图的子图对象。如果为None，则创建新的图形。默认为None。
        swap_axes: bool。表示是否交换x轴和y轴。默认为False。
        show_error_in_legend: bool。表示是否在图例中显示误差。默认为True。
        use_lon_lat: bool。表示是否使用经纬度坐标进行绘制。默认为False。
        fault: Fault。表示fault对象。默认为None。

        返回值:
        无。这个函数会绘制图像并显示。
        """
        if use_lon_lat and fault is None:
            raise ValueError("If use_lon_lat is True, fault cannot be None.")

        if use_lon_lat:
            lon, lat = fault.xy2ll(self.x_values, self.y_values)
            x_values = lon
            y_values = lat
        else:
            x_values = self.x_values
            y_values = self.y_values

        # Set default properties for plotting
        with plt.style.context(style):
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.formatter.use_mathtext'] = False
            plt.rcParams['text.usetex'] = False
            plt.rcParams['mathtext.fontset'] = 'dejavusans'
            plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica','DejaVu Sans', 'Bitstream Vera Sans', 
                                            'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 
                                            'Lucid', 'Avant Garde', 'sans-serif']

            if fontsize is not None:
                plt.rcParams['axes.labelsize'] = fontsize
                plt.rcParams['xtick.labelsize'] = fontsize
                plt.rcParams['ytick.labelsize'] = fontsize
                plt.rcParams['legend.fontsize'] = fontsize
                plt.rcParams['font.size'] = fontsize
            
            if figsize is not None:
                plt.rcParams['figure.figsize'] = figsize

            line_width = 2

            if show:
                default_scatter_props = {'color': "black", 's': 30, 'alpha': 0.6}
                scatter_props = scatter_props if scatter_props is not None else {}
                scatter_props = {**default_scatter_props, **scatter_props}  # Update default properties with provided ones

                if swap_axes:
                    (ax if ax is not None else plt).scatter(y_values, x_values, **scatter_props)
                else:
                    (ax if ax is not None else plt).scatter(x_values, y_values, **scatter_props)
                # (ax if ax is not None else plt).title("Comparison of Different Fitting Methods", fontsize=16)
                if use_lon_lat:
                    (ax if ax is not None else plt).xlabel("Latitude" if swap_axes else "Longitude", fontsize=10)
                    (ax if ax is not None else plt).ylabel("Longitude" if swap_axes else "Latitude", fontsize=10)
                else:
                    (ax if ax is not None else plt).xlabel("Y" if swap_axes else "X", fontsize=10)
                    (ax if ax is not None else plt).ylabel("X" if swap_axes else "Y", fontsize=10)

            if x_fit is None:
                X_smooth = np.linspace(self.x_values.min(), self.x_values.max(), 100)
            else:
                X_smooth = x_fit

            for method, model in models.items():
                y_plot = model.predict(X_smooth[:, np.newaxis])
                if use_lon_lat:
                    lon, lat = fault.xy2ll(X_smooth, y_plot)
                    x_predict = lon
                    y_predict = lat
                else:
                    x_predict = X_smooth
                    y_predict = y_plot
                if ax is not None or show:
                    label = method if not show_error_in_legend or mses is None else '%s: error = %.3f' % (method, mses[method])
                    if swap_axes:
                        (ax if ax is not None else plt).plot(y_predict, x_predict, color=self.color_dict[method], linestyle=self.line_style_dict[method],
                                    linewidth=line_width, label=label)
                    else:
                        (ax if ax is not None else plt).plot(x_predict, y_predict, color=self.color_dict[method], linestyle=self.line_style_dict[method],
                                    linewidth=line_width, label=label)

            if show:
                if use_lon_lat:
                    formatter = DegreeFormatter()
                    if ax is None:
                        ax = plt.gca()
                    ax.xaxis.set_major_formatter(formatter)
                    ax.yaxis.set_major_formatter(formatter)
                (ax if ax is not None else plt).legend(fontsize=10)
                if save_fig is not None:
                    plt.savefig(save_fig, dpi=dpi)
                plt.show()


if __name__ ==  '__main__':
    pass