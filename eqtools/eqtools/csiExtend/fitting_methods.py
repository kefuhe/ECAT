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

# Define method information outside the class
METHOD_INFO = {
    'ols': {
        'short_name': 'OLS',
        'full_name': 'Ordinary Least Squares',
        'color': '#1f77b4',
        'line_style': '-',
        'fit_method': 'fit_ols'
    },
    'theil_sen': {
        'short_name': 'Theil-Sen',
        'full_name': 'Theil-Sen Estimator',
        'color': '#ff7f0e',
        'line_style': '--',
        'fit_method': 'fit_theil_sen'
    },
    'ransac': {
        'short_name': 'RANSAC',
        'full_name': 'Random Sample Consensus',
        'color': '#2ca02c',
        'line_style': '-.',
        'fit_method': 'fit_ransac'
    },
    'huber': {
        'short_name': 'Huber',
        'full_name': 'Huber Regression',
        'color': '#d62728',
        'line_style': ':',
        'fit_method': 'fit_huber'
    },
    'lasso': {
        'short_name': 'LASSO',
        'full_name': 'Least Absolute Shrinkage and Selection Operator',
        'color': '#9467bd',
        'line_style': '-',
        'fit_method': 'fit_lasso'
    },
    'ridge': {
        'short_name': 'Ridge',
        'full_name': 'Ridge Regression',
        'color': '#8c564b',
        'line_style': '--',
        'fit_method': 'fit_ridge'
    },
    'elasticnet': {
        'short_name': 'ElasticNet',
        'full_name': 'Elastic Net Regression',
        'color': '#e377c2',
        'line_style': '-.',
        'fit_method': 'fit_elasticnet'
    },
    'quantile': {
        'short_name': 'Quantile',
        'full_name': 'Quantile Regression',
        'color': '#7f7f7f',
        'line_style': ':',
        'fit_method': 'fit_quantile'
    },
    'groupby_interpolation': {
        'short_name': 'GroupBy Interpolation',
        'full_name': 'GroupBy Interpolation',
        'color': '#bcbd22',
        'line_style': '--',
        'fit_method': 'fit_groupby_interpolation'
    },
    'polynomial': {
        'short_name': 'Polynomial',
        'full_name': 'Polynomial Regression',
        'color': '#17becf',
        'line_style': '-.',
        'fit_method': 'fit_polynomial'
    },
    'svr': {
        'short_name': 'SVR',
        'full_name': 'Support Vector Regression',
        'color': '#1f77b4',
        'line_style': '-',
        'fit_method': 'fit_svr'
    },
    'decision_tree': {
        'short_name': 'Decision Tree',
        'full_name': 'Decision Tree Regression',
        'color': '#ff7f0e',
        'line_style': '--',
        'fit_method': 'fit_decision_tree'
    }
}

class RegressionFitter:
    def __init__(self, x_values, y_values, degree=3):
        self.x_values = x_values
        self.y_values = y_values
        self.degree = degree
        self.fit_methods = {key: getattr(self, value['fit_method']) for key, value in METHOD_INFO.items()}
        self.color_dict = {method: METHOD_INFO[method]['color'] for method in METHOD_INFO}
        self.line_style_dict = {method: METHOD_INFO[method]['line_style'] for method in METHOD_INFO}
        self.short_name_dict = {method: METHOD_INFO[method]['short_name'] for method in METHOD_INFO}

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
        # Fit polynomial model
        poly_model = np.polyfit(self.x_values, self.y_values, self.degree)
        poly_model_fn = np.poly1d(poly_model)
        return PolynomialModel(poly_model_fn)

    def plot_fit(self, models, mses=None, x_fit=None, show=False, 
                 save_fig=None, dpi=600, style=['science'], ax=None, swap_axes=False, 
                 show_error_in_legend=True, use_lon_lat=False, fault=None,
                 fontsize=None, figsize=None, scatter_props=None, custom_data=None):
        """
        Plot fitted curves.
    
        Parameters:
        models: dict. Keys are method names, values are corresponding models.
        mses: dict. Keys are method names, values are corresponding mse values. Default is None.
        x_fit: numpy array. Represents x coordinates.
        show: bool. Indicates whether to show the plot. Default is False.
        save_fig: str. Indicates the filename to save the plot. If None, the plot is not saved. Default is None.
        dpi: int. Indicates the resolution of the saved plot. Default is 600.
        style: str. Indicates the style. Default is 'science'.
        ax: matplotlib.axes.Axes. Indicates the subplot object for plotting. If None, a new figure is created. Default is None.
        swap_axes: bool. Indicates whether to swap x and y axes. Default is False.
        show_error_in_legend: bool. Indicates whether to show error in the legend. Default is True.
        use_lon_lat: bool. Indicates whether to use longitude and latitude coordinates for plotting. Default is False.
        fault: Fault. Indicates the fault object. Default is None.
        fontsize: int. Font size for the plot. Default is None.
        figsize: tuple. Figure size for the plot. Default is None.
        scatter_props: dict. Properties for scatter plot. Default is None.
        custom_data: dict. Custom data for plotting. Keys are labels, values are (x, y) tuples. Default is None.
    
        Returns:
        None. This function will plot the figure and display it.
        """
        from ..plottools import sci_plot_style

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
        with sci_plot_style(style, fontsize=fontsize, figsize=figsize):
    
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
                    label = self.short_name_dict[method] if not show_error_in_legend or mses is None else '%s: error = %.3f' % (self.short_name_dict[method], mses[method])
                    if swap_axes:
                        (ax if ax is not None else plt).plot(y_predict, x_predict, color=self.color_dict[method], linestyle=self.line_style_dict[method],
                                    linewidth=line_width, label=label)
                    else:
                        (ax if ax is not None else plt).plot(x_predict, y_predict, color=self.color_dict[method], linestyle=self.line_style_dict[method],
                                    linewidth=line_width, label=label)
    
            # Plot custom data if provided
            if custom_data:
                for label, (x_custom, y_custom) in custom_data.items():
                    if swap_axes:
                        (ax if ax is not None else plt).plot(y_custom, x_custom, color='red', linestyle='-', linewidth=line_width, label=label)
                    else:
                        (ax if ax is not None else plt).plot(x_custom, y_custom, color='red', linestyle='-', linewidth=line_width, label=label)
    
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