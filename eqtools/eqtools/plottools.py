from os import listdir
from os.path import isdir, join
import matplotlib.pyplot as plt
import scienceplots
from matplotlib import cm
import matplotlib as mpl
from contextlib import contextmanager

# # 保存scienceplots的样式库
# scienceplots_styles = plt.style.library.copy()
# # 将scienceplots的样式库合并到当前的样式库中
# plt.style.library.update(scienceplots_styles)
# # 更新可用的样式列表
# plt.style.available[:] = sorted(plt.style.library.keys())

class DegreeFormatter(mpl.ticker.ScalarFormatter):
    def __call__(self, x, pos=None):
        # Call the parent class to get the original label
        label = super().__call__(x, pos)
        # Add the degree symbol
        return label + '°'


def register_science_styles():
    # register the included stylesheet in the matplotlib style library
    scienceplots_path = scienceplots.__path__[0]
    styles_path = join(scienceplots_path, 'styles')

    # Reads styles in /styles
    stylesheets = plt.style.core.read_style_directory(styles_path)
    # Reads styles in /styles subfolders
    for inode in listdir(styles_path):
        new_data_path = join(styles_path, inode)
        if isdir(new_data_path):
            new_stylesheets = plt.style.core.read_style_directory(new_data_path)
            stylesheets.update(new_stylesheets)

    # Update dictionary of styles
    plt.style.core.update_nested_dict(plt.style.library, stylesheets)
    # Update `plt.style.available`, ensuring all styles are registered
    plt.style.core.available[:] = sorted(plt.style.library.keys())

def set_plot_style(style=['science', 'no-latex'], figsize=(3.5, 2.8), use_degree=False):
    with plt.style.context(style=style):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.formatter.use_mathtext'] = False
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.fontset'] = 'dejavusans'
        plt.rcParams['font.sans-serif'] = ['Arial']
        fig, ax = plt.subplots(figsize=figsize)
        if use_degree:
            formatter = DegreeFormatter()
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
        return fig, ax

def set_degree_formatter(ax, axis='both'):
    formatter = DegreeFormatter()
    if axis in ['x', 'both']:
        ax.xaxis.set_major_formatter(formatter)
    if axis in ['y', 'both']:
        ax.yaxis.set_major_formatter(formatter)

def update_style_library():
    import matplotlib.pyplot as plt
    import scienceplots
    # 保存scienceplots的样式库
    scienceplots_styles = plt.style.library.copy()
    # 将scienceplots的样式库合并到当前的样式库中
    plt.style.library.update(scienceplots_styles)
    # 更新可用的样式列表
    plt.style.available[:] = sorted(plt.style.library.keys())

@contextmanager
def sci_plot_style(style=['science', 'no-latex'], legend_frame=False, use_tes=False, 
                   use_mathtext=False, serif=False, fontsize=None, figsize=None):
    # 动态注册scienceplots的样式
    register_science_styles()

    plt.style.use(style)
    if legend_frame:
        plt.rc('legend', frameon=True, framealpha=0.7,
            fancybox=True, numpoints=1)
    if serif:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times', 'Palatino', 'New Century Schoolbook', 
                                       'Bookman', 'Computer Modern Roman']
    else:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'Bitstream Vera Sans', 
                                            'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 
                                            'Lucid', 'Avant Garde', 'sans-serif']
    plt.rcParams['axes.formatter.use_mathtext'] = use_mathtext
    plt.rcParams['text.usetex'] = use_tes
    plt.rcParams['mathtext.fontset'] = 'dejavusans'

    if fontsize is not None:
        plt.rcParams['axes.labelsize'] = fontsize
        plt.rcParams['xtick.labelsize'] = fontsize
        plt.rcParams['ytick.labelsize'] = fontsize
        plt.rcParams['legend.fontsize'] = fontsize
        plt.rcParams['font.size'] = fontsize
    
    if figsize is not None:
        plt.rcParams['figure.figsize'] = figsize

    try:
        yield
    finally:
        plt.rcdefaults()