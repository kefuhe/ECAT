config = {
    "do_covar": False,
    "do_downsample": True,
    "plot_downsample": True,
    "output_check": True,
    "default_top_depth": 0.0,
    "default_bottom_depth": 20.0,
    "show_raw_sar": False,
    "covar": {
        "function": 'exp',
        "frac": 0.02,
        "every": 2.0,
        "distmax": 100.,
        "rampEst": True
    },
    "downsample": {
        "minimumsize": 1.0,
        "tolerance": 0.1,
        "plot": False,
        "decimorig": 10,
        "max_samples": 2000,
        "change_threshold": 5,
        "smooth_factor": 0.25,
        "slipdirection": 'd',
    },
    "faults": [
        {
            "trace_file": r'/mnt/e/geocodes/eqtools/example_tests/SMC_Geometry_Slip_Bayesian_Inversion/China/Tibet/Dingri_20250107_Mw7_1/OtherInformations/Trace_From_SunJianbao.txt',
            "dip_angle": 60,
            "dip_direction": 280,
            "top_size": 2.0,
            "bottom_size": 4.0,
            "top_depth": 0.0,
            "bottom_depth": 20.0
        },
        # 添加更多的断层配置
    ],
    "lon0": 87.5,
    "lat0": 28.5,
    "outName": 'S1_T121D',
    "prefix": r'geo_20250101_20250113',
    "sar_dict": {
        'phsname': 'roff_20241206_20250107.phs',
        'rscname': 'roff_20241206_20250107.phs.rsc',
        'azifile': 'off_20241206_20250107.azi',
        'incfile': 'off_20241206_20250107.inc',
    },
    "use_offset_sar": False,
    "maskOut": [87.2, 88.5, 28.2, 29.3],
    "downsample_box": {
        "minlat": None,
        "maxlat": None,
        "minlon": None,
        "maxlon": None
    },
    "plot_box": {
        "plotMinLat": None,
        "plotMaxLat": None,
        "plotMinLon": None,
        "plotMaxLon": None
    },
    "plot_raw_sar": {
        "save_fig": True,
        "rawdownsample4plot": 10,
        "colorbar_x": 0.3,
        "colorbar_y": 0.25,
        "colorbar_length": 0.3,
        "vmin": -100,
        "vmax": 100
    }
}