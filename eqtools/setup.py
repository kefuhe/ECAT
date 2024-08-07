from setuptools import find_packages, setup

setup(name = 'eqtools',
    version = '1.1.5',
    author = 'Kefeng He',
    author_email = 'kefenghe@whu.edu.cn',
    url = 'https://github.com/kefuhe/eqtools',
    description = 'Earthquake Cycle Analysis Toolkit (ECAT)',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'eqtools': ['cpt/*'], # , 'examples/*'
        'Tectonic_Utils': ['README.md', 'cover_picture.png']
    },
    classifiers=[
        # Choose classifiers from https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    )