# Tectonic Utils

A simple library with utility functions for "everyday use" in active tectonics research. 
There are geodesy and seismology utility functions in Python, 
some originally from libraries in Matlab or other languages. 
There are read-write functions to access commonly used formats. 
Please explore and let me know if you'd like to contribute! 

Functions in this library that I use regularly:
* Haversine formula
* Euler Pole rotations
* Geographic to XYZ Earth-centered coordinates
* InSAR look vectors (Fialko et al., 2001)
* Fault vector operations (Aki and Richards, 1980)
* Moment magnitude conversions (Hanks and Kanamori, 1979)
* Earthquake magnitude scaling (Wells and Coppersmith, 1994)
* Reading GMT multi-segment files into Python

![cover](https://github.com/kmaterna/Tectonic_Utils/blob/master/Tectonic_Utils/cover_picture.png)

## Dependencies
* Python 3
* numpy
* scipy 
* netCDF4 (pip install netCDF4)


## Installation
On Unix (other systems not tested yet), install from PyPI using: 

```pip install Tectonic-Utils```

To test that it works, import a utility in Python. I use an import structure like: 
```
$ python
>>> from Tectonic_Utils.geodesy import haversine
```

## Documentation
The preliminary sphinx html docs can be found here: https://kmaterna.github.io/Tectonic_Utils/index.html

## Additional Thoughts
I wrote this library for my own research because (**in theory**) 
putting all my Python utility functions 
into one unit-tested repository reduces bugs, increases code reproducibility, and allows for faster experimentation. 
The library is open source and available on Pip in case it would help others too. 

Disclaimer: Completely bug-free code can never be guaranteed. As things come up, I occasionally fix and update the package on Pip/PyPI.    

If you're using this package and you'd like to contribute or you find bugs, please let me know!  

## References

* Aki, K., and P. G. Richards (1980), Quantitative Seismology: Theory and Methods, W. H. Freeman, New York.
* Fialko, Y., Simons, M., & Agnew, D. (2001). The complete (3-D) surface displacement field in the epicentral area of the 1999 Mw7.1 Hector Mine earthquake, California, from space geodetic observations. Geophysical Research Letters, 28(16), 3063–3066.
* Hanks, T., & Kanamori, H. (1979). A moment magnitude scale. Journal of Geophysical Research, 84(B5), 2348–2350.
* Wells, D. L., & Coppersmith, K. J. (1994). New Empirical Relationships among Magnitude, Rupture Length, Rupture Width, Rupture Area, and Surface Displacement. Bulletin of the Seismological Society of America, 84(4), 974–1002. 
