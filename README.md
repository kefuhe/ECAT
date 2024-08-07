<p align="center">

<img src="image/logo.png" alt="Logo">

</p>

# eqtools

---

**A set of tools related to the forward and inverse *earthquake cycle*.**

**Continue updating...**

* Stable code will gradually be incorporated
* New code will gradually be added
* Bug will gradually be fixed

---

## Installation

### Case 1

1. Download and untar the github package "eqtools"
2. cd eqtools
3. pip install .

### Case 2

1. Download and untar the github package "eqtools"
2. cd eqtools
3. Install eqtools by python setup.py

```
# Create a build file 
python setup.py build 
# Generate a release package
python setup.py sdist 
# Go into the dist directory to unzip the package, then go into the unzip file, and do
python setup.py install --user
```

## Coseismic

### Building fault with variance dip angle determined by relocated aftershocks

1. **Case 1: 2021 Maduo earthquake**

<p align="center">
  <img src="image/README/1702140126132.png" alt="Image Description"/>
</p>

![1705485485549](image/README/1705485485549.png)

2. **Case 2: 2023 Kahramanmaraş Türkiye earthquake doublet**

![1708366709425](image/README/1708366709425.png)

![1708350669168](image/README/1708350669168.png)

![1708435466354](image/README/1708435466354.png)

![1710499654420](image/README/1710499654420.png)

3. **Case 3: 2022 Luding, China earthquake** (Pre-Clustering Selection Followed by Fitting)

![1709000559665](image/README/1709000559665.png)

### Building fault with isocurve determined by depth-focused aftershocks

<p align="center">
  <img src="image/README/1702349779348.png" alt="Image Description"/>
</p

![1706253529470](image/README/1706253529470.png)

***Note:*** Various ***easy*** or ***robust*** curve fitting methods are only used as a reference to determine the ***isodepth curve***, and ultimately the final fitted curve may need to be extracted based on the user's own *background knowledge* or *visual interaction judgment*.

### Reference

[1] He, K., Y. Wen, C. Xu, and Y. Zhao (2021), Fault Geometry and Slip Distribution of the 2021 Mw 7.4 Maduo, China, Earthquake Inferred from InSAR Measurements and Relocated Aftershocks, Seismol. Res. Lett., 93(1), 8-20.

[2] [jolivetr/csi: Classic Slip Inversion (github.com)](https://github.com/jolivetr/csi)

[3] [Gmsh 4.11.1](https://gmsh.info/doc/texinfo/gmsh.html)

[4] ...

## Postseismic

- pscmp_visco_postseismic.py
  - Simple calculating codes about ***Combined model of viscoelastic relaxation and stress-driven afterslip***
- Deformation components
  - CV (Viscoelastic relaxation due to coseismic): Function ***cv_cum(t, tobs, disp)***
  - AS (Afterslip): Function ***as_cum(t, disp0, tau_as=0.25, alpha=1.0)***
  - AV (Viscoelastic relaxation due to afterslip): Function ***calAS_AV(pscmpts, obsdate, eqdate, alpha, tau, unit='m', intp_tunit='Y', onlyAV=True, mcpu=4)***

### Reference

[1] Diao, F., R. Wang, Y. Wang, X. Xiong, and T. R. Walter (2018), Fault behavior and lower crustal rheology inferred from the first seven years of postseismic GPS data after the 2008 Wenchuan earthquake, Earth Planet. Sci. Lett., 495, 202-212, doi:10.1016/j.epsl.2018.05.020.

[2] Diao, F., R. Wang, X. Xiong, and C. Liu (2021), Overlapped Postseismic Deformation Caused by Afterslip and Viscoelastic Relaxation Following the 2015 *Mw* 7.8 Gorkha (Nepal) Earthquake, J. Geophys. Res.: Sol. Ea., 126(3), doi:10.1029/2020jb020378.

[3] He Ke., Y. Wen, C. Xu, W. Xiong, J. Zang (2023), In preparation.

## Interseismic

- pscmp_visco_interseismic.py
  - Calculating velocity related to ***earthquake cycle***
- Main function
  - ***calviscoGfromPscmp***(pscmpts, T=None, diffint=None, unit='m')

### Reference

[1] Diao, F., X. Xiong, R. Wang, T. R. Walter, Y. Wang, and K. Wang (2019), Slip Rate Variation Along the Kunlun Fault (Tibet): Results From New GPS Observations and a Viscoelastic Earthquake‐Cycle Deformation Model, Geophys. Res. Lett., 46(5), 2524-2533, doi:10.1029/2019gl081940.

[2] Zhu, Y., K. Wang, and J. He (2020), Effects of Earthquake Recurrence on Localization of Interseismic Deformation Around Locked Strike‐Slip Faults, J. Geophys. Res.: Sol. Ea., 125(8), doi:10.1029/2020jb019817.

[3] Wang, K., Y. Zhu, E. Nissen, and Z. K. Shen (2021), On the Relevance of Geodetic Deformation Rates to Earthquake Potential, Geophys. Res. Lett., 48(11), doi:10.1029/2021gl093231.
