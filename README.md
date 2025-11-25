<div align="center">
  <img height="300" src="image/logo.jpg"  alt="Logo" />
</div>

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=kefuhe.ECAT) ![GitHub Stars](https://img.shields.io/github/stars/kefuhe/ECAT?style=social)![GitHub Forks](https://img.shields.io/github/forks/kefuhe/ECAT?style=social)

![Version](https://img.shields.io/badge/version-1.5.0-blue) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13730101.svg)](https://doi.org/10.5281/zenodo.13730101) ![Repository Size](https://img.shields.io/github/repo-size/kefuhe/ECAT) ![GitHub Language](https://img.shields.io/github/languages/top/kefuhe/ECAT) ![GitHub Last Commit](https://img.shields.io/github/last-commit/kefuhe/ECAT)

</p>

# ECAT (Earthquake Cycle Analyse Toolkits)

---

## Forward Modeling Engines

***ECAT*** supports layered Green's function calculations for both rectangular and triangular elements, utilizing `edcmp` and `pscmp`.
For more details and configuration instructions, see the `README` file in the `csi` subdirectory.
![1755593711201](image/README/1755593711201.png)

**A set of tools related to the forward and inverse *earthquake cycle*.**

**Continue updating...**

* Stable code will gradually be incorporated
* New code will gradually be added
* Bug will gradually be fixed

---

## Installation

- See Readme ***Install.md***

## Download ECAT Case Library

The full ECAT case library, including research cases and advanced examples, is stored in a separate repository. To download the full case library, you have two options:

### Option 1: Enable Submodule Download in the Installation Script

1. Open the install.bat (Windows) or [install.sh](http://install.sh) (Linux/Mac) file.
2. Uncomment the following line:

```Bash
git submodule update --init --recursive
```

### Option 2: Download the Case Library Manually

If you prefer to download the case library manually, you can do so with the following script:

```bash
./download_cases.sh
```

Or download the case manually:

```bash
git clone https://github.com/kefuhe/ECAT-cases.git
```

## Coseismic （Uploaded）

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

<p align="center">
  <img src="image/README/1723047070456.png" alt="描述" />
</p>

### Building fault with isocurve determined by depth-focused aftershocks

<p align="center">
  <img src="image/README/1702349779348.png" alt="Image Description"/>
</p

![1706253529470](image/README/1706253529470.png)

***Note:*** Various ***easy*** or ***robust*** curve fitting methods are only used as a reference to determine the ***isodepth curve***, and ultimately the final fitted curve may need to be extracted based on the user's own *background knowledge* or *visual interaction judgment*.

### Citation

If you use this package in your research, please cite it as follows:

He K., C. Xu, Y. Wen, Y. Zhao, G. Xu, L. Sun, and J. Wang (2025), A Unified Framework for Adaptive Fault Modeling: Methods and Applications, SCIENCE CHINA Earth Sciences. doi: [10.1007/s11430-025-1773-0](https://doi.org/10.1007/s11430-025-1773-0 "https://doi.org/10.1007/s11430-025-1773-0").

## Postseismic （Pending)

- pscmp_visco_postseismic.py
  - Simple calculating codes about ***Combined model of viscoelastic relaxation and stress-driven afterslip***
- Deformation components
  - CV (Viscoelastic relaxation due to coseismic): Function ***cv_cum(t, tobs, disp)***
  - AS (Afterslip): Function ***as_cum(t, disp0, tau_as=0.25, alpha=1.0)***
  - AV (Viscoelastic relaxation due to afterslip): Function ***calAS_AV(pscmpts, obsdate, eqdate, alpha, tau, unit='m', intp_tunit='Y', onlyAV=True, mcpu=4)***

## Interseismic (Pending)

- pscmp_visco_interseismic.py
  - Calculating velocity related to ***earthquake cycle***
- Main function
  - ***calviscoGfromPscmp***(pscmpts, T=None, diffint=None, unit='m')
