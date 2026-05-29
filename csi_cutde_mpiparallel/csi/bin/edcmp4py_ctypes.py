"""
edcmp4py_ctypes.py - Python wrapper for EDCMP layered model via ctypes.

Loads the shared library (libedcmp4py.dll/.so) compiled from edcmp4py_ctypes.f90
and provides a Pythonic interface identical to the f2py version.

Advantages over f2py:
  - Independent of Python version (no .pyd/.so per Python version)
  - Independent of numpy version (no numpy.distutils/meson dependency)
  - Only requires gfortran to compile the shared library once

Usage:
    from edcmp4py_ctypes import EdcmpLayeredCtypes

    model = EdcmpLayeredCtypes()  # auto-finds libedcmp4py.dll/.so
    info = model.load_greenfunctions('edgrnhs.ss', 'edgrnhs.ds', 'edgrnhs.cl')
    disp, strain, tilt, nwarn = model.compute_single(
        slip, xs, ys, zs, length, width, strike, dip, rake, nrec, xrec, yrec)
    disp_sum, strain_sum, tilt_sum, nwarn = model.compute_bundle_sum(
        slip_arr, xs_arr, ys_arr, zs_arr, length_arr, width_arr,
        strike_arr, dip_arr, rake_arr, nrec, xrec, yrec)
"""
import os
import sys
import ctypes
import platform
import numpy as np
from numpy.ctypeslib import ndpointer


def _find_library(lib_dir=None):
    """Find the shared library file."""
    if lib_dir is None:
        lib_dir = os.path.dirname(os.path.abspath(__file__))

    system = platform.system()
    if system == 'Windows':
        names = ['libedcmp4py.dll', 'edcmp4py.dll']
        platform_subdir = 'windows'
    elif system == 'Darwin':
        names = ['libedcmp4py.dylib', 'libedcmp4py.so']
        platform_subdir = None
    else:
        names = ['libedcmp4py.so']
        platform_subdir = 'ubuntu20.04'

    # Search lib_dir itself, then platform-specific subdirectory
    search_dirs = [lib_dir]
    if platform_subdir:
        search_dirs.append(os.path.join(lib_dir, platform_subdir))

    for d in search_dirs:
        for name in names:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path

    raise FileNotFoundError(
        f"Cannot find shared library in {lib_dir}. "
        f"Looked for: {', '.join(names)}. "
        f"Run build_edcmp4py_ctypes.py first."
    )


class EdcmpLayeredCtypes:
    """EDCMP layered model via ctypes (iso_c_binding)."""

    def __init__(self, lib_path=None):
        """Initialize and load the shared library.

        Args:
            lib_path: Path to libedcmp4py.dll/.so. If None, auto-detect.
        """
        if lib_path is None:
            lib_path = _find_library()

        self.lib = ctypes.CDLL(lib_path)
        self._setup_prototypes()

    def _setup_prototypes(self):
        """Set up C function signatures."""
        c_int_p = ctypes.POINTER(ctypes.c_int)
        c_double_p = ctypes.POINTER(ctypes.c_double)

        # edcmp_load_greenfunctions
        self.lib.edcmp_load_greenfunctions.restype = None
        self.lib.edcmp_load_greenfunctions.argtypes = [
            ctypes.c_char * 256,   # grnss
            ctypes.c_char * 256,   # grnds
            ctypes.c_char * 256,   # grncl
            c_int_p,               # nr_out
            c_int_p,               # nz_out
            c_double_p,            # r1_out
            c_double_p,            # r2_out
            c_double_p,            # z1_out
            c_double_p,            # z2_out
            c_double_p,            # zrec0_out
            c_double_p,            # lambda_out
            c_double_p,            # mu_out
            c_int_p,               # ierr
        ]

        # edcmp_compute_single
        self.lib.edcmp_compute_single.restype = None
        self.lib.edcmp_compute_single.argtypes = [
            ctypes.c_double,       # slip (value)
            ctypes.c_double,       # xs0 (value)
            ctypes.c_double,       # ys0 (value)
            ctypes.c_double,       # zs0 (value)
            ctypes.c_double,       # flen (value)
            ctypes.c_double,       # fwid (value)
            ctypes.c_double,       # fstrike (value)
            ctypes.c_double,       # fdip (value)
            ctypes.c_double,       # frake (value)
            ctypes.c_int,          # nrec (value)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # xrec
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # yrec
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # disp (out)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # strain (out)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # tilt (out)
            c_int_p,               # nwarn (out)
        ]

        # edcmp_compute_bundle_sum
        self.lib.edcmp_compute_bundle_sum.restype = None
        self.lib.edcmp_compute_bundle_sum.argtypes = [
            ctypes.c_int,          # ns (value)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # slip(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # xs0(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # ys0(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # zs0(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # flen(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # fwid(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # fstrike(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # fdip(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # frake(ns)
            ctypes.c_int,          # nrec (value)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # xrec
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # yrec
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # disp (out)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # strain (out)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # tilt (out)
            c_int_p,               # nwarn (out)
        ]

        # edcmp_compute_single_disp
        self.lib.edcmp_compute_single_disp.restype = None
        self.lib.edcmp_compute_single_disp.argtypes = [
            ctypes.c_double,       # slip (value)
            ctypes.c_double,       # xs0 (value)
            ctypes.c_double,       # ys0 (value)
            ctypes.c_double,       # zs0 (value)
            ctypes.c_double,       # flen (value)
            ctypes.c_double,       # fwid (value)
            ctypes.c_double,       # fstrike (value)
            ctypes.c_double,       # fdip (value)
            ctypes.c_double,       # frake (value)
            ctypes.c_int,          # nrec (value)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # xrec
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # yrec
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # disp (out)
            c_int_p,               # nwarn (out)
        ]

        # edcmp_compute_bundle_sum_disp
        self.lib.edcmp_compute_bundle_sum_disp.restype = None
        self.lib.edcmp_compute_bundle_sum_disp.argtypes = [
            ctypes.c_int,          # ns (value)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # slip(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # xs0(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # ys0(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # zs0(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # flen(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # fwid(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # fstrike(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # fdip(ns)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # frake(ns)
            ctypes.c_int,          # nrec (value)
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # xrec
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # yrec
            ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # disp (out)
            c_int_p,               # nwarn (out)
        ]

        # edcmp_get_greenfunctions
        _f64_1d = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        self.lib.edcmp_get_greenfunctions.restype = None
        self.lib.edcmp_get_greenfunctions.argtypes = [
            ctypes.c_int,          # nr (value)
            ctypes.c_int,          # nz (value)
            _f64_1d,               # grnss_out (10*nr*nz)
            _f64_1d,               # grnds_out (10*nr*nz)
            _f64_1d,               # grncl_out (7*nr*nz)
        ]

        # edcmp_set_greenfunctions
        self.lib.edcmp_set_greenfunctions.restype = None
        self.lib.edcmp_set_greenfunctions.argtypes = [
            ctypes.c_int,          # nr (value)
            ctypes.c_int,          # nz (value)
            ctypes.c_double,       # r1 (value)
            ctypes.c_double,       # r2 (value)
            ctypes.c_double,       # z1 (value)
            ctypes.c_double,       # z2 (value)
            ctypes.c_double,       # zrec0 (value)
            ctypes.c_double,       # lambda (value)
            ctypes.c_double,       # mu (value)
            _f64_1d,               # grnss_in (10*nr*nz)
            _f64_1d,               # grnds_in (10*nr*nz)
            _f64_1d,               # grncl_in (7*nr*nz)
            c_int_p,               # ierr (out)
        ]

    @staticmethod
    def _make_c_string(s, size=256):
        """Convert Python string to null-terminated C char array."""
        if isinstance(s, str):
            s = s.encode('utf-8')
        buf = (ctypes.c_char * size)()
        buf.value = s
        return buf

    def load_greenfunctions(self, grnss_file, grnds_file, grncl_file):
        """Load Green's functions from EDGRN output files.

        Returns: (nr, nz, r1, r2, z1, z2, zrec0, lam, mu, ierr)
        """
        grnss = self._make_c_string(grnss_file)
        grnds = self._make_c_string(grnds_file)
        grncl = self._make_c_string(grncl_file)

        nr = ctypes.c_int()
        nz = ctypes.c_int()
        r1 = ctypes.c_double()
        r2 = ctypes.c_double()
        z1 = ctypes.c_double()
        z2 = ctypes.c_double()
        zrec0 = ctypes.c_double()
        lam = ctypes.c_double()
        mu = ctypes.c_double()
        ierr = ctypes.c_int()

        self.lib.edcmp_load_greenfunctions(
            grnss, grnds, grncl,
            ctypes.byref(nr), ctypes.byref(nz),
            ctypes.byref(r1), ctypes.byref(r2),
            ctypes.byref(z1), ctypes.byref(z2),
            ctypes.byref(zrec0), ctypes.byref(lam), ctypes.byref(mu),
            ctypes.byref(ierr)
        )

        self.nr = nr.value
        self.nz = nz.value
        self.r1 = r1.value
        self.r2 = r2.value
        self.z1 = z1.value
        self.z2 = z2.value
        self.zrec0 = zrec0.value
        self.lam = lam.value
        self.mu = mu.value
        self._grn_loaded = True

        if ierr.value == 0:
            self._export_greenfunctions()

        return (nr.value, nz.value, r1.value, r2.value,
                z1.value, z2.value, zrec0.value, lam.value, mu.value,
                ierr.value)

    def _export_greenfunctions(self):
        """Export Fortran internal Green's function arrays to Python attributes."""
        nr, nz = self.nr, self.nz
        self.grnss = np.zeros(10 * nr * nz, dtype=np.float64)
        self.grnds = np.zeros(10 * nr * nz, dtype=np.float64)
        self.grncl = np.zeros(7 * nr * nz, dtype=np.float64)
        self.lib.edcmp_get_greenfunctions(
            ctypes.c_int(nr), ctypes.c_int(nz),
            self.grnss, self.grnds, self.grncl,
        )

    def _inject_greenfunctions(self):
        """Inject Python Green's function arrays into Fortran internal state."""
        ierr = ctypes.c_int()
        self.lib.edcmp_set_greenfunctions(
            ctypes.c_int(self.nr), ctypes.c_int(self.nz),
            ctypes.c_double(self.r1), ctypes.c_double(self.r2),
            ctypes.c_double(self.z1), ctypes.c_double(self.z2),
            ctypes.c_double(self.zrec0), ctypes.c_double(self.lam), ctypes.c_double(self.mu),
            np.ascontiguousarray(self.grnss, dtype=np.float64),
            np.ascontiguousarray(self.grnds, dtype=np.float64),
            np.ascontiguousarray(self.grncl, dtype=np.float64),
            ctypes.byref(ierr),
        )
        if ierr.value != 0:
            raise RuntimeError(f"edcmp_set_greenfunctions failed (ierr={ierr.value})")

    @classmethod
    def from_shared_arrays(cls, grnss, grnds, grncl, model_info, lib_path=None):
        """Reconstruct a model from pre-loaded Green's function arrays.

        Parameters
        ----------
        grnss, grnds, grncl : np.ndarray
            Green's function arrays (may be read-only shared memory views).
        model_info : dict
            Scalar metadata with keys: nr, nz, r1, r2, z1, z2, zrec0, lam, mu.
        lib_path : str, optional
            Path to shared library.
        """
        obj = cls(lib_path=lib_path)
        obj._grn_loaded = True
        obj.grnss = np.ascontiguousarray(grnss, dtype=np.float64)
        obj.grnds = np.ascontiguousarray(grnds, dtype=np.float64)
        obj.grncl = np.ascontiguousarray(grncl, dtype=np.float64)
        for attr in ('nr', 'nz', 'r1', 'r2', 'z1', 'z2', 'zrec0', 'lam', 'mu'):
            setattr(obj, attr, model_info[attr])
        obj._inject_greenfunctions()
        return obj

    @staticmethod
    def _prepare_source_vector(values, ns=None, name="array"):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 0:
            if ns is None:
                return np.ascontiguousarray(np.atleast_1d(arr), dtype=np.float64)
            return np.full(int(ns), float(arr), dtype=np.float64)

        arr = np.ascontiguousarray(np.atleast_1d(arr), dtype=np.float64)
        if ns is not None and arr.size != int(ns):
            raise ValueError(f"{name} must have length {int(ns)}, got {arr.size}")
        return arr

    def compute_single(self, slip, xs0, ys0, zs0, flen, fwid, fstrike, fdip, frake,
                       nrec, xrec, yrec):
        """Compute deformation for a single rectangular source.

        Returns: (disp(nrec,3), strain(nrec,6), tilt(nrec,2), nwarn)
        """
        xrec = np.ascontiguousarray(xrec, dtype=np.float64)
        yrec = np.ascontiguousarray(yrec, dtype=np.float64)

        disp_flat = np.zeros(nrec * 3, dtype=np.float64)
        strain_flat = np.zeros(nrec * 6, dtype=np.float64)
        tilt_flat = np.zeros(nrec * 2, dtype=np.float64)
        nwarn = ctypes.c_int()

        self.lib.edcmp_compute_single(
            ctypes.c_double(float(slip)),
            ctypes.c_double(float(xs0)),
            ctypes.c_double(float(ys0)),
            ctypes.c_double(float(zs0)),
            ctypes.c_double(float(flen)),
            ctypes.c_double(float(fwid)),
            ctypes.c_double(float(fstrike)),
            ctypes.c_double(float(fdip)),
            ctypes.c_double(float(frake)),
            ctypes.c_int(int(nrec)),
            xrec, yrec,
            disp_flat, strain_flat, tilt_flat,
            ctypes.byref(nwarn)
        )

        disp = disp_flat.reshape(nrec, 3)
        strain = strain_flat.reshape(nrec, 6)
        tilt = tilt_flat.reshape(nrec, 2)

        return disp, strain, tilt, nwarn.value

    def compute_single_disp(self, slip, xs0, ys0, zs0, flen, fwid, fstrike, fdip, frake,
                            nrec, xrec, yrec):
        """Compute displacement only for a single rectangular source."""
        xrec = np.ascontiguousarray(xrec, dtype=np.float64)
        yrec = np.ascontiguousarray(yrec, dtype=np.float64)

        disp_flat = np.zeros(nrec * 3, dtype=np.float64)
        nwarn = ctypes.c_int()

        self.lib.edcmp_compute_single_disp(
            ctypes.c_double(float(slip)),
            ctypes.c_double(float(xs0)),
            ctypes.c_double(float(ys0)),
            ctypes.c_double(float(zs0)),
            ctypes.c_double(float(flen)),
            ctypes.c_double(float(fwid)),
            ctypes.c_double(float(fstrike)),
            ctypes.c_double(float(fdip)),
            ctypes.c_double(float(frake)),
            ctypes.c_int(int(nrec)),
            xrec, yrec,
            disp_flat,
            ctypes.byref(nwarn)
        )

        disp = disp_flat.reshape(nrec, 3)
        return disp, nwarn.value

    def compute_bundle_sum(self, slip, xs0, ys0, zs0, flen, fwid,
                           fstrike, fdip, frake, nrec, xrec, yrec):
        """Compute accumulated deformation for a source bundle.

        Each entry in the source arrays describes one rectangular source. The
        returned displacement/strain/tilt are the sum over the full bundle.
        """
        slip = self._prepare_source_vector(slip, name="slip")
        ns = slip.size
        xs0 = self._prepare_source_vector(xs0, ns=ns, name="xs0")
        ys0 = self._prepare_source_vector(ys0, ns=ns, name="ys0")
        zs0 = self._prepare_source_vector(zs0, ns=ns, name="zs0")
        flen = self._prepare_source_vector(flen, ns=ns, name="flen")
        fwid = self._prepare_source_vector(fwid, ns=ns, name="fwid")
        fstrike = self._prepare_source_vector(fstrike, ns=ns, name="fstrike")
        fdip = self._prepare_source_vector(fdip, ns=ns, name="fdip")
        frake = self._prepare_source_vector(frake, ns=ns, name="frake")

        xrec = np.ascontiguousarray(xrec, dtype=np.float64)
        yrec = np.ascontiguousarray(yrec, dtype=np.float64)

        disp_flat = np.zeros(nrec * 3, dtype=np.float64)
        strain_flat = np.zeros(nrec * 6, dtype=np.float64)
        tilt_flat = np.zeros(nrec * 2, dtype=np.float64)
        nwarn = ctypes.c_int()

        self.lib.edcmp_compute_bundle_sum(
            ctypes.c_int(int(ns)),
            slip, xs0, ys0, zs0, flen, fwid, fstrike, fdip, frake,
            ctypes.c_int(int(nrec)),
            xrec, yrec,
            disp_flat, strain_flat, tilt_flat,
            ctypes.byref(nwarn)
        )

        disp = disp_flat.reshape(nrec, 3)
        strain = strain_flat.reshape(nrec, 6)
        tilt = tilt_flat.reshape(nrec, 2)

        return disp, strain, tilt, nwarn.value

    def compute_bundle_sum_disp(self, slip, xs0, ys0, zs0, flen, fwid,
                                fstrike, fdip, frake, nrec, xrec, yrec):
        """Compute accumulated displacement only for a source bundle."""
        slip = self._prepare_source_vector(slip, name="slip")
        ns = slip.size
        xs0 = self._prepare_source_vector(xs0, ns=ns, name="xs0")
        ys0 = self._prepare_source_vector(ys0, ns=ns, name="ys0")
        zs0 = self._prepare_source_vector(zs0, ns=ns, name="zs0")
        flen = self._prepare_source_vector(flen, ns=ns, name="flen")
        fwid = self._prepare_source_vector(fwid, ns=ns, name="fwid")
        fstrike = self._prepare_source_vector(fstrike, ns=ns, name="fstrike")
        fdip = self._prepare_source_vector(fdip, ns=ns, name="fdip")
        frake = self._prepare_source_vector(frake, ns=ns, name="frake")

        xrec = np.ascontiguousarray(xrec, dtype=np.float64)
        yrec = np.ascontiguousarray(yrec, dtype=np.float64)

        disp_flat = np.zeros(nrec * 3, dtype=np.float64)
        nwarn = ctypes.c_int()

        self.lib.edcmp_compute_bundle_sum_disp(
            ctypes.c_int(int(ns)),
            slip, xs0, ys0, zs0, flen, fwid, fstrike, fdip, frake,
            ctypes.c_int(int(nrec)),
            xrec, yrec,
            disp_flat,
            ctypes.byref(nwarn)
        )

        disp = disp_flat.reshape(nrec, 3)
        return disp, nwarn.value

    def compute_batch(self, ns, slip, xs0, ys0, zs0, flen, fwid,
                      fstrike, fdip, frake, nrec, xrec, yrec):
        """Batch computation: loop over sources in Python.

        Returns: (disp_all(ns,nrec,3), strain_all(ns,nrec,6),
                  tilt_all(ns,nrec,2), nwarn_total)
        """
        disp_all = np.zeros((ns, nrec, 3))
        strain_all = np.zeros((ns, nrec, 6))
        tilt_all = np.zeros((ns, nrec, 2))
        nwarn_total = 0

        for i in range(ns):
            d, s, t, nw = self.compute_single(
                slip[i], xs0[i], ys0[i], zs0[i],
                flen[i], fwid[i], fstrike[i], fdip[i], frake[i],
                nrec, xrec, yrec)
            disp_all[i] = d
            strain_all[i] = s
            tilt_all[i] = t
            nwarn_total += nw

        return disp_all, strain_all, tilt_all, nwarn_total
