"""
Source Adapter Layer for Multi-Source Inversion

Provides a unified interface for different CSI source types (Fault, Pressure, Sbarbot)
so the inversion framework can operate on any source through a common API.

Design pattern: Adapter + Simple Factory
"""

from abc import ABC, abstractmethod
import warnings
import numpy as np


class SourceAdapter(ABC):
    """Abstract base class for source adapters."""

    def __init__(self, source, **kwargs):
        self.source = source

    @property
    def name(self):
        return self.source.name

    @property
    @abstractmethod
    def source_type(self):
        """Return the CSI type string: 'Fault', 'Pressure', or 'Sbarbot'."""
        ...

    # ── Parameter description ──────────────────────────────────────────

    @abstractmethod
    def get_param_names(self):
        """Return ordered list of parameter component names.

        Examples: ['strikeslip','dipslip'], ['pressure'], ['eps12','eps13']
        """
        ...

    @abstractmethod
    def get_n_source_params(self):
        """Total number of source parameters (excluding poly)."""
        ...

    @abstractmethod
    def get_n_params_per_component(self):
        """Return dict {component_name: count}."""
        ...

    @abstractmethod
    def get_n_spatial_elements(self):
        """Number of spatial elements (patches / volumes / 1 for point source)."""
        ...

    # ── Green's function construction ──────────────────────────────────

    @abstractmethod
    def build_gfs(self, data, vertical, method='homogeneous', options=None,
                  verbose=None, convergence=None):
        """Build GFs with the correct call signature for this source type."""
        ...

    @abstractmethod
    def get_gf_column_keys(self):
        """Keys into ``source.G[data_name]`` that correspond to source params."""
        ...

    # ── Smoothing ──────────────────────────────────────────────────────

    @classmethod
    def supports_smoothing(cls):
        """Whether Laplacian smoothing is meaningful for this source."""
        return False

    def build_laplacian(self, **kwargs):
        """Build and return a Laplacian matrix, or *None*."""
        return None

    # ── Depth-Equalized Smoothing (DES) ────────────────────────────────

    def get_depths(self):
        """Return list of element depths, or *None*."""
        return None

    def get_des_config(self, ss_indices, ds_indices, poly_indices):
        """Return DES config dict, or *None*."""
        return None

    # ── Result distribution ────────────────────────────────────────────

    @abstractmethod
    def distribute_results(self, mpost_segment):
        """Write solved parameters back onto the source object."""
        ...

    # ── Area / volume ──────────────────────────────────────────────────

    def compute_patch_areas(self):
        """Return area array (for moment computation) or *None*."""
        return None

    # ── Initial assembly helpers ───────────────────────────────────────

    def initialize_solution(self):
        """Initialize the source's solution storage (slip / pressure / strain)."""
        pass  # Default no-op; overridden per type

    def assemble_data(self, datas, verbose=False):
        """Call the source-appropriate ``assembled`` / ``assembledata``."""
        self.source.assembled(datas, verbose=verbose)

    @abstractmethod
    def assemble_gfs(self, datas, polys=None, verbose=False, custom=False):
        """Call the source-appropriate ``assembleGFs``."""
        ...

    def assemble_cd(self, datas, verbose=False, add_prediction=None):
        """Call the source-appropriate ``assembleCd``."""
        self.source.assembleCd(datas, verbose=verbose, add_prediction=add_prediction)

    # ── GF method inference ────────────────────────────────────────────

    def infer_gf_method(self):
        """Infer default GF method from source properties, or return None."""
        return None

    # ── Constraint capability ──────────────────────────────────────────

    def get_supported_constraints(self):
        """Return list of constraint type names this source supports.

        Each name corresponds to a ``generate_<name>_constraints`` method that
        the constraint manager can invoke.

        Returns
        -------
        list of str
            e.g. ``['rake_angle', 'fixed_rake', 'euler', 'zero_edge_slip']`` for
            Fault; ``['positive_pressure']`` for Pressure; ``[]`` for Sbarbot.
        """
        return []

    def generate_source_inequality_constraints(self, constraint_cfg, param_start, n_total_params):
        """Generate source-specific inequality constraints ``A @ x <= b``.

        Parameters
        ----------
        constraint_cfg : dict
            Per-source constraint configuration parsed from ``source_constraints``
            YAML section.  Structure: ``{constraint_name: {rule params ...}}``.
        param_start : int
            Starting column index for this source's parameters in the global
            parameter vector.
        n_total_params : int
            Total number of columns in the constraint matrix.

        Returns
        -------
        list of tuple
            ``[(name, A, b), ...]`` where *A* is ``(n_constraints, n_total_params)``
            and *b* is ``(n_constraints,)``.  Empty list if no constraints.
        """
        return []

    def generate_source_equality_constraints(self, constraint_cfg, param_start, n_total_params):
        """Generate source-specific equality constraints ``A @ x = b``.

        Same parameter/return convention as
        :meth:`generate_source_inequality_constraints`.
        """
        return []


# ═══════════════════════════════════════════════════════════════════════
# Concrete adapters
# ═══════════════════════════════════════════════════════════════════════

class FaultAdapter(SourceAdapter):
    """Adapter for CSI Fault objects (rectangular / triangular patches)."""

    _DEFAULT_SLIPDIR = 'sd'

    def __init__(self, source, slipdir=None):
        super().__init__(source)
        self._slipdir = slipdir or getattr(source, 'slipdir', self._DEFAULT_SLIPDIR)

    @property
    def source_type(self):
        return 'Fault'

    # ── slipdir helpers ────────────────────────────────────────────────

    @property
    def slipdir(self):
        """Slip direction string.

        After ``assembleGFs()`` the source stores ``slipdir`` as an
        attribute; that value takes precedence so the adapter always
        stays in sync with the assembled GF matrix.  Before that call
        we fall back to the value configured at construction time
        (default ``'sd'``).
        """
        return getattr(self.source, 'slipdir', self._slipdir)

    @slipdir.setter
    def slipdir(self, value):
        self._slipdir = value

    _CHAR_TO_NAME = {'s': 'strikeslip', 'd': 'dipslip',
                     't': 'tensile', 'c': 'coupling'}

    def _slipdir_names(self):
        return [self._CHAR_TO_NAME[c] for c in self.slipdir]

    # ── Parameter description ──────────────────────────────────────────

    def get_param_names(self):
        return self._slipdir_names()

    def get_n_source_params(self):
        return len(self.source.patch) * len(self.slipdir)

    def get_n_params_per_component(self):
        n = len(self.source.patch)
        return {name: n for name in self._slipdir_names()}

    def get_n_spatial_elements(self):
        return len(self.source.patch)

    # ── GF construction ────────────────────────────────────────────────

    def build_gfs(self, data, vertical, method='homogeneous', options=None,
                  verbose=None, convergence=None):
        if verbose is None:
            source_verbose = getattr(self.source, 'verbose', False)
            verbose = source_verbose if isinstance(source_verbose, bool) else False
        self.source.buildGFs(data, vertical=vertical,
                             slipdir=self.slipdir,
                             method=method, verbose=verbose,
                             convergence=convergence,
                             options=options)

    def get_gf_column_keys(self):
        return self._slipdir_names()

    # ── Smoothing ──────────────────────────────────────────────────────

    @classmethod
    def supports_smoothing(cls):
        return True

    def build_laplacian(self, **kwargs):
        from scipy.sparse import block_diag as sp_block_diag
        lap = self.source.buildLaplacian(**kwargs)
        if len(self.slipdir) == 1:
            from scipy.sparse import csr_matrix
            return csr_matrix(lap)
        return sp_block_diag([lap for _ in self.slipdir]).tocsr()

    # ── DES ────────────────────────────────────────────────────────────

    def get_depths(self):
        centers = self.source.getcenters()
        return [c[2] for c in centers]

    def get_des_config(self, ss_indices, ds_indices, poly_indices):
        return {
            'name': self.source.name,
            'ss': ss_indices,
            'ds': ds_indices,
            'poly': poly_indices,
            'depths': self.get_depths(),
        }

    # ── Result distribution ────────────────────────────────────────────

    def distribute_results(self, mpost_segment):
        """Write slip values back to ``source.slip`` / ``source.coupling``."""
        st = 0
        n = self.source.slip.shape[0]
        for char in self.slipdir:
            se = st + n
            if char == 's':
                self.source.slip[:, 0] = mpost_segment[st:se]
            elif char == 'd':
                self.source.slip[:, 1] = mpost_segment[st:se]
            elif char == 't':
                self.source.slip[:, 2] = mpost_segment[st:se]
            elif char == 'c':
                self.source.coupling = mpost_segment[st:se]
            st = se

    # ── Area ───────────────────────────────────────────────────────────

    def compute_patch_areas(self):
        return self.source.compute_patch_areas()

    # ── Initial assembly helpers ───────────────────────────────────────

    def initialize_solution(self):
        self.source.initializeslip()

    def assemble_gfs(self, datas, polys=None, verbose=False, custom=False):
        self.source.assembleGFs(datas, polys=polys, slipdir=self.slipdir,
                                verbose=verbose, custom=custom)

    def infer_gf_method(self):
        if hasattr(self.source, 'patchType'):
            if self.source.patchType == 'triangle':
                return 'cutde'
            elif self.source.patchType == 'rectangle':
                return 'okada'
        return None

    # ── Constraint capability ──────────────────────────────────────────

    def get_supported_constraints(self):
        return ['rake_angle', 'fixed_rake', 'euler', 'zero_edge_slip']

    def _parse_zero_edge_slip_rule(self, rule_str):
        """Parse a zero_edge_slip rule string into (edges, slip_modes).

        Accepted formats::

            zero_edge_slip(top, strikeslip)
            zero_edge_slip(top+bottom, ss+ds)
            zero_edge_slip(top, strikeslip, dipslip)
            zero_edge_slip(left+right, ds)

        Returns
        -------
        edges : list of str
        slip_modes : list of str   (normalised to 'strikeslip' / 'dipslip')
        """
        import re
        m = re.match(r'zero_edge_slip\((.+)\)', rule_str.replace(' ', ''))
        if not m:
            return None, None
        tokens = [t.strip() for t in m.group(1).split(',')]
        # First token group = edges (separated by +)
        edges = [e.strip() for e in tokens[0].split('+')]
        # Remaining tokens = slip modes (each may be +-separated)
        raw_modes = []
        for t in tokens[1:]:
            raw_modes.extend(t.split('+'))

        _ALIASES = {
            'strikeslip': 'strikeslip', 'strike_slip': 'strikeslip',
            'ss': 'strikeslip', 's': 'strikeslip', 'strike': 'strikeslip',
            'dipslip': 'dipslip', 'dip_slip': 'dipslip',
            'ds': 'dipslip', 'd': 'dipslip', 'dip': 'dipslip',
        }
        modes = []
        for rm in raw_modes:
            norm = rm.lower().replace(' ', '').replace('_', '')
            if norm not in _ALIASES:
                raise ValueError(
                    f"Unknown slip mode '{rm}' in zero_edge_slip rule. "
                    f"Known: {list(_ALIASES.keys())}"
                )
            canonical = _ALIASES[norm]
            if canonical not in modes:
                modes.append(canonical)
        return edges, modes

    def generate_source_inequality_constraints(self, constraint_cfg, param_start, n_total_params):
        results = []
        n_patches = self.get_n_spatial_elements()
        n_slip_dirs = len(self.slipdir)

        for cname, cparams in constraint_cfg.items():
            ctype = cparams.get('type', 'inequality')
            if ctype != 'inequality':
                continue

            rule = cparams.get('rule', '')
            rule_lower = rule.lower().replace(' ', '')

            # Positive strikeslip: strikeslip >= 0  →  -strikeslip <= 0
            if rule_lower == 'strikeslip>=0':
                A = np.zeros((n_patches, n_total_params))
                b = np.zeros(n_patches)
                A[np.arange(n_patches), param_start + np.arange(n_patches)] = -1.0
                results.append((cname, A, b))

            # Positive dipslip: dipslip >= 0  →  -dipslip <= 0
            elif rule_lower == 'dipslip>=0' and n_slip_dirs >= 2:
                A = np.zeros((n_patches, n_total_params))
                b = np.zeros(n_patches)
                offset = param_start + n_patches  # dipslip offset
                A[np.arange(n_patches), offset + np.arange(n_patches)] = -1.0
                results.append((cname, A, b))

            # Negative dipslip: dipslip <= 0
            elif rule_lower == 'dipslip<=0' and n_slip_dirs >= 2:
                A = np.zeros((n_patches, n_total_params))
                b = np.zeros(n_patches)
                offset = param_start + n_patches
                A[np.arange(n_patches), offset + np.arange(n_patches)] = 1.0
                results.append((cname, A, b))

        return results

    def generate_source_equality_constraints(self, constraint_cfg, param_start, n_total_params):
        results = []
        n_patches = self.get_n_spatial_elements()
        n_slip_dirs = len(self.slipdir)

        for cname, cparams in constraint_cfg.items():
            ctype = cparams.get('type', 'inequality')
            if ctype != 'equality':
                continue

            rule = cparams.get('rule', '')
            rule_lower = rule.lower().replace(' ', '')

            # Zero strikeslip: strikeslip == 0
            if rule_lower == 'strikeslip==0':
                A = np.zeros((n_patches, n_total_params))
                b = np.zeros(n_patches)
                A[np.arange(n_patches), param_start + np.arange(n_patches)] = 1.0
                results.append((cname, A, b))

            # Zero dipslip: dipslip == 0
            elif rule_lower == 'dipslip==0' and n_slip_dirs >= 2:
                A = np.zeros((n_patches, n_total_params))
                b = np.zeros(n_patches)
                offset = param_start + n_patches
                A[np.arange(n_patches), offset + np.arange(n_patches)] = 1.0
                results.append((cname, A, b))

            # ── Zero edge slip: zero_edge_slip(edges, modes) ───────────
            elif rule_lower.startswith('zero_edge_slip('):
                edges, modes = self._parse_zero_edge_slip_rule(rule_lower)
                if edges is None:
                    continue
                if not hasattr(self.source, 'edge_triangles_indices'):
                    raise AttributeError(
                        f"Source '{self.source.name}' has no 'edge_triangles_indices'. "
                        "Run edge detection first."
                    )
                for edge in edges:
                    if edge not in self.source.edge_triangles_indices:
                        available = list(self.source.edge_triangles_indices.keys())
                        raise KeyError(
                            f"Edge '{edge}' not found in source '{self.source.name}'. "
                            f"Available: {available}"
                        )
                    tri_idx = np.asarray(self.source.edge_triangles_indices[edge])
                    for mode in modes:
                        if mode == 'strikeslip':
                            offset = param_start
                        else:  # dipslip
                            if n_slip_dirs < 2:
                                raise ValueError(
                                    f"Source '{self.source.name}' has no dip-slip "
                                    f"component (slipdir={self.slipdir})."
                                )
                            offset = param_start + n_patches
                        global_idx = tri_idx + offset
                        n_c = len(global_idx)
                        A = np.zeros((n_c, n_total_params))
                        A[np.arange(n_c), global_idx] = 1.0
                        b = np.zeros(n_c)
                        sub_name = f"{cname}_{edge}_{mode}"
                        results.append((sub_name, A, b))

        return results


class PressureAdapter(SourceAdapter):
    """Adapter for CSI Pressure objects (Mogi, Yang, CDM, pCDM)."""

    @property
    def source_type(self):
        return 'Pressure'

    # ── Parameter description ──────────────────────────────────────────

    def get_param_names(self):
        if self.source.source == 'pCDM':
            return ['pressureDVx', 'pressureDVy', 'pressureDVz']
        return ['pressure']

    def get_n_source_params(self):
        return 3 if self.source.source == 'pCDM' else 1

    def get_n_params_per_component(self):
        if self.source.source == 'pCDM':
            return {'pressureDVx': 1, 'pressureDVy': 1, 'pressureDVz': 1}
        return {'pressure': 1}

    def get_n_spatial_elements(self):
        return 1  # point source

    # ── GF construction ────────────────────────────────────────────────

    def build_gfs(self, data, vertical, method='homogeneous', options=None,
                  verbose=None, convergence=None):
        self.source.buildGFs(data, vertical=vertical,
                             method=method, verbose=verbose or False)

    def get_gf_column_keys(self):
        return self.get_param_names()

    # ── Result distribution ────────────────────────────────────────────

    def distribute_results(self, mpost_segment):
        src = self.source
        if src.source in ('Mogi', 'Yang'):
            src.deltapressure = float(mpost_segment[0])
        elif src.source == 'pCDM':
            src.DVx = float(mpost_segment[0])
            src.DVy = float(mpost_segment[1])
            src.DVz = float(mpost_segment[2])
            if src.DVtot is None:
                src.computeTotalpotency()
        elif src.source == 'CDM':
            src.deltaopening = float(mpost_segment[0])

    # ── Initial assembly helpers ───────────────────────────────────────

    def initialize_solution(self):
        self.source.initializepressure()

    def assemble_gfs(self, datas, polys=None, verbose=False, custom=False):
        self.source.assembleGFs(datas, polys=polys, verbose=verbose, custom=custom)

    def assemble_cd(self, datas, verbose=False, add_prediction=None):
        self.source.assembleCd(datas, add_prediction=add_prediction, verbose=verbose)

    def infer_gf_method(self):
        return 'homogeneous'

    # ── Constraint capability ──────────────────────────────────────────

    def get_supported_constraints(self):
        if self.source.source == 'pCDM':
            return ['positive_pressure', 'volume_sign']
        return ['positive_pressure']

    def generate_source_inequality_constraints(self, constraint_cfg, param_start, n_total_params):
        results = []
        param_names = self.get_param_names()
        n_params = self.get_n_source_params()

        for cname, cparams in constraint_cfg.items():
            ctype = cparams.get('type', 'inequality')
            if ctype != 'inequality':
                continue

            rule = cparams.get('rule', '')
            rule_lower = rule.lower().replace(' ', '')

            # positive_pressure: pressure >= 0  →  -pressure <= 0
            if rule_lower == 'pressure>=0':
                A = np.zeros((n_params, n_total_params))
                b = np.zeros(n_params)
                for i in range(n_params):
                    A[i, param_start + i] = -1.0
                results.append((cname, A, b))

            # negative_pressure: pressure <= 0
            elif rule_lower == 'pressure<=0':
                A = np.zeros((n_params, n_total_params))
                b = np.zeros(n_params)
                for i in range(n_params):
                    A[i, param_start + i] = 1.0
                results.append((cname, A, b))

            # volume_positive (pCDM): DVx + DVy + DVz >= 0
            elif rule_lower in ('dvx+dvy+dvz>=0', 'volume>=0') and self.source.source == 'pCDM':
                A = np.zeros((1, n_total_params))
                b = np.zeros(1)
                for i in range(n_params):
                    A[0, param_start + i] = -1.0
                results.append((cname, A, b))

            # Per-component: e.g., pressuredvx>=0
            else:
                for idx, pname in enumerate(param_names):
                    if rule_lower == f'{pname.lower()}>=0':
                        A = np.zeros((1, n_total_params))
                        b = np.zeros(1)
                        A[0, param_start + idx] = -1.0
                        results.append((cname, A, b))
                        break
                    elif rule_lower == f'{pname.lower()}<=0':
                        A = np.zeros((1, n_total_params))
                        b = np.zeros(1)
                        A[0, param_start + idx] = 1.0
                        results.append((cname, A, b))
                        break

        return results

    def generate_source_equality_constraints(self, constraint_cfg, param_start, n_total_params):
        results = []
        param_names = self.get_param_names()

        for cname, cparams in constraint_cfg.items():
            ctype = cparams.get('type', 'inequality')
            if ctype != 'equality':
                continue

            rule = cparams.get('rule', '')
            rule_lower = rule.lower().replace(' ', '')

            # Per-component: e.g., pressure==0
            for idx, pname in enumerate(param_names):
                if rule_lower == f'{pname.lower()}==0':
                    A = np.zeros((1, n_total_params))
                    b = np.zeros(1)
                    A[0, param_start + idx] = 1.0
                    results.append((cname, A, b))
                    break

        return results


class SbarbotAdapter(SourceAdapter):
    """Adapter for CSI Sbarbot objects (strain volumes)."""

    _DEFAULT_STRAIN_COMPONENTS = ['eps11', 'eps12', 'eps13', 'eps22', 'eps23', 'eps33']

    def __init__(self, source, strain_components=None):
        super().__init__(source)
        self._strain_components = (
            list(strain_components) if strain_components is not None
            else getattr(source, 'strain_components', None)
            or list(self._DEFAULT_STRAIN_COMPONENTS)
        )

    @property
    def source_type(self):
        return 'Sbarbot'

    # ── strain_components helpers ──────────────────────────────────────

    @property
    def strain_components(self):
        """Strain component list.

        After ``buildGFs()`` the source stores ``strain_components``;
        that value takes precedence so the adapter stays in sync with
        the assembled GF matrix.  Before that call we fall back to the
        value configured at construction time (default
        ``['eps12', 'eps13']``).
        """
        source_val = getattr(self.source, 'strain_components', None)
        return source_val if source_val is not None else self._strain_components

    @strain_components.setter
    def strain_components(self, value):
        self._strain_components = list(value)

    # ── Parameter description ──────────────────────────────────────────

    def get_param_names(self):
        return list(self.strain_components)

    def get_n_source_params(self):
        return len(self.source.volumes) * len(self.strain_components)

    def get_n_params_per_component(self):
        n = len(self.source.volumes)
        return {comp: n for comp in self.strain_components}

    def get_n_spatial_elements(self):
        return len(self.source.volumes)

    # ── GF construction ────────────────────────────────────────────────

    def build_gfs(self, data, vertical, method='homogeneous', options=None,
                  verbose=None, convergence=None):
        if method not in (None, 'homogeneous', 'volume'):
            warnings.warn(
                f"SbarbotAdapter ignores method='{method}'; "
                f"Sbarbot sources always use their built-in GF computation.",
                UserWarning,
                stacklevel=2,
            )
        self.source.buildGFs(data, vertical=vertical,
                             strain_components=self.strain_components,
                             verbose=verbose or False)

    def get_gf_column_keys(self):
        return list(self.strain_components)

    # ── Result distribution ────────────────────────────────────────────

    def distribute_results(self, mpost_segment):
        n_vol = len(self.source.volumes)
        n_comp = len(self.strain_components)
        self.source.strain = mpost_segment.reshape(n_comp, n_vol).T

    # ── Initial assembly helpers ───────────────────────────────────────

    def initialize_solution(self):
        self.source.initializeStrain()

    def assemble_data(self, datas, verbose=False):
        self.source.assembledata(datas)

    def assemble_gfs(self, datas, polys=None, verbose=False, custom=False):
        self.source.assembleGFs(datas, polys=polys,
                                strain_components=self.strain_components,
                                verbose=verbose, custom=custom)

    def assemble_cd(self, datas, verbose=False, add_prediction=None):
        self.source.assembleCd(datas, verbose=verbose)

    # ── Constraint capability ──────────────────────────────────────────

    def get_supported_constraints(self):
        return ['strain_sign', 'incompressible']

    def generate_source_inequality_constraints(self, constraint_cfg, param_start, n_total_params):
        results = []
        n_vol = self.get_n_spatial_elements()
        comp_names = self.get_param_names()
        params_per_comp = self.get_n_params_per_component()

        for cname, cparams in constraint_cfg.items():
            ctype = cparams.get('type', 'inequality')
            if ctype != 'inequality':
                continue

            rule = cparams.get('rule', '')
            rule_lower = rule.lower().replace(' ', '')

            # Per-component sign constraints: e.g., eps12>=0, eps13<=0
            offset = param_start
            for comp in comp_names:
                n = params_per_comp[comp]
                if rule_lower == f'{comp.lower()}>=0':
                    A = np.zeros((n, n_total_params))
                    b = np.zeros(n)
                    A[np.arange(n), offset + np.arange(n)] = -1.0
                    results.append((cname, A, b))
                    break
                elif rule_lower == f'{comp.lower()}<=0':
                    A = np.zeros((n, n_total_params))
                    b = np.zeros(n)
                    A[np.arange(n), offset + np.arange(n)] = 1.0
                    results.append((cname, A, b))
                    break
                offset += n

        return results

    def generate_source_equality_constraints(self, constraint_cfg, param_start, n_total_params):
        results = []
        comp_names = self.get_param_names()
        params_per_comp = self.get_n_params_per_component()
        n_vol = self.get_n_spatial_elements()

        for cname, cparams in constraint_cfg.items():
            ctype = cparams.get('type', 'inequality')
            if ctype != 'equality':
                continue

            rule = cparams.get('rule', '')
            rule_lower = rule.lower().replace(' ', '')

            # ── Incompressibility: eps11 + eps22 + eps33 = 0 (per volume) ──
            if rule_lower == 'incompressible':
                required = ['eps11', 'eps22', 'eps33']
                missing = [r for r in required if r not in comp_names]
                if missing:
                    raise ValueError(
                        f"Incompressibility requires {required}, "
                        f"but strain_components={comp_names} is missing {missing}"
                    )
                # Compute column offsets for required components
                offsets = {}
                off = param_start
                for comp in comp_names:
                    n = params_per_comp[comp]
                    if comp in required:
                        offsets[comp] = off
                    off += n
                A = np.zeros((n_vol, n_total_params))
                b = np.zeros(n_vol)
                for i in range(n_vol):
                    A[i, offsets['eps11'] + i] = 1.0
                    A[i, offsets['eps22'] + i] = 1.0
                    A[i, offsets['eps33'] + i] = 1.0
                results.append((cname, A, b))
                continue

            # ── Per-component zero: e.g. eps12 == 0 ──
            offset = param_start
            for comp in comp_names:
                n = params_per_comp[comp]
                if rule_lower == f'{comp.lower()}==0':
                    A = np.zeros((n, n_total_params))
                    b = np.zeros(n)
                    A[np.arange(n), offset + np.arange(n)] = 1.0
                    results.append((cname, A, b))
                    break
                offset += n

        return results


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════

_ADAPTER_MAP = {
    'Fault': FaultAdapter,
    'Pressure': PressureAdapter,
    'Sbarbot': SbarbotAdapter,
}


def make_adapter(source, **kwargs):
    """Create the correct adapter for a CSI source object.

    Parameters
    ----------
    source : object
        A CSI source with a ``.type`` attribute ('Fault', 'Pressure', 'Sbarbot').
    **kwargs
        Forwarded to the adapter constructor.

        - ``FaultAdapter`` accepts ``slipdir`` (default ``'sd'``).
        - ``SbarbotAdapter`` accepts ``strain_components`` (default: all 6).

    Returns
    -------
    SourceAdapter
    """
    cls = _ADAPTER_MAP.get(source.type)
    if cls is None:
        raise ValueError(
            f"No adapter registered for source type '{source.type}'. "
            f"Known types: {list(_ADAPTER_MAP.keys())}"
        )
    return cls(source, **kwargs)
