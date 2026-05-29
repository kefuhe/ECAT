import warnings
from dataclasses import dataclass


@dataclass
class PscmpOptions:
    """PSCMP Green's function configuration.

    Pass as ``options=`` to ``buildGFs(method='pscmp', options=...)``.
    Use ``PscmpOptions.describe_options()`` to see all fields.
    """
    grn_dir: str = "psgrnfcts"
    output_dir: str = "pscmpgrns"
    workdir: str = "pscmp_ecat"
    n_jobs: int = 4
    cleanup_inp: bool = True
    force_recompute: bool = True

    _FIELD_DESCRIPTIONS = {
        'grn_dir':          ('str',  'psgrnfcts', "PSGRN Green's function directory (relative to workdir)"),
        'output_dir':       ('str',  'pscmpgrns', 'PSCMP output directory (relative to workdir)'),
        'workdir':          ('str',  'pscmp_ecat','Working directory for all intermediate files'),
        'n_jobs':           ('int',  '4',         'Number of parallel workers'),
        'cleanup_inp':      ('bool', 'True',      'Remove intermediate .inp files after computation'),
        'force_recompute':  ('bool', 'True',      'Recompute even if output files exist'),
    }

    def __post_init__(self):
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Build from a dict.  Only canonical field names are accepted."""
        valid = set(cls.__dataclass_fields__)
        unknown = set(kwargs) - valid
        if unknown:
            warnings.warn(
                f"Unknown PscmpOptions keys ignored: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}",
                UserWarning, stacklevel=2,
            )
        return cls(**{k: v for k, v in kwargs.items() if k in valid})

    @classmethod
    def to_commented_map(cls, instance=None):
        """Return a ruamel.yaml CommentedMap with inline comments."""
        from ruamel.yaml.comments import CommentedMap
        obj = instance or cls()
        cm = CommentedMap()
        for name in cls.__dataclass_fields__:
            cm[name] = getattr(obj, name)
            desc = cls._FIELD_DESCRIPTIONS.get(name)
            if desc:
                cm.yaml_add_eol_comment(desc[2], name)
        return cm

    @classmethod
    def describe_yaml(cls):
        """Return a YAML string showing all options with inline comments."""
        from ruamel.yaml import YAML
        from io import StringIO
        y = YAML()
        y.indent(mapping=2, sequence=4, offset=2)
        buf = StringIO()
        y.dump(cls.to_commented_map(), buf)
        return buf.getvalue()

    @classmethod
    def describe_options(cls):
        """Print a human-readable summary of all available PSCMP options."""
        lines = ["PscmpOptions — PSCMP Green's function configuration", "=" * 55]
        for name in cls.__dataclass_fields__:
            desc = cls._FIELD_DESCRIPTIONS.get(name, ('', '', ''))
            lines.append(f"  {name:.<30s} type={desc[0]}, default={desc[1]}")
            if desc[2]:
                lines.append(f"    {desc[2]}")
        lines.append("")
        lines.append("Pass as: fault.buildGFs(data, method='pscmp', options=PscmpOptions(...))")
        lines.append("")
        lines.append("YAML example:")
        lines.append("  options:")
        for line in cls.describe_yaml().splitlines():
            lines.append(f"    {line}")
        print("\n".join(lines))
