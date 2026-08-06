"""Generate and audit ECAT's one supported conda dependency file.

This tool deliberately does not run ``conda list``, ``pip freeze``, or query a
package index. Those commands describe one developer environment rather than
ECAT. The source of truth is the direct ``install_requires`` metadata in the
two distributable packages:

* csi_cutde_mpiparallel/setup.py
* eqtools/setup.py

Usage:

    python scripts/generate_requirements.py          # rewrite the txt file
    python scripts/generate_requirements.py --check  # verify it is current
    python scripts/generate_requirements.py --audit-only \
        --csi-project PATH --eqtools-project PATH

The import audit is strict in both directions. Known optional or legacy
backends are listed explicitly below; every other import must be declared by
the package that imports it, and every base dependency must be backed by an
import in that same package. The rendered file separates shared, CSI-only, and
eqtools-only dependencies so output order cannot be mistaken for ownership.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SETUP_FILES = {
    "csi": ROOT / "csi_cutde_mpiparallel" / "setup.py",
    "eqtools": ROOT / "eqtools" / "setup.py",
}
SOURCE_ROOTS = {
    "csi": ROOT / "csi_cutde_mpiparallel" / "csi",
    "eqtools": ROOT / "eqtools" / "eqtools",
}
OUTPUT = ROOT / "requirements" / "ecat-requirements.txt"

# okada4py is required by CSI, but it must be supplied as a platform-matched
# wheel before the package install. It can be a release wheel or one built from
# its public source. It is therefore declared in CSI's metadata but not emitted
# into a conda package list.
WHEEL_ONLY = {"okada4py"}
FORBIDDEN = {"pymc", "pymc-base", "pytensor", "pytensor-base", "theano"}

IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "shapefile": "pyshp",
    "mpl_toolkits": "matplotlib",
    "ruamel": "ruamel-yaml",
    "osgeo": "gdal",
}
FIRST_PARTY = {
    "csi",
    "eqtools",
    "Tectonic_Utils",
    "input_adapter",
    "generate_nonlinear_geometry_config",
    "seismo_tools",
    "yangfull",
}
# These imports occur inside explicitly optional, legacy, or external-backend
# methods. They are deliberately excluded from the base ECAT environment.
KNOWN_NON_BASE = {
    "altarexplore",  # legacy AlTar posterior helpers
    "arviz",         # optional posterior diagnostic plotting
    "global-share",  # case-local legacy __main__ demonstration module
    "mayavi",        # obsolete optional 3-D plotting method
    "obspy",         # optional beachball plotting in earthquake clients
    "pyastronomy",   # legacy GPS time-series fitting utility
    "sacpy",         # external kinematic waveform backend
    "sqlalchemy",    # optional GPS SQL-file reader
    "statsmodels",   # legacy GPS outlier helper
    "sympy",         # optional symbolic rheology derivation helper
    "tsinsar",       # optional legacy InSAR time-series backend
}
SKIPPED_DIRECTORIES = {"__pycache__", "build", "docs", "examples", "test", "tests"}


def normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if not match:
        raise ValueError(f"Cannot determine distribution name from {requirement!r}")
    return normalize_name(match.group(1))


def literal_strings(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        raise TypeError(f"Expected a literal sequence, got {type(node).__name__}")
    values: list[str] = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            raise TypeError("Dependency metadata must use literal strings")
        values.append(element.value)
    return values


def read_setup_dependencies(path: Path) -> tuple[list[str], list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    setup_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Name) and node.func.id == "setup")
             or (isinstance(node.func, ast.Attribute) and node.func.attr == "setup"))
    ]
    if len(setup_calls) != 1:
        raise RuntimeError(f"Expected one setup() call in {path}")

    base: list[str] = []
    optional: list[str] = []
    for keyword in setup_calls[0].keywords:
        if keyword.arg == "install_requires":
            base = literal_strings(keyword.value)
        elif keyword.arg == "extras_require":
            if not isinstance(keyword.value, ast.Dict):
                raise TypeError("extras_require must be a literal dictionary")
            for value in keyword.value.values:
                optional.extend(literal_strings(value))
    if not base:
        raise RuntimeError(f"No install_requires metadata found in {path}")
    return base, optional


def unique_requirements(requirements: Iterable[str]) -> list[str]:
    selected: dict[str, str] = {}
    for requirement in requirements:
        name = requirement_name(requirement)
        previous = selected.get(name)
        if previous is not None and previous != requirement:
            raise ValueError(
                f"Conflicting direct requirements for {name}: {previous!r} and {requirement!r}"
            )
        selected[name] = requirement
    return [selected[name] for name in sorted(selected)]


def requirements_by_name(requirements: Iterable[str]) -> dict[str, str]:
    """Return normalized distribution names mapped to requirement strings."""

    return {
        requirement_name(requirement): requirement
        for requirement in unique_requirements(requirements)
    }


def render_requirements(csi_base: list[str], eqtools_base: list[str]) -> str:
    csi = requirements_by_name(
        item for item in csi_base if requirement_name(item) not in WHEEL_ONLY
    )
    eqtools = requirements_by_name(
        item for item in eqtools_base if requirement_name(item) not in WHEEL_ONLY
    )

    shared_names = sorted(set(csi) & set(eqtools))
    for name in shared_names:
        if csi[name] != eqtools[name]:
            raise ValueError(
                f"Conflicting direct requirements for shared dependency {name}: "
                f"CSI={csi[name]!r}, eqtools={eqtools[name]!r}"
            )

    shared = [csi[name] for name in shared_names]
    csi_only = [csi[name] for name in sorted(set(csi) - set(eqtools))]
    eqtools_only = [eqtools[name] for name in sorted(set(eqtools) - set(csi))]

    lines = [
        "# ECAT's supported runtime environment: direct dependencies only.",
        "#",
        "# Create the recommended Python 3.10 environment with:",
        "#   conda create -n ecat -c conda-forge python=3.10 --file requirements/ecat-requirements.txt",
        "# Choose any explicit MKL/MPI profile before environment creation; see Install.md.",
        "#",
        "# This file is generated by scripts/generate_requirements.py from the",
        "# package install_requires metadata. It uses compatibility ranges instead",
        "# of a machine-specific export or complete dependency closure.",
        "#",
        "# IMPORTANT: csi imports okada4py at package import time. Install a",
        "# wheel matching Python 3.10, 3.11, or 3.12 before installing ECAT.",
        "# See Install.md for the release-wheel and source-build routes.",
        "",
        "python>=3.10,<3.13",
        "",
        "# Shared direct runtime dependencies of CSI and eqtools.",
        *shared,
        "",
        "# CSI-only direct runtime dependencies.",
        *csi_only,
        "",
        "# eqtools-only direct runtime dependencies.",
        *eqtools_only,
        "",
    ]
    return "\n".join(lines)


def iter_source_files(source_root: Path) -> Iterable[Path]:
    for path in source_root.rglob("*.py"):
        relative_parts = path.relative_to(source_root).parts
        if any(part in SKIPPED_DIRECTORIES for part in relative_parts):
            continue
        yield path


def scan_imports(
    source_roots: dict[str, Path] | None = None,
) -> tuple[
    dict[str, set[str]],
    dict[str, dict[str, set[str]]],
]:
    source_roots = source_roots or SOURCE_ROOTS
    stdlib = set(getattr(sys, "stdlib_module_names", set())) | set(sys.builtin_module_names)
    distributions: dict[str, set[str]] = {name: set() for name in source_roots}
    locations: dict[str, dict[str, set[str]]] = {name: {} for name in source_roots}
    for package_name, source_root in source_roots.items():
        for path in iter_source_files(source_root):
            relative = (
                Path(source_root.parent.name)
                / source_root.name
                / path.relative_to(source_root)
            ).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            except (OSError, UnicodeDecodeError, SyntaxError) as error:
                print(f"WARNING: cannot parse {relative}: {error}")
                continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".", 1)[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".", 1)[0]]
                for module in names:
                    if module in stdlib or module in FIRST_PARTY:
                        continue
                    distribution = normalize_name(IMPORT_TO_DISTRIBUTION.get(module, module))
                    distributions[package_name].add(distribution)
                    locations[package_name].setdefault(distribution, set()).add(relative)
    return distributions, locations


def audit_imports(
    declared: dict[str, set[str]],
    base_declared: dict[str, set[str]],
    source_roots: dict[str, Path] | None = None,
) -> set[str]:
    source_roots = source_roots or SOURCE_ROOTS
    imported_by_package, locations_by_package = scan_imports(source_roots)
    issues: set[str] = set()

    for package_name in source_roots:
        imported = imported_by_package[package_name]
        locations = locations_by_package[package_name]

        forbidden_imports = imported & FORBIDDEN
        if forbidden_imports:
            print(f"Forbidden probabilistic imports in {package_name}:")
            for name in sorted(forbidden_imports):
                print(f"  {name}: {', '.join(sorted(locations[name]))}")
                issues.add(f"{package_name}:{name}")

        known_non_base = imported & KNOWN_NON_BASE
        if known_non_base:
            print(f"Known optional/legacy imports excluded from {package_name} base:")
            for name in sorted(known_non_base):
                print(f"  {name}: {', '.join(sorted(locations[name]))}")

        uncovered = imported - declared[package_name] - FORBIDDEN - KNOWN_NON_BASE
        if uncovered:
            print(f"Unclassified imports missing from {package_name} metadata:")
            for name in sorted(uncovered):
                print(f"  {name}: {', '.join(sorted(locations[name]))}")
                issues.add(f"{package_name}:{name}")
        else:
            print(
                f"Source-import audit ({package_name}): every classified import "
                "is declared by that package as base or optional metadata."
            )

        overdeclared = base_declared[package_name] - imported
        if overdeclared:
            print(f"Base dependencies declared by {package_name} but not imported there:")
            for name in sorted(overdeclared):
                print(f"  {name}")
                issues.add(f"{package_name}:overdeclared:{name}")
        else:
            print(
                f"Dependency-ownership audit ({package_name}): "
                "every base declaration is imported by that package."
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the generated txt file is stale")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="audit package metadata and imports without reading or writing the ECAT txt file",
    )
    parser.add_argument(
        "--csi-project",
        type=Path,
        help="CSI project root to use with --audit-only (defaults to ECAT's embedded copy)",
    )
    parser.add_argument(
        "--eqtools-project",
        type=Path,
        help="eqtools project root to use with --audit-only (defaults to ECAT's embedded copy)",
    )
    args = parser.parse_args()

    if args.check and args.audit_only:
        parser.error("--check and --audit-only are mutually exclusive")
    if (args.csi_project or args.eqtools_project) and not args.audit_only:
        parser.error("--csi-project and --eqtools-project require --audit-only")

    setup_files = dict(SETUP_FILES)
    source_roots = dict(SOURCE_ROOTS)
    project_overrides = {
        "csi": args.csi_project,
        "eqtools": args.eqtools_project,
    }
    package_directories = {"csi": "csi", "eqtools": "eqtools"}
    for package_name, project_root in project_overrides.items():
        if project_root is None:
            continue
        resolved_root = project_root.resolve()
        setup_files[package_name] = resolved_root / "setup.py"
        source_roots[package_name] = (
            resolved_root / package_directories[package_name]
        )

    csi_base, csi_optional = read_setup_dependencies(setup_files["csi"])
    eqtools_base, eqtools_optional = read_setup_dependencies(setup_files["eqtools"])
    rendered = render_requirements(csi_base, eqtools_base)

    declared_by_package = {
        "csi": {
            requirement_name(item)
            for item in [*csi_base, *csi_optional]
        },
        "eqtools": {
            requirement_name(item)
            for item in [*eqtools_base, *eqtools_optional]
        },
    }
    base_declared_by_package = {
        "csi": {requirement_name(item) for item in csi_base},
        "eqtools": {requirement_name(item) for item in eqtools_base},
    }
    declared = set().union(*declared_by_package.values())
    forbidden_declared = declared & FORBIDDEN
    if forbidden_declared:
        print(f"Forbidden package metadata: {', '.join(sorted(forbidden_declared))}")
    issues = {f"metadata:{name}" for name in forbidden_declared}
    issues |= audit_imports(
        declared_by_package,
        base_declared_by_package,
        source_roots,
    )
    if issues:
        return 1

    if args.audit_only:
        print("OK: dependency metadata ownership audit passed.")
        return 0

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if current != rendered:
            print(f"STALE: {OUTPUT.relative_to(ROOT)} differs from package metadata.")
            print("Run: python scripts/generate_requirements.py")
            return 1
        print(f"OK: {OUTPUT.relative_to(ROOT)} is current.")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
