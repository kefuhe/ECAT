"""Backend selection helpers for optional compute engines.

The cutde backend is selected when ``cutde.backend`` is imported. Entry
points must call :func:`configure_cutde_backend` before importing CSI modules
that may import cutde.
"""

from __future__ import annotations

import os
import sys
from typing import Mapping, MutableMapping


CUTDE_BACKEND_CHOICES = ("cpp", "cuda", "opencl", "auto")


def normalize_cutde_backend(value: str | None, default: str = "cpp") -> str:
    backend = default if value is None else str(value)
    backend = backend.strip().replace("-", "_").lower()
    if backend not in CUTDE_BACKEND_CHOICES:
        raise ValueError(
            "cutde_backend must be one of "
            f"{CUTDE_BACKEND_CHOICES}; got {value!r}."
        )
    return backend


def get_active_cutde_backend(import_backend: bool = False) -> str | None:
    module = sys.modules.get("cutde.backend")
    if module is None and import_backend:
        import cutde.backend as module  # noqa: WPS433
    if module is None:
        return None
    return getattr(module, "which_backend", None)


def configure_cutde_backend(
    requested: str | None,
    *,
    default: str = "cpp",
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str | None]:
    """Configure the cutde backend before cutde is imported.

    ``auto`` intentionally leaves ``CUTDE_USE_BACKEND`` unchanged so advanced
    users can rely on cutde's normal environment/default backend selection.
    Explicit values (``cpp``, ``cuda``, ``opencl``) set the environment and
    fail if cutde has already been imported with a different backend.
    """

    env = os.environ if env is None else env
    backend = normalize_cutde_backend(requested, default=default)
    active = get_active_cutde_backend(import_backend=False)

    if active is not None and backend != "auto" and active != backend:
        raise RuntimeError(
            "cutde backend was already initialized as "
            f"{active!r}, but the configuration requested {backend!r}. "
            "Configure the backend before importing cutde/CSI modules, or "
            "start a new Python process."
        )

    if backend != "auto":
        env["CUTDE_USE_BACKEND"] = backend

    return {
        "requested_backend": backend,
        "environment_backend": env.get("CUTDE_USE_BACKEND"),
        "active_backend": active,
    }


def cutde_backend_summary(
    requested: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    env = os.environ if env is None else env
    summary = {
        "requested_backend": requested,
        "environment_backend": env.get("CUTDE_USE_BACKEND"),
        "active_backend": get_active_cutde_backend(import_backend=False),
    }
    return summary
