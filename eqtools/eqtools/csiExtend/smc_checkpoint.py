"""Shared HDF5 checkpoint serialization for SMC backends.

The SMC engines own the sample-state datasets and tempering metadata.  A
scientific caller may additionally provide an opaque sample-layout manifest;
the backend persists it without interpreting fault, geometry, or parameter
semantics.  This keeps MPI checkpoint writing generic while allowing the
Bayesian inversion layer to verify that stored columns still mean the same
thing when a file is loaded later.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import h5py
import numpy as np

from .smc_tempering import write_smc_tempering_metadata


SAMPLE_LAYOUT_SCHEMA_VERSION = 1
SAMPLE_LAYOUT_JSON_ATTR = "smc_sample_layout_json"
SAMPLE_LAYOUT_SHA256_ATTR = "smc_sample_layout_sha256"


def canonical_sample_layout_json(manifest):
    """Return the deterministic JSON representation of one layout manifest."""
    if manifest is None:
        return None
    if not isinstance(manifest, Mapping):
        raise TypeError("sample_layout_manifest must be a mapping or None")
    return json.dumps(
        dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def write_sample_layout_manifest(attrs, manifest):
    """Write an opaque, canonical sample-layout manifest to HDF5 attrs."""
    payload = canonical_sample_layout_json(manifest)
    if payload is None:
        return
    attrs[SAMPLE_LAYOUT_JSON_ATTR] = payload
    attrs[SAMPLE_LAYOUT_SHA256_ATTR] = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def read_sample_layout_manifest(attrs):
    """Read and integrity-check a sample-layout manifest from HDF5 attrs.

    ``None`` identifies a historical checkpoint that predates this contract.
    A malformed or internally inconsistent new-format manifest is rejected.
    """
    if SAMPLE_LAYOUT_JSON_ATTR not in attrs:
        return None
    payload = attrs[SAMPLE_LAYOUT_JSON_ATTR]
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    payload = str(payload)
    expected_hash = attrs.get(SAMPLE_LAYOUT_SHA256_ATTR)
    if isinstance(expected_hash, bytes):
        expected_hash = expected_hash.decode("ascii")
    actual_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if expected_hash is None or str(expected_hash) != actual_hash:
        raise ValueError(
            "SMC checkpoint sample-layout metadata failed its SHA-256 "
            "integrity check."
        )
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "SMC checkpoint sample-layout metadata is not valid JSON."
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("SMC checkpoint sample-layout manifest must be an object.")
    return manifest


def _write_h5_value(group, key, value):
    """Write one sample-state value, including nested diagnostic mappings."""
    if value is None:
        return
    if isinstance(value, Mapping):
        subgroup = group.create_group(str(key))
        for subkey, subvalue in value.items():
            _write_h5_value(subgroup, subkey, subvalue)
        return
    array = np.asarray(value)
    if array.dtype.kind in {"U", "O"}:
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(
            str(key),
            data=np.asarray(value, dtype=dtype),
            dtype=dtype,
        )
        return
    group.create_dataset(str(key), data=value)


def write_smc_checkpoint(
    filename,
    samples,
    *,
    tempering_policy,
    sample_layout_manifest=None,
):
    """Write one SMC state using the common checkpoint contract.

    State fields whose value is ``None`` are intentionally absent. Readers
    must restore optional fields as ``None`` while continuing to require the
    core sample, posterior, beta, and stage datasets.
    """
    with h5py.File(filename, "w") as handle:
        for key, value in samples._asdict().items():
            _write_h5_value(handle, key, value)
        write_smc_tempering_metadata(handle.attrs, tempering_policy)
        write_sample_layout_manifest(handle.attrs, sample_layout_manifest)


__all__ = [
    "SAMPLE_LAYOUT_SCHEMA_VERSION",
    "canonical_sample_layout_json",
    "read_sample_layout_manifest",
    "write_sample_layout_manifest",
    "write_smc_checkpoint",
]
