"""Compile named block Euler definitions into equality rows.

Physical vectors and column resolution belong to interseismic_parameter_model;
this module only assembles equalities in the caller's active linear space.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple
import numpy as np

from ..interseismic_parameter_model import (
    get_fault_loading_config, get_blocks_config, resolve_transform_columns,
    _block_datasets, _block_euler_mode, _block_anchor_dataset,
    _block_fixed_euler_vector, _physical_euler_to_model_units,
)


def generate_block_euler_equality_constraints(
    inversion: Any,
    interseismic_config: Mapping[str, Any],
    n_total: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Generate equality rows tying dataset Euler vectors to block Euler.

    For ``mode: estimate``, every dataset in a block shares one estimated
    ``eulerrotation`` vector.  For ``mode: fixed_pole`` or ``mode:
    fixed_vector``, any listed dataset that has an ``eulerrotation`` transform
    is fixed to the block Euler vector.
    """
    loading = get_fault_loading_config(interseismic_config)
    blocks = get_blocks_config(interseismic_config)
    if not loading.get("enabled", False) or not blocks.get("enabled", False):
        return None, None

    rows = []
    rhs = []
    for block_name, block in blocks.get("items", {}).items():
        datasets = _block_datasets(block)
        if not datasets:
            continue
        euler_config = block.get("euler", {})
        mode = _block_euler_mode(block)
        owner_source = euler_config.get("owner_source")

        if mode == "estimate":
            if len(datasets) <= 1:
                continue
            anchor_dataset = _block_anchor_dataset(block)
            ref_columns, _ = resolve_transform_columns(
                inversion,
                anchor_dataset,
                owner_source=owner_source,
                n_total=n_total,
            )
            if ref_columns.size != 3:
                raise ValueError(f"Expected 3 Euler parameters for block '{block_name}' anchor dataset")
            for dataset in datasets:
                if dataset == anchor_dataset:
                    continue
                columns, _ = resolve_transform_columns(
                    inversion,
                    dataset,
                    owner_source=owner_source,
                    n_total=n_total,
                )
                if columns.size != 3:
                    raise ValueError(f"Expected 3 Euler parameters for block '{block_name}' dataset '{dataset}'")
                for local_idx in range(3):
                    row = np.zeros(int(n_total), dtype=float)
                    row[columns[local_idx]] = 1.0
                    row[ref_columns[local_idx]] = -1.0
                    rows.append(row)
                    rhs.append(0.0)
        elif mode in {"fixed_pole", "fixed_vector"}:
            fixed_vector = _physical_euler_to_model_units(inversion, _block_fixed_euler_vector(block))
            for dataset in datasets:
                try:
                    columns, _ = resolve_transform_columns(
                        inversion,
                        dataset,
                        owner_source=owner_source,
                        n_total=n_total,
                    )
                except ValueError:
                    # A fixed block may list datasets whose rigid Euler motion
                    # has already been removed before inversion.  Preflight
                    # diagnostics report that case; constraint generation can
                    # safely skip missing eulerrotation columns.
                    continue
                if columns.size != 3:
                    raise ValueError(f"Expected 3 Euler parameters for block '{block_name}' dataset '{dataset}'")
                for local_idx in range(3):
                    row = np.zeros(int(n_total), dtype=float)
                    row[columns[local_idx]] = 1.0
                    rows.append(row)
                    rhs.append(float(fixed_vector[local_idx]))
        else:
            raise ValueError(f"Unsupported block Euler mode '{mode}'")

    if not rows:
        return None, None
    return np.vstack(rows), np.asarray(rhs, dtype=float)
