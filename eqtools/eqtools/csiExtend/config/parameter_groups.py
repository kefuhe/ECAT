"""Shared cardinality contract for sigma and alpha parameter groups.

The inversion stack works in three related spaces: physical members (data
sets or smoothing-capable sources), parameter groups, and the subset of groups
that are sampled or iteratively updated.  Keeping those mappings together
prevents callers from inferring a group count from the number of members.
"""

from collections.abc import Mapping

import numpy as np


_GROUP_MODES = {"single", "individual", "grouped"}


def resolve_group_layout(
    member_names,
    mode,
    groups=None,
    *,
    member_label="member",
    single_group_name="all",
    individual_prefix="group_",
):
    """Resolve member-to-group membership without assigning parameter values.

    Parameters
    ----------
    member_names : sequence of str
        Ordered physical members.  Their order defines the expansion back to
        data-set or source space.
    mode : {"single", "individual", "grouped"}
        Grouping rule.
    groups : mapping, optional
        Named group-to-member mapping required by ``grouped`` mode.
    member_label : str, optional
        Noun used in validation errors.
    single_group_name : str, optional
        Stable internal name used for the shared group.
    individual_prefix : str, optional
        Prefix for deterministic individual group names.

    Returns
    -------
    dict
        A small, serializable layout plus ``member_param_indices`` for ordered
        numerical expansion.  No values or update flags are inferred here.
    """

    members = list(member_names)
    if len(members) != len(set(members)):
        raise ValueError(f"{member_label} names must be unique")
    if any(not isinstance(name, str) or not name for name in members):
        raise ValueError(f"Every {member_label} name must be a non-empty string")

    mode = str(mode).lower()
    if mode not in _GROUP_MODES:
        allowed = ", ".join(sorted(_GROUP_MODES))
        raise ValueError(f"Unknown grouping mode '{mode}'. Expected one of: {allowed}")

    if mode == "single":
        members_by_group = {str(single_group_name): members.copy()}
    elif mode == "individual":
        members_by_group = {
            f"{individual_prefix}{member}": [member]
            for member in members
        }
    else:
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError(
                "groups parameter must be provided and cannot be empty in grouped mode"
            )
        members_by_group = {}
        for raw_name, raw_members in groups.items():
            group_name = str(raw_name)
            if not group_name:
                raise ValueError("Group names must be non-empty strings")
            if group_name in members_by_group:
                raise ValueError(f"Duplicate group name '{group_name}'")
            if isinstance(raw_members, str) or not isinstance(
                raw_members, (list, tuple)
            ):
                raise ValueError(
                    f"Group '{group_name}' must contain a list of {member_label} names"
                )
            if not raw_members:
                raise ValueError(f"Group '{group_name}' cannot be empty")
            members_by_group[group_name] = list(raw_members)

    known = set(members)
    member_to_group = {}
    for group_name, group_members in members_by_group.items():
        for member in group_members:
            if member not in known:
                raise ValueError(
                    f"Unknown {member_label} '{member}' in group '{group_name}'"
                )
            if member in member_to_group:
                raise ValueError(
                    f"{member_label.capitalize()} '{member}' is assigned to multiple groups"
                )
            member_to_group[member] = group_name

    missing = [member for member in members if member not in member_to_group]
    if missing:
        raise ValueError(
            f"{member_label.capitalize()}s not assigned to any group: "
            + ", ".join(missing)
        )

    group_names = list(members_by_group)
    group_index = {name: index for index, name in enumerate(group_names)}
    member_param_indices = np.asarray(
        [group_index[member_to_group[member]] for member in members],
        dtype=int,
    )
    return {
        "mode": mode,
        "member_names": members,
        "group_names": group_names,
        "members_by_group": members_by_group,
        "member_to_group": member_to_group,
        "member_param_indices": member_param_indices,
        "total_params": len(group_names),
    }


def normalize_group_vector(
    value,
    group_names,
    *,
    value_name,
    default_value,
    dtype=float,
    key_aliases=None,
):
    """Normalize one scalar/list/dict value per resolved parameter group.

    A scalar is the only implicit broadcast.  Sequence and dictionary inputs
    must describe every group exactly, so a misspelled or omitted key cannot
    silently acquire a scientifically meaningful default.
    """

    names = list(group_names)
    if value is None:
        value = default_value

    if isinstance(value, Mapping):
        aliases = {} if key_aliases is None else dict(key_aliases)
        accepted = {}
        for group_name in names:
            keys = [group_name]
            alias = aliases.get(group_name)
            if alias is not None and alias != group_name:
                keys.append(alias)
            matches = [key for key in keys if key in value]
            if len(matches) > 1:
                raise ValueError(
                    f"{value_name} provides both canonical and alias keys for "
                    f"group '{group_name}'"
                )
            if matches:
                accepted[group_name] = matches[0]

        missing = [name for name in names if name not in accepted]
        known_keys = {key for keys in ([name, aliases.get(name)] for name in names)
                      for key in keys if key is not None}
        unknown = [key for key in value if key not in known_keys]
        if missing or unknown:
            details = []
            if missing:
                details.append("missing groups: " + ", ".join(missing))
            if unknown:
                details.append("unknown keys: " + ", ".join(map(str, unknown)))
            raise ValueError(f"Invalid {value_name} mapping ({'; '.join(details)})")
        raw = [value[accepted[name]] for name in names]
    elif np.isscalar(value):
        raw = [value] * len(names)
    else:
        array = np.asarray(value)
        if array.ndim != 1 or array.size != len(names):
            raise ValueError(
                f"{value_name} must contain exactly {len(names)} value(s), "
                f"one per parameter group"
            )
        raw = array.tolist()

    try:
        result = np.asarray(raw, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{value_name} contains values of the wrong type") from exc
    if result.ndim != 1 or result.size != len(names):
        raise ValueError(
            f"{value_name} must contain exactly {len(names)} value(s), "
            "one per parameter group"
        )
    return result


def attach_group_parameters(
    layout,
    *,
    values,
    update,
    value_name,
    default_value,
    value_key_aliases=None,
):
    """Attach group values and sampled/update positions to a resolved layout."""

    group_names = layout["group_names"]
    values_by_group = normalize_group_vector(
        values,
        group_names,
        value_name=value_name,
        default_value=default_value,
        dtype=float,
        key_aliases=value_key_aliases,
    )
    update_by_group = normalize_group_vector(
        update,
        group_names,
        value_name="update",
        default_value=True,
        dtype=bool,
    )

    next_sample = 0
    sample_index_by_group = []
    for should_update in update_by_group:
        if should_update:
            sample_index_by_group.append(next_sample)
            next_sample += 1
        else:
            sample_index_by_group.append(-1)

    return {
        **layout,
        "values_by_group": values_by_group,
        "update_by_group": update_by_group,
        "sample_index_by_group": np.asarray(sample_index_by_group, dtype=int),
        "updatable_params": next_sample,
    }
