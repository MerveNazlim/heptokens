"""Schema helpers for continuous event-sequence baselines."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


CONTINUOUS_SCHEMA_METADATA_KEY = b"heptokens_continuous_schema"

ROLE_IDS = {
    "padding": 0,
    "cls": 1,
    "event": 2,
    "object": 3,
    "separator": 4,
}

# Semantic groups follow the feature-group organization used in the original
# BNL hierarchical baseline. Values are H5 feature basenames, not column indices,
# so the schema remains correct if a datamodule reorders its inputs.
SEMANTIC_FEATURE_GROUPS = {
    "jets": {
        "kinematics": ("pt", "eta", "phi", "mass"),
        "substructure": ("n_trk", "QG_nTracks", "QG_tracksWidth", "QG_tracksC1"),
        "btagging": (
            "DL1d_pb",
            "DL1d_pc",
            "DL1d_pu",
            "GN2_pb",
            "GN2_pc",
            "GN2_pu",
        ),
    },
    "electrons": {
        "kinematics": ("pt", "eta", "phi", "charge"),
        "id": ("LHMedium", "LHTight"),
        "isolation": ("ptvarcone30", "topoetcone20"),
    },
    "muons": {
        "kinematics": ("pt", "eta", "phi", "charge"),
        "isolation": ("trk_iso03", "ptvarcone30", "topoetcone20"),
    },
    "taus": {
        "kinematics": ("pt", "eta", "phi", "charge"),
        "id": ("NNDecayMode",),
        "rnn": ("RNNJetScore", "RNNEleScore"),
    },
    "photons": {
        "kinematics": ("pt", "eta", "phi"),
        "id": ("isTight",),
        "isolation": ("trk_iso03", "ptcone20", "topoetcone20", "topoetcone40"),
    },
    "tracks": {
        "kinematics": ("pt", "eta", "phi", "qOverP"),
        "impact": ("d0", "z0"),
        "quality": ("chiSquared", "nDoF"),
    },
}


# Opt-in grouping for new hierarchical models. The original mapping above is
# retained for existing Parquet metadata, checkpoints and seed replications.
# Only assignments change: no feature is added, removed, reordered or rescaled.
SLIDE_KINEMATICS_FEATURE_GROUPS = {
    "jets": dict(SEMANTIC_FEATURE_GROUPS["jets"]),
    "electrons": {
        "kinematics": ("pt", "eta", "phi"),
        "categorical": ("charge", "LHMedium", "LHTight"),
        "isolation": SEMANTIC_FEATURE_GROUPS["electrons"]["isolation"],
    },
    "muons": {
        "kinematics": ("pt", "eta", "phi"),
        "charge": ("charge",),
        "isolation": SEMANTIC_FEATURE_GROUPS["muons"]["isolation"],
    },
    "taus": {
        "kinematics": ("pt", "eta", "phi"),
        "categorical": ("charge", "NNDecayMode"),
        "rnn": SEMANTIC_FEATURE_GROUPS["taus"]["rnn"],
    },
    "photons": dict(SEMANTIC_FEATURE_GROUPS["photons"]),
    "tracks": {
        "kinematics": ("pt", "eta", "phi"),
        "impact": ("d0", "z0"),
        "tracking": ("qOverP", "chiSquared", "nDoF"),
    },
}


# User-selected revision: only jets and tracks retain three groups. All other
# species combine their non-kinematic inputs into one auxiliary sub-token.
KINEMATICS_AUXILIARY_FEATURE_GROUPS = {
    object_name: (
        dict(groups)
        if object_name in {"jets", "tracks"}
        else {
            "kinematics": groups["kinematics"],
            "auxiliary": tuple(
                feature
                for group_name, features in groups.items()
                if group_name != "kinematics"
                for feature in features
            ),
        }
    )
    for object_name, groups in SLIDE_KINEMATICS_FEATURE_GROUPS.items()
}

CONTINUOUS_FEATURE_GROUPINGS = {
    "slide_kinematics_v1": SLIDE_KINEMATICS_FEATURE_GROUPS,
    "kinematics_auxiliary_v1": KINEMATICS_AUXILIARY_FEATURE_GROUPS,
}


def regroup_continuous_schema(schema: Mapping, *, grouping: str = "schema") -> dict:
    """Select hierarchical groups without modifying an existing dataset schema.

    ``schema`` preserves the stored groups, including those used by old models.
    The named presets reassign existing feature indices for a new model, so they
    can use the original continuous Parquets without re-exporting or changing
    preprocessing. ``kinematics_auxiliary_v1`` uses two groups except for jets
    and tracks, which retain three. The resolved mapping is saved by the model.
    """
    if grouping != "schema" and grouping not in CONTINUOUS_FEATURE_GROUPINGS:
        raise ValueError(f"Unknown continuous feature grouping: {grouping!r}")
    resolved = deepcopy(dict(schema))
    if grouping == "schema":
        return resolved
    validate_continuous_schema(resolved)
    for object_name, spec in resolved["objects"].items():
        requested_groups = CONTINUOUS_FEATURE_GROUPINGS[grouping].get(object_name)
        if requested_groups is None:
            raise ValueError(f"No {grouping} groups configured for {object_name!r}")
        feature_names = spec.get("feature_names")
        if not feature_names or len(feature_names) != int(spec["feature_count"]):
            raise ValueError(f"Complete feature_names are required to regroup {object_name}")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError(f"Duplicate feature names for {object_name}")
        indices_by_name = {name: index for index, name in enumerate(feature_names)}
        groups = {
            group_name: [indices_by_name[name] for name in names if name in indices_by_name]
            for group_name, names in requested_groups.items()
        }
        groups = {name: indices for name, indices in groups.items() if indices}
        assigned = [index for indices in groups.values() for index in indices]
        if sorted(assigned) != list(range(len(feature_names))):
            missing = [
                name for index, name in enumerate(feature_names) if index not in assigned
            ]
            raise ValueError(
                f"{grouping} groups must cover every {object_name} feature exactly "
                f"once; unassigned features: {missing}"
            )
        spec["groups"] = groups
    resolved["feature_grouping"] = grouping
    return resolved


def _feature_name(path: str) -> str:
    return str(path).rstrip("/").split("/")[-1]


def build_continuous_schema(
    *,
    collections: Mapping[str, Mapping],
    object_order: list[str],
    event_token_specs: list[dict],
    type_ids: Mapping[str, int],
    include_decoded_q8: bool = False,
) -> dict:
    """Build and validate metadata for paired continuous sequence columns."""
    objects = {}
    max_feature_dim = 1
    for object_name in object_order:
        collection = collections.get(object_name)
        if collection is None:
            continue
        feature_paths = [str(path) for path in collection.get("inputs", [])]
        feature_names = [_feature_name(path) for path in feature_paths]
        if not feature_names:
            raise ValueError(f"Continuous schema has no features for {object_name}")
        group_names = SEMANTIC_FEATURE_GROUPS.get(object_name)
        if group_names is None:
            raise ValueError(
                f"No semantic continuous feature groups are configured for {object_name!r}"
            )
        groups = {}
        assigned = set()
        for group_name, requested_names in group_names.items():
            indices = [
                feature_names.index(name)
                for name in requested_names
                if name in feature_names
            ]
            if indices:
                groups[group_name] = indices
                assigned.update(indices)
        missing = [
            feature_names[index]
            for index in range(len(feature_names))
            if index not in assigned
        ]
        if missing:
            raise ValueError(
                f"Continuous features for {object_name} are not assigned to a semantic "
                f"group: {missing}"
            )
        objects[object_name] = {
            "type_id": int(type_ids[object_name]),
            "feature_paths": feature_paths,
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "groups": groups,
        }
        max_feature_dim = max(max_feature_dim, len(feature_names))

    event_inputs = []
    for index, spec in enumerate(event_token_specs):
        event_inputs.append(
            {
                "index": index,
                "input": str(spec["input"]),
                "name": _feature_name(spec["input"]),
                "range": [float(spec["range"][0]), float(spec["range"][1])],
            }
        )

    feature_columns = {
        "raw": "continuous_features",
        "mask": "continuous_feature_mask",
        "roles": "position_role_ids",
    }
    if include_decoded_q8:
        feature_columns["decoded_q8"] = "decoded_continuous_features"

    return {
        "version": 2 if include_decoded_q8 else 1,
        "layout": "aligned_grouped_event_sequence",
        "preprocessing": {
            "objects": "matching fitted VQ-VAE input transformer",
            "event": "linear scaling to [0, 1] with event-token vocabulary range",
        },
        "max_feature_dim": max_feature_dim,
        "feature_columns": feature_columns,
        "decoded_q8": (
            {
                "objects": "VQ-VAE decoder output from the complete residual-code tuple",
                "event_inputs": "identical to the scaled raw-continuous event inputs",
            }
            if include_decoded_q8
            else None
        ),
        "role_ids": dict(ROLE_IDS),
        "event_inputs": event_inputs,
        "objects": objects,
    }


def validate_continuous_schema(schema: Mapping) -> None:
    """Fail early when a model receives incomplete paired-data metadata."""
    required = {"max_feature_dim", "role_ids", "event_inputs", "objects"}
    missing = required - set(schema)
    if missing:
        raise ValueError(f"Continuous schema is missing keys: {sorted(missing)}")
    if int(schema["max_feature_dim"]) < 1:
        raise ValueError("Continuous max_feature_dim must be positive")
    if not schema["objects"]:
        raise ValueError("Continuous schema contains no object definitions")
    if schema.get("decoded_q8") is not None and schema.get("feature_columns", {}).get(
        "decoded_q8"
    ) != "decoded_continuous_features":
        raise ValueError("Decoded-Q8 schema does not declare its feature column")
