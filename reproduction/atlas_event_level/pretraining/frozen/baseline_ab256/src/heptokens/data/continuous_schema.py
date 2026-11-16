"""Schema helpers for continuous event-sequence baselines."""

from __future__ import annotations

from collections.abc import Mapping


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
