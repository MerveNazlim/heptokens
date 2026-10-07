"""Export one event-sequence position per object, preserving residual codes."""

from __future__ import annotations

import json
import sys

import numpy as np

import benchmark_tokenize_objects_to_parquet_with_atlasopenmagic_metadata as flat_export
from heptokens.data.continuous_schema import ROLE_IDS

FLAT_MAKE_TABLE = flat_export.make_table
FLAT_BUILD_VOCABULARY = flat_export.build_vocabulary


def build_grouped_vocabulary(models, event_token_inputs, event_range_map, args) -> dict:
    vocabulary = FLAT_BUILD_VOCABULARY(
        models,
        event_token_inputs,
        event_range_map,
        args,
    )
    vocabulary["sequence_layout"] = "one_position_per_object"
    vocabulary["includes_cls"] = not args.no_cls
    vocabulary["max_quantizers"] = max(
        [1]
        + [int(spec["num_quantizers"]) for spec in vocabulary["objects"].values()]
    )
    return vocabulary


def assemble_grouped_rows(
    encoded_objects: dict,
    event_tokens: np.ndarray | None,
    batch_size: int,
    vocabulary: dict,
    args,
    event_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]
]:
    """Build [event, position, quantizer] token arrays."""
    max_quantizers = int(vocabulary["max_quantizers"])
    tokens = np.full(
        (batch_size, args.max_seq_length, max_quantizers),
        args.pad_token_id,
        dtype=np.int64,
    )
    mask = np.zeros((batch_size, args.max_seq_length), dtype=bool)
    type_ids = np.zeros((batch_size, args.max_seq_length), dtype=np.int64)
    continuous_schema = vocabulary.get("continuous_schema")
    write_continuous = continuous_schema is not None
    write_decoded_q8 = bool(
        write_continuous
        and continuous_schema.get("feature_columns", {}).get("decoded_q8")
    )
    if write_continuous:
        max_feature_dim = int(continuous_schema["max_feature_dim"])
        continuous_features = np.zeros(
            (batch_size, args.max_seq_length, max_feature_dim), dtype=np.float32
        )
        continuous_feature_mask = np.zeros_like(continuous_features, dtype=bool)
        if write_decoded_q8:
            decoded_continuous_features = np.zeros_like(continuous_features)
        position_role_ids = np.full(
            (batch_size, args.max_seq_length), ROLE_IDS["padding"], dtype=np.int64
        )

    for event_idx in range(batch_size):
        position = 0

        def add_position(
            codes: list[int],
            type_id: int,
            *,
            role_id: int,
            features: np.ndarray | None = None,
            decoded_features: np.ndarray | None = None,
        ) -> None:
            nonlocal position
            if position >= args.max_seq_length:
                return
            tokens[event_idx, position, : len(codes)] = codes
            mask[event_idx, position] = True
            type_ids[event_idx, position] = type_id
            if write_continuous:
                position_role_ids[event_idx, position] = role_id
                if features is not None:
                    values = np.asarray(features, dtype=np.float32).reshape(-1)
                    count = min(len(values), max_feature_dim)
                    continuous_features[event_idx, position, :count] = values[:count]
                    continuous_feature_mask[event_idx, position, :count] = True
                if write_decoded_q8 and decoded_features is not None:
                    decoded_values = np.asarray(
                        decoded_features, dtype=np.float32
                    ).reshape(-1)
                    decoded_count = min(len(decoded_values), max_feature_dim)
                    decoded_continuous_features[
                        event_idx, position, :decoded_count
                    ] = decoded_values[:decoded_count]
            position += 1

        if not args.no_cls:
            add_position([args.cls_token_id], 0, role_id=ROLE_IDS["cls"])

        if event_tokens is not None and not args.no_event_token:
            for event_index, token in enumerate(np.atleast_1d(event_tokens[event_idx])):
                features = (
                    np.asarray([event_values[event_idx, event_index]], dtype=np.float32)
                    if event_values is not None
                    else None
                )
                add_position(
                    [int(token)],
                    flat_export.TYPE_IDS["event"],
                    role_id=ROLE_IDS["event"],
                    features=features,
                    decoded_features=features,
                )

        for object_name in args.object_order:
            if object_name not in encoded_objects:
                continue
            payload = encoded_objects[object_name]
            indices, object_mask = payload[:2]
            object_features = payload[2] if len(payload) > 2 else None
            decoded_object_features = payload[3] if len(payload) > 3 else None
            if write_continuous and object_features is None:
                raise ValueError(
                    f"Continuous export is missing preprocessed features for {object_name}"
                )
            if write_decoded_q8 and decoded_object_features is None:
                raise ValueError(
                    f"Decoded-Q8 export is missing decoded features for {object_name}"
                )
            object_vocab = vocabulary["objects"][object_name]
            event_indices = indices[event_idx]
            event_mask = object_mask[event_idx]
            for object_idx in np.flatnonzero(event_mask.cpu().numpy()):
                codes = []
                for quantizer_idx, code_value in enumerate(event_indices[object_idx]):
                    code = int(code_value)
                    if code < 0:
                        continue
                    quantizer_vocab = object_vocab["quantizers"][quantizer_idx]
                    if code >= quantizer_vocab["size"]:
                        raise ValueError(
                            f"{object_name} q{quantizer_idx} produced out-of-range code {code}"
                        )
                    codes.append(int(quantizer_vocab["base"] + code))
                if codes:
                    add_position(
                        codes,
                        flat_export.TYPE_IDS[object_name],
                        role_id=ROLE_IDS["object"],
                        features=(
                            object_features[event_idx, object_idx].cpu().numpy()
                            if object_features is not None
                            else None
                        ),
                        decoded_features=(
                            decoded_object_features[event_idx, object_idx]
                            .cpu()
                            .numpy()
                            if decoded_object_features is not None
                            else None
                        ),
                    )

            if not args.no_separators:
                add_position(
                    [args.sep_token_id],
                    0,
                    role_id=ROLE_IDS["separator"],
                )

    if not write_continuous:
        return tokens, mask, type_ids
    extra_columns = {
        "continuous_features": continuous_features,
        "continuous_feature_mask": continuous_feature_mask,
        "position_role_ids": position_role_ids,
    }
    if write_decoded_q8:
        extra_columns["decoded_continuous_features"] = decoded_continuous_features
    return tokens, mask, type_ids, extra_columns


def make_grouped_table(
    tokens: np.ndarray,
    mask: np.ndarray,
    type_ids: np.ndarray,
    **kwargs,
):
    import pyarrow as pa

    table = FLAT_MAKE_TABLE(tokens[:, :, 0], mask, type_ids, **kwargs)
    seq_len = tokens.shape[1]
    max_quantizers = tokens.shape[2]
    grouped_tokens = pa.array(
        tokens.tolist(),
        type=pa.list_(pa.list_(pa.int64(), max_quantizers), seq_len),
    )
    table = table.set_column(table.schema.get_field_index("tokens"), "tokens", grouped_tokens)
    if "input_ids" in table.column_names:
        table = table.set_column(
            table.schema.get_field_index("input_ids"), "input_ids", grouped_tokens
        )
    metadata = dict(table.schema.metadata or {})
    vocabulary = kwargs["vocabulary"]
    metadata[b"heptokens_token_vocabulary"] = json.dumps(
        vocabulary, sort_keys=True
    ).encode()
    metadata[b"heptokens_sequence_layout"] = b"one_position_per_object"
    return table.replace_schema_metadata(metadata)


def main() -> None:
    for flag in ("--write-continuous-features", "--write-decoded-q8-features"):
        if flag not in sys.argv:
            sys.argv.append(flag)
    flat_export.build_vocabulary = build_grouped_vocabulary
    flat_export.assemble_rows = assemble_grouped_rows
    flat_export.make_table = make_grouped_table
    flat_export.main()


if __name__ == "__main__":
    main()
