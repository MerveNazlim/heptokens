"""Export one event-sequence position per object, preserving residual codes."""

from __future__ import annotations

import json

import numpy as np

import tokenize_objects_to_parquet_with_atlasopenmagic_metadata as flat_export

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build [event, position, quantizer] token arrays."""
    max_quantizers = int(vocabulary["max_quantizers"])
    tokens = np.full(
        (batch_size, args.max_seq_length, max_quantizers),
        args.pad_token_id,
        dtype=np.int64,
    )
    mask = np.zeros((batch_size, args.max_seq_length), dtype=bool)
    type_ids = np.zeros((batch_size, args.max_seq_length), dtype=np.int64)

    for event_idx in range(batch_size):
        position = 0

        def add_position(codes: list[int], type_id: int) -> None:
            nonlocal position
            if position >= args.max_seq_length:
                return
            tokens[event_idx, position, : len(codes)] = codes
            mask[event_idx, position] = True
            type_ids[event_idx, position] = type_id
            position += 1

        if not args.no_cls:
            add_position([args.cls_token_id], 0)

        if event_tokens is not None and not args.no_event_token:
            for token in np.atleast_1d(event_tokens[event_idx]):
                add_position([int(token)], flat_export.TYPE_IDS["event"])

        for object_name in args.object_order:
            if object_name not in encoded_objects:
                continue
            indices, object_mask = encoded_objects[object_name]
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
                    add_position(codes, flat_export.TYPE_IDS[object_name])

            if not args.no_separators:
                add_position([args.sep_token_id], 0)

    return tokens, mask, type_ids


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
    flat_export.build_vocabulary = build_grouped_vocabulary
    flat_export.assemble_rows = assemble_grouped_rows
    flat_export.make_table = make_grouped_table
    flat_export.main()


if __name__ == "__main__":
    main()
