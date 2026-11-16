#!/usr/bin/env python3
"""Add original continuous features to the EXACT saved hybrid/Q4 sample rows.

No tokenizer inference, decoding, new split, resampling or preprocessing fit.
The original Q8 columns are retained unchanged as an auditable join reference.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace
import zlib

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from heptokens.data.continuous_schema import build_continuous_schema
from hybrid_q4_downstream import SPLITS, classes, sequence_numpy, sha256, validate_data
from prepare_q4_downstream_from_h5 import verify_inventory, source_event_mapping, source_events_in_slice
from package_final_q4_tokenizer_artifacts import PREPROCESS_SHA256
import tokenize_objects_to_grouped_parquet as grouped
import tokenize_objects_to_parquet_with_atlasopenmagic_metadata as flat


def read_continuous_slice(handle, event_slice, offsets, config, vocabulary, schema, preprocessors):
    encoded = {}
    for name in vocabulary['object_order']:
        collection = flat.collection_map(config['object_collections'])[name]
        features, mask = flat.read_collection_batch(handle, collection, event_slice,
            num_objects=None, global_mask_input=config.get('mask_input'))
        features, mask = features[offsets], mask[offsets]
        features = flat.apply_object_preprocessing(features, mask, preprocessors[name], name)
        if not torch.isfinite(features[mask]).all():
            raise ValueError(f'Non-finite original preprocessed features: {name}')
        # Structural placeholders ONLY: the assembler uses them to lay out the
        # same objects; these IDs are discarded, never saved or used by a model.
        placeholder = torch.zeros((*mask.shape, 8), dtype=torch.long)
        encoded[name] = (placeholder, mask, features)
    event_tokens = flat.read_event_token_values(handle, vocabulary['event_tokens'], event_slice)[offsets]
    event_values = flat.read_normalized_event_values(handle, vocabulary['event_tokens'], event_slice)[offsets]
    assembly_vocab = dict(vocabulary, continuous_schema=schema)
    options = SimpleNamespace(max_seq_length=256, no_cls=False, no_event_token=False,
        no_separators=False, object_order=vocabulary['object_order'],
        **{f'{key}_token_id': int(value) for key, value in vocabulary['special_tokens'].items()})
    return grouped.assemble_grouped_rows(encoded, event_tokens, len(offsets), assembly_vocab,
                                         options, event_values=event_values)


def cache_blob(features, feature_mask, roles, tokens, mask, types):
    stream = features.astype('<f4').tobytes() + feature_mask.astype(np.uint8).tobytes()
    stream += roles.astype(np.uint8).tobytes() + tokens[:, 0].astype('<i4').tobytes()
    stream += mask.astype(np.uint8).tobytes() + types.astype(np.uint8).tobytes()
    return zlib.compress(stream, level=1)


def cached_columns(blob, width):
    raw = zlib.decompress(blob)
    size = 256 * width
    expected = size * 5 + 256 * 7
    if len(raw) != expected:
        raise ValueError('Invalid cached continuous event size')
    features = np.frombuffer(raw, '<f4', count=size).reshape(256, width)
    valid = np.frombuffer(raw, np.uint8, count=size, offset=size * 4).reshape(256, width).astype(bool)
    offset = size * 5
    roles = np.frombuffer(raw, np.uint8, count=256, offset=offset)
    tokens = np.frombuffer(raw, '<i4', count=256, offset=offset + 256)
    mask = np.frombuffer(raw, np.uint8, count=256, offset=offset + 1280).astype(bool)
    types = np.frombuffer(raw, np.uint8, count=256, offset=offset + 1536)
    return features, valid, roles, tokens, mask, types


def attach_original_features(batch, connection, schema):
    table = pa.Table.from_batches([batch])
    col = lambda key, dtype: sequence_numpy(table[key].combine_chunks(), dtype)
    original, masks, types = col('tokens', np.int64), col('mask', bool), col('type_ids', np.int64)
    features, feature_masks, roles = [], [], []
    physical = [spec['type_id'] for spec in schema['objects'].values()]
    for index, (source, event) in enumerate(zip(table['source_file'].to_pylist(), table['event_index'].to_pylist(), strict=True)):
        record = connection.execute('SELECT features FROM identities WHERE source=? AND event=?', (source, int(event))).fetchone()
        if record is None or record[0] is None:
            raise ValueError(f'Missing original continuous event {source}:{event}')
        values, valid, role, dummy, mask, type_ids = cached_columns(record[0], schema['max_feature_dim'])
        if not np.array_equal(mask, masks[index]) or not np.array_equal(type_ids, types[index]):
            raise ValueError(f'Original H5 object order/selection differs: {source}:{event}')
        nonphysical = ~np.isin(type_ids, physical) | ~mask
        if not np.array_equal(dummy[nonphysical], original[index, nonphysical, 0]):
            raise ValueError(f'CLS/context/SEP/padding differs: {source}:{event}')
        features.append(values); feature_masks.append(valid); roles.append(role)
    for name, values, dtype in (
        ('continuous_features', features, pa.list_(pa.list_(pa.float32(), schema['max_feature_dim']), 256)),
        ('continuous_feature_mask', feature_masks, pa.list_(pa.list_(pa.bool_(), schema['max_feature_dim']), 256)),
        ('position_role_ids', roles, pa.list_(pa.int64(), 256)),
    ):
        if name in table.column_names:
            raise ValueError(f'Refusing to replace an existing {name} column')
        table = table.append_column(name, pa.array(np.asarray(values).tolist(), type=dtype))
    metadata = dict(table.schema.metadata or {})
    metadata[b'heptokens_continuous_schema'] = json.dumps(schema, sort_keys=True).encode()
    return table.replace_schema_metadata(metadata)


def prepare(args):
    source, output, cache = args.source_dir, args.output_dir, args.cache_dir
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Refusing to overwrite prepared continuous classification data')
    inventory = verify_inventory(source, args.source_manifest_sha256, args.input_inventory, args.input_inventory_sha256)
    manifest, definitions, vocabulary = validate_data(source, args.source_manifest_sha256, 'hybrid', cache)
    if inventory['classes'] != manifest['classes'] or inventory['split_counts'] != manifest['split_counts']:
        raise ValueError('Original H5 inventory differs from the selected multiclass samples')
    config = flat.load_data_config(str(args.feature_config))
    collections = flat.collection_map(config['object_collections'])
    paths = {name: str(args.preprocessing_dir / f'{name}.joblib') for name in vocabulary['object_order']}
    if set(paths) != set(PREPROCESS_SHA256) or any(sha256(path) != PREPROCESS_SHA256[name] for name, path in paths.items()):
        raise ValueError('Fitted preprocessing differs from original Q1/Q8/continuous export')
    preprocessors = flat.load_preprocessors(paths)
    schema = build_continuous_schema(collections=collections, object_order=vocabulary['object_order'],
        event_token_specs=vocabulary['event_tokens'], type_ids=flat.TYPE_IDS, include_decoded_q8=False)
    for name, spec in schema['objects'].items():
        transformer = preprocessors[name]
        size = getattr(transformer, 'n_features_in_', None)
        if size is None:
            size = getattr(getattr(transformer, 'final_transformer', None), 'n_features_in_', None)
        if size != spec['feature_count']:
            raise ValueError(f'Fitted input dimension differs for {name}')
    connection = sqlite3.connect(cache / 'classification_identity_check.sqlite')
    connection.execute('PRAGMA cache_size=-8192')
    connection.execute('ALTER TABLE identities ADD COLUMN features BLOB')
    counts = dict(connection.execute('SELECT source,count(*) FROM identities GROUP BY source'))
    if counts != {name: spec['selected_events'] for name, spec in inventory['h5_sources'].items()}:
        raise ValueError('Original H5 membership differs')
    downloads = cache / 'downloaded_h5'; downloads.mkdir()
    mappings = {}
    try:
        for name, spec in sorted(inventory['h5_sources'].items()):
            if Path(name).name != name or not spec['uri'].startswith('gs://'):
                raise ValueError('Unsafe original H5 reference')
            path = args.h5_dir / name if args.h5_dir else downloads / name
            if not args.h5_dir:
                subprocess.run(['gcloud', 'storage', 'cp', spec['uri'], str(path)], check=True)
            if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
                raise ValueError(f'Original H5 checksum/size mismatch: {name}')
            with h5py.File(path, 'r') as handle:
                n = flat.infer_n_events(handle, config)
                mapping = source_event_mapping(connection, name, n, spec['selected_events'])
                mappings[name] = mapping
                for start in range(0, n, args.read_batch_size):
                    stop = min(start + args.read_batch_size, n)
                    events, offsets = source_events_in_slice(connection, name, start, stop, mapping['offset'])
                    if not events:
                        continue
                    tokens, masks, types, extra = read_continuous_slice(handle, slice(start, stop), offsets,
                        config, vocabulary, schema, preprocessors)
                    records = [(cache_blob(extra['continuous_features'][i], extra['continuous_feature_mask'][i],
                                extra['position_role_ids'][i], tokens[i], masks[i], types[i]), name, int(event))
                               for i, event in enumerate(events)]
                    connection.executemany('UPDATE identities SET features=? WHERE source=? AND event=?', records)
                connection.commit()
            print(f'PASS original continuous H5 features and verified export offset: {name}', flush=True)
        if connection.execute('SELECT count(*) FROM identities WHERE features IS NULL').fetchone()[0]:
            raise ValueError('Missing continuous event features')
        partial.mkdir(parents=True)
        shards = {}
        for split in SPLITS:
            for name, _ in definitions:
                for path in sorted((source / split / name).glob('*.parquet')):
                    target = partial / path.relative_to(source); target.parent.mkdir(parents=True, exist_ok=True)
                    with pq.ParquetWriter(target, attach_original_features(next(pq.ParquetFile(path).iter_batches(batch_size=1)),
                            connection, schema).schema, compression='snappy') as writer:
                        for batch in pq.ParquetFile(path).iter_batches(batch_size=args.read_batch_size):
                            writer.write_table(attach_original_features(batch, connection, schema), row_group_size=4096)
                    shards[str(target.relative_to(partial))] = dict(sha256=sha256(target), source_sha256=sha256(path))
        result = copy.deepcopy(manifest)
        result['continuous_preparation'] = dict(source_manifest_sha256=args.source_manifest_sha256,
            input_inventory_sha256=args.input_inventory_sha256, preprocessing_sha256=PREPROCESS_SHA256,
            source='original H5, saved preprocessing; never decoded token predictions',
            splits='unchanged', all_original_columns='unchanged', index_mappings=mappings, shards=shards)
        for key in ('shards', 'files', 'outputs'):
            if key in result:
                result['continuous_preparation'][f'source_{key}'] = result.pop(key)
        (partial / 'manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        (partial / 'manifest.sha256').write_text(sha256(partial / 'manifest.json') + '\n')
        partial.rename(output)
        print('PASS exact original events, splits, labels, Q8 IDs and object positions retained', flush=True)
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir', 'output-dir', 'cache-dir', 'input-inventory', 'feature-config', 'preprocessing-dir'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--input-inventory-sha256', required=True)
    parser.add_argument('--h5-dir', type=Path)
    parser.add_argument('--read-batch-size', type=int, default=128)
    args = parser.parse_args()
    if args.read_batch_size < 1:
        parser.error('read-batch-size must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
