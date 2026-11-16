#!/usr/bin/env python3
"""Re-encode token-only classification data from SHA-pinned original H5 files.

The existing Q8 Parquets are authoritative for event identity, order, class and
split. A disk-backed cache joins final Q4 codes onto those exact rows. No new
split, sampling, decoding of Q8 or tokenizer/preprocessor fitting occurs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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

from hybrid_q4_downstream import SPLITS, classes, sequence_numpy, sha256, validate_data
from prepare_q4_downstream_from_features import make_vocabulary
import q4_data_grl_campaign as campaign
import tokenize_objects_to_grouped_parquet as grouped
import tokenize_objects_to_parquet_with_atlasopenmagic_metadata as flat
from package_final_q4_tokenizer_artifacts import PREPROCESS_SHA256


def pack_row(tokens, mask, types):
    if tokens.shape != (256, 4) or mask.shape != (256,) or types.shape != (256,):
        raise ValueError('Unexpected cached Q4 row shape')
    if np.any(tokens < 0) or np.any(tokens >= campaign.VOCAB_SIZE) or np.any(types < 0) or np.any(types > 255):
        raise ValueError('Out-of-range cached Q4 token/type ID')
    data = (tokens.astype('<i4').tobytes() + mask.astype(np.uint8).tobytes()
            + types.astype(np.uint8).tobytes())
    return zlib.compress(data, level=1)


def unpack_row(blob):
    data = zlib.decompress(blob)
    if len(data) != 256 * 4 * 4 + 512:
        raise ValueError('Invalid cached Q4 byte length')
    tokens = np.frombuffer(data, dtype='<i4', count=1024).reshape(256, 4)
    mask = np.frombuffer(data, dtype=np.uint8, count=256, offset=4096).astype(bool)
    types = np.frombuffer(data, dtype=np.uint8, count=256, offset=4352)
    return tokens, mask, types


def source_event_mapping(connection, name, n_events, expected_count):
    """Prove a bijection before translating cumulative export IDs to H5 rows.

    The matched multiclass data include every row of each original H5 exactly
    once across the three splits. Only under that verified condition is its
    contiguous saved-ID interval sufficient to determine the export offset.
    A subset could start after H5 row zero: never guess its offset from min(ID).
    """
    count, first, last = connection.execute(
        'SELECT count(*), min(event), max(event) FROM identities WHERE source=?',
        (name,)).fetchone()
    if count != expected_count or count != n_events or first is None or first < 0 \
            or last - first + 1 != n_events:
        raise ValueError(
            f'Cannot prove complete H5 row mapping for {name}: '
            f'{count} saved rows, range {first}..{last}, H5 rows {n_events}. '
            'An authoritative export offset is required for partial/non-contiguous coverage.')
    return dict(offset=int(first), h5_rows=int(n_events), saved_rows=int(count),
                saved_event_index_min=int(first), saved_event_index_max=int(last),
                rule='h5_row = saved_event_index - offset',
                verification='unique contiguous IDs cover every original H5 row exactly once')


def source_events_in_slice(connection, name, start, stop, offset):
    """Return saved identities and the corresponding offsets within an H5 slice."""
    identities = [row[0] for row in connection.execute(
        'SELECT event FROM identities WHERE source=? AND event>=? AND event<? ORDER BY event',
        (name, start + offset, stop + offset))]
    local_offsets = np.asarray(identities, dtype=np.int64) - offset - start
    if np.any(local_offsets < 0) or np.any(local_offsets >= stop - start):
        raise ValueError(f'Invalid mapped H5 slice offsets: {name}')
    return identities, local_offsets


@torch.inference_mode()
def encode_raw_slice(handle, event_slice, selected_offsets, models, preprocessors,
                     config, vocabulary, device, object_batch_size=1024):
    """Encode only selected events, with bounded per-object inference batches."""
    collections = flat.collection_map(config['object_collections'])
    encoded = {}
    for name in vocabulary['object_order']:
        features, mask = flat.read_collection_batch(handle, collections[name], event_slice,
            num_objects=None, global_mask_input=config.get('mask_input'))
        features, mask = features[selected_offsets], mask[selected_offsets]
        features = flat.apply_object_preprocessing(features, mask, preprocessors[name], name)
        if not torch.isfinite(features[mask]).all():
            raise ValueError(f'Non-finite preprocessed {name} features')
        values = features[mask]
        codes = torch.full((*mask.shape, 4), -1, dtype=torch.long)
        local = []
        for start in range(0, len(values), object_batch_size):
            chunk = values[start:start + object_batch_size][None]
            local.append(flat.encode_object_batch(models[name], chunk,
                torch.ones(chunk.shape[:2], dtype=torch.bool), device)[0])
        if local:
            result = torch.cat(local)
            if result.shape != (int(mask.sum()), 4):
                raise ValueError(f'Unexpected Q4 encoder shape for {name}')
            codes[mask] = result
        encoded[name] = (codes, mask)
    context = flat.read_event_token_values(handle, vocabulary['event_tokens'], event_slice)
    context = context[selected_offsets] if context is not None else None
    args = SimpleNamespace(max_seq_length=256, no_cls=False, no_event_token=False,
        no_separators=False, object_order=vocabulary['object_order'],
        **{f'{key}_token_id': int(value) for key, value in vocabulary['special_tokens'].items()})
    return grouped.assemble_grouped_rows(encoded, context, len(selected_offsets), vocabulary, args)


def replace_from_cache(batch, connection, vocabulary):
    table = pa.Table.from_batches([batch])
    original = sequence_numpy(table['tokens'].combine_chunks(), np.int64)
    mask = sequence_numpy(table['mask'].combine_chunks(), bool)
    types = sequence_numpy(table['type_ids'].combine_chunks(), np.int64)
    source = table['source_file'].to_pylist()
    events = table['event_index'].to_pylist()
    rows = []
    physical_ids = [spec['type_id'] for spec in vocabulary['objects'].values()]
    for index, (name, event) in enumerate(zip(source, events, strict=True)):
        found = connection.execute('SELECT codes FROM identities WHERE source=? AND event=?', (name, int(event))).fetchone()
        if found is None or found[0] is None:
            raise ValueError(f'Missing encoded event: {name}:{event}')
        tokens, new_mask, new_types = unpack_row(found[0])
        if not np.array_equal(new_mask, mask[index]) or not np.array_equal(new_types, types[index]):
            raise ValueError(f'Original H5 object selection/order differs from saved Q8 positions: {name}:{event}')
        nonphysical = ~np.isin(types[index], physical_ids) | ~mask[index]
        if not np.array_equal(tokens[nonphysical], original[index, nonphysical, :4]):
            raise ValueError(f'CLS/context/SEP/padding changed: {name}:{event}')
        rows.append(tokens)
    array = pa.array(np.stack(rows).astype(np.int64).tolist(), type=pa.list_(pa.list_(pa.int64(), 4), 256))
    for key in ('tokens', 'input_ids'):
        index = table.schema.get_field_index(key)
        if index >= 0:
            table = table.set_column(index, key, array)
    metadata = dict(table.schema.metadata or {})
    metadata[b'heptokens_token_vocabulary'] = json.dumps(vocabulary, sort_keys=True).encode()
    return table.replace_schema_metadata(metadata)


def verify_inventory(source, manifest_hash, inventory_path, inventory_hash):
    if sha256(inventory_path) != inventory_hash:
        raise ValueError('H5/input inventory checksum mismatch')
    inventory = json.loads(Path(inventory_path).read_text())
    if inventory['source_manifest_sha256'] != manifest_hash or sha256(source / 'manifest.json') != manifest_hash:
        raise ValueError('Inventory belongs to a different classification dataset')
    expected = set()
    for item in inventory['q8_shards']:
        relative = Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts or relative in expected:
            raise ValueError('Invalid/duplicate inventory path')
        expected.add(relative)
        path = source / relative
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'Classification shard checksum/size mismatch: {relative}')
        if pq.ParquetFile(path).metadata.num_rows != item['rows']:
            raise ValueError(f'Classification shard count mismatch: {relative}')
    observed = {path.relative_to(source) for split in SPLITS for path in (source / split).rglob('*.parquet')}
    if observed != expected:
        raise ValueError('Uploaded classification shard inventory differs from Zephyr')
    return inventory


def prepare(args):
    source, output = args.source_dir, args.output_dir
    partial = output.with_name(output.name + '.partial')
    if output.exists() or partial.exists():
        raise FileExistsError('Refusing to overwrite Q4 classification output')
    inventory = verify_inventory(source, args.source_manifest_sha256,
                                 args.input_inventory, args.input_inventory_sha256)
    manifest, definitions, source_vocab = validate_data(source, args.source_manifest_sha256,
                                                       'hybrid', args.cache_dir)
    if inventory['classes'] != manifest['classes'] or inventory['split_counts'] != manifest['split_counts']:
        raise ValueError('Inventory class/split definitions differ')
    connection = sqlite3.connect(args.cache_dir / 'classification_identity_check.sqlite')
    connection.execute('PRAGMA cache_size=-8192')
    connection.execute('ALTER TABLE identities ADD COLUMN codes BLOB')
    source_counts = dict(connection.execute('SELECT source, count(*) FROM identities GROUP BY source'))
    expected_counts = {name: int(spec['selected_events']) for name, spec in inventory['h5_sources'].items()}
    if source_counts != expected_counts:
        raise ValueError('Original H5 membership/counts differ from the uploaded inventory')
    config = flat.load_data_config(str(campaign.feature_config_path()))
    device = torch.device(args.device)
    models = flat.load_models({name: str(args.bundle_root / 'q4' / f'{name}.ckpt') for name in campaign.Q4_SIZES}, device)
    campaign.validate_q4_models(models)
    paths = {name: str(args.bundle_root / 'preprocessing' / f'{name}.joblib') for name in models}
    if any(sha256(path) != PREPROCESS_SHA256[name] for name, path in paths.items()):
        raise ValueError('Preprocessing differs from the matched Q1/Q8 preprocessing')
    preprocessors = flat.load_preprocessors(paths)
    campaign.validate_q4_feature_schema(models, flat.collection_map(config['object_collections']), preprocessors)
    vocabulary = make_vocabulary(source_vocab, models)
    download_dir = args.cache_dir / 'downloaded_h5'
    download_dir.mkdir()
    index_mappings = {}
    try:
        for name, spec in sorted(inventory['h5_sources'].items()):
            if Path(name).name != name or not name.endswith('.h5') or not spec['uri'].startswith('gs://'):
                raise ValueError('Invalid original H5 reference')
            path = args.h5_dir / name if args.h5_dir else download_dir / name
            if not args.h5_dir:
                subprocess.run(['gcloud', 'storage', 'cp', spec['uri'], str(path)], check=True)
            if path.stat().st_size != spec['bytes'] or sha256(path) != spec['sha256']:
                raise ValueError(f'Original H5 checksum/size mismatch: {name}')
            encoded_count = 0
            with h5py.File(path, 'r') as handle:
                n_events = flat.infer_n_events(handle, config)
                mapping = source_event_mapping(connection, name, n_events, spec['selected_events'])
                index_mappings[name] = dict(mapping, original_h5_sha256=spec['sha256'])
                print(f'Verified {name}: saved IDs {mapping["saved_event_index_min"]}..'
                      f'{mapping["saved_event_index_max"]} map to all {n_events:,} original H5 rows', flush=True)
                for start in range(0, n_events, args.read_batch_size):
                    stop = min(start + args.read_batch_size, n_events)
                    indices, local_offsets = source_events_in_slice(
                        connection, name, start, stop, mapping['offset'])
                    if not indices:
                        continue
                    tokens, masks, types = encode_raw_slice(handle, slice(start, stop),
                        local_offsets, models, preprocessors, config, vocabulary, device)
                    records = [(pack_row(tokens[i], masks[i], types[i]), name, event) for i, event in enumerate(indices)]
                    cursor = connection.executemany('UPDATE identities SET codes=? WHERE source=? AND event=? AND codes IS NULL', records)
                    if cursor.rowcount != len(indices):
                        raise ValueError('Repeated/missing Q4 event encoding')
                    encoded_count += len(indices)
                connection.commit()
            if encoded_count != spec['selected_events']:
                raise ValueError(f'Incomplete Q4 encoding: {name}')
            print(f'Encoded {name}: {encoded_count:,} selected events', flush=True)
            if not args.h5_dir:
                path.unlink()  # Only this job's downloaded scratch copy, not original H5.
        if connection.execute('SELECT count(*) FROM identities WHERE codes IS NULL').fetchone()[0]:
            raise ValueError('Some classification events have not been encoded')
        partial.mkdir(parents=True)
        output_inventory = {}
        for split in SPLITS:
            for name, _ in definitions:
                digest, count = hashlib.sha256(), 0
                for path in sorted((source / split / name).glob('*.parquet')):
                    destination = partial / path.relative_to(source)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    writer = None
                    try:
                        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                            table = replace_from_cache(batch, connection, vocabulary)
                            if writer is None:
                                writer = pq.ParquetWriter(destination, table.schema, compression='snappy')
                            writer.write_table(table)
                            for s, e in zip(table['source_file'].to_pylist(), table['event_index'].to_pylist(), strict=True):
                                digest.update(json.dumps([s, int(e)], separators=(',', ':')).encode() + b'\n')
                            count += table.num_rows
                    finally:
                        if writer:
                            writer.close()
                    output_inventory[str(path.relative_to(source))] = dict(sha256=sha256(destination),
                        source_sha256=sha256(path), rows=pq.ParquetFile(destination).metadata.num_rows)
                expected = inventory['memberships'][f'{split}/{name}']
                if count != expected['rows'] or digest.hexdigest() != expected['ordered_identity_sha256']:
                    raise ValueError(f'Event identity/order changed for {split}/{name}')
        result = copy.deepcopy(manifest)
        result.update(token_shape=[256, 4], representation='final_q4')
        result['q4_reencoding'] = dict(source_manifest_sha256=args.source_manifest_sha256,
            tokenizer_bundle_sha256=campaign.BUNDLE_SHA, input_inventory_sha256=args.input_inventory_sha256,
            source='SHA-pinned original H5 features with matching frozen preprocessors',
            split_policy='unchanged', selection='unchanged', non_token_columns='unchanged',
            ordered_memberships=inventory['memberships'], original_h5_sources=inventory['h5_sources'],
            event_index_mapping=index_mappings, shards=output_inventory)
        for key in ('shards', 'files', 'outputs'):
            if key in result:
                result['q4_reencoding'][f'source_{key}'] = result.pop(key)
        (partial / 'manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        (partial / 'manifest.sha256').write_text(sha256(partial / 'manifest.json') + '\n')
        partial.rename(output)
        print('PASS: original H5 -> final Q4, every original event/split/label/order retained', flush=True)
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir', 'output-dir', 'bundle-root', 'input-inventory', 'cache-dir'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--input-inventory-sha256', required=True)
    parser.add_argument('--h5-dir', type=Path, help='Read existing originals instead of downloading; never modify them')
    parser.add_argument('--read-batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.read_batch_size < 1:
        parser.error('Read batch size must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
