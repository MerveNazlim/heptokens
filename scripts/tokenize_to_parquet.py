"""Apply a trained VQ-VAE tokenizer checkpoint and save token sequences to Parquet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from heptokens.data.atlas_event_mappable import AtlasEventMapDataset
from heptokens.models.vq_vae import LitVqVae

log = logging.getLogger(__name__)


DATAMODULE_KEYS = {
    "_target_",
    "data_path",
    "data_paths",
    "train_frac",
    "val_frac",
    "test_frac",
    "seed",
    "n_classes",
    "num_workers",
    "batch_size",
    "pin_memory",
    "persistent_workers",
    "multiprocessing_context",
    "transforms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained heptokens VQ-VAE checkpoint to an HDF5 file."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.ckpt or last.ckpt.")
    parser.add_argument("--output", required=True, help="Output parquet file.")
    parser.add_argument(
        "--datamodule-config",
        default="configs/datamodule/atlas_event_mappable.yaml",
        help="Datamodule YAML that defines event_inputs/object_collections.",
    )
    parser.add_argument("--data-path", help="Override data_path from the datamodule config.")
    parser.add_argument("--h5-files", nargs="+", help="One or more HDF5 files to tokenize.")
    parser.add_argument("--output-mode", help="Override output_mode from the datamodule config.")
    parser.add_argument("--object-type", help="Override object_type from the datamodule config.")
    parser.add_argument("--num-events", type=int, help="Cap number of events to encode.")
    parser.add_argument("--num-objects", type=int, help="Cap number of objects per event.")
    parser.add_argument("--batch-size", type=int, help="Override batch size from the config.")
    parser.add_argument("--num-workers", type=int, help="Override number of dataloader workers.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--token-offset",
        type=int,
        default=4,
        help="Offset valid VQ codes so 0-3 remain reserved special tokens.",
    )
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--cls-token-id", type=int, default=1)
    parser.add_argument("--no-cls", action="store_true", help="Do not prepend a CLS token.")
    parser.add_argument(
        "--write-legacy-columns",
        action="store_true",
        help="Also write input_ids/attention_mask/token_type_ids column aliases.",
    )
    return parser.parse_args()


def dataset_kwargs_from_config(
    cfg_path: str,
    args: argparse.Namespace,
) -> tuple[list[str], dict, int, int]:
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    if args.h5_files:
        data_paths = list(args.h5_files)
    elif args.data_path:
        data_paths = [args.data_path]
    elif cfg.get("data_paths"):
        data_paths = list(cfg["data_paths"])
    else:
        data_paths = [cfg["data_path"]]
    batch_size = args.batch_size or cfg.get("batch_size", 1024)
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = cfg.get("num_workers", 0)

    dataset_kwargs = {key: value for key, value in cfg.items() if key not in DATAMODULE_KEYS}
    if args.output_mode is not None:
        dataset_kwargs["output_mode"] = args.output_mode
    if args.object_type is not None:
        dataset_kwargs["object_type"] = args.object_type
    if args.num_events is not None:
        dataset_kwargs["num_events"] = args.num_events
    if args.num_objects is not None:
        dataset_kwargs["num_objects"] = args.num_objects
    if dataset_kwargs.get("output_mode") == "separate":
        raise ValueError(
            "Token parquet export needs a batch with csts/mask. Use output_mode=combined "
            "or output_mode=object, not separate."
        )

    return data_paths, dataset_kwargs, int(batch_size), int(num_workers)


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_token_batch(
    indices: torch.Tensor,
    batch: dict,
    *,
    token_offset: int,
    pad_token_id: int,
    cls_token_id: int,
    add_cls: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if indices.dim() == 2:
        indices = indices.unsqueeze(-1)

    object_mask = batch["mask"].detach().cpu().bool()
    indices = indices.detach().cpu().long()
    valid_mask = object_mask.unsqueeze(-1).expand_as(indices) & indices.ge(0)

    tokens = indices + token_offset
    tokens = torch.where(valid_mask, tokens, torch.full_like(tokens, pad_token_id))

    if "type_ids" in batch:
        type_ids = batch["type_ids"].detach().cpu().long()
    else:
        type_ids = torch.zeros_like(object_mask, dtype=torch.long)
    type_ids = type_ids.unsqueeze(-1).expand_as(indices)
    type_ids = torch.where(valid_mask, type_ids, torch.zeros_like(type_ids))

    batch_size = tokens.shape[0]
    tokens = tokens.reshape(batch_size, -1)
    mask = valid_mask.reshape(batch_size, -1)
    type_ids = type_ids.reshape(batch_size, -1)

    if add_cls:
        cls_tokens = torch.full((batch_size, 1), cls_token_id, dtype=tokens.dtype)
        cls_mask = torch.ones((batch_size, 1), dtype=torch.bool)
        cls_types = torch.zeros((batch_size, 1), dtype=type_ids.dtype)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        mask = torch.cat([cls_mask, mask], dim=1)
        type_ids = torch.cat([cls_types, type_ids], dim=1)

    labels = batch.get("labels")
    if labels is not None:
        labels = labels.detach().cpu().long().numpy()

    return tokens.numpy(), mask.numpy(), type_ids.numpy(), labels


def make_table(
    tokens: np.ndarray,
    mask: np.ndarray,
    type_ids: np.ndarray,
    labels: np.ndarray | None,
    *,
    start_index: int,
    source_file: str,
    write_legacy_columns: bool,
) -> pa.Table:
    seq_len = tokens.shape[1]
    fields = {
        "tokens": pa.array(tokens.tolist(), type=pa.list_(pa.int64(), seq_len)),
        "mask": pa.array(mask.tolist(), type=pa.list_(pa.bool_(), seq_len)),
        "type_ids": pa.array(type_ids.tolist(), type=pa.list_(pa.int64(), seq_len)),
        "event_index": pa.array(
            np.arange(start_index, start_index + len(tokens), dtype=np.int64)
        ),
        "source_file": pa.array([source_file] * len(tokens)),
    }
    if labels is not None:
        fields["labels"] = pa.array(labels.astype(np.int64))
    if write_legacy_columns:
        fields["input_ids"] = fields["tokens"]
        fields["attention_mask"] = fields["mask"]
        fields["token_type_ids"] = fields["type_ids"]
    return pa.table(fields)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    data_paths, dataset_kwargs, batch_size, num_workers = dataset_kwargs_from_config(
        args.datamodule_config,
        args,
    )
    device = choose_device(args.device)
    model = LitVqVae.load_from_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    n_written = 0
    with torch.no_grad():
        for data_path in data_paths:
            log.info("Tokenizing %s", data_path)
            dataset = AtlasEventMapDataset(data_path, **dataset_kwargs)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
            source_file = Path(data_path).name
            for batch_idx, batch in enumerate(loader):
                batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                indices = model(batch)
                tokens, mask, type_ids, labels = build_token_batch(
                    indices,
                    batch,
                    token_offset=args.token_offset,
                    pad_token_id=args.pad_token_id,
                    cls_token_id=args.cls_token_id,
                    add_cls=not args.no_cls,
                )
                table = make_table(
                    tokens,
                    mask,
                    type_ids,
                    labels,
                    start_index=n_written,
                    source_file=source_file,
                    write_legacy_columns=args.write_legacy_columns,
                )
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
                writer.write_table(table)
                n_written += len(tokens)
                if batch_idx == 0 or batch_idx % 100 == 0:
                    log.info("Encoded %s events", n_written)

    if writer is not None:
        writer.close()

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info("Wrote %s events to %s (%.1f MB)", n_written, output_path, size_mb)


if __name__ == "__main__":
    main()
