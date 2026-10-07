#!/usr/bin/env bash
set -euo pipefail

# Keep the identity projection from Q8-direct pretraining for HZZ fine-tuning.
ROOT=${ROOT:-/home/magaras/heptok_fork/heptokens}
RESULTS=${RESULTS:-${ROOT}/results}
SOURCE_DIR=${SOURCE_DIR:-${RESULTS}/q8_direct_no_projection_google/cluster_164_proc_0}
PREPARED_DIR=${PREPARED_DIR:-${RESULTS}/grouped_representation_full_prepared/20260907-060234/hzz_classification}
GPU=${GPU:-3}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
SEED=${SEED:-42}
DRY_RUN=${DRY_RUN:-0}
PROJECT_NAME=${PROJECT_NAME:-atlas_hzz_q8_direct_200m_data_pretrained_comparison}
NETWORK_NAME=${NETWORK_NAME:-q8_direct_200m_data_pretrained_hzz_finetuned_cls_mlp}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.pixi/envs/default/bin/python}
PYARROW_PKG=${PYARROW_PKG:-/home/magaras/.cache/rattler/cache/pkgs/pyarrow-core-16.1.0-py311hf1d6e26_2_cpu}

if [[ -z "${PRETRAIN_CKPT:-}" ]]; then
  [[ -d "$SOURCE_DIR" ]] || { echo "Missing copied run: $SOURCE_DIR" >&2; exit 1; }
  checkpoints=()
  while IFS= read -r path; do checkpoints+=("$path"); done < <(
    find "$SOURCE_DIR" -type f -path '*/checkpoints/last.ckpt' -print
  )
  if [[ ${#checkpoints[@]} -ne 1 ]]; then
    printf 'Expected one last.ckpt under %s; found %s. Set PRETRAIN_CKPT explicitly.\n' "$SOURCE_DIR" "${#checkpoints[@]}" >&2
    exit 1
  fi
  PRETRAIN_CKPT=${checkpoints[0]}
fi
for path in "$PRETRAIN_CKPT" "$PREPARED_DIR/manifest.json"; do
  [[ -s "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }
done
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 1; }
RUN_DIR=${RESULTS}/${PROJECT_NAME}/${NETWORK_NAME}
[[ ! -e "$RUN_DIR" ]] || { echo "Output already exists; choose a new NETWORK_NAME: $RUN_DIR" >&2; exit 1; }

CACHE_LIBS=${CACHE_LIBS:-$(find /home/magaras/.cache/rattler/cache/pkgs -type d -path '*/lib' -print | paste -sd: -)}
export PYTHONPATH="$ROOT/src:$PYARROW_PKG/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$CACHE_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPLBACKEND=Agg

"$PYTHON_BIN" - "$PRETRAIN_CKPT" "$PREPARED_DIR" <<'PY'
import inspect
import json
import sys
from pathlib import Path

import torch
from heptokens.models import foundation_grouped_cls_classifier as cls

if "object_projection_mode" not in inspect.signature(cls.LitGroupedCLSClassifier.__init__).parameters:
    raise RuntimeError("Zephyr needs the current foundation_grouped.py and foundation_grouped_cls_classifier.py with identity-projection support")

path, prepared = map(Path, sys.argv[1:])
checkpoint = cls._load_checkpoint_state(str(path))
hparams = checkpoint["hyper_parameters"]
expected = {
    "object_projection_mode": "identity", "max_quantizers": 8,
    "quantizer_embedding_dim": 32, "hidden_dim": 256,
    "num_layers": 4, "num_heads": 8, "max_seq_length": 256,
}
for name, value in expected.items():
    if hparams.get(name) != value:
        raise RuntimeError(f"Expected {name}={value!r}; checkpoint contains {hparams.get(name)!r}")
if checkpoint.get("epoch") != 9:
    raise RuntimeError(f"Expected completed 10-epoch pretraining (epoch 9), found {checkpoint.get('epoch')}")

manifest = json.loads((prepared / "manifest.json").read_text())
if manifest.get("token_shape") != [256, 8] or manifest.get("identity_overlap_detected") is not False:
    raise RuntimeError("Expected audited grouped-CLS Q8 classification shards [256,8]")
if manifest["signal"]["dsid"] != 345060 or manifest["background"]["dsid"] != 700600:
    raise RuntimeError("Expected the same ggF 345060 versus ZZ continuum 700600 task")
if manifest["signal"]["label"] != 1 or manifest["background"]["label"] != 0:
    raise RuntimeError("Expected signal=1 and background=0")

prefix = "model.backbone."
state = {name.removeprefix(prefix): value for name, value in checkpoint["state_dict"].items() if name.startswith(prefix)}
if not state or any(not torch.isfinite(value).all() for value in state.values() if torch.is_floating_point(value)):
    raise RuntimeError("Missing or non-finite pretrained backbone weights")
model = cls.LitGroupedCLSClassifier(
    **expected, vocab_size=int(hparams["vocab_size"]), n_classes=2,
    freeze_backbone=False, classifier_hidden_dim=256,
    num_type_ids=int(hparams.get("num_type_ids", 12)),
    use_type_embedding=bool(hparams.get("use_type_embedding", True)),
    use_position_embedding=bool(hparams.get("use_position_embedding", True)),
)
model.backbone.load_state_dict(state, strict=True)
if not isinstance(model.backbone.object_projection, torch.nn.Identity):
    raise RuntimeError("Classifier recreated a learned projection")
print("PASS: Q8-direct backbone strictly loaded; projection is Identity")
print("Pretraining checkpoint:", path)
print("Pretraining epoch:", checkpoint["epoch"], "global step:", checkpoint.get("global_step"))
print("Classification rows:", manifest["split_counts"])
PY

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
UNIT=${UNIT:-atlas-hzz-q8-direct-finetune-gpu${GPU}-${STAMP}}
TMP_DIR=${RESULTS}/tmp/${PROJECT_NAME}
mkdir -p "$TMP_DIR"
WORKER=${TMP_DIR}/${UNIT}.sh
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'cd %q\n' "$ROOT"
  printf 'export CUDA_VISIBLE_DEVICES=%q MPLBACKEND=Agg HYDRA_FULL_ERROR=1 WANDB_MODE=offline WANDB_INIT_TIMEOUT=300\n' "$GPU"
  printf 'export PYTHONPATH=%q\n' "$PYTHONPATH"
  printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"
  printf '%q ' "$PYTHON_BIN" "$ROOT/scripts/benchmark_train.py" \
    datamodule=token_parquet_grouped_classification model=foundation_grouped_cls_mlp_classifier \
    callbacks=grouped_classification "project_name=$PROJECT_NAME" "network_name=$NETWORK_NAME" \
    "output_dir=$RESULTS" "seed=$SEED" ckpt_path=null weight_ckpt_path=null full_resume=false \
    logger.offline=true "datamodule.prepared_dir=$PREPARED_DIR" datamodule.n_classes=2 \
    "datamodule.batch_size=$BATCH_SIZE" datamodule.num_workers=0 \
    datamodule.stream_batch_size=4096 datamodule.shuffle_buffer_size=8192 \
    datamodule.persistent_workers=false "model.backbone_ckpt_path=$PRETRAIN_CKPT" \
    model.freeze_backbone=false model.max_quantizers=8 model.quantizer_embedding_dim=32 \
    model.hidden_dim=256 ++model.object_projection_mode=identity model.learning_rate=1e-4 \
    model.optimizer.lr=1e-4 "trainer.max_epochs=$EPOCHS" trainer.accelerator=gpu \
    trainer.devices=1 trainer.check_val_every_n_epoch=1 trainer.val_check_interval=1.0 \
    trainer.limit_val_batches=1.0 +trainer.log_every_n_steps=50
  printf '\n'
} > "$WORKER"
bash -n "$WORKER"
echo "GPU: $GPU; epochs: $EPOCHS; batch size: $BATCH_SIZE; projection: identity"
echo "Output: $RUN_DIR"
echo "Worker: $WORKER"
if [[ "$DRY_RUN" == 1 ]]; then
  echo "DRY RUN: would submit $UNIT.service"
  exit 0
fi
systemd-run --unit="$UNIT" --description="Fine-tune Q8 direct no-projection backbone on HZZ" \
  --collect --property=WorkingDirectory="$ROOT" /bin/bash "$WORKER"
echo "Follow: journalctl -u $UNIT.service -f -o cat"
