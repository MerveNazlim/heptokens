# heptokens: object tokenizers and event Parquet

Train one VQ-VAE/RVQ tokenizer per object type, evaluate its reconstruction, and
convert H5 events into grouped Q1, Q8 and continuous Parquet datasets.

This guide covers the tokenizer and Parquet code in the event_level branch.
Grouped foundation-model implementations are not included here yet; pretraining
and fine-tuning will be documented when those models are integrated.

## Setup

Run commands from the repository root. For a fresh Pixi environment:

    pixi install --locked
    pixi run python -c "import torch, pyarrow; print(torch.__version__, pyarrow.__version__)"

PyArrow 24.0.0 and vector-quantize-pytorch 0.2.2 are pinned. Saved preprocessing
joblibs need the scikit-learn version used to fit them. Use the separate cloud
runtime/lock for B300 jobs; the repository's Linux lock uses CUDA 12.1.

The current H5 inputs are produced by the
[BNL conversion code](https://gitlab.cern.ch/vcavalie/bnl-treasure/-/blob/master/download_and_convert.py?ref_type=heads).

## 1. Configure the H5 inputs

Use configs/datamodule/atlas_event_object_iterable.yaml for chunked tokenizer
training. Its default shuffle mode is buffered: batches mix events across files.
Set datamodule.shuffle_mode=legacy to reproduce the original Google ordering.
Both modes use the same seeded global 70/15/15 event-level split.

The eager loader is still available as atlas_event_object. MC, data and mixed
input lists remain supported. The iterable loader does not implement weighted
domain sampling.

Object features and masks have shape [events, objects]; event scalars have shape
[events]. Paths and feature order must match the chosen YAML:

    common/event/...                 event scalars
    common/met/...                   MET scalars
    common/<object>/...              shared object features and mask
    atlas/<object>/...               ATLAS-specific object features

The six collections are electrons, muons, taus, photons, jets and tracks.
To use different paths or features, copy the datamodule YAML and edit its
object_collections. Select the same configuration for fitting, training and
conversion. Do not reorder features after training.

The iterable defaults are 4,096-event chunks, a 16,384-event shuffle buffer per
worker and a 16-file mixing window. Features stay on disk; split assignment
still uses memory proportional to the event count.

## 2. Fit preprocessing

Fit once per object on training events, then reuse that exact joblib for training,
evaluation and conversion. Do not fit on validation or test objects.

Example for jets:

    H5=(/data/mc_001.h5 /data/data_001.h5)
    WORK=/work/heptokens

    pixi run python scripts/get_atlas_object_preprocessing.py \
      --h5-files "${H5[@]}" \
      --datamodule-config configs/datamodule/atlas_event_object_iterable.yaml \
      --object-type jets --fit-split train \
      --mode log_standard --log-features pt,mass,n_trk,QG_nTracks \
      --max-objects 1000000 --seed 42 \
      --output-dir "$WORK/preprocessing" --output-name jets_log_standard

--fit-split train matches the tokenizer loader's training membership. Use the
same ordered H5 list, event caps, split fractions and seed during training.
The old default, --fit-split all, is retained for training-only H5 copies and
existing callers; it does not exclude held-out events from mixed input files.

The fitter reads feature arrays per file and stops at --max-objects in input
order. This cap is not a representative sample across all files: choose the
input list accordingly. It writes a .joblib and a .json describing the fit.

Repeat for the other objects with their configured feature lists:

| Object | Log features |
| --- | --- |
| jets | pt,mass,n_trk,QG_nTracks |
| electrons | pt,ptvarcone30 |
| muons | pt,ptvarcone30 |
| photons | pt,ptcone20 |
| taus | pt |
| tracks | pt,chiSquared |

## 3. Train and evaluate tokenizers

One tokenizer uses one GPU. num_quantizers is residual-code depth, not GPU count.
For the default direct-conversion campaign, use latent dimension 8 and:

| Objects | Q1 codebook | Q8 codebook |
| --- | ---: | ---: |
| electrons, muons, photons, jets | 16384 × 1 | 2048 × 8 |
| taus, tracks | 16384 × 1 | 4096 × 8 |

Other Q/codebook settings are supported by the tokenizer and standalone exporter,
but must not be substituted into a campaign with a different conversion policy.

Example Q1 jet training, using the same input list as preprocessing:

    pixi run python scripts/train.py \
      datamodule=atlas_event_object_iterable model=vqvae callbacks=event_tokenizer \
      seed=42 output_dir="$WORK/results" project_name=object_tokenizers \
      network_name=jets_q1 \
      "datamodule.data_paths=[/data/mc_001.h5,/data/data_001.h5]" \
      "datamodule.data_domains=[mc,data]" \
      datamodule.seed=42 datamodule.object_type=jets \
      datamodule.batch_size=1024 datamodule.num_workers=3 \
      datamodule.multiprocessing_context=spawn datamodule.shuffle_mode=buffered \
      model.codebook_dim=8 model.codebook_size=16384 model.num_quantizers=1 \
      +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \
      +datamodule.transforms.preprocess._partial_=true \
      +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \
      +datamodule.transforms.preprocess.cst_fn.filename="$WORK/preprocessing/jets_log_standard.joblib" \
      trainer.max_epochs=20 trainer.accelerator=gpu trainer.devices=1 \
      trainer.val_check_interval=1.0 trainer.check_val_every_n_epoch=1 \
      logger.offline=true

The epoch validation overrides are needed for small runs; the base training
config otherwise validates every 5,000 batches. Checkpoints and full_config.yaml
are saved under $WORK/results/object_tokenizers/jets_q1/.

For a short smoke test, use trainer.max_epochs=1, +trainer.limit_train_batches=20,
trainer.limit_val_batches=2 and +trainer.num_sanity_val_steps=0. A CPU test can
use trainer.accelerator=cpu and datamodule.num_workers=0. Keep the same event cap
when fitting and training.

For sparse objects, check that training/validation batches contain valid objects.
All-empty object batches are not supported by the tokenizer loss yet.
Codebook initialization and reset are optional and off by default; their settings
are in configs/model/vqvae.yaml. This pipeline does not change loss or optimizer
settings.

Evaluate a trained tokenizer:

    pixi run python scripts/analyze_vqvae_tokenizer.py \
      --run-dir "$WORK/results/object_tokenizers/jets_q1" \
      --split val --device auto

The evaluator uses the saved configuration, preprocessing and best.ckpt (falling
back to last.ckpt). It writes reconstruction plots/metrics and per-quantizer
codebook usage under figures/tokenizer_analysis/. Use --split test for final
held-out results. Changing the H5 list or event cap changes split membership.

The root analyze_vqvae_tokenizer.py delegates to the same evaluator. Evaluation
does not train a model or create Parquets.

## 4. Convert H5 directly to Parquet

Use scripts/tokenize_data_grl_multi_representation.py for the Google/data-GRL
pipeline. Settings are in configs/datamodule/data_grl_q1_q8_continuous_conversion.yaml;
the conversion policy is in src/heptokens/data/data_grl_conversion.py.

It reads each H5 in bounded batches, applies saved joblibs without refitting,
and runs frozen Q1/Q8 tokenizers. All three outputs share deterministic
approximately 90/10 train/validation membership based on source identity,
local event index and seed 42.

Defaults: batch size 1,024; at most 50,000 events per shard; 4,096 rows per Parquet
row group; sequence length 256. Each event contains CLS, four context positions,
one position per physical object, six separators and padding. Q1 tokens are
[256,1], Q8 tokens [256,8]. Continuous output stores the preprocessed features,
feature masks and position roles, without unused token columns.

There is no intermediate giant signal.parquet/background.parquet and no separate
resharing step. A shard can contain events from several H5 inputs; this is not
strictly one H5 to one Parquet. Long sequences are truncated in object order.

Stage the input H5s locally. Example group manifest at $WORK/group-00000.json:

    {
      "group_id": "group-00000",
      "files": [
        {
          "local_path": "/work/heptokens/inputs/data_001.h5",
          "source_uri": "gs://my-bucket/inputs/data_001.h5"
        }
      ]
    }

source_uri must identify the original file and remain unchanged across staging
locations and retries. Assign each source to exactly one group. The converter
does local file I/O; the cloud launcher handles GCS downloads/uploads.

Supply matching Q1/Q8 checkpoints and the six joblibs. Q1 and Q8 must use the same
saved preprocessing in this paired converter. For a bundle arranged as
artifacts/q1/<object>.ckpt, artifacts/q8/<object>.ckpt and
artifacts/preprocessing/<object>.joblib:

    A="$WORK/artifacts"
    Q1=()
    Q8=()
    PREPROCESSORS=()
    for object in electrons muons taus photons jets tracks; do
      Q1+=("$object=$A/q1/$object.ckpt")
      Q8+=("$object=$A/q8/$object.ckpt")
      PREPROCESSORS+=("$object=$A/preprocessing/$object.joblib")
    done

    pixi run python scripts/tokenize_data_grl_multi_representation.py \
      --conversion-config configs/datamodule/data_grl_q1_q8_continuous_conversion.yaml \
      --input-manifest "$WORK/group-00000.json" \
      --datamodule-config configs/datamodule/atlas_event_object_iterable.yaml \
      --q1-tokenizer-checkpoints "${Q1[@]}" \
      --q8-tokenizer-checkpoints "${Q8[@]}" \
      --preprocess-transformers "${PREPROCESSORS[@]}" \
      --output-dir "$WORK/prepared" --device cuda

Use a fresh output directory for a smoke test, with --num-events-per-file 128.
The default campaign is collision data only; --allow-mc explicitly permits MC.
Zero-byte/unreadable H5s are recorded and skipped. Review invalid-input counts.
Zero-event conversion fails rather than writing a success marker.

Output layout:

    prepared/
      q1/{train,val}/part-group-00000-*.parquet
      q8/{train,val}/part-group-00000-*.parquet
      continuous/{train,val}/part-group-00000-*.parquet
      conversion_status/group-00000.json
      conversion_status/group-00000.SUCCESS.txt

The JSON records counts, settings, aligned part lists, checksums and a membership
fingerprint. Vocabulary/continuous schema metadata is embedded in the Parquets.

## 5. Finalize manifests

The Parquets already have their train/validation split. Do not pass them through
prepare_token_parquet_pretrain_shards.py.

Example $WORK/campaign_plan.json for the single input above:

    {
      "groups": [{"group_id": "group-00000", "manifest": "group-00000.json"}],
      "input_file_count": 1,
      "inventory_sha256": "<SHA256 of your selected input inventory>"
    }

Group manifest paths are relative to the plan directory. Use the actual input
count and inventory digest. Once every group has completed:

    pixi run python scripts/finalize_data_grl_conversion_manifests.py \
      --campaign-plan "$WORK/campaign_plan.json" \
      --status-dir "$WORK/prepared/conversion_status" \
      --output-dir "$WORK/prepared/manifests"

This validates group/source assignments and aligned shard counts without
rewriting data. Place each output manifest next to its train/val directories:

    for representation in q1 q8 continuous; do
      cp "$WORK/prepared/manifests/${representation}_manifest.json" \
        "$WORK/prepared/$representation/manifest.json"
    done

The prepared directories are $WORK/prepared/q1, q8 and continuous. Tokenizer
training splits, conversion train/validation membership and downstream test
selection are separate. Do not use capped smoke outputs for production.

## Other exporters

These remain available but are not steps in the direct Google pipeline:

| Script | Use |
| --- | --- |
| tokenize_objects_to_grouped_parquet.py | One grouped --output from the supplied H5s; optional continuous columns. No split. |
| tokenize_objects_to_parquet_with_atlasopenmagic_metadata.py | Older flat token layout; no continuous output. |
| prepare_token_parquet_pretrain_shards.py | Split already-exported Parquets. Not needed after direct conversion. |
| prepare_grouped_hzz_classification_shards.py | Labeled shards for the specific HZZ 345060 vs ZZ 700600 task. |

All four are under scripts/. For standalone grouped exports, set
--max-seq-length 256 --event-bins 128 to match the layout above.
--write-continuous-features adds direct continuous inputs;
--write-decoded-q8-features additionally stores decoded reconstructions.

## Tests

    pixi run python -m unittest discover -s tests -v

Tests cover event splitting/shuffling, saved preprocessing, checkpoint evaluation,
nested Arrow values, aligned Q1/Q8/continuous conversion and streaming readback.
End-to-end tests use small synthetic H5s and untrained fixture checkpoints; they
do not establish reconstruction quality.

The direct converter also passed a Zephyr GPU smoke with real H5 data and the
saved production tokenizers: 128 events (116 train, 12 validation), exact values
preserved across shard sizes and aligned outputs for all three representations.
