# heptokens (event-level implementation)

> [!CAUTION]
> This implementation is done with the first version of heptokens, this will be moved to https://github.com/Treasure-AmSC/heptokens-atlas later in the project.



Train object-level VQ-VAE/RVQ tokenizers, convert collider events to grouped token sequences, pretrain a masked event transformer, and fine-tune it for event-level classification.

Follow the stages in this order:

1. Prepare HDF5 inputs and a datamodule configuration.
2. Fit one preprocessing transform per object type.
3. Train one tokenizer per object type.
4. Export grouped event Parquet files.
5. Build train/validation Parquet shards.
6. Pretrain the masked event model.
7. Build labeled shards and fine-tune a classifier.

The project assumes a fixed object order and one tokenizer configuration across every export, pretraining, and downstream run.

## 1. Install

Create the project environment with Pixi:

    pixi install

Run all commands from the repository root:

    cd /path/to/heptokens

Use the same environment for every stage:

    pixi run python --version

## 2. Prepare HDF5 data
Current HDF5 files are produced using the following code: https://gitlab.cern.ch/vcavalie/bnl-treasure/-/blob/master/download_and_convert.py?ref_type=heads

The default configuration is:

    configs/datamodule/atlas_event_object.yaml

Your HDF5 files must use the same collection and feature paths declared in the selected datamodule YAML. Arrays must be event-major:

| Content | Required shape |
| --- | --- |
| Object feature | [n_events, max_objects] |
| Object mask | [n_events, max_objects] |
| Event feature | [n_events] |

The default ATLAS layout is:

    file.h5
    ├── common/
    │   ├── event/         # pvx, pvy, pvz, mu
    │   └── met/           # pt, phi, sumet
    ├── jets/              # object features and mask
    ├── electrons/
    ├── muons/
    ├── photons/
    ├── taus/
    └── tracks/

For each object collection, define:

- a mask with the same first two dimensions as its features;
- pT, eta, and phi for objects where they exist;
- every feature used by the tokenizer;
- the same paths for tokenizer training and Parquet export.

### Use data stored elsewhere

Copy the default datamodule config and edit paths, object names, masks, feature names, and feature order for your files:

    cp configs/datamodule/atlas_event_object.yaml \
      configs/datamodule/my_event_data.yaml

Select this config in training with:

    datamodule=my_event_data

Pass the full YAML path to standalone scripts:

    --datamodule-config configs/datamodule/my_event_data.yaml

Do not change feature order after training a tokenizer. The export config must match the config used for that tokenizer.

If your HDF5 inputs already contain sample metadata, disable Atlas Open Magic metadata lookup during export:

    --metadata-source h5 --atlasopenmagic-release ""

## 3. Fit preprocessing transforms

Fit one transform per object type before tokenizer training. Save the transform and reuse the identical file when training and exporting that object.

Example for jets:

    pixi run python scripts/get_atlas_object_preprocessing.py \
      --h5-files /data/events/train_001.h5 /data/events/train_002.h5 \
      --datamodule-config configs/datamodule/my_event_data.yaml \
      --object-type jets \
      --mode log_standard \
      --log-features pt,mass,n_trk,QG_nTracks \
      --output-dir /work/heptokens/preprocessing \
      --output-name jets_log_standard

Typical log features used by the ATLAS configuration are:

| Object | Log features |
| --- | --- |
| jets | pt,mass,n_trk,QG_nTracks |
| electrons | pt,ptvarcone30 |
| muons | pt,ptvarcone30 |
| photons | pt,ptcone20 |
| taus | pt |
| tracks | pt,chiSquared |

Check that every expected transform exists before training:

    ls /work/heptokens/preprocessing/*.joblib

## 4. Train object tokenizers

Train one independent VQ-VAE/RVQ tokenizer per object type. Each tokenizer has its own codebook and preprocessing transform.

The main residual-quantization settings are:

| Setup | num_quantizers | codebook_size | codebook_dim |
| --- | ---: | ---: | ---: |
| q1 control | 1 | 16384 | 8 |
| q4 full | 4 | chosen per object | 16 |
| q8 scan/full | 8 | chosen per object | 8 |

The q1 cb16384 dim8 control has the same total codebook-vector capacity as q8 cb2048 dim8:

    1 × 16384 × 8 = 8 × 2048 × 8

Example: train a q1 jet tokenizer on a GPU:

    pixi run python scripts/train.py \
      datamodule=my_event_data model=vqvae callbacks=event_tokenizer \
      output_dir=/work/heptokens/results \
      project_name=my_tokenizers \
      network_name=jets_full_dim8_cb16384_q1_e20 \
      "datamodule.data_paths=[/data/events/train_001.h5,/data/events/train_002.h5]" \
      datamodule.object_type=jets \
      datamodule.batch_size=1024 \
      model.codebook_dim=8 \
      model.codebook_size=16384 \
      model.num_quantizers=1 \
      +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \
      +datamodule.transforms.preprocess._partial_=true \
      +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \
      +datamodule.transforms.preprocess.cst_fn.filename=/work/heptokens/preprocessing/jets_log_standard.joblib \
      trainer.max_epochs=20 \
      trainer.accelerator=gpu \
      trainer.devices=1

Expected training output:

    /work/heptokens/results/my_tokenizers/
      jets_full_dim8_cb16384_q1_e20/
        full_config.yaml
        checkpoints/best.ckpt
        checkpoints/last.ckpt

Use best.ckpt for evaluation and Parquet export. Use last.ckpt only to resume an interrupted run.

### Train one shared tokenizer across object types

Use a shared tokenizer only for collections configured with exactly the same feature names and feature order. For example, all object types can share a kinematics-only tokenizer with:

    pt, eta, phi

The repository already provides this shared-tokenizer implementation in configs/datamodule/atlas_event_mappable.yaml. It uses output_mode: combined and defines the common kinematics inputs for its selected collections. Add tracks to that config with pt, eta, phi when you want it in the shared tokenizer. Keep the separate full-feature configuration for object-specific tokenizers; its feature lists are intentionally heterogeneous.


Use or copy the combined datamodule configuration:

    cp configs/datamodule/atlas_event_mappable.yaml \
      configs/datamodule/my_shared_kinematics.yaml

Set output_mode to combined. Give every selected collection the same inputs, in the same order. Combined mode concatenates valid objects from those collections into one object axis and trains one tokenizer on that pooled object sample.

Fit one preprocessing transform across all selected object collections:

    pixi run python scripts/get_atlas_combined_object_preprocessing.py \
      --h5-files /data/events/train_001.h5 /data/events/train_002.h5 \
      --datamodule-config configs/datamodule/my_shared_kinematics.yaml \
      --object-types jets electrons muons taus photons \
      --mode log_standard \
      --log-features pt \
      --output-dir /work/heptokens/preprocessing \
      --output-name combined_kinematics_log_standard

Train the shared tokenizer:

    pixi run python scripts/train.py \
      datamodule=my_shared_kinematics \
      model=transformer_vqvae \
      callbacks=event_tokenizer \
      output_dir=/work/heptokens/results \
      project_name=my_tokenizers \
      network_name=combined_kinematics_q1 \
      "datamodule.data_paths=[/data/events/train_001.h5,/data/events/train_002.h5]" \
      datamodule.batch_size=1024 \
      model.codebook_dim=8 \
      model.codebook_size=16384 \
      model.num_quantizers=1 \
      +datamodule.transforms.preprocess._target_=heptokens.data.collation.preprocess_objects_batch \
      +datamodule.transforms.preprocess._partial_=true \
      +datamodule.transforms.preprocess.cst_fn._target_=joblib.load \
      +datamodule.transforms.preprocess.cst_fn.filename=/work/heptokens/preprocessing/combined_kinematics_log_standard.joblib \
      trainer.max_epochs=20 \
      trainer.accelerator=gpu \
      trainer.devices=1

When exporting a shared tokenizer, use the same checkpoint and the same combined preprocessing transform for every object collection included in that shared setup.

Expected training output:

    /work/heptokens/results/my_tokenizers/
      jets_full_dim8_cb16384_q1_e20/
        full_config.yaml
        checkpoints/best.ckpt
        checkpoints/last.ckpt

Use best.ckpt for evaluation and Parquet export. Use last.ckpt only to resume an interrupted run.

## 5. Export grouped token Parquet

The grouped format keeps each physical object at one sequence position. The token vector at that position contains Q residual code IDs:

    tokens:   [n_events, sequence_length, Q]
    mask:     [n_events, sequence_length]
    type_ids: [n_events, sequence_length]

For a q8 object, the eight residual code IDs q0 through q7 remain together at that one position:

    [q0, q1, q2, q3, q4, q5, q6, q7]
      → 8 code embeddings of dimension 32
      → concatenate: 8 × 32 = 256
      → linear object projection: 256 → d_model = 256
      → add object-type embedding and position embedding
      → one vector passed to the event transformer

      
This is a concatenate-and-project operation, not mean pooling. q1 and q4 follow the same procedure with one or four code embeddings. In a mixed-Q export, its common tensor uses the largest Q and pads unused code slots. Therefore one physical object always contributes one event-sequence position, independent of Q.


### Set and check sequence length

The exported Parquet arrays are padded to --max-seq-length. The current pretraining and classification models use:

    max_seq_length = 256

With the default export options and six object collections, the number of valid positions in an event is:

    1 [CLS] + 4 event-context tokens + number of valid objects + 6 [SEP] tokens

Thus an event with no valid physics objects has 11 valid positions, while the maximum is 256. The stored arrays always have length 256; sum(mask) gives the actual valid length of each event.

The exporter appends positions in the supplied --object-order. When an event exceeds 256 positions, later positions are dropped. Put the most important collections first, keep the same order in every export, and measure truncation before the production export:

    pixi run python scripts/analyze_event_sequence_lengths.py \
      --h5-files /data/events/sample_001.h5 \
      --datamodule-config configs/datamodule/my_event_data.yaml \
      --object-quantizers jets=8 electrons=8 muons=8 photons=8 taus=8 tracks=8 \
      --object-order jets electrons muons photons taus tracks \
      --max-seq-length 256 \
      --output-json /work/heptokens/sequence_length_report.json

Use the quantizer count from the selected tokenizer for each object. For example, replace every 8 above with 1 for a q1 export, or use the actual mixed q settings when applicable.

The sequence includes [CLS], event-context tokens, object positions grouped by type and separated by [SEP], and padding. Object order must be fixed across signal, background, data, pretraining, and fine-tuning exports.


Run a small two-object export first:

    TOKENIZERS=( \
      jets=/work/heptokens/results/my_tokenizers/jets_full_dim8_cb16384_q1_e20/checkpoints/best.ckpt \
      electrons=/work/heptokens/results/my_tokenizers/electrons_full_dim8_cb16384_q1_e20/checkpoints/best.ckpt \
    )

    PREPROCESSORS=( \
      jets=/work/heptokens/preprocessing/jets_log_standard.joblib \
      electrons=/work/heptokens/preprocessing/electrons_log_standard.joblib \
    )

    pixi run python scripts/tokenize_objects_to_grouped_parquet.py \
      --h5-files /data/events/sample_001.h5 \
      --output /work/heptokens/parquet/events_q1.parquet \
      --datamodule-config configs/datamodule/my_event_data.yaml \
      --tokenizer-checkpoints "${TOKENIZERS[@]}" \
      --preprocess-transformers "${PREPROCESSORS[@]}" \
      --object-order jets electrons \
      --max-seq-length 256 \
      --device cuda \
      --metadata-source h5 \
      --atlasopenmagic-release ""

For a production export:

1. Supply all selected object checkpoints.
2. Supply all matching preprocessing transforms.
3. Use one fixed object order.
4. Export signal, background, and data separately.
5. Inspect the Parquet schema and metadata before preparing shards.

> [!CAUTION]
> The current code merges many HDF5 inputs into one Parquet file per sample category: signal, background, and data. This is good for the existing runs, but it is not the intended scalable layout. Future production processing should convert each HDF5 file independently to a corresponding Parquet file, then build pretraining or classification shards from the complete set of per-file Parquets. Preserve source_file and event_index so splits remain deterministic and no source event is duplicated across splits.

The Parquet schema metadata stores the token vocabulary. The downstream model reads this metadata to choose the correct vocabulary/output heads. Do not hardcode a vocabulary from another q setup.

For the paired continuous benchmark, use:

    --write-continuous-features --write-decoded-q8-features

This adds aligned fields:

    continuous_features
    continuous_feature_mask
    position_role_ids
    decoded_continuous_features

Use these fields only when comparing direct continuous or decoded-reconstruction baselines with the tokenized input.

## 6. Prepare pretraining shards

Create source-file-stratified train/validation shards from the exported grouped Parquet files:

    pixi run python scripts/prepare_token_parquet_pretrain_shards.py \
      --input-parquets \
        /work/heptokens/parquet/signal_q1.parquet \
        /work/heptokens/parquet/background_q1.parquet \
        /work/heptokens/parquet/data_q1.parquet \
      --output-dir /work/heptokens/shards/q1_pretrain \
      --train-frac 0.90 \
      --seed 42 \
      --shard-rows 50000

The output directory must contain a manifest and train/validation shards. Train from this prepared directory, not directly from one large input Parquet file.

For a smoke test, add:

    --max-rows-per-input 10000

Do not use a smoke-test shard directory for final results.

## 7. Pretrain the grouped event transformer

Pretraining masks complete object token groups and predicts their residual code IDs. It learns cross-object structure from the grouped event sequences.

Set model.max_quantizers to the same Q used during Parquet export:

| Exported tokens | Required setting |
| --- | ---: |
| q1 | model.max_quantizers=1 |
| q4 | model.max_quantizers=4 |
| q8 | model.max_quantizers=8 |

Example q1 pretraining run:

    pixi run python scripts/train.py \
      datamodule=token_parquet_pretrain \
      model=foundation_grouped_pretrain \
      callbacks=pretrain \
      output_dir=/work/heptokens/results \
      project_name=my_foundation \
      network_name=q1_grouped_pretrain \
      datamodule.prepared_dir=/work/heptokens/shards/q1_pretrain \
      model.max_quantizers=1 \
      trainer.max_epochs=10 \
      trainer.accelerator=gpu \
      trainer.devices=1

Keep the same tokenizer setup, Parquet vocabulary, sequence layout, and maximum number of quantizers throughout pretraining.

## 8. Prepare labeled classification shards

For the supplied HZZ signal-versus-background workflow, create balanced train/validation/test shards:

    pixi run python scripts/prepare_grouped_hzz_classification_shards.py \
      --signal-parquet /work/heptokens/parquet/signal_q1.parquet \
      --background-parquet /work/heptokens/parquet/background_q1.parquet \
      --signal-dsid 345060 \
      --background-dsid 700600 \
      --output-dir /work/heptokens/shards/q1_hzz_classification \
      --train-frac 0.70 \
      --val-frac 0.15 \
      --seed 42 \
      --shard-rows 50000

For another downstream task, create equivalent train/validation/test shards with:

- a single label column;
- source_file and event_index retained for deterministic splits;
- no event overlap between splits;
- a manifest describing the shard set.

This helper is specific to one binary HZZ signal/background selection. 

## 9. Fine-tune a classification head

Start from a pretrained checkpoint and train the grouped [CLS] classification head:

    pixi run python scripts/train.py \
      datamodule=token_parquet_grouped_classification \
      model=foundation_grouped_cls_classifier \
      callbacks=grouped_classification \
      output_dir=/work/heptokens/results \
      project_name=my_downstream \
      network_name=q1_hzz_finetuned \
      datamodule.prepared_dir=/work/heptokens/shards/q1_hzz_classification \
      model.backbone_ckpt_path=/work/heptokens/results/my_foundation/q1_grouped_pretrain/checkpoints/best.ckpt \
      model.freeze_backbone=false \
      model.max_quantizers=1 \
      trainer.max_epochs=10 \
      trainer.accelerator=gpu \
      trainer.devices=1

To measure the value of pretraining, run the same classifier configuration without model.backbone_ckpt_path. Keep the tokenizer setup, prepared shards, split seed, and training budget fixed.

### Use masked mean pooling instead of [CLS]

Use the mean-pooling classifier when you want the downstream head to average the final hidden states at all valid sequence positions, rather than classify from the final [CLS] state alone. This changes only the classification head. Keep the tokenizer checkpoints, grouped Parquet files, shard directory, and pretrained backbone unchanged.

Run the mean-pooling version by replacing the classifier model:

    pixi run python scripts/train.py \
      datamodule=token_parquet_grouped_classification \
      model=foundation_grouped_mean_classifier \
      callbacks=grouped_classification \
      output_dir=/work/heptokens/results \
      project_name=my_downstream \
      network_name=q1_hzz_mean_pool_finetuned \
      datamodule.prepared_dir=/work/heptokens/shards/q1_hzz_classification \
      model.backbone_ckpt_path=/work/heptokens/results/my_foundation/q1_grouped_pretrain/checkpoints/best.ckpt \
      model.freeze_backbone=false \
      model.max_quantizers=1 \
      trainer.max_epochs=10 \
      trainer.accelerator=gpu \
      trainer.devices=1

The masked mean includes [CLS] by default. Exclude it when you want the average to use only event and object positions:

    model.exclude_first_position=true

For a controlled [CLS] versus mean-pooling comparison, change only model=foundation_grouped_cls_classifier to model=foundation_grouped_mean_classifier, and keep every other setting fixed.

## 10. Resume, monitor, and validate

Resume tokenizer training from its last checkpoint:

    pixi run python scripts/train.py \
      ... \
      ckpt_path=/work/heptokens/results/my_tokenizers/jets_full_dim8_cb16384_q1_e20/checkpoints/last.ckpt

Before launching a large run, check:

    test -f /work/heptokens/preprocessing/jets_log_standard.joblib
    test -f /work/heptokens/results/my_tokenizers/jets_full_dim8_cb16384_q1_e20/checkpoints/best.ckpt
    test -f /work/heptokens/parquet/signal_q1.parquet
    test -d /work/heptokens/shards/q1_pretrain

For every final comparison, record:

1. HDF5 data selection and datamodule YAML revision.
2. Object feature lists and preprocessing transforms.
3. Tokenizer checkpoint for every object.
4. Q, codebook size, codebook dimension, and sequence length.
5. Parquet vocabulary metadata.
6. Shard seed and split fractions.
7. Pretraining checkpoint and fine-tuning configuration.

## Repository map

| Path | Purpose |
| --- | --- |
| configs/datamodule/atlas_event_object.yaml | Default object features, masks, and HDF5 paths |
| scripts/get_atlas_object_preprocessing.py | Fit object preprocessing transforms |
| scripts/train.py | Hydra training entry point |
| scripts/tokenize_objects_to_grouped_parquet.py | Encode object checkpoints into grouped event Parquet |
| scripts/prepare_token_parquet_pretrain_shards.py | Build pretraining shards |
| scripts/prepare_grouped_hzz_classification_shards.py | Build balanced HZZ classification shards |
| configs/model/foundation_grouped_pretrain.yaml | Masked grouped-token pretraining model |
| configs/model/foundation_grouped_cls_classifier.yaml | Grouped [CLS] classification model |

