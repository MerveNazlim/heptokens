# ATLAS four-lepton downstream studies

This is a source snapshot for the **five-class full four-lepton study (DAG
521)** and the **three-arm Paper 0 binary study (jobs 520.0–520.2)**. It does
not replace the repository's tokenizer, converter or generic Parquet pipeline.

- `source/` preserves the paths of the downstream study entrypoints, model
  code, pinned lock files and their study-specific helpers. Archived source
  members are byte-identical. The few local data-preparation or plotting
  recipes that were never worker archive members are identified separately in
  [source_provenance.json](source_provenance.json), including a fixed-threshold
  source restoration and one import-only adaptation.
- `launch/core521/` and `launch/binary520/` are the exact generated text
  members of the historical launch bundles. These are records, not
  instructions to resubmit existing jobs. Their references to the original
  site's transfer credential are paths only; no credentials are included. An
  authorized operator uses their own site configuration in a separate working
  copy.

No event data, membership arrays, Parquet shards, pretrained weights, private
keys or runtime tarballs are included.

## Exact reproduction

Use the original, checksum-verified runtime in an isolated working directory
for each campaign. The historical workers check the archive SHA-256, so a
freshly assembled archive is not interchangeable, even if much of its source
matches. Some generic modules in the historic archives differ from those in
the repository; do not overwrite the repository's versions to install this
snapshot. The launcher receipts and `source_provenance.json` give the original
artifact URIs and checksums. Fetch only with authorized access and verify
content digests before extraction or loading checkpoints.

| Campaign | Worker runtime SHA-256 | Test population |
| --- | --- | ---: |
| DAG 521, five-class Paper 0 core | `413caa8972a69a4bbac5f4a8460d66d355e03b90bec128b7f13945cfa8db0c62` | 64,403 |
| Job 520, Paper 0 binary | `d5fa5c90e1d267c626833e3cffcee9a215746f06b568f27a5f0b091643d632df` | 47,705 |

The respective worker scripts pin the locked Linux runtime and unpack the
marked dataset archives. Inspect the workers for exact commands and pretraining
assets; the launchers are not required for inspecting code or existing
predictions.

## Five-class full four-lepton (DAG 521)

The reconstructed four-lepton candidate requires each lepton to have
`pT > 10 GeV`, with `pT > 20 GeV` for the leading lepton. It also requires
two disjoint opposite-sign same-flavor pairs, the original Z-mass, separation
and low-mass-pair cuts, and `m4l > 70 GeV`.
The cut is applied to the original event IDs without resplitting. The
`hzz_4l_paper0_core_v1` cohort contains 300,279 / 64,264 / 64,403 events
(train / validation / test). Only the selected rows are copied; all normal
input objects remain, including reconstructed taus.

- Local preparation: `scripts/prepare_hzz_4l_core_multiclass.py select`
  consumes the pinned original Q8 Parquet, original canonical index, H5
  inventory and raw H5 sources. It verifies the checksum-pinned candidate
  helper; this is not the later variant that removed the 20-GeV leading cut. `filter`
  applies frozen IDs to Q1/Q4/Q8 and the verified continuous superset.
  `build-index` writes the canonical index and 30 train schedules. The
  archived `prepare_hierarchical_multiclass_from_h5.py` supplies the original
  H5 continuous feature preparation when the superset is not already
  available. The continuous values are preprocessed original features, not
  decoded Q8 predictions.
- Worker training: `scripts/hzz_4l_core_multiclass_study.py` calls the
  archived shared trainer with the core cohort and budget (1,172 batches per
  epoch, ten epochs). Five model families have each of fine-tune, scratch and
  frozen conditions, with three replicates. Scratch varies supervised seed and
  loads no pretraining checkpoint; the other two conditions vary pretraining
  seed with classifier seed 42. Frozen backbone parameters and trainability
  are audited separately from full fine-tuning. Checkpoints are chosen by
  validation macro one-versus-rest AUC.
- Worker result export: `scripts/collect_hzz_4l_core_results.py` and
  `scripts/plot_hzz_4l_core_multiclass.py` check saved IDs and turn the
  original held-out probabilities into seed-aware figures.

The fixed candidate helper staged here has the digest in the frozen cohort
receipt. For local H5 preparation only, use the copied recipes within the
isolated historical runtime and ensure the ROOT audit helper's optional
`uproot`/Awkward imports are available. No ROOT conversion or tokenizer
retraining is part of these downstream reproduction steps.

## Paper 0 binary (job 520)

Use `scripts/prepare_paper0_atlas_hzz_selected_q1.py` for the fixed-20-GeV
candidate selection and `scripts/prepare_paper0_atlas_hzz_selected_continuous.py`
for the exact matched continuous view. The latter uses the historical
exporter's matching feature readers as an import-only substitution; it does
not alter their source, retrain preprocessing or derive features from tokens.
The sealed datasets retain original full Q1 event splits, without balancing
by discarding events. They are event-disjoint, not H5-file-disjoint.

`scripts/train_atlas_paper0_q1_4l.py` runs Q1 scratch, Q1 fine-tuning from
pretraining job 130.0 or continuous fine-tuning from 133.0. There is no frozen
Paper 0 binary arm. The worker emits per-arm held-out predictions and ROC
arrays; test AUC is unweighted. The local
`scripts/plot_atlas_paper0_validation_test.py` recipe combines validation and
test only for an explicitly labeled diagnostic, not as the independent paper
test result. The separate range-only replot is also included.

Packaging recipes included for traceability require the named frozen base
runtime and external input artifacts; they do not recreate private/site
credentials or provide authorization to launch.
