# Parallel Q8 and flat continuous

This directory contains the two main ATLAS baselines: their archived model
source, two pretraining configs, and the five-class reconstructed four-lepton
fit code and plan. It does not change the tokenizer or Parquet pipeline.

The Python files here are readable source extracts. Run the commands from
the matching **unpacked archival runtime**, which supplies the shared code
and original environment. Do not combine pretraining and downstream
model files from different archives.

| Runtime | SHA-256 |
| --- | --- |
| Pretraining: `heptokens-google-data-grl-b300.tar.gz` | `ab256c9feb6aab2195570a67f5ede231f95ed5bd26dad329407b6a2b8f88d558` |
| Four-lepton: `hzz_4l_core_multiclass_v1/runtime.tar.gz` | `413caa8972a69a4bbac5f4a8460d66d355e03b90bec128b7f13945cfa8db0c62` |

## Pretraining

| Model | Seeds 42 / 43 / 44 | Prepared manifest SHA-256 |
| --- | --- | --- |
| Parallel Q8 | 131.0 / 136.0 / 136.1 | `20ae12ae48c8420e0605159ddd67430560de38f48194caf2001673f217cf7829` |
| Flat continuous | 133.0 / 137.0 / 137.1 | `6e4d519a003efd3fc8e4d79a01d761323dd25dd0affa7b48be01f2dce1a14628` |

Use the prepared Data16 shards for the selected representation. The original
environment lock is pinned by SHA-256
`ba154627f585bf1013dbfbca459589a4a78fd04a3d004f9b84a6b74717507aa9`.
Set `ATLAS_PREPARED_DIR` and `ATLAS_OUTPUT_DIR` to absolute input and new
output paths. From the pretraining runtime, with `PYTHONPATH` pointing to
its absolute `src/` path, run one of:

```sh
python scripts/train.py --config-path "$configs" --config-name q8
python scripts/benchmark_train.py --config-path "$configs" --config-name continuous
```

Here `$configs` is the absolute path of this repository's
[`pretraining/`](pretraining/) directory. The two YAML files use the saved
settings, with portable paths and seed-derived run names. Both train for
10 epochs on 8 GPUs at 448 events per GPU. Use `seed=43` or `seed=44` for
the other runs; flat continuous keeps data-loader seed 42, as in the originals.

## Five-class four-lepton fits

Use the original event splits for ggF, VBF, VH, top-associated Higgs, and
continuum ZZ. The cut requires active reconstructed electrons or muons
with unit charge, finite kinematics, `pT > 10 GeV`, and
`abs(eta) < 2.47` (electron) or `2.5` (muon). The leading candidate lepton
must satisfy `pT > 20 GeV`. Require two disjoint same-flavour opposite-sign
pairs, `40 < mZ1 < 120 GeV`, `12 < mZ2 < 120 GeV`,
`deltaR > 0.02` for every pair, opposite-sign pair masses above 4 GeV,
and `m4l > 70 GeV`. There is no narrow Higgs-mass cut. Taus and other
objects remain model inputs. The [candidate kernel](four_lepton/candidate.py)
takes the leptons that pass the upstream cuts.

The frozen sample has **300,279 train / 64,264 validation / 64,403 test**
events. The [fit plan](four_lepton/plan.json) retains just the Q8 and flat
continuous model and checkpoint records from DAG 521. Use the verified
`common.tar.gz` (selection and schedules) and only the representation's
`q8.tar.gz` or `continuous.tar.gz`. The archived inputs are under
`gs://fcc-teststorage/bucket_results/foundation_pretrain/downstream_data/hzz_4l_core_multiclass_v1/`.
Their SHA-256 values are:

| Input | SHA-256 |
| --- | --- |
| Common | `458b2a8625f28135020ff043bdc99f2d2e22f2cac7eccd39dcd3afd3f00b5112` |
| Q8 | `ee9a5f2cde951ebaceabd87e250e78b8f6a5b53e42e485423dc162fc79dc5c0a` |
| Continuous | `bb8874a512dbc4e066cf86da9aecfb5fce7d65df892c1a23d043388dfdb44eca` |

Unpack common and the chosen representation into one dataset root. In the
four-lepton runtime, set `PYTHONPATH=src:scripts` and run one fit, e.g.:

```sh
python scripts/hzz_4l_core_multiclass_study.py \
  --plan "$plan" --dataset-root "$dataset" --output-dir "$out" \
  --run-key q8_finetune_p42 \
  --membership-sha256 b9f2ac8bba919c9ace97679a4dabd37b3dc39560df76e73348dfc0214288f3c6 \
  --source-dir "$source"
```

Use `q8` or `continuous_flat` as the run-key prefix. Fine-tuned and frozen
keys end in `_finetune_p42` or `_frozen_p42` (also 43 and 44); their
classifier seed is 42. Stage the plan's pinned source assets as
`checkpoint.ckpt`, `config.yaml`, `status.txt`, and, when listed,
`vocabulary.parquet`. Scratch keys end in `_scratch_s42` (also 43 and 44)
and omit `--source-dir`. Use `--preflight-only` to check inputs and one
batch without training.

The original trainer uses one GPU, ten epochs, batch 256, matched
canonical schedules, and a linear five-class head. It selects the best
validation macro-AUC checkpoint and exports held-out predictions.
