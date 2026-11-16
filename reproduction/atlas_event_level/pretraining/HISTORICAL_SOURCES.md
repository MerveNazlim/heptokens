# Historical pretraining sources

These are the source profiles behind the five-model, four-lepton multiclass
benchmark. The downstream cohort has 64,403 held-out events; these pretrained
checkpoints were trained earlier on the full aligned event corpus, not that cohort.
Only training source, configuration, and selected provenance fields are included.
No data, checkpoint weights, credentials, or run logs are included.

| Paper configuration | Pretraining jobs (seeds 42 / 43 / 44) | Frozen layer(s), in application order | Actual entrypoint |
| --- | --- | --- | --- |
| RVQ-1, 32-dimensional token lookup with learned projection | `130.0 / 135.0 / 135.1` | `baseline_ab256` | `scripts/train.py` |
| RVQ-4, independent tokenizers and 4×64 lookup with identity projection | `197.0 / 297.0 / 298.0` | `variant_2be1e53` + `q4_2cbb72e` | `scripts/train.py` |
| Parallel RVQ-8, 8×32 lookup with learned projection | `131.0 / 136.0 / 136.1` | `baseline_ab256` | `scripts/train.py` |
| Flat continuous | `133.0 / 137.0 / 137.1` | `baseline_ab256` | `scripts/benchmark_train.py` |
| Hierarchical continuous, kinematics/auxiliary grouping | `287.0 / 299.0 / 300.0` | `variant_2be1e53` + `hierarchical_497ac79` | `scripts/benchmark_train.py` |

Each directory under [`frozen/`](frozen/) is a **partial, byte-for-byte source
extract**, with its own `source_provenance.json` linking every file to the exact
archive and member. It is not a complete archived environment. When interpreting a
profile with an overlay, a path present in both layers means the overlay replaced
that path. Existing tokenizer and data-export code is not part of these extracts.
Model input readers are included because they were used during training.

The full SHA-256 values are:

| Archive | SHA-256 |
| --- | --- |
| `heptokens-google-data-grl-b300.tar.gz` | `ab256c9feb6aab2195570a67f5ede231f95ed5bd26dad329407b6a2b8f88d558` |
| `heptokens-google-data-grl-variants-b300-v2.tar.gz` | `2be1e53caa4eabe65a878d5bbea3d84a1118618585b5a6a6e9b52327869f4b2f` |
| `heptokens-q4-campaign-update.tar.gz` | `2cbb72e615432973909d0735c194a9ee847a95273cf47bcc46fc704248ff0934` |
| `heptokens-hierarchical-kin-aux-update-v2.tar.gz` | `497ac79701f35f355d62c6fe2cfcc5098b8b9536858b128b8e45d98c6f609b39` |

Both base archive checksums were independently verified on the retained Condor
copies. The variant-base bytes were retrieved read-only and checked again before
extraction. The overlays match the hashes recorded by the corresponding jobs.
Original model artifacts and prepared-input manifest hashes are pinned in
[`reference/run_index.json`](reference/run_index.json) and the linked
[`reference/receipts/`](reference/receipts/) records. All 15 original
`full_config.yaml` and `job_status.txt` hashes match the pins in the four-lepton
benchmark launch plan. The published receipts retain only allowlisted configuration
and completion fields; the original private log paths and host information are omitted.
The resolved Pixi lock was also verified for every executed run; all 15 match
[`reference/condor/q4-latest-hier-seed42-pixi.lock`](reference/condor/q4-latest-hier-seed42-pixi.lock)
(`ba154627f585bf1013dbfbca459589a4a78fd04a3d004f9b84a6b74717507aa9`).
The later submit-host `pixi-b300.lock` did not match and is not included.

The files in [`reference/condor/`](reference/condor/) are historical **references**,
not ready-to-submit commands. Site-specific bucket names and credential filenames
are replaced by placeholders. The reference ledger preserves the original hashes
separately from the published file hashes. For the original RVQ-1/RVQ-8 workers in
particular, the retained worker script has changed since the reported jobs: its
later projection-mode override is not supported by the original archived constructor
and did not appear in the executed resolved configs. Use the frozen source and
resolved settings in the checked receipts as the record of those jobs, not the
currently retained worker. The published RVQ-4 and hierarchical wrappers also
refer to site-specific staging and additional historical files that are not in this
source-only supplement.

## Experimental distinctions

- All five models used 10 epochs, 15% initial masking, hidden width 256, four outer
  transformer layers and eight heads. For RVQ-1 and RVQ-8, an empty mask is repaired
  only at the minibatch level on each rank. RVQ-4 additionally ensures a masked
  physical object in each event that has a maskable physical object. Event-context
  masking remains a separate Bernoulli draw.
- RVQ-1, RVQ-4 and RVQ-8 use code-weighted cross-entropy; the continuous
  models use masked preprocessed-feature MSE, feature-averaged within positions
  and then averaged over masked positions. These are local-rank loss reductions.
- RVQ-1, parallel RVQ-8 and flat continuous used 8 GPUs × 448 events. RVQ-4 used
  8 × 448 for seed 42 and 6 × 597 for seeds 43/44. Latest hierarchical used
  6 × 597 for all three seeds. Flat continuous kept the data-module seed at 42
  for all three runs; the model training seed changed.
- The latest kinematics/auxiliary grouping is not the original hierarchical
  grouping. The hybrid, autoregressive, no-projection, direct RVQ-1, and altered
  jet-codebook experiments are separate from the five reported profiles.
