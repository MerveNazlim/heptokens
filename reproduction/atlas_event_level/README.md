# Archived ATLAS event-foundation workflows

This directory records the exact pretraining and classifier sources used for
the ATLAS four-lepton studies. It is separate from the checkout's tokenizer
and H5-to-Parquet workflow. No tokenizer, export, or existing Parquet loader
in the root tree is replaced.

[Pretraining sources and run map](pretraining/HISTORICAL_SOURCES.md) ·
[Four-lepton downstream workflows](downstream/README.md)

## Which result uses which workflow?

| Workflow | Scope |
| --- | --- |
| Original data16 pretraining | Projected RVQ-1, parallel RVQ-8, and flat continuous |
| Other pretraining variants in the five-class comparison | Independently trained RVQ-4 and hierarchical continuous with the kinematics/auxiliary grouping |
| Five-class full reconstructed four-lepton benchmark | DAG 521: five representations, fine-tuned/scratch/frozen, three replicates per combination |
| Paper 0 binary benchmark | Job 520: projected RVQ-1 scratch, projected RVQ-1 fine-tuned, and flat continuous fine-tuned |

The two downstream benchmarks use the reconstructed four-electron/muon
candidate selection (including the pairing and mass cuts, without a narrow
Higgs mass window). The five-class test set has 64,403 events; the binary
ggF-versus-ZZ test set has 47,705. These workflows are not the inclusive
benchmark or the multiplicity-only four-lepton comparison.

## Use the matching runtime

The source snapshots expose the model, training code, configs, and
study-specific orchestration for review. They are **not** drop-in overlays
for the root checkout or a substitute for the historical runtime archive.
Some runs used different versions of shared modules, environment locks, or
site wrappers. Use the original, checksum-pinned bundles and their documented
overlay order to reproduce a run. Do not assemble an environment by mixing
files from different profiles.

Each profile's `source_provenance.json` files record digests and their
sources. The profile documentation distinguishes byte-exact runtime members
from preparation/analysis scripts that ran outside the workers, and from
portable launch records. Check the committed snapshots from the repository
root without credentials or training data:

```sh
python3 reproduction/atlas_event_level/verify_sources.py --syntax
```

The reported results require the pinned prepared shards, membership and
schedule artifacts, and pretrained checkpoints. This source-only release
does not include weights, events, runtime tarballs, or credentials. Do not
submit archived launch records to the original output locations: use a
separate run destination and your own site configuration.
