"""Fixed Paper-0 candidate cohort for the matched five-class rerun."""
from __future__ import annotations

import prepare_hzz_4l_multiclass as base


SELECTION_ID = "hzz_4l_paper0_core_v1"
PREDICATE = dict(
    base.PREDICATE,
    no_mass_or_pairing_cut=False,
    candidate_core="paper0_atlas_hzz_cms_core_q1_v1",
    candidate_code_sha256="e40de3c7aa7e634a464439a48e29fbb6e40f512233693645b6df4036860a074d",
)
COUNTS = {
    "train": (216803, 22480, 24240, 30954, 5802),
    "val": (46632, 4734, 4991, 6645, 1262),
    "test": (46424, 4826, 5261, 6611, 1281),
}
TEST_EVENTS = sum(COUNTS["test"])


def activate():
    """Bind only this process to the distinct cohort; leave the base sources unchanged."""
    base.SELECTION_ID, base.PREDICATE = SELECTION_ID, PREDICATE


def validate_selection(selection_dir):
    activate()
    receipt, membership = base.load_selection(selection_dir)
    expected = {split: dict(zip(base.NAMES, counts)) | {"total": sum(counts)}
                for split, counts in COUNTS.items()}
    if receipt["split_counts"] != expected or receipt["source_manifest_sha256"] != base.SOURCE_SHA:
        raise ValueError("Wrong Paper 0 core membership or source dataset")
    return receipt, membership
