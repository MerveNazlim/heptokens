#!/usr/bin/env python3
"""Run yesterday's matched multiclass protocol on the Paper-0 core cohort."""
from __future__ import annotations

import hzz_4l_multiclass_study as study
from hzz_4l_core_cohort import COUNTS, PREDICATE, SELECTION_ID, activate


def main():
    activate()
    study.COUNTS = COUNTS
    study.STEPS_PER_EPOCH = sum(COUNTS["train"]) // study.BATCH_SIZE
    study.PROTOCOL = dict(
        study.PROTOCOL, cohort=SELECTION_ID, selection_predicate=PREDICATE,
        full_optimizer_steps=study.EPOCHS * study.STEPS_PER_EPOCH,
    )
    study.main()


if __name__ == "__main__":
    main()
