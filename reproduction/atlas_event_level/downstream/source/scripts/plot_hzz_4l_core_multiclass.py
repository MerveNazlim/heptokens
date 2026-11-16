#!/usr/bin/env python3
"""Plot 45 matched five-class fits on the Paper-0 core selection."""
from __future__ import annotations

import plot_hzz_4l_multiclass as plot
from hzz_4l_core_cohort import TEST_EVENTS, activate


def main():
    activate()
    plot.SELECTED_TEST_EVENTS = TEST_EVENTS
    plot.CONFIG = plot.ROOT / "configs/plots/hzz_4l_core_multiclass.json"
    plot.main()


if __name__ == "__main__":
    main()
