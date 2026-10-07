"""Regression tests for conversion campaign source-membership validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from finalize_data_grl_conversion_manifests import finalize  # noqa: E402


class TestFinalizeDataGrlConversionManifests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "groups").mkdir()
        (self.root / "status").mkdir()
        self.sources = ["gs://bucket/b.h5", "gs://bucket/a.h5"]
        self.plan = {
            "groups": [{"group_id": "group-00000"}],
            "input_file_count": 2,
            "inventory_sha256": "inventory-sha256",
        }
        self.status = self._write_group("group-00000", self.sources)
        self._write_json(self.root / "plan.json", self.plan)
        self.args = argparse.Namespace(
            campaign_plan=self.root / "plan.json",
            status_dir=self.root / "status",
            output_dir=self.root / "output",
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        # Match the converter: dictionary keys are sorted, file lists are not.
        path.write_text(json.dumps(payload, sort_keys=True))

    def _write_group(self, group_id: str, sources: list[str]) -> dict:
        self._write_json(
            self.root / "groups" / f"{group_id}.json",
            {"files": [{"source_uri": uri} for uri in sources]},
        )
        status = {
            "group_id": group_id,
            "seed": 42,
            "train_fraction_requested": 0.9,
            "split_method": "test-split",
            "max_seq_length": 256,
            "q1_vocab_size": 98820,
            "q1_max_quantizers": 1,
            "q8_vocab_size": 131588,
            "q8_max_quantizers": 8,
            "continuous_max_feature_dim": 14,
            "membership_sha256": f"membership-{group_id}",
            "train_rows": len(sources),
            "val_rows": len(sources),
            "source_counts": {uri: {"processed": 2} for uri in sources},
            "outputs": {
                representation: {
                    split: [{
                        "path": f"{representation}/{split}/part-{group_id}-00000.parquet",
                        "rows": len(sources),
                        "bytes": 100,
                    }]
                    for split in ("train", "val")
                }
                for representation in ("q1", "q8", "continuous")
            },
        }
        self._write_json(self.root / "status" / f"{group_id}.json", status)
        return status

    def test_json_key_order_does_not_change_source_membership(self) -> None:
        summary = finalize(self.args)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["total_rows"], 4)
        self.assertEqual(summary["input_file_count"], 2)
        digest = hashlib.sha256()
        digest.update(b"group-00000")
        digest.update(self.status["membership_sha256"].encode())
        self.assertEqual(summary["membership_sha256"], digest.hexdigest())
        for representation in ("q1", "q8", "continuous"):
            manifest = json.loads(
                (self.root / "output" / f"{representation}_manifest.json").read_text()
            )
            self.assertEqual(set(manifest["source_counts"]), set(self.sources))
            self.assertEqual(manifest["membership_sha256"], digest.hexdigest())

    def test_missing_and_unexpected_sources_are_still_rejected(self) -> None:
        for observed in ([self.sources[0]], [*self.sources, "gs://bucket/c.h5"]):
            with self.subTest(observed=observed):
                status = dict(self.status)
                status["source_counts"] = {uri: {"processed": 2} for uri in observed}
                self._write_json(self.root / "status/group-00000.json", status)
                with self.assertRaisesRegex(ValueError, "source list differs"):
                    finalize(self.args)

    def test_duplicate_sources_in_group_assignment_are_rejected(self) -> None:
        self._write_json(
            self.root / "groups/group-00000.json",
            {"files": [{"source_uri": uri} for uri in [*self.sources, self.sources[0]]]},
        )
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            finalize(self.args)

    def test_source_assigned_to_two_groups_is_still_rejected(self) -> None:
        self._write_group("group-00001", [self.sources[0]])
        self.plan["groups"].append({"group_id": "group-00001"})
        self._write_json(self.root / "plan.json", self.plan)
        with self.assertRaisesRegex(ValueError, "more than one completed group"):
            finalize(self.args)


if __name__ == "__main__":
    unittest.main()
