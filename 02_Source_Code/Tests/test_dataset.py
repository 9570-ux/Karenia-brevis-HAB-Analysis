"""Read-only test against real downloaded data; never creates a fixture."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("inspect_dataset", ROOT / "scripts" / "inspect_dataset.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestRawDataset(unittest.TestCase):
    def test_original_csv_exists_is_readable_and_unchanged(self):
        files = sorted((ROOT / "data" / "raw").glob("*.csv"))
        self.assertTrue(files, "No original CSV downloaded into data/raw; Phase 1 is incomplete")
        manifest = json.loads((ROOT / "docs" / "download_manifest.json").read_text(encoding="utf-8"))
        for path in files:
            with self.subTest(file=path.name):
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(path.name, manifest["filename"])
                self.assertEqual(path.stat().st_size, manifest["size_bytes"])
                self.assertEqual(before, manifest["sha256"])
                report = MODULE.inspect(path)
                self.assertGreater(report["row_count"], 0)
                self.assertTrue(report["column_names"])
                self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(before, report["sha256"])


if __name__ == "__main__":
    unittest.main()
