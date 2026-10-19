"""Integration tests use the genuine NOAA archive; no fabricated fixtures."""
import csv
from datetime import datetime
import importlib.util
from itertools import zip_longest
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('preprocess_dataset', ROOT / 'scripts/preprocess_dataset.py')
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


class TestPhase2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = PIPELINE.raw_snapshot()
        cls.write_attempts = []
        original_open = Path.open

        def protected_open(path, mode='r', *args, **kwargs):
            if (ROOT / 'data/raw') in path.resolve().parents and any(c in mode for c in 'wax+'):
                cls.write_attempts.append(str(path))
                raise AssertionError('Pipeline attempted raw write')
            return original_open(path, mode, *args, **kwargs)

        cls.outputs = [ROOT / 'data/processed/habsos_cleaned.csv',
                       ROOT / 'reports/phase2_data_quality_report.json',
                       ROOT / 'reports/phase2_record_issues.csv']
        with patch.object(Path, 'open', protected_open):
            cls.report = PIPELINE.run()
            cls.first_hashes = [PIPELINE.sha(p) for p in cls.outputs]
            PIPELINE.run()
            cls.second_hashes = [PIPELINE.sha(p) for p in cls.outputs]
        cls.after = PIPELINE.raw_snapshot()

    def test_raw_exists_and_matches_phase1(self):
        manifest = json.loads((ROOT / 'docs/download_manifest.json').read_text())
        self.assertEqual(PIPELINE.sha(ROOT / 'data/raw' / manifest['filename']), manifest['sha256'])

    def test_raw_directory_never_written_or_changed(self):
        self.assertEqual(self.write_attempts, [])
        self.assertEqual(self.before, self.after)
        self.assertTrue(self.report['raw_directory_unchanged'])

    def test_pipeline_runs_and_all_outputs_load(self):
        for path in self.outputs:
            self.assertTrue(path.is_file())
        json.loads(self.outputs[1].read_text(encoding='utf-8'))
        with self.outputs[0].open(encoding='utf-8', newline='') as source:
            reader = csv.reader(source)
            self.assertEqual(len(next(reader)), self.report['processed']['columns'])
            self.assertEqual(sum(1 for _ in reader), self.report['raw']['rows'])

    def test_all_original_cells_and_rows_preserved_no_added_observations(self):
        raw = ROOT / self.report['raw']['filename']
        processed = self.outputs[0]
        with raw.open(encoding='utf-8-sig', newline='') as a, processed.open(encoding='utf-8', newline='') as b:
            source, dest = csv.reader(a), csv.reader(b)
            original_header, output_header = next(source), next(dest)
            self.assertEqual(output_header[:28], [name or f'RAW_UNNAMED_{i}' for i, name in enumerate(original_header, 1)])
            count = 0
            for count, (raw_row, output_row) in enumerate(zip_longest(source, dest), 1):
                self.assertIsNotNone(raw_row)
                self.assertIsNotNone(output_row)
                self.assertEqual(output_row[:28], raw_row)
                self.assertEqual(output_row[28], str(count))
            self.assertEqual(count, 211834)

    def test_required_columns_and_row_accounting(self):
        self.assertEqual(self.report['raw']['rows'], self.report['processed']['rows'])
        self.assertEqual(self.report['columns_removed'], [])
        self.assertEqual(self.report['duplicates']['removed'], 0)
        self.assertEqual(self.report['invalid_values']['removed_records'], 0)
        for field in ('CELLCOUNT', 'CATEGORY', 'SALINITY', 'WATER_TEMP', 'SAMPLE_DATE', 'SAMPLE_TIME'):
            self.assertIn(field, self.report['processed']['column_names'])

    def test_missing_and_parsed_timestamps_do_not_invent_times(self):
        with self.outputs[0].open(encoding='utf-8', newline='') as source:
            for row in csv.DictReader(source):
                if row['TIME_STATUS'] != 'valid':
                    self.assertEqual(row['SAMPLE_TIME_UTC_PARSED'], '')
                    self.assertEqual(row['SAMPLE_DATETIME_UTC'], '')
                elif row['SAMPLE_DATETIME_UTC']:
                    parsed = datetime.fromisoformat(row['SAMPLE_DATETIME_UTC'].replace('Z', '+00:00'))
                    self.assertEqual(parsed.date().strftime('%Y/%m/%d'), row['SAMPLE_DATE'].strip())
                    self.assertEqual(parsed.strftime('%H:%M'), datetime.strptime(row['SAMPLE_TIME'].strip(), '%H:%M').strftime('%H:%M'))

    def test_qa9_is_exposed_even_when_source_number_present(self):
        found = 0
        with self.outputs[0].open(encoding='utf-8', newline='') as source:
            for row in csv.DictReader(source):
                no_data = set(row['QA_NO_DATA_FIELDS'].split(';'))
                for field in ('CELLCOUNT', 'SALINITY', 'WATER_TEMP', 'WIND_DIR', 'WIND_SPEED'):
                    if row[field + '_QA'].strip() == '9':
                        self.assertIn(field, no_data)
                        if row[field].strip():
                            found += 1
                            self.assertIn(field, row['SUSPICIOUS_FIELDS'].split(';'))
        self.assertGreater(found, 0)

    def test_reports_cover_every_source_column_and_missingness(self):
        self.assertEqual(len(self.report['columns']), 28)
        for column in self.report['columns']:
            self.assertAlmostEqual(column['missing_percentage'], column['missing'] * 100 / column['total'], places=5)
            self.assertEqual(column['action'], 'retain original values and blanks; no imputation')

    def test_duplicate_report_matches_exact_original_rows(self):
        self.assertEqual(self.report['duplicates']['extra_records'], len(self.report['duplicates']['records']))
        with self.outputs[0].open(encoding='utf-8', newline='') as source:
            links = {int(row['SOURCE_ROW_NUMBER']): int(row['EXACT_DUPLICATE_OF_SOURCE_ROW'])
                     for row in csv.DictReader(source) if row['EXACT_DUPLICATE_OF_SOURCE_ROW']}
        self.assertEqual(links, {entry['source_row']: entry['duplicate_of_source_row'] for entry in self.report['duplicates']['records']})

    def test_second_run_and_missing_handling_are_byte_deterministic(self):
        self.assertEqual(self.first_hashes, self.second_hashes)


if __name__ == '__main__':
    unittest.main()
