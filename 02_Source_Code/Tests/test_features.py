"""Phase 3 integration checks against genuine source records; no fake fixtures."""
import csv
from datetime import datetime
import importlib.util
from itertools import zip_longest
import json
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('build_features', ROOT / 'scripts/build_features.py')
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class TestFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = BUILD.protected_snapshot()
        cls.write_attempts = []
        real_open = Path.open

        def protected_open(path, mode='r', *args, **kwargs):
            absolute = path.resolve()
            protected = (ROOT / 'data/raw') in absolute.parents or absolute == ROOT / 'data/processed/habsos_cleaned.csv'
            if protected and any(c in mode for c in 'wax+'):
                cls.write_attempts.append(str(path))
                raise AssertionError('Phase 3 attempted to write a protected input')
            return real_open(path, mode, *args, **kwargs)

        paths = [ROOT / 'data/ml' / f for f in ('features.csv', 'target.csv', 'row_metadata.csv', 'feature_metadata.json')]
        paths.append(ROOT / 'reports/phase3_feature_report.json')
        with patch.object(Path, 'open', protected_open):
            cls.report = BUILD.run()
            first = [BUILD.sha(p) for p in paths]
            BUILD.run()
            second = [BUILD.sha(p) for p in paths]
        cls.repeated_identically = first == second
        cls.after = BUILD.protected_snapshot()

    def aligned_rows(self):
        files = [ROOT / 'data/processed/habsos_cleaned.csv', ROOT / 'data/ml/features.csv',
                 ROOT / 'data/ml/target.csv', ROOT / 'data/ml/row_metadata.csv']
        handles = [path.open(encoding='utf-8', newline='') for path in files]
        try:
            yield from zip_longest(*(csv.DictReader(f) for f in handles))
        finally:
            for f in handles:
                f.close()

    def test_input_exists_and_raw_and_phase2_content_unchanged(self):
        self.assertEqual(self.before, self.after)
        self.assertEqual(self.write_attempts, [])
        self.assertTrue(self.report['protected_inputs_unchanged'])
        expected = json.loads((ROOT / 'reports/phase2_data_quality_report.json').read_text(encoding='utf-8'))['processed']['sha256']
        self.assertEqual(BUILD.sha(ROOT / 'data/processed/habsos_cleaned.csv'), expected)

    def test_two_builds_are_deterministic(self):
        self.assertTrue(self.repeated_identically)

    def test_rows_lineage_and_all_original_source_fields_preserved(self):
        count = 0
        for count, rows in enumerate(self.aligned_rows(), 1):
            self.assertTrue(all(r is not None for r in rows))
            source, x, y, metadata = rows
            for row in rows:
                self.assertEqual(row['SOURCE_ROW_NUMBER'], str(count))
            for key, value in source.items():
                self.assertEqual(metadata[key], value)
        self.assertEqual(count, 211834)
        self.assertEqual(count, self.report['output_rows'])

    def test_calendar_features_use_only_recorded_date(self):
        for source, x, y, metadata in self.aligned_rows():
            date = datetime.strptime(source['SAMPLE_DATE'], '%Y/%m/%d').date()
            self.assertEqual(int(x['observation_year']), date.year)
            self.assertEqual(int(x['observation_month']), date.month)
            self.assertEqual(int(x['day_of_year']), (date - date.replace(month=1, day=1)).days + 1)

    def test_identifiers_and_direct_leakage_are_not_predictors(self):
        forbidden = {'OBJECTID', 'SOURCE_ROW_NUMBER', 'CATEGORY', 'GENUS', 'SPECIES', 'CELLCOUNT',
                     'CELLCOUNT_UNIT', 'CELLCOUNT_QA', 'SUSPICIOUS_FIELDS', 'INVALID_FIELDS', 'QA_NO_DATA_FIELDS', 'TARGET_STATUS'}
        predictors = set(self.report['predictor_columns'])
        self.assertFalse(predictors & forbidden)
        for candidate in self.report['candidate_features']:
            if candidate['target_related']:
                self.assertFalse(candidate['allowed_as_feature'])
        self.assertEqual(predictors, set(BUILD.SELECTED))

    def test_target_is_verbatim_source_category_and_not_binary_or_future_risk(self):
        eligible = 0
        for source, x, y, metadata in self.aligned_rows():
            if y['TARGET_STATUS'] == 'eligible':
                eligible += 1
                self.assertEqual(y[BUILD.TARGET], source['CATEGORY'])
                self.assertIn(y[BUILD.TARGET], BUILD.CLASSES)
                self.assertEqual((source['GENUS'], source['SPECIES']), ('Karenia', 'brevis'))
                self.assertEqual(source['CELLCOUNT_UNIT'], 'cells/L')
                self.assertIn(source['CELLCOUNT_QA'], ('1', '5'))
            else:
                self.assertEqual(y[BUILD.TARGET], '')
            self.assertEqual(metadata['TARGET_STATUS'], y['TARGET_STATUS'])
        self.assertEqual(eligible, self.report['target_definition']['eligible_observations'])
        self.assertIsNone(self.report['target_definition']['forecast_horizon'])

    def test_features_never_fill_missing_values_or_modify_measurements(self):
        for source, x, y, metadata in self.aligned_rows():
            for field in ('WATER_TEMP', 'SALINITY', 'SAMPLE_DEPTH'):
                if not source[field].strip() or source.get(field + '_QA') == '9':
                    self.assertEqual(x[field], '')
                elif source.get(field + '_QA', '1') in ('1', '4', '5'):
                    self.assertEqual(x[field], source[field])
            for field in ('LATITUDE', 'LONGITUDE'):
                self.assertEqual(x[field], source[field])

    def test_qa4_is_retained_qa9_masked_and_metadata_kept(self):
        caution = no_data = 0
        for source, x, y, metadata in self.aligned_rows():
            for field in ('WATER_TEMP', 'SALINITY'):
                if source[field + '_QA'] == '4':
                    caution += 1
                    self.assertEqual(x[field], source[field])
                    self.assertEqual(metadata[field + '_FEATURE_QUALITY'], 'caution')
                elif source[field + '_QA'] == '9':
                    no_data += 1
                    self.assertEqual(x[field], '')
                    self.assertEqual(metadata[field], source[field])
        self.assertGreater(caution, 0)
        self.assertGreater(no_data, 0)

    def test_metadata_covers_every_candidate_and_feature_role(self):
        r = self.report
        self.assertEqual(r['candidate_feature_count'], 40)
        self.assertEqual(r['selected_feature_count'], 8)
        self.assertEqual(r['excluded_feature_count'], 32)
        self.assertEqual(len(r['candidate_features']), 40)
        for candidate in r['candidate_features']:
            for key in ('source_column', 'type', 'unit', 'origin', 'missing_percentage', 'leakage_risk', 'selected', 'reason'):
                self.assertIn(key, candidate)
        self.assertEqual(r['split_strategy']['status'], 'proposal only; no split files or fitted transformers')

    def test_class_counts_reconcile_and_unavailable_labels_not_filled(self):
        target = self.report['target_definition']
        self.assertEqual(sum(d['count'] for d in target['class_distribution'].values()), target['eligible_observations'])
        self.assertEqual(target['eligible_observations'] + target['ineligible_observations'], self.report['output_rows'])
        self.assertAlmostEqual(sum(d['percentage_of_eligible'] for d in target['class_distribution'].values()), 100, places=4)


if __name__ == '__main__':
    unittest.main()
