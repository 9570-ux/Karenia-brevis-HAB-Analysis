"""Phase 6 uses genuine existing rows; invalid variants are validation probes only."""
import csv
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import inference as api


def genuine_rows():
    complete = missing = None
    with (ROOT / 'data/ml/features.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            values = {key: float(row[source]) if row[source] else None for key, source in api.FIELDS.items()}
            if all(values[k] is not None for k in api.OPTIONAL) and complete is None:
                complete = values
            if any(values[k] is None for k in api.OPTIONAL) and missing is None:
                missing = values
            if complete is not None and missing is not None:
                return complete, missing
    raise AssertionError('Expected genuine complete and incomplete source rows')


class TestInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.complete, cls.missing = genuine_rows()
        cls.model, cls.metadata = api.load_model()
        cls.baseline = json.loads((ROOT / 'reports/phase6_integrity_baseline.json').read_text(encoding='utf-8'))

    def test_model_exists_and_loads_with_correct_schema(self):
        self.assertTrue(api.ARTIFACT.is_file())
        self.assertEqual(api.sha256(api.ARTIFACT), self.metadata['model_sha256'])
        self.assertEqual(list(self.model.feature_names_in_), list(api.FIELDS.values()))

    def test_valid_real_row_prediction(self):
        result = api.predict_observation(self.complete)
        self.assertEqual(result['terminology'], 'Predicted recorded abundance category')
        self.assertEqual(result['input_validation']['status'], 'PASS')
        self.assertIn(result['predicted_class'], api.CLASSES)
        self.assertEqual(result['model']['sha256'], self.metadata['model_sha256'])

    def test_exact_classes_and_probability_mapping(self):
        result = api.predict_observation(self.complete)
        self.assertEqual(list(result['class_probabilities']), list(api.CLASSES))
        frame, _ = api.validate_observation(self.complete)
        expected = dict(zip(self.model.classes_, self.model.predict_proba(frame)[0]))
        self.assertEqual(result['class_probabilities'], expected)
        self.assertEqual(result['predicted_class'], self.model.predict(frame)[0])

    def test_finite_normalized_probabilities(self):
        probabilities = api.predict_observation(self.complete)['class_probabilities'].values()
        self.assertTrue(all(isinstance(p, float) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1, places=12)

    def test_schema_rejects_unknown_missing_and_nonmapping(self):
        for value in ([], dict(self.complete, ph=None), {k:v for k,v in self.complete.items() if k != 'latitude'}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'mapping|schema'):
                api.predict_observation(value)

    def test_invalid_numeric_rejected(self):
        for value in ('warm', '25', True, float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'temperature'):
                api.predict_observation(dict(self.complete, temperature=value))

    def test_invalid_calendar_bounds(self):
        for field, value in (('month',0),('month',13),('day_of_year',0),('day_of_year',367),('year',0),('year',10000),('year',2024.5)):
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                api.predict_observation(dict(self.complete, **{field:value}))

    def test_calendar_consistency_and_leap_year(self):
        for update in ({'year':2023,'month':12,'day_of_year':366}, {'month':1,'day_of_year':180}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                api.validate_observation(dict(self.complete, **update))

    def test_missing_environmental_real_row_uses_saved_imputer(self):
        frame, missing = api.validate_observation(self.missing)
        self.assertTrue(missing)
        self.assertTrue(all(math.isnan(frame.iloc[0][api.FIELDS[k]]) for k in missing))
        before = self.model.named_steps['imputer'].statistics_.copy()
        with patch.object(type(self.model), 'fit', side_effect=AssertionError('Inference must not fit')):
            result = api.predict_observation(self.missing)
        self.assertEqual(result['input_validation']['missing_environmental_fields'], missing)
        self.assertTrue((before == self.model.named_steps['imputer'].statistics_).all())

    def test_omitted_environmental_equals_none(self):
        omitted = {k:v for k,v in self.missing.items() if v is not None}
        self.assertEqual(api.predict_observation(omitted), api.predict_observation(self.missing))

    def test_required_context_and_coordinates(self):
        for field in ('latitude','longitude','year','month','day_of_year'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                api.predict_observation(dict(self.complete, **{field:None}))
        for field,value in (('latitude',91),('longitude',181)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'latitude'):
                api.predict_observation(dict(self.complete, **{field:value}))

    def test_identical_input_repeats_exactly(self):
        self.assertEqual(api.predict_observation(self.complete), api.predict_observation(self.complete))

    def test_no_writes_during_inference(self):
        original = Path.open
        def read_only(path, mode='r', *args, **kwargs):
            if any(flag in mode for flag in 'wa+x'):
                raise AssertionError(f'Unexpected write: {path}')
            return original(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', read_only):
            api.predict_observation(self.complete)
            api.predict_observation(self.missing)

    def test_raw_and_all_prior_artifacts_unchanged(self):
        exceptions = self.baseline['compatibility_changes']
        for name, digest in self.baseline['prior_hashes'].items():
            expected = exceptions.get(name, {}).get('after_sha256', digest)
            with self.subTest(path=name):
                self.assertEqual(api.sha256(ROOT / name), expected)
        self.assertEqual(api.sha256(ROOT / 'data/raw/habsos_20240430.csv'),
                         '0c561afc3239fc885afd161bbc0c3e9826969019f8266ffaade4adca7f700596')

    def test_no_new_dataset_or_dashboard_hardware_files(self):
        allowed = {'scripts/inference.py','scripts/run_inference.py','tests/test_inference.py',
                   'tests/__init__.py','docs/phase6_inference.md','reports/phase6_integrity_baseline.json',
                   'reports/phase6_inference_report.json','reports/phase6_test_results.txt',
                   'reports/phase6_dedicated_test_results.txt'}
        # Phase 7 is now authorized; allow only its explicitly scoped files.
        allowed.update({'tests/test_dashboard.py', 'docs/phase7_dashboard.md',
                        'reports/phase7_integrity_baseline.json',
                        'reports/phase7_dashboard_report.json',
                        'reports/phase7_dashboard_test_results.txt',
                        'reports/phase7_test_results.txt',
                        'reports/phase7_launch_log.txt'})
        for folder in ('data','models','reports','docs','scripts','tests'):
            for path in (ROOT / folder).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts:
                    name = path.relative_to(ROOT).as_posix()
                    self.assertTrue(name in self.baseline['prior_hashes'] or name in allowed, name)


if __name__ == '__main__':
    unittest.main()
