"""Read-only Phase 7 checks and Streamlit UI tests on genuine existing rows."""
import ast
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dashboard
from streamlit.testing.v1 import AppTest


class TestDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts = dashboard.load_artifacts()
        cls.history = dashboard.load_history()
        cls.example = dashboard.genuine_example()['observation']
        cls.baseline = json.loads((ROOT / 'reports/phase7_integrity_baseline.json').read_text())

    def new_app(self, page='Overview'):
        at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=45).run()
        if page != 'Overview':
            at.sidebar.radio[0].set_value(page).run()
        self.assertFalse(at.exception, [e.message for e in at.exception])
        return at

    def test_dashboard_imports_without_running_ui(self):
        self.assertTrue(callable(importlib.import_module('app').main))

    def test_inference_is_direct_phase6_import_and_delegation(self):
        phase6 = importlib.import_module('scripts.inference')
        self.assertIs(dashboard.inference, phase6)
        with patch.object(phase6, 'predict_observation', wraps=phase6.predict_observation) as call:
            actual = dashboard.predict_for_dashboard(self.example)
            call.assert_called_once_with(self.example)
        self.assertEqual(actual, phase6.predict_observation(self.example))

    def test_required_files_and_classes(self):
        for name in ('app.py','dashboard.py','requirements_phase7.txt','docs/phase7_dashboard.md',
                     'reports/phase7_integrity_baseline.json','tests/test_dashboard.py'):
            self.assertTrue((ROOT / name).is_file(), name)
        self.assertEqual(dashboard.CLASSES, ('not observed','very low','low','medium','high'))

    def test_all_prior_data_models_results_inference_unchanged(self):
        exceptions = self.baseline['compatibility_changes']
        for name, digest in self.baseline['prior_hashes'].items():
            with self.subTest(path=name):
                self.assertEqual(dashboard.inference.sha256(ROOT / name),
                                 exceptions.get(name,{}).get('after_sha256',digest))
        self.assertEqual(dashboard.inference.sha256(ROOT / 'models/phase5_best_model.joblib'),
                         self.artifacts['phase5']['model_sha256'])
        self.assertEqual(dashboard.inference.sha256(ROOT / 'data/raw/habsos_20240430.csv'),
                         '0c561afc3239fc885afd161bbc0c3e9826969019f8266ffaade4adca7f700596')

    def test_required_metrics_read_from_locked_reports(self):
        metrics = self.artifacts['phase5']['test_metrics']
        self.assertAlmostEqual(metrics['accuracy'], .691437802907916)
        self.assertAlmostEqual(metrics['macro_f1'], .2925618359931265)
        self.assertAlmostEqual(metrics['balanced_accuracy'], .3034199516735775)
        self.assertAlmostEqual(metrics['per_class']['high']['recall'], 3/90)
        self.assertEqual(len(self.artifacts['comparison']), 2)
        self.assertEqual(len(self.artifacts['importance']), 12)

    def test_scientific_limitations_visible_in_source_and_ui(self):
        self.assertIn('not a validated future HAB-risk or early-warning model', dashboard.LIMITATION)
        self.assertIn('not be interpreted as probability of future HAB occurrence', dashboard.PROBABILITY_NOTE)
        at = self.new_app()
        self.assertTrue(any(w.value == dashboard.LIMITATION for w in at.warning))
        self.assertTrue(any(w.value == dashboard.HIGH_WARNING for w in at.warning))

    def test_no_hardware_training_or_model_loading_implementation(self):
        forbidden_modules = {'paho','serial','machine','RPi','joblib','sklearn'}
        for file in ('app.py','dashboard.py'):
            tree = ast.parse((ROOT / file).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    self.assertFalse({n.name.split('.')[0] for n in node.names} & forbidden_modules)
                if isinstance(node,ast.ImportFrom):
                    self.assertNotIn((node.module or '').split('.')[0], forbidden_modules)
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr, ('fit','fit_transform','to_csv','to_json','dump','publish','subscribe'))

    def test_repeat_inference_through_helper_is_identical(self):
        self.assertEqual(dashboard.predict_for_dashboard(self.example), dashboard.predict_for_dashboard(self.example))

    def test_historical_rows_categories_and_missingness_preserved(self):
        import pandas as pd
        self.assertEqual(len(self.history), 211834)
        feature = pd.read_csv(ROOT / 'data/ml/features.csv')
        for name in dashboard.ENV_UNITS:
            pd.testing.assert_series_equal(self.history[name], feature[name])
        counts = dashboard.category_counts(self.history)
        self.assertEqual(int(counts.Observations.sum()) + int(self.history.CATEGORY.isna().sum()), len(self.history))
        self.assertEqual(int(self.history.CATEGORY.isna().sum()),856)

    def test_map_subset_contains_only_original_coordinates(self):
        points, stride = dashboard.map_subset(self.history)
        self.assertLessEqual(len(points),4000)
        self.assertGreater(stride,1)
        self.assertTrue(points.SOURCE_ROW_NUMBER.isin(self.history.SOURCE_ROW_NUMBER).all())
        self.assertTrue(points.equals(self.history.dropna(subset=['LATITUDE','LONGITUDE']).iloc[::stride]))

    def test_no_new_scientific_files_and_prediction_no_writes(self):
        expected = {name for name in self.baseline['prior_hashes'] if name.startswith(('data/','models/'))}
        actual = {p.relative_to(ROOT).as_posix() for folder in ('data','models') for p in (ROOT/folder).rglob('*') if p.is_file()}
        self.assertEqual(actual, expected)
        original = Path.open
        def readonly(path, mode='r', *args, **kwargs):
            if any(c in mode for c in 'wax+'):
                raise AssertionError(f'Unexpected dashboard write: {path}')
            return original(path,mode,*args,**kwargs)
        with patch.object(Path,'open',readonly):
            dashboard.predict_for_dashboard(self.example)

    def test_all_six_pages_render(self):
        at = self.new_app()
        for page in dashboard.PAGES:
            with self.subTest(page=page):
                at.sidebar.radio[0].set_value(page).run()
                self.assertFalse(at.exception, [e.message for e in at.exception])

    def test_blank_form_has_no_invented_defaults_and_rejects_input(self):
        at = self.new_app('Prediction')
        self.assertTrue(all(widget.value is None for widget in at.number_input))
        at.button[1].click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any('Input validation: FAIL' in e.value for e in at.error))
        self.assertNotIn('prediction_result',at.session_state)

    def test_genuine_row_form_predicts_exact_phase6_result(self):
        at = self.new_app('Prediction')
        at.button[0].click().run()
        at.button[1].click().run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        self.assertEqual(at.session_state['prediction_result'], dashboard.inference.predict_observation(self.example))
        self.assertEqual(at.session_state['submitted_observation'], self.example)

    def test_invalid_resubmission_clears_old_prediction(self):
        at = self.new_app('Prediction')
        at.button[0].click().run()
        at.button[1].click().run()
        self.assertIn('prediction_result',at.session_state)
        at.number_input(key='input_month').set_value(13)
        at.button[1].click().run()
        self.assertFalse(at.exception)
        self.assertNotIn('prediction_result',at.session_state)
        self.assertTrue(any('month must be' in e.value for e in at.error))

    def test_missing_environmental_form_passes_none_to_phase6(self):
        row = self.history.loc[self.history[list(dashboard.ENV_UNITS)].isna().any(axis=1)].iloc[0]
        obs = {k:None if __import__('pandas').isna(row[v]) else float(row[v]) for k,v in dashboard.inference.FIELDS.items()}
        at = self.new_app('Prediction')
        for field,value in obs.items():
            if value is not None:
                at.number_input(key='input_'+field).set_value(int(value) if field in ('year','month','day_of_year') else value)
        at.button[1].click().run()
        self.assertFalse(at.exception,[e.message for e in at.exception])
        self.assertEqual(at.session_state['prediction_result'],dashboard.inference.predict_observation(obs))
        self.assertTrue(at.session_state['prediction_result']['input_validation']['missing_environmental_fields'])


if __name__ == '__main__':
    unittest.main()
