"""Phase 5 audits use genuine source rows and saved fixed experiments only."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('phase5_model_improvement', ROOT / 'scripts/phase5_model_improvement.py')
P5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P5)


class TestPhase5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((ROOT / 'reports/phase5_model_metadata.json').read_text(encoding='utf-8'))
        cls.baseline = json.loads((ROOT / 'reports/phase5_baseline.json').read_text(encoding='utf-8'))
        cls.frame, cls.feature_meta = P5.p4.load_inputs()
        cls.masks = P5.p4.split_masks(cls.frame)
        cls.model = P5.p4.joblib.load(ROOT / 'models/phase5_best_model.joblib')

    def test_phase4_baseline_exists_and_all_locked_artifacts_unchanged(self):
        self.assertEqual(P5.snapshot(), self.baseline['locked_hashes'])
        self.assertEqual(self.metadata['locked_hashes'], self.baseline['locked_hashes'])
        self.assertAlmostEqual(self.baseline['test']['macro_f1'], .19697460958001423, places=12)
        original = json.loads((ROOT / 'reports/model_metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(P5.p4.sha(ROOT / 'models/best_model.joblib'), original['artifact_sha256'])

    def test_raw_processed_phase3_and_target_definition_preserved(self):
        for prefix in ('data/raw/', 'data/processed/', 'data/ml/'):
            entries = {k:v for k,v in self.baseline['locked_hashes'].items() if k.startswith(prefix)}
            self.assertTrue(entries)
            for name, digest in entries.items():
                self.assertEqual(P5.p4.sha(ROOT / name), digest)
        self.assertEqual(self.metadata['target_definition'], self.feature_meta['target_definition'])
        self.assertTrue(self.frame.loc[self.frame.eligible, 'target'].isin(P5.CLASSES).all())

    def test_temporal_splits_disjoint_and_unaltered(self):
        expected = {'train':184845, 'validation':18494, 'test':7428}
        for name, count in expected.items():
            self.assertEqual(int(self.masks[name].sum()), count)
        self.assertTrue(self.frame.loc[self.masks['train'], 'observation_year'].le(2020).all())
        self.assertTrue(self.frame.loc[self.masks['validation'], 'observation_year'].between(2021,2022).all())
        self.assertTrue(self.frame.loc[self.masks['test'], 'observation_year'].between(2023,2024).all())
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse((self.masks[a] & self.masks[b]).any())
            self.assertFalse(set(self.frame.loc[self.masks[a], 'date']) & set(self.frame.loc[self.masks[b], 'date']))

    def test_group_outputs_exclude_target_ids_and_quality_fields(self):
        allowed = set(P5.p4.FEATURES + P5.CYCLIC)
        forbidden = {'CATEGORY','CELLCOUNT','CELLCOUNT_QA','SOURCE_ROW_NUMBER','OBJECTID','GENUS','SPECIES','TARGET_STATUS'}
        for group, fields in P5.GROUPS.items():
            self.assertTrue(set(fields) <= allowed)
            self.assertFalse(set(fields) & forbidden)
        self.assertEqual(list(self.model.feature_names_in_), P5.p4.FEATURES)
        self.assertEqual(self.metadata['selected_features'], P5.GROUPS[self.metadata['feature_group']])

    def test_cyclic_transforms_are_deterministic_real_calendar_functions(self):
        X = self.frame[P5.p4.FEATURES]
        helper = P5.Phase5FeatureGroup('D_environmental_temporal_spatial_cyclic').fit(X)
        a, b = helper.transform(X), helper.transform(X)
        self.assertTrue(a.equals(b))
        P5.np.testing.assert_allclose(a.month_sin**2 + a.month_cos**2, 1, atol=1e-12, rtol=0)
        P5.np.testing.assert_allclose(a.day_of_year_sin**2 + a.day_of_year_cos**2, 1, atol=1e-12, rtol=0)
        days = P5.np.where(self.frame.date.dt.is_leap_year, 366, 365)
        angle = 2*P5.np.pi*(self.frame.day_of_year.to_numpy()-1)/days
        P5.np.testing.assert_allclose(a.day_of_year_sin, P5.np.sin(angle), atol=1e-12, rtol=0)
        for field in P5.ENV + P5.SPACE:
            self.assertTrue(a[field].equals(X[field]))

    def test_imputation_training_only_and_no_resampling(self):
        train = self.frame.loc[self.masks['train'], P5.p4.FEATURES]
        recipes = P5.experiments()
        self.assertEqual(len(self.metadata['candidate_imputation_audit']), 15)
        for event in self.metadata['candidate_imputation_audit']:
            self.assertEqual(event['fit_split'], 'train')
            self.assertEqual(event['fit_rows'], 184845)
            self.assertEqual(event['fit_row_hash'], self.metadata['split_summary']['train']['lineage_sha256'])
            group = recipes[event['experiment']]['group']
            helper = P5.Phase5FeatureGroup(group).fit(train)
            expected = helper.transform(train).median().to_numpy()
            P5.np.testing.assert_allclose(event['imputer_statistics'], expected, atol=1e-12, rtol=0)
        final_X = self.frame.loc[self.masks['train'] | self.masks['validation'], P5.p4.FEATURES]
        expected_final = self.model.named_steps['features'].transform(final_X).median().to_numpy()
        P5.np.testing.assert_allclose(self.model.named_steps['imputer'].statistics_, expected_final, atol=1e-12, rtol=0)
        self.assertEqual(self.metadata['final_fit_rows'], 203339)
        self.assertEqual(self.metadata['resampling'], 'none; no synthetic observations')

    def test_all_ablation_groups_and_model_families_present(self):
        results = P5.pd.read_csv(ROOT / 'reports/phase5_ablation_results.csv')
        self.assertEqual(len(results), 15)
        self.assertEqual(set(results.feature_group), set(P5.GROUPS))
        self.assertEqual(set(results.model), {'logistic','random_forest','hist_gradient'})
        self.assertEqual(set(results.experiment_name), set(P5.experiments()))
        anchor = self.metadata['validation_metrics']['A_original8__logistic__balanced']['macro_f1']
        self.assertAlmostEqual(anchor, self.baseline['validation']['macro_f1'], places=12)

    def test_selection_validation_only_and_saved_test_rows_correct(self):
        metrics = self.metadata['validation_metrics']
        rank = lambda name: (-metrics[name]['macro_f1'], -P5.np.mean([metrics[name]['per_class'][c]['recall'] for c in ('low','medium','high')]), name)
        selected = sorted(metrics, key=rank)[0]
        self.assertEqual(selected, self.metadata['selected_experiment'])
        freeze = json.loads((ROOT / 'reports/phase5_selection.json').read_text(encoding='utf-8'))
        self.assertEqual(freeze['test_predictions_made_this_run'], 0)
        saved = P5.pd.read_csv(ROOT / 'reports/phase5_test_predictions.csv')
        test = self.frame.loc[self.masks['test']]
        self.assertEqual(saved.SOURCE_ROW_NUMBER.tolist(), test.SOURCE_ROW_NUMBER.tolist())
        self.assertEqual(saved.recorded_category.tolist(), test.target.tolist())
        self.assertTrue(saved.predicted_category.isin(P5.CLASSES).all())
        self.assertEqual(self.metadata['test_evaluations_per_run'], 1)
        cm = P5.p4.confusion_matrix(saved.recorded_category, saved.predicted_category, labels=P5.CLASSES)
        self.assertEqual(cm.tolist(), self.metadata['test_metrics']['confusion_matrix'])

    def test_selected_model_loads_and_predicts_valid_classes(self):
        self.assertEqual(P5.p4.sha(ROOT / 'models/phase5_best_model.joblib'), self.metadata['model_sha256'])
        sample = self.frame.loc[self.masks['train'], P5.p4.FEATURES].iloc[:32]
        with P5.p4.threadpool_limits(limits=1):
            predictions = self.model.predict(sample)
        self.assertTrue(set(predictions) <= set(P5.CLASSES))

    def test_required_phase5_outputs_exist(self):
        for name in ('phase5_baseline.json','phase5_error_analysis.csv','phase5_candidate_features.json','phase5_ablation_results.csv',
                     'phase5_vs_phase4.csv','phase5_feature_importance.csv','phase5_model_metadata.json','phase5_reproducibility.json'):
            self.assertGreater((ROOT / 'reports' / name).stat().st_size, 0)
        for name in ('phase5_error_confusion_matrix.png','phase5_feature_importance.png','phase5_final_confusion_matrix.png'):
            self.assertGreater((ROOT / 'reports/figures' / name).stat().st_size, 0)
        self.assertTrue((ROOT / 'docs/phase5_model_improvement.md').is_file())

    def test_two_complete_reproductions_match_metrics_and_probabilities(self):
        report = json.loads((ROOT / 'reports/phase5_reproducibility.json').read_text(encoding='utf-8'))
        self.assertGreaterEqual(report['complete_runs'], 2)
        self.assertTrue(report['verified'])
        self.assertTrue(report['same_selected_experiment'])
        self.assertLessEqual(report['maximum_metric_absolute_difference'], 1e-12)
        self.assertLessEqual(report['maximum_test_probability_absolute_difference'], 1e-12)

    def test_phase4_error_audit_and_comparison_are_honest(self):
        error = P5.pd.read_csv(ROOT / 'reports/phase5_error_analysis.csv')
        high = error.loc[error['class'].eq('high')].iloc[0]
        self.assertEqual(int(high.support), 90)
        self.assertEqual(int(high.true_positive), 0)
        comparison = P5.pd.read_csv(ROOT / 'reports/phase5_vs_phase4.csv')
        self.assertEqual(len(comparison), 2)
        self.assertAlmostEqual(comparison.iloc[0].macro_f1, self.baseline['test']['macro_f1'])
        self.assertAlmostEqual(comparison.iloc[1].macro_f1, self.metadata['test_metrics']['macro_f1'])
        self.assertEqual(self.metadata['test_macro_f1_improved'], comparison.iloc[1].macro_f1 > comparison.iloc[0].macro_f1)


if __name__ == '__main__':
    unittest.main()
