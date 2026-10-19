"""Audit saved real-data experiments without reevaluating the held-out test."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('train_models', ROOT / 'scripts/train_models.py')
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.feature_meta = TRAIN.load_inputs()
        cls.masks = TRAIN.split_masks(cls.frame)
        cls.metadata = json.loads((ROOT / 'reports/model_metadata.json').read_text(encoding='utf-8'))
        cls.metrics = json.loads((ROOT / 'reports/final_test_metrics.json').read_text(encoding='utf-8'))
        cls.model = TRAIN.joblib.load(ROOT / 'models/best_model.joblib')
        cls.predictions = TRAIN.pd.read_csv(ROOT / 'reports/phase4_test_predictions.csv')

    def test_phase3_inputs_exist_and_source_hashes_unchanged(self):
        self.assertEqual(TRAIN.protected_snapshot(), self.metadata['protected_input_hashes'])
        self.assertTrue(self.metadata['protected_inputs_unchanged'])

    def test_predictor_allowlist_has_no_target_or_identifier(self):
        self.assertEqual(self.feature_meta['predictor_columns'], TRAIN.FEATURES)
        self.assertEqual(list(self.model.feature_names_in_), TRAIN.FEATURES)
        self.assertFalse(set(TRAIN.FEATURES) & {'SOURCE_ROW_NUMBER', 'OBJECTID', 'CATEGORY', 'CELLCOUNT', 'GENUS', 'SPECIES', 'TARGET_STATUS'})

    def test_source_target_values_are_known_and_only_eligible_used(self):
        self.assertTrue(self.frame.loc[self.frame.eligible, 'target'].isin(TRAIN.CLASSES).all())
        self.assertEqual(sum(int(m.sum()) for m in self.masks.values()), 210767)

    def test_temporal_boundaries_and_disjoint_observations(self):
        years = self.frame.observation_year
        self.assertTrue(years[self.masks['train']].le(2020).all())
        self.assertTrue(years[self.masks['validation']].between(2021, 2022).all())
        self.assertTrue(years[self.masks['test']].between(2023, 2024).all())
        self.assertEqual([int(self.masks[k].sum()) for k in ('train', 'validation', 'test')], [184845, 18494, 7428])
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            self.assertFalse((self.masks[a] & self.masks[b]).any())
            self.assertFalse(set(self.frame.loc[self.masks[a], 'date']) & set(self.frame.loc[self.masks[b], 'date']))

    def test_every_pipeline_contains_imputation_and_only_logistic_scales(self):
        for name, pipeline in TRAIN.make_candidates().items():
            self.assertIn('imputer', pipeline.named_steps)
            self.assertEqual(pipeline.named_steps['imputer'].strategy, 'median')
            self.assertEqual('scaler' in pipeline.named_steps, name.startswith('logistic'))

    def test_training_only_candidate_imputation_and_final_refit_pool(self):
        expected = self.frame.loc[self.masks['train'], TRAIN.FEATURES].median().to_numpy()
        events = self.metadata['imputation_audit']
        candidates = [a for a in events if a['stage'] == 'candidate_training_and_repeat']
        self.assertEqual(len(candidates), 7)
        for event in candidates:
            self.assertEqual(event['fit_split'], 'train')
            self.assertEqual(event['fit_rows'], 184845)
            self.assertEqual(event['fit_lineage_sha256'], self.metadata['split_summary']['train']['lineage_sha256'])
            TRAIN.np.testing.assert_allclose(event['imputer_statistics'], expected, atol=1e-12, rtol=0)
        final_mask = self.masks['train'] | self.masks['validation']
        TRAIN.np.testing.assert_allclose(self.model.named_steps['imputer'].statistics_, self.frame.loc[final_mask, TRAIN.FEATURES].median(), atol=1e-12, rtol=0)
        self.assertEqual(self.metadata['final_fit_rows'], 203339)

    def test_all_candidate_families_trained_and_repeated_deterministically(self):
        names = set(self.metadata['validation_metrics'])
        self.assertEqual(names, set(TRAIN.make_candidates()))
        for event in self.metadata['imputation_audit']:
            self.assertLessEqual(event['max_probability_difference_on_validation'], 1e-12)
        self.assertEqual(self.metadata['random_seed'], 42)

    def test_majority_baseline_matches_training_majority_reproducibly(self):
        training_labels = self.frame.loc[self.masks['train'], 'target']
        majority = training_labels.value_counts().index[0]
        valid = self.frame.loc[self.masks['validation'], 'target']
        expected = float(valid.eq(majority).mean())
        self.assertAlmostEqual(self.metadata['validation_metrics']['majority_baseline']['accuracy'], expected)
        self.assertEqual(majority, 'not observed')

    def test_selection_uses_validation_macro_f1_only(self):
        metrics = self.metadata['validation_metrics']
        selected = sorted(metrics, key=lambda n: (-metrics[n]['macro_f1'], -metrics[n]['macro_recall'], n))[0]
        selection = json.loads((ROOT / 'reports/phase4_selection.json').read_text(encoding='utf-8'))
        self.assertEqual(selected, self.metadata['selected_model'])
        self.assertEqual(selection['test_predictions_made'], 0)

    def test_saved_test_predictions_only_match_heldout_source_rows(self):
        source = self.frame.loc[self.masks['test']]
        self.assertEqual(self.predictions.SOURCE_ROW_NUMBER.tolist(), source.SOURCE_ROW_NUMBER.tolist())
        self.assertEqual(self.predictions.recorded_category.tolist(), source.target.tolist())
        self.assertTrue(self.predictions.predicted_category.isin(TRAIN.CLASSES).all())
        self.assertEqual(self.metrics['test_evaluations_per_run'], 1)
        matrix = TRAIN.confusion_matrix(self.predictions.recorded_category, self.predictions.predicted_category, labels=TRAIN.CLASSES)
        self.assertEqual(matrix.tolist(), self.metrics['confusion_matrix'])
        self.assertEqual(int(matrix.sum()), 7428)

    def test_artifact_loads_and_model_predicts_known_classes_on_training_rows(self):
        sample = self.frame.loc[self.masks['train'], TRAIN.FEATURES].iloc[:32]
        with TRAIN.threadpool_limits(limits=1):
            prediction = self.model.predict(sample)
        self.assertTrue(set(prediction) <= set(TRAIN.CLASSES))
        self.assertEqual(TRAIN.sha(ROOT / 'models/best_model.joblib'), self.metadata['artifact_sha256'])

    def test_metrics_comparison_and_figures_exist(self):
        comparison = TRAIN.pd.read_csv(ROOT / 'reports/model_comparison.csv')
        self.assertEqual(len(comparison), 7)
        self.assertEqual(int(comparison.selected.sum()), 1)
        self.assertTrue(comparison.loc[~comparison.selected, 'test_macro_f1'].isna().all())
        for name in ('confusion_matrix', 'feature_importance'):
            for extension in ('png', 'svg', 'pdf'):
                self.assertGreater((ROOT / 'reports/figures' / (name + '.' + extension)).stat().st_size, 0)
        self.assertTrue((ROOT / 'reports/feature_importance.csv').is_file())
        self.assertEqual(self.metrics['class_order'], TRAIN.CLASSES)
        self.assertIn('not be interpreted as a validated future HAB-risk', self.metadata['scientific_limitation'])


if __name__ == '__main__':
    unittest.main()
