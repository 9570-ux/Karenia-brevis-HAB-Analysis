"""Phase 4 only: temporal classification of NOAA source-recorded abundance.

Training-fitted preprocessing, fixed validation selection, one final test pass.
No source dataset is rewritten. No future-risk/early-warning claim is made.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.phase4_dependencies'))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLCONFIGDIR'] = str(ROOT / '.phase4_dependencies/mplconfig')
import hashlib
import importlib.metadata as package_versions
import json
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, log_loss, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
CLASSES = ['not observed', 'very low', 'low', 'medium', 'high']
FEATURES = ['WATER_TEMP', 'SALINITY', 'SAMPLE_DEPTH', 'LATITUDE', 'LONGITUDE',
            'observation_year', 'observation_month', 'day_of_year']
CLAIM = 'The model predicts the source-recorded abundance category represented in the NOAA HABSOS data. It should not be interpreted as a validated future HAB-risk or early-warning model.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_snapshot():
    paths = []
    for directory in ('data/raw', 'data/processed', 'data/ml'):
        paths.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
    paths.extend([ROOT / 'scripts/build_features.py'])
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def load_inputs():
    r = json.loads((ROOT / 'reports/phase3_feature_report.json').read_text(encoding='utf-8'))
    meta = json.loads((ROOT / 'data/ml/feature_metadata.json').read_text(encoding='utf-8'))
    if meta['predictor_columns'] != FEATURES or r['predictor_columns'] != FEATURES:
        raise ValueError('Predictor allowlist mismatch')
    for item in r['outputs'].values():
        if sha(ROOT / item['filename']) != item['sha256']:
            raise ValueError('Phase 3 source file hash mismatch')
    phase2 = json.loads((ROOT / 'reports/phase2_data_quality_report.json').read_text(encoding='utf-8'))
    if sha(ROOT / phase2['processed']['filename']) != phase2['processed']['sha256']:
        raise ValueError('Phase 2 hash mismatch')
    x = pd.read_csv(ROOT / 'data/ml/features.csv')
    y = pd.read_csv(ROOT / 'data/ml/target.csv', keep_default_na=False)
    row_meta = pd.read_csv(ROOT / 'data/ml/row_metadata.csv', usecols=['SOURCE_ROW_NUMBER', 'SAMPLE_DATE'])
    if list(x.columns) != ['SOURCE_ROW_NUMBER'] + FEATURES:
        raise ValueError('Unexpected feature columns')
    if not (len(x) == len(y) == len(row_meta) == r['output_rows']):
        raise ValueError('Row count mismatch')
    if not x.SOURCE_ROW_NUMBER.equals(y.SOURCE_ROW_NUMBER) or not x.SOURCE_ROW_NUMBER.equals(row_meta.SOURCE_ROW_NUMBER):
        raise ValueError('Row alignment mismatch')
    if not x.SOURCE_ROW_NUMBER.is_unique:
        raise ValueError('Duplicate lineage keys')
    target = meta['target_definition']['name']
    if target != 'observed_abundance_category' or set(y[target].unique()) - set(CLASSES + ['']):
        raise ValueError('Target schema mismatch')
    eligible = y.TARGET_STATUS.eq('eligible')
    if not y.loc[eligible, target].isin(CLASSES).all() or not y.loc[~eligible, target].eq('').all():
        raise ValueError('Phase 3 target eligibility mismatch')
    dates = pd.to_datetime(row_meta.SAMPLE_DATE, format='%Y/%m/%d', errors='raise')
    if not np.array_equal(dates.dt.year.to_numpy(), x.observation_year.to_numpy()):
        raise ValueError('Observation year differs from source date')
    if np.isinf(x[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError('Infinite predictor')
    frame = x.copy()
    frame['target'] = y[target]
    frame['eligible'] = eligible
    frame['date'] = dates
    return frame, meta


def split_masks(frame):
    year = frame.observation_year
    return {'train': frame.eligible & (year <= 2020),
            'validation': frame.eligible & year.between(2021, 2022),
            'test': frame.eligible & year.between(2023, 2024)}


def make_candidates():
    models = {'majority_baseline': DummyClassifier(strategy='most_frequent', random_state=SEED)}
    for weight in (None, 'balanced'):
        suffix = 'unweighted' if weight is None else 'balanced'
        models['logistic_' + suffix] = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, class_weight=weight, random_state=SEED)
        models['random_forest_' + suffix] = RandomForestClassifier(n_estimators=80, max_depth=18, min_samples_leaf=5, max_features='sqrt', class_weight=weight, random_state=SEED, n_jobs=1)
        models['hist_gradient_' + suffix] = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, min_samples_leaf=30, learning_rate=0.1, l2_regularization=1.0, early_stopping=False, class_weight=weight, random_state=SEED)
    result = {}
    for name, estimator in models.items():
        steps = [('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True))]
        if name.startswith('logistic'):
            steps.append(('scaler', StandardScaler()))
        result[name] = Pipeline(steps + [('classifier', estimator)])
    return result


def evaluate(y, prediction, probabilities, probability_classes):
    result = {'accuracy': float(accuracy_score(y, prediction)),
              'balanced_accuracy': float(balanced_accuracy_score(y, prediction))}
    for average in ('macro', 'weighted'):
        precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, labels=CLASSES, average=average, zero_division=0)
        result.update({average + '_precision': float(precision), average + '_recall': float(recall), average + '_f1': float(f1)})
    result['per_class'] = {k: v for k, v in classification_report(y, prediction, labels=CLASSES, output_dict=True, zero_division=0).items() if k in CLASSES}
    result['class_order'] = CLASSES
    result['confusion_matrix'] = confusion_matrix(y, prediction, labels=CLASSES).tolist()
    result['log_loss'] = float(log_loss(y, probabilities, labels=probability_classes))
    result['probability_column_order'] = list(probability_classes)
    return result


def split_summary(frame, masks):
    return {name: {'rows': int(mask.sum()), 'earliest_date': str(frame.loc[mask, 'date'].min().date()),
                   'latest_date': str(frame.loc[mask, 'date'].max().date()),
                   'years': sorted(int(v) for v in frame.loc[mask, 'observation_year'].unique()),
                   'class_counts': {c: int((frame.loc[mask, 'target'] == c).sum()) for c in CLASSES},
                   'all_classes_present': all((frame.loc[mask, 'target'] == c).any() for c in CLASSES),
                   'lineage_sha256': hashlib.sha256(','.join(map(str, frame.loc[mask, 'SOURCE_ROW_NUMBER'])).encode()).hexdigest()}
            for name, mask in masks.items()}


def draw_figures(metrics, importance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    directory = ROOT / 'reports/figures'
    directory.mkdir(exist_ok=True)
    matrix = np.asarray(metrics['confusion_matrix'])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout='constrained')
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    for ax, values, title in zip(axes, (matrix, normalized), ('Test confusion matrix — counts', 'Test confusion matrix — within-class recall')):
        image = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if values is normalized else None)
        for i in range(5):
            for j in range(5):
                label = f'{values[i,j]:.1%}' if values is normalized else str(values[i,j])
                ax.text(j, i, label, ha='center', va='center', color='white' if values[i,j] > values.max() * .55 else 'black', fontsize=10)
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES, xlabel='Predicted category', ylabel='Recorded category', title=title)
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
        fig.colorbar(image, ax=ax, shrink=.75)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(directory / ('confusion_matrix.' + extension), dpi=300)
    plt.close(fig)
    display = importance.sort_values('importance')
    fig, ax = plt.subplots(figsize=(9, max(4.5, len(display) * .36)), layout='constrained')
    ax.barh(display.feature, display.importance, color='#236B8E')
    if metrics['selected_model'].startswith('logistic'):
        ax.set(xlabel='Mean absolute standardized class coefficient', title='Selected Logistic Regression\nModel-associated coefficient magnitude')
    else:
        ax.set(xlabel='Model-associated importance (method in model metadata)', title='Selected abundance-category classifier')
    ax.axvline(0, color='black', linewidth=.7)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(directory / ('feature_importance.' + extension), dpi=300)
    plt.close(fig)


def run():
    before = protected_snapshot()
    frame, feature_meta = load_inputs()
    masks = split_masks(frame)
    if sum(int(m.sum()) for m in masks.values()) != int(frame.eligible.sum()):
        raise ValueError('Eligible row outside declared temporal split')
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if (masks[left] & masks[right]).any():
            raise ValueError('Temporal split overlap')
    reports = ROOT / 'reports'
    (ROOT / 'models').mkdir(exist_ok=True)
    splits = split_summary(frame, masks)
    write_json(reports / 'phase4_split_summary.json', splits)
    assignment = frame[['SOURCE_ROW_NUMBER', 'date', 'eligible']].copy()
    assignment['split'] = 'ineligible'
    for name, mask in masks.items():
        assignment.loc[mask, 'split'] = name
    assignment.to_csv(reports / 'phase4_split_assignments.csv', index=False)
    train, valid = frame.loc[masks['train']], frame.loc[masks['validation']]
    X, y = train[FEATURES], train.target
    Xv, yv = valid[FEATURES], valid.target
    candidates, trained, comparison, validation_results, audit = make_candidates(), {}, [], {}, []
    # Candidate recipes are fixed before any test prediction; repeats verify determinism.
    with threadpool_limits(limits=1):
        for name, pipeline in candidates.items():
            print('Training and reproducibility check: ' + name, flush=True)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                pipeline.fit(X, y)
            probability = pipeline.predict_proba(Xv)
            prediction = pipeline.predict(Xv)
            replica = clone(pipeline).fit(X, y)
            repeated = replica.predict_proba(Xv)
            difference = float(np.max(np.abs(probability - repeated)))
            if not np.allclose(probability, repeated, rtol=0, atol=1e-12):
                raise RuntimeError('Training not reproducible: ' + name)
            metrics = evaluate(yv, prediction, probability, pipeline.classes_)
            statistics = pipeline.named_steps['imputer'].statistics_
            if not np.allclose(statistics, X.median().to_numpy(), rtol=0, atol=1e-12):
                raise RuntimeError('Candidate imputer not fitted to training median')
            audit.append({'stage': 'candidate_training_and_repeat', 'model': name, 'fit_split': 'train',
                          'fit_rows': len(X), 'fit_lineage_sha256': splits['train']['lineage_sha256'],
                          'imputer_statistics': statistics.tolist(), 'max_probability_difference_on_validation': difference,
                          'warnings': [str(w.message) for w in caught], 'training_seconds_including_repeat': time.perf_counter() - started})
            validation_results[name] = metrics
            row = {'model': name}
            row.update({'validation_' + k: v for k, v in metrics.items() if isinstance(v, float)})
            comparison.append(row)
            trained[name] = pipeline
            write_json(reports / 'validation_metrics.json', validation_results)
            print(f'{name}: validation macro F1={metrics["macro_f1"]:.6f}', flush=True)
        winner = sorted(validation_results, key=lambda n: (-validation_results[n]['macro_f1'], -validation_results[n]['macro_recall'], n))[0]
        # Freeze selection on disk before fitting the final pipeline or predicting test.
        write_json(reports / 'phase4_selection.json', {'selected_model': winner, 'primary_metric': 'validation_macro_f1',
                   'tie_breaks': ['validation_macro_recall', 'model name ascending'], 'test_predictions_made': 0,
                   'candidate_hyperparameters': {n: p.named_steps['classifier'].get_params() for n, p in candidates.items()}})
        print('Validation-selected model: ' + winner, flush=True)
        # If needed, interpretation uses the training-only winner and validation, never test.
        classifier = trained[winner].named_steps['classifier']
        permutation = None
        if not hasattr(classifier, 'feature_importances_') and not hasattr(classifier, 'coef_') and winner != 'majority_baseline':
            permutation = permutation_importance(trained[winner], Xv, yv, scoring='f1_macro', n_repeats=5, random_state=SEED, n_jobs=1)
        final_rows = masks['train'] | masks['validation']
        Xf, yf = frame.loc[final_rows, FEATURES], frame.loc[final_rows, 'target']
        final = clone(candidates[winner]).fit(Xf, yf)
        replicate_final = clone(candidates[winner]).fit(Xf, yf)
        difference = float(np.max(np.abs(final.predict_proba(Xv) - replicate_final.predict_proba(Xv))))
        if difference > 1e-12:
            raise RuntimeError('Final refit not reproducible')
        audit.append({'stage': 'final_refit_and_repeat', 'model': winner, 'fit_split': 'train+validation',
                      'fit_rows': len(Xf), 'imputer_statistics': final.named_steps['imputer'].statistics_.tolist(),
                      'max_probability_difference_on_validation': difference})
        # Sole held-out test evaluation: the selected/refitted pipeline only.
        test = frame.loc[masks['test']]
        probability = final.predict_proba(test[FEATURES])
        prediction = final.classes_[np.argmax(probability, axis=1)]
        final_metrics = evaluate(test.target, prediction, probability, final.classes_)
        final_metrics['selected_model'] = winner
        final_metrics['test_evaluations_per_run'] = 1
        final_metrics['calendar_subperiod_metrics'] = {}
        for year in (2023, 2024):
            mask = test.observation_year.eq(year).to_numpy()
            # Descriptive breakdown of the same saved predictions; never rerun selection.
            final_metrics['calendar_subperiod_metrics'][str(year)] = {'rows': int(mask.sum()),
                'accuracy': float(accuracy_score(test.target.iloc[np.flatnonzero(mask)], prediction[mask])),
                'macro_f1_fixed_five_classes': float(precision_recall_fscore_support(test.target.iloc[np.flatnonzero(mask)], prediction[mask], labels=CLASSES, average='macro', zero_division=0)[2]),
                'absent_recorded_classes': [c for c in CLASSES if not test.target.iloc[np.flatnonzero(mask)].eq(c).any()]}
    saved = pd.DataFrame({'SOURCE_ROW_NUMBER': test.SOURCE_ROW_NUMBER.to_numpy(), 'recorded_category': test.target.to_numpy(), 'predicted_category': prediction})
    for i, c in enumerate(final.classes_):
        saved['probability_' + c.replace(' ', '_')] = probability[:, i]
    saved.to_csv(reports / 'phase4_test_predictions.csv', index=False)
    feature_names = final.named_steps['imputer'].get_feature_names_out(FEATURES)
    classifier = final.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        importance = pd.DataFrame({'feature': feature_names, 'importance': classifier.feature_importances_})
        interpretation = 'Final refitted forest impurity-based model-associated importance; includes missingness indicators; correlated/high-cardinality features can bias attribution; not causal.'
    elif hasattr(classifier, 'coef_'):
        importance = pd.DataFrame({'feature': feature_names, 'importance': np.mean(np.abs(classifier.coef_), axis=0)})
        pd.DataFrame(classifier.coef_, index=classifier.classes_, columns=feature_names).to_csv(reports / 'logistic_coefficients.csv', index_label='class')
        interpretation = 'Mean absolute class coefficient after training-pool imputation and scaling, plus class-specific coefficient table; not causal.'
    elif permutation is not None:
        importance = pd.DataFrame({'feature': FEATURES, 'importance': permutation.importances_mean, 'std': permutation.importances_std})
        interpretation = 'Validation permutation decrease in macro F1 of the selected training-only pipeline before refit (5 repeats, seed 42); diagnostic permutations are not saved as research observations; not causal.'
    else:
        importance = pd.DataFrame({'feature': FEATURES, 'importance': np.zeros(len(FEATURES))})
        interpretation = 'Majority baseline ignores all features; zeros indicate no predictor dependence.'
    importance.to_csv(reports / 'feature_importance.csv', index=False)
    draw_figures(final_metrics, importance)
    for row in comparison:
        row['selected'] = row['model'] == winner
        for key, value in final_metrics.items():
            if isinstance(value, float):
                row['test_' + key] = value if row['selected'] else None
    pd.DataFrame(comparison).to_csv(reports / 'model_comparison.csv', index=False)
    write_json(reports / 'final_test_metrics.json', final_metrics)
    joblib.dump(final, ROOT / 'models/best_model.joblib', compress=3)
    if protected_snapshot() != before:
        raise RuntimeError('Protected source content changed')
    import matplotlib
    metadata = {'target_definition': feature_meta['target_definition'], 'scientific_limitation': CLAIM,
                'selected_model': winner, 'selected_features': FEATURES, 'random_seed': SEED,
                'transformed_feature_names': list(feature_names),
                'internal_missingness_indicators': [str(n) for n in feature_names if str(n).startswith('missingindicator_')],
                'preprocessing': 'Pipeline: training-fitted median imputation + missingness indicators; StandardScaler only for logistic regression. No source data written.',
                'hyperparameters': final.named_steps['classifier'].get_params(), 'split_summary': splits,
                'selection': 'Highest validation macro F1, then macro recall, then model name; no test-based tuning',
                'validation_metrics': validation_results, 'final_test_metrics': final_metrics,
                'imputation_audit': audit, 'train_only_medians': X.median().to_dict(), 'final_train_validation_medians': Xf.median().to_dict(),
                'final_fit_rows': len(Xf), 'interpretation': interpretation,
                'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__, 'matplotlib': matplotlib.__version__, 'scipy': package_versions.version('scipy'), 'threadpoolctl': package_versions.version('threadpoolctl')},
                'protected_input_hashes': before, 'protected_inputs_unchanged': True,
                'artifact_sha256': sha(ROOT / 'models/best_model.joblib'), 'reproducibility_tolerance': 1e-12,
                'limitations': ['Same-sample source category only, not validated future HAB-risk, toxicity, health risk or early warning', 'Temporal and collection/site distribution shift; same sites can occur across periods', 'Strong imbalance, rare high class and partial-year 2024; 2024 has no high/medium/low recorded labels', 'QA=4 and converted QA=5 source values retained; missingness indicators may proxy sampling practices', 'Model scores do not establish causality or operational deployment readiness', 'Training times, serialized bytes and figure metadata may vary; compare metrics/probabilities within tolerance, not wall time']}
    write_json(reports / 'model_metadata.json', metadata)
    print(json.dumps({'selected_model': winner, 'test_macro_f1': final_metrics['macro_f1'], 'test_accuracy': final_metrics['accuracy'], 'protected_inputs_unchanged': True}), flush=True)
    return metadata


if __name__ == '__main__':
    run()
