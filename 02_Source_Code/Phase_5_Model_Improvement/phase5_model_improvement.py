"""Locked-baseline Phase 5 study. Only Phase 5-specific output paths are used.

Run twice: python scripts/phase5_model_improvement.py
Each fixed reproduction selects on validation, then evaluates its winner once.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import train_models as p4
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from phase5_transformers import Phase5FeatureGroup, GROUPS, ENV, SPACE, TIME, CYCLIC

SEED = 42
CLASSES = p4.CLASSES
CLAIM = 'The target represents source-recorded abundance categories and is not a validated future HAB-risk or early-warning target.'


def snapshot():
    # A frozen lock covers its original files, not later phases' new files.
    lock = ROOT / 'reports/phase5_baseline.json'
    if lock.is_file():
        names = json.loads(lock.read_text(encoding='utf-8'))['locked_hashes']
        return {name: p4.sha(ROOT / name) for name in names}
    files = []
    for folder in ('data', 'models', 'reports', 'docs', 'scripts', 'tests'):
        for path in (ROOT / folder).rglob('*'):
            relative = path.relative_to(ROOT)
            if not path.is_file() or '__pycache__' in relative.parts:
                continue
            if any(part.startswith(('phase5', 'test_phase5')) for part in relative.parts):
                continue
            files.append(path)
    files.append(ROOT / 'requirements_phase4.txt')
    return {str(p.relative_to(ROOT).as_posix()): p4.sha(p) for p in sorted(files)}


def write(name, value):
    if not Path(name).name.startswith('phase5'):
        raise ValueError('Refusing non-Phase-5 report output')
    p4.write_json(ROOT / 'reports' / name, value)


def experiments():
    result = {}
    for group in GROUPS:
        for family in ('logistic', 'random_forest', 'hist_gradient'):
            weights = ('balanced', None) if group.startswith('D_') else ('balanced',)
            for weight in weights:
                if family == 'logistic':
                    model = LogisticRegression(C=1, solver='lbfgs', max_iter=1000, class_weight=weight, random_state=SEED)
                elif family == 'random_forest':
                    model = RandomForestClassifier(n_estimators=40, max_depth=14, min_samples_leaf=8, max_features='sqrt', class_weight=weight, random_state=SEED, n_jobs=1)
                else:
                    model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, min_samples_leaf=30, learning_rate=.1, l2_regularization=1, early_stopping=False, class_weight=weight, random_state=SEED)
                steps = [('features', Phase5FeatureGroup(group)), ('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True))]
                if family == 'logistic':
                    steps.append(('scaler', StandardScaler()))
                name = f'{group}__{family}__{weight or "unweighted"}'
                result[name] = {'group': group, 'family': family, 'weight': weight or 'unweighted',
                                'pipeline': Pipeline(steps + [('classifier', model)])}
    return result


def candidate_report(frame):
    phase3 = json.loads((ROOT / 'data/ml/feature_metadata.json').read_text(encoding='utf-8'))
    phase2 = json.loads((ROOT / 'reports/phase2_data_quality_report.json').read_text(encoding='utf-8'))
    lookup = {p['name']: p for p in phase2['columns']}
    candidates = []
    for field in p4.FEATURES:
        old = next(c for c in phase3['candidate_features'] if c['feature_name'] == field)
        candidates.append({'name': field, 'source_column': old['source_column'],
                           'meaning': lookup.get(field, {}).get('meaning', 'Recorded calendar component'),
                           'unit': old['unit'], 'missing_percentage': old['missing_percentage'],
                           'included': True, 'reason': 'Existing approved Phase 3 predictor; controlled group ablation',
                           'leakage_risk': 'No direct target content; collection/spatial/temporal confounding remains',
                           'available_at_observation_time': 'Recorded at collection; prospective sensor availability not validated' if field in ENV else 'Date/location context known at collection'})
    for field in CYCLIC:
        candidates.append({'name': field, 'source_column': 'Phase 3 observation_month/day_of_year/observation_year from SAMPLE_DATE',
                           'meaning': 'Deterministic calendar-cycle coordinate, not a marine season label', 'unit': 'dimensionless',
                           'missing_percentage': 0, 'included': True, 'reason': 'Represent calendar wrap-around without a learned phase or physical threshold',
                           'leakage_risk': 'Same-observation date only; no future value', 'available_at_observation_time': 'Calendar context known at collection'})
    for old in phase3['candidate_features']:
        field = old['feature_name']
        if field in p4.FEATURES:
            continue
        candidates.append({'name': field, 'source_column': old['source_column'], 'meaning': lookup.get(field, {}).get('meaning', 'Source metadata'),
                           'unit': old['unit'] or lookup.get(field, {}).get('unit'), 'missing_percentage': old['missing_percentage'], 'included': False,
                           'reason': old['reason'], 'leakage_risk': old['leakage_risk'],
                           'available_at_observation_time': 'Not used; laboratory/admin timing or sensor availability not established'})
    return {'source': 'NOAA dictionary and unchanged Phase 1–3 documentation/metadata', 'candidates': candidates,
            'groups': GROUPS, 'new_scientific_variables_added': [],
            'formulas': {'month_angle': '2*pi*(month-1)/12', 'day_angle': '2*pi*(day_of_year-1)/(366 if leap year else 365)',
                         'cyclic_coordinates': 'sin(angle), cos(angle)'},
            'calendar_design': 'D replaces raw year/month/day with four calendar-cycle coordinates. This bundles cyclic encoding with absolute-year removal; ablation cannot isolate these effects.',
            'rejected_transformations': {'wind_vectors': 'Reference/from-to convention undocumented and measurements extremely sparse',
                                         'spatial_distance_or_bins': 'No documented coordinate datum/projection or scientifically justified region/site bins',
                                         'environmental_log_or_interactions': 'No prespecified scientific/measurement rationale for additional transforms; tree models already allow nonlinear associations',
                                         'additional_sensors': 'pH, turbidity, TDS, oxygen, chlorophyll-a unavailable'},
            'leakage_exclusions': ['CATEGORY', 'CELLCOUNT', 'CELLCOUNT_QA', 'GENUS', 'SPECIES', 'OBJECTID', 'SOURCE_ROW_NUMBER', 'compound target-quality flags']}


def draw_confusion(metrics, filename, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix = np.asarray(metrics['confusion_matrix'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    for ax, normalized in zip(axes, (False, True)):
        data = matrix / matrix.sum(axis=1, keepdims=True) if normalized else matrix
        im = ax.imshow(data, cmap='Blues', vmin=0, vmax=1 if normalized else None)
        for i in range(5):
            for j in range(5):
                text = f'{data[i,j]:.1%}' if normalized else str(matrix[i,j])
                ax.text(j, i, text, ha='center', va='center', fontsize=9, color='white' if data[i,j] > .55 * data.max() else 'black')
        ax.set(xticks=range(5), yticks=range(5), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted category', ylabel='Recorded category', title='Within-class recall' if normalized else 'Counts')
        plt.setp(ax.get_xticklabels(), rotation=35, ha='right')
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle(title, fontsize=12)
    fig.savefig(ROOT / 'reports/figures' / filename, dpi=300)
    plt.close(fig)


def error_analysis(baseline):
    saved = pd.read_csv(ROOT / 'reports/phase4_test_predictions.csv')
    cm = p4.confusion_matrix(saved.recorded_category, saved.predicted_category, labels=CLASSES)
    report = p4.classification_report(saved.recorded_category, saved.predicted_category, labels=CLASSES, output_dict=True, zero_division=0)
    rows = []
    for i, c in enumerate(CLASSES):
        item = report[c]
        rows.append({'record_type': 'class', 'class': c, 'support': int(item['support']), 'precision': item['precision'], 'recall': item['recall'],
                     'f1': item['f1-score'], 'true_positive': int(cm[i,i]), 'false_negative': int(cm[i,:].sum()-cm[i,i]), 'false_positive': int(cm[:,i].sum()-cm[i,i])})
    for avg in ('macro avg', 'weighted avg'):
        item = report[avg]
        rows.append({'record_type': 'aggregate', 'class': avg, 'support': len(saved), 'precision': item['precision'], 'recall': item['recall'], 'f1': item['f1-score']})
    balanced = float(p4.balanced_accuracy_score(saved.recorded_category, saved.predicted_category))
    rows.append({'record_type': 'aggregate', 'class': 'balanced_accuracy', 'support': len(saved), 'balanced_accuracy': balanced})
    pd.DataFrame(rows).to_csv(ROOT / 'reports/phase5_error_analysis.csv', index=False)
    if cm.tolist() != baseline['test']['confusion_matrix'] or abs(report['macro avg']['f1-score'] - baseline['test']['macro_f1']) > 1e-12:
        raise ValueError('Phase 4 saved predictions do not reproduce locked metrics')
    draw_confusion(baseline['test'], 'phase5_error_confusion_matrix.png', 'Phase 5 descriptive audit of locked Phase 4 predictions')


def run():
    before = snapshot()
    previous_lock_path = ROOT / 'reports/phase5_baseline.json'
    if previous_lock_path.exists():
        expected = json.loads(previous_lock_path.read_text(encoding='utf-8'))['locked_hashes']
        if expected != before:
            raise ValueError('Existing Phase 1–4 lock differs; stop without overwriting Phase 4')
    p4meta = json.loads((ROOT / 'reports/model_metadata.json').read_text(encoding='utf-8'))
    if p4.sha(ROOT / 'models/best_model.joblib') != p4meta['artifact_sha256']:
        raise ValueError('Locked Phase 4 model hash mismatch')
    if p4meta['selected_model'] != 'logistic_balanced':
        raise ValueError('Unexpected Phase 4 baseline')
    baseline = {'selected_model': p4meta['selected_model'], 'validation': p4meta['validation_metrics']['logistic_balanced'],
                'test': p4meta['final_test_metrics'], 'locked_hashes': before,
                'analysis_boundary': 'Phase 4 test mistakes audited descriptively; no test-covariate search or test-driven tuning. Reused test is not a new independent holdout.'}
    write('phase5_baseline.json', baseline)
    error_analysis(baseline)
    frame, feature_meta = p4.load_inputs()
    masks = p4.split_masks(frame)
    summaries = p4.split_summary(frame, masks)
    write('phase5_candidate_features.json', candidate_report(frame))
    candidates = experiments()
    plan = {'seed': SEED, 'groups': GROUPS, 'experiment_count': len(candidates),
            'recipes': {n: {'feature_group': e['group'], 'family': e['family'], 'weight': e['weight'], 'hyperparameters': e['pipeline'].named_steps['classifier'].get_params()} for n, e in candidates.items()},
            'selection': 'validation macro F1, then mean low/medium/high recall, then experiment name ascending',
            'split_summary': summaries, 'resampling': 'none', 'claim': CLAIM}
    plan_path = ROOT / 'reports/phase5_experiment_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8')) != plan:
        raise ValueError('Frozen Phase 5 experiment plan changed; no automatic retuning')
    write('phase5_experiment_plan.json', plan)
    prior_run_path = ROOT / 'reports/phase5_model_metadata.json'
    previous = json.loads(prior_run_path.read_text(encoding='utf-8')) if prior_run_path.exists() else None
    train, valid = frame.loc[masks['train']], frame.loc[masks['validation']]
    X, y, Xv, yv = train[p4.FEATURES], train.target, valid[p4.FEATURES], valid.target
    comparisons, metrics_by_name, audits = [], {}, []
    best = None
    with p4.threadpool_limits(limits=1):
        for name, experiment in candidates.items():
            print('Phase 5 experiment: ' + name, flush=True)
            pipeline = experiment['pipeline']
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                pipeline.fit(X, y)
            probabilities = pipeline.predict_proba(Xv)
            prediction = pipeline.classes_[np.argmax(probabilities, axis=1)]
            metric = p4.evaluate(yv, prediction, probabilities, pipeline.classes_)
            metrics_by_name[name] = metric
            group_frame = pipeline.named_steps['features'].transform(X)
            statistics = pipeline.named_steps['imputer'].statistics_
            if not np.allclose(statistics, group_frame.median().to_numpy(), atol=1e-12, rtol=0):
                raise ValueError('Phase 5 candidate imputer training-only check failed')
            audits.append({'experiment': name, 'fit_split': 'train', 'fit_rows': len(X),
                           'fit_row_hash': summaries['train']['lineage_sha256'],
                           'transformed_columns': list(group_frame.columns), 'imputer_statistics': statistics.tolist(), 'warnings': [str(w.message) for w in captured]})
            row = {'experiment_name': name, 'feature_group': experiment['group'], 'model': experiment['family'], 'class_weight_strategy': experiment['weight']}
            for key in ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1'):
                row['validation_' + key] = metric[key]
            for c in ('low', 'medium', 'high'):
                row[c + '_recall'] = metric['per_class'][c]['recall']
            comparisons.append(row)
            rank = (-row['validation_macro_f1'], -np.mean([row[c+'_recall'] for c in ('low', 'medium', 'high')]), name)
            if best is None or rank < best[0]:
                best = (rank, name, pipeline)
            print(f'validation macro F1={metric["macro_f1"]:.6f}; high recall={row["high_recall"]:.6f}', flush=True)
            pd.DataFrame(comparisons).to_csv(ROOT / 'reports/phase5_ablation_results.csv', index=False)
        _, selected, winning_train_pipeline = best
        frozen_selection = {'experiment': selected, 'validation_metrics': metrics_by_name[selected], 'test_predictions_made_this_run': 0, 'plan_sha256': p4.sha(plan_path)}
        if previous and previous['selected_experiment'] != selected:
            raise ValueError('Reproduction changed validation winner; stop before test')
        write('phase5_selection.json', frozen_selection)
        print('Phase 5 validation selection frozen: ' + selected, flush=True)
        candidate = candidates[selected]
        model = winning_train_pipeline.named_steps['classifier']
        permutation = None
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            # Interpretation of selected training-only model on validation, never test.
            permutation = permutation_importance(winning_train_pipeline, Xv, yv, scoring='f1_macro', n_repeats=3, random_state=SEED, n_jobs=1)
        fit_mask = masks['train'] | masks['validation']
        final = clone(candidate['pipeline']).fit(frame.loc[fit_mask, p4.FEATURES], frame.loc[fit_mask, 'target'])
        final_group = final.named_steps['features'].transform(frame.loc[fit_mask, p4.FEATURES])
        if not np.allclose(final.named_steps['imputer'].statistics_, final_group.median().to_numpy(), atol=1e-12, rtol=0):
            raise ValueError('Final imputer fitted outside train+validation pool')
        # Exactly one final test probability pass per independent fixed reproduction.
        test = frame.loc[masks['test']]
        probabilities = final.predict_proba(test[p4.FEATURES])
        prediction = final.classes_[np.argmax(probabilities, axis=1)]
        final_metrics = p4.evaluate(test.target, prediction, probabilities, final.classes_)
    original_names = final.named_steps['features'].get_feature_names_out()
    transformed_names = final.named_steps['imputer'].get_feature_names_out(original_names)
    model = final.named_steps['classifier']
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({'feature': transformed_names, 'importance': model.feature_importances_})
        method = 'Final refitted forest impurity-based model-associated feature importance; biased by correlations/feature cardinality; not causal'
    elif hasattr(model, 'coef_'):
        importance = pd.DataFrame({'feature': transformed_names, 'importance': np.mean(np.abs(model.coef_), axis=0)})
        pd.DataFrame(model.coef_, index=model.classes_, columns=transformed_names).to_csv(ROOT / 'reports/phase5_coefficients.csv', index_label='class')
        method = 'Mean absolute standardized class coefficient of final refit; signed coefficients separate; not causal'
    else:
        importance = pd.DataFrame({'feature': p4.FEATURES, 'importance': permutation.importances_mean, 'std': permutation.importances_std})
        method = 'Permutation decrease in validation macro F1 of selected training-only pipeline, 3 diagnostic repeats seed42; grouped source inputs before final refit; not causal'
    importance.to_csv(ROOT / 'reports/phase5_feature_importance.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    display = importance.sort_values('importance')
    fig, ax = plt.subplots(figsize=(10, max(5, len(display)*.36)), layout='constrained')
    ax.barh(display.feature, display.importance, color='#236B8E')
    ax.set(title='Selected Phase 5 model\nModel-associated feature importance', xlabel='Importance (method documented in Phase 5 metadata)')
    fig.savefig(ROOT / 'reports/figures/phase5_feature_importance.png', dpi=300)
    plt.close(fig)
    draw_confusion(final_metrics, 'phase5_final_confusion_matrix.png', 'Selected Phase 5 recorded-abundance classifier — test')
    predictions = pd.DataFrame({'SOURCE_ROW_NUMBER': test.SOURCE_ROW_NUMBER.to_numpy(), 'recorded_category': test.target.to_numpy(), 'predicted_category': prediction})
    for i, c in enumerate(final.classes_):
        predictions['probability_' + c.replace(' ', '_')] = probabilities[:,i]
    prior_prediction_path = ROOT / 'reports/phase5_test_predictions.csv'
    probability_difference = None
    if previous:
        old_predictions = pd.read_csv(prior_prediction_path)
        identity_columns = ['SOURCE_ROW_NUMBER', 'recorded_category', 'predicted_category']
        if not predictions[identity_columns].equals(old_predictions[identity_columns]):
            raise RuntimeError('Full reproduction changed test identities/labels/predictions')
        probability_columns = [c for c in predictions.columns if c.startswith('probability_')]
        probability_difference = float(np.max(np.abs(predictions[probability_columns].to_numpy()-old_predictions[probability_columns].to_numpy())))
        if probability_difference > 1e-12:
            raise RuntimeError('Full reproduction probabilities differ beyond tolerance')
    predictions.to_csv(prior_prediction_path, index=False)
    p4.joblib.dump(final, ROOT / 'models/phase5_best_model.joblib', compress=3)
    comparison = []
    for phase, metric in [('Phase4_locked', baseline['test']), ('Phase5_selected', final_metrics)]:
        row = {'phase': phase}
        for key in ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1'):
            row[key] = metric[key]
        for c in ('low', 'medium', 'high'):
            row[c+'_recall'] = metric['per_class'][c]['recall']
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(ROOT / 'reports/phase5_vs_phase4.csv', index=False)
    if snapshot() != before:
        raise RuntimeError('Phase 1–4 locked artifacts changed')
    metadata = {'scientific_limitation': CLAIM, 'selected_experiment': selected, 'feature_group': candidate['group'],
                'model': candidate['family'], 'class_weight': candidate['weight'], 'seed': SEED, 'experiment_count': len(candidates),
                'target_definition': feature_meta['target_definition'], 'selected_features': list(original_names), 'internal_model_features': list(transformed_names),
                'validation_metrics': metrics_by_name, 'test_metrics': final_metrics, 'test_evaluations_per_run': 1,
                'hyperparameters': model.get_params(), 'split_summary': summaries, 'candidate_imputation_audit': audits,
                'final_fit_rows': int(fit_mask.sum()), 'final_imputer_statistics': final.named_steps['imputer'].statistics_.tolist(),
                'resampling': 'none; no synthetic observations', 'preprocessing': 'Group selection/cycles then training-fitted median+indicators; scaling only logistic; refit pool train+validation only',
                'interpretation_method': method, 'versions': p4meta['versions'], 'locked_hashes': before, 'phase1_to_4_unchanged': True,
                'plan_sha256': p4.sha(plan_path), 'model_sha256': p4.sha(ROOT / 'models/phase5_best_model.joblib'),
                'test_macro_f1_delta_vs_phase4': final_metrics['macro_f1']-baseline['test']['macro_f1'],
                'test_macro_f1_improved': final_metrics['macro_f1'] > baseline['test']['macro_f1'],
                'limitations': ['Same previously examined 2023–2024 test reused by explicit Phase 5 design, not independent external validation',
                                'Phase 4 descriptive error audit does not establish causal failure mechanism',
                                'Engineered group bundles cyclic encoding and removal of absolute year, not an isolated intervention',
                                'Sampling/site/time dependence, missingness and source-category quality persist',
                                'No validated future HAB-risk, toxicity, health-risk or early-warning capability']}
    # Compare actual complete reproductions, excluding runtime/serialized artifact bytes.
    reproducibility_path = ROOT / 'reports/phase5_reproducibility.json'
    old_repro = json.loads(reproducibility_path.read_text(encoding='utf-8')) if reproducibility_path.exists() else {}
    max_difference = None
    if previous:
        differences = []
        for name in metrics_by_name:
            for key in ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1', 'log_loss'):
                differences.append(abs(metrics_by_name[name][key]-previous['validation_metrics'][name][key]))
            if metrics_by_name[name]['confusion_matrix'] != previous['validation_metrics'][name]['confusion_matrix']:
                raise RuntimeError('Validation confusion matrix changed on reproduction')
            for c in CLASSES:
                for key in ('precision', 'recall', 'f1-score', 'support'):
                    differences.append(abs(metrics_by_name[name]['per_class'][c][key]-previous['validation_metrics'][name]['per_class'][c][key]))
        for key in ('accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_precision', 'weighted_recall', 'weighted_f1', 'log_loss'):
            differences.append(abs(final_metrics[key]-previous['test_metrics'][key]))
        max_difference = max(differences)
        if max_difference > 1e-12:
            raise RuntimeError('Phase 5 full reproduction differs beyond tolerance')
    write('phase5_model_metadata.json', metadata)
    write('phase5_reproducibility.json', {'complete_runs': old_repro.get('complete_runs', 0)+1,
          'same_selected_experiment': previous is not None and previous['selected_experiment']==selected,
          'maximum_metric_absolute_difference': max_difference, 'tolerance': 1e-12,
          'maximum_test_probability_absolute_difference': probability_difference,
          'verified': max_difference is not None and max_difference<=1e-12,
          'protocol': 'Two complete fixed-recipe reproductions; one selected-model test evaluation per run. Second run repeats the frozen experiment; no retuning after test.'})
    print(json.dumps({'selected': selected, 'validation_macro_f1': metrics_by_name[selected]['macro_f1'], 'test_macro_f1': final_metrics['macro_f1'],
                      'delta_vs_phase4': metadata['test_macro_f1_delta_vs_phase4'], 'locked_artifacts_unchanged': True}), flush=True)
    return metadata


if __name__ == '__main__':
    run()
