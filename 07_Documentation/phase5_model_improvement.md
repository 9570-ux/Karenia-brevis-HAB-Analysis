# Phase 5 — controlled model improvement and error analysis

## Objective and scientific boundary

Investigate the locked Phase 4 model's minority-category failures using documented source variables, deterministic feature representations, training-only imbalance handling, and a fixed validation-selected classical-model comparison.

**The target represents source-recorded abundance categories and is not a validated future HAB-risk or early-warning target.**

No target is reconstructed, relabeled, merged or converted into binary risk. No future bloom, toxicity, health-risk, causal or operational-alert capability is claimed.

## Locked Phase 4 baseline

The original Balanced Logistic Regression model in `models/best_model.joblib` and all existing data/reports/source files are locked by SHA-256. The Phase 5 script captures 52 existing Phase 1–4 files, including raw/processed/Phase 3 inputs, model/results, figures, documentation and source/tests; Python caches and new Phase 5-prefixed files are excluded. `reports/phase5_baseline.json` records the lock and exact original metrics. Every run verifies that the lock matches and checks it again after completion. Only Phase 5-specific research outputs are written.

Phase 4 validation macro F1: **0.20273098770614473**. Phase 4 final test macro F1: **0.19697460958001423**; accuracy **0.6441841680129241**. Low, medium and high test recall were all zero. The old model and its reports/figures are not overwritten or retrained by Phase 5.

## Error analysis and evaluation independence

Error analysis recomputes confusion matrix, per-class precision/recall/F1, macro/weighted averages and balanced accuracy from **saved Phase 4 test predictions**, without calling the Phase 4 model again. The recomputed matrix and macro F1 must match the locked report.

The baseline missed all 431 low, 297 medium and 90 high observations. Its high predictions were all false positives: 744 false high predictions and zero true high predictions. This is descriptive error accounting, not evidence for a specific causal failure mechanism. The report does not search test covariates for thresholds, regional rules or parameter choices.

Outputs: `reports/phase5_error_analysis.csv` and `reports/figures/phase5_error_confusion_matrix.png`. The original Phase 4 confusion matrix remains untouched.

**The 2023–March 2024 test has already been examined in Phase 4 and reused by the user's explicit Phase 5 design. It is not a new independent external holdout.** Phase 5 selection still uses validation only, and the experiment plan is frozen before new Phase 5 test predictions. Subsequent independent/prospective data would be needed for stronger generalization claims.

## Candidate variables and leakage prevention

The existing NOAA dictionary, Phase 2 profiles and Phase 3 feature metadata were inspected. Candidate source measurements remain temperature (deg. C), salinity (ppt), depth (m), latitude/longitude (decimal degrees), and calendar components derived from the same observation's original SAMPLE_DATE. Missing percentages remain approximately 49.62% temperature, 48.97% salinity, 2.60% depth, and zero for recorded coordinates/calendar components.

`reports/phase5_candidate_features.json` lists each candidate's source, meaning, unit, missingness, inclusion/exclusion reason, leakage risk and observation-time availability. Environmental values are documented as recorded at collection; prospective sensor availability is not established. Date/location context is known at observation time. No new measured environmental variable is introduced.

Wind direction remains excluded because reference/from-to convention is not documented; winds are also extremely sparse. Time-of-day, unresolved site names, arbitrary geographic bins/projections and unmotivated environmental log transforms/interactions are not added. No physical threshold is invented. pH, turbidity, TDS, dissolved oxygen and chlorophyll-a remain unavailable.

CATEGORY, CELLCOUNT, CELLCOUNT_QA and other target-quality fields, biological target context, compound count-related flags, IDs and row ordering are never predictors. SOURCE_ROW_NUMBER remains solely an audit key. Original Phase 3 target eligibility and feature QA handling are consumed unchanged through the read-only Phase 4 loader. No target-derived feature is introduced.

## Four feature groups and deterministic engineering

| Group | Model-level source/derived features before imputation |
|---|---|
| A_original8 | Original 8 Phase 3 features, unchanged |
| B_temporal_spatial | LATITUDE, LONGITUDE, observation_year, observation_month, day_of_year |
| C_environmental | WATER_TEMP, SALINITY, SAMPLE_DEPTH |
| D_environmental_temporal_spatial_cyclic | WATER_TEMP, SALINITY, SAMPLE_DEPTH, LATITUDE, LONGITUDE, month_sin, month_cos, day_of_year_sin, day_of_year_cos |

All group transforms are inside the fitted sklearn pipeline. No Phase 3 CSV or feature-builder logic is changed. `scripts/phase5_transformers.py` provides the serializable Phase 5-only transformer.

Formulas use known calendar arithmetic, with no fitted phase/period:

```text
month_angle = 2*pi*(observation_month - 1)/12
day_angle = 2*pi*(day_of_year - 1)/(366 if calendar leap year else 365)
month_sin = sin(month_angle); month_cos = cos(month_angle)
day_of_year_sin = sin(day_angle); day_of_year_cos = cos(day_angle)
```

The new coordinates are dimensionless representations of calendar wrap-around, not fabricated observations, bloom thresholds or scientifically asserted marine seasons. Observation year is used to calculate known leap-year length, but is **not an absolute-year predictor in D**. Environmental and coordinate values are preserved. No spatial transform requiring an undocumented coordinate datum is used.

D jointly replaces raw month/day encoding and removes absolute-year trend. Therefore differences between A and D cannot be attributed uniquely to cyclic encoding or uniquely to removing year. They are a bundled feature-design experiment, not a causal ablation. Month and day-of-year cycles can also be correlated.

## Imbalance, missing data and model experiment plan

There are **15 predeclared experiments**:

- All four groups × Logistic Regression, Random Forest, HistGradientBoosting, with balanced class weights: 12 fits.
- D × the same three models with unweighted classes: 3 controlled weighting comparisons.

This is not a full factorial weighting comparison across every group. The restricted weighting controls are explicit to limit search size. No extra algorithms, massive grid search, threshold optimization, SMOTE, over/undersampling or synthetic observations are used. All training rows are retained; class weights are fitted from the current training labels only. Validation/test distributions are unchanged.

| Family | Predeclared capacity/settings |
|---|---|
| Logistic Regression | C=1, lbfgs, max_iter=1000, seed 42 |
| Random Forest | 40 trees, max_depth=14, min_samples_leaf=8, max_features=sqrt, n_jobs=1, seed 42 |
| HistGradientBoosting | 60 iterations, max_leaf_nodes=15, min_samples_leaf=30, learning_rate=.1, l2_regularization=1, early_stopping=False, seed 42 |

Within Phase 5 each family's settings are held fixed across groups/weights. Tree capacities are intentionally bounded and differ from Phase 4, so the cross-phase gain is not an isolated feature-only effect. The A balanced Logistic Regression recipe exactly reproduces the Phase 4 training/validation control and its validation macro F1.

Every pipeline applies group selection/calendar arithmetic, then `SimpleImputer(median, add_indicator=True, keep_empty_features=True)`. Scaling is added only for Logistic Regression. The imputer and scaler fit only training data for candidate evaluation. Missing indicators, as in Phase 4, are internal preprocessing signals rather than new source variables. No filled research dataset is written. Each imputer's statistics are checked against the independently computed median of its transformed training-only group. These modeling replacements do not alter source measurements.

The plan, exact hyperparameters, seed, split summary and selection rule are frozen in `reports/phase5_experiment_plan.json`. If the plan changes on a later reproduction, the script stops rather than silently retuning.

## Temporal validation and selection

The same original eligible rows are used:

- Train: year ≤2020, **184,845** observations.
- Validation: 2021–2022, **18,494** observations.
- Test: 2023–2024-03-25, **7,428** observations.

No random cross-time split occurs. Same-date records stay in the same calendar period. Class distributions and row-ID hashes match the earlier phase. Candidate training never includes validation or test rows.

Selection ranks **validation macro F1**, then mean recall across low/medium/high for exact ties, then experiment name. Accuracy is descriptive only. The selection freeze records zero test predictions before final refitting. The selected recipe is refit on **203,339 train+validation rows**, with preprocessing refit on exactly that pool; test rows never enter the fit. Only that frozen winner is evaluated once on the fixed test per complete reproducibility run. No test outcome changes features, parameters, class weights or selection.

## Ablation results and selected model

Complete results for all 15 experiments—including validation accuracy, balanced accuracy, macro precision/recall/F1, weighted F1 and low/medium/high recall—are in `reports/phase5_ablation_results.csv`. Full per-class metrics and confusion matrices are retained in Phase 5 metadata.

**Selected experiment:** `D_environmental_temporal_spatial_cyclic__random_forest__balanced`.

Validation macro F1: **0.25255772755920713**, compared with the locked Phase 4 **0.20273098770614473**. Its validation high recall is **0.33676975945017185**. D balanced HistGradientBoosting scored macro F1 0.241669, with higher high recall but lower primary macro F1, and therefore was not selected. The environmental-only experiments produced relatively high high-class recall but poor overall macro F1; minority sensitivity alone did not override the predefined primary metric.

The selected pipeline uses the nine D variables plus three fitted missingness indicators. Original absolute year/raw month/day are not classifier inputs in this group. The saved pipeline still accepts the original approved eight-column DataFrame because its first step performs this deterministic selection/engineering.

## Final test results and comparison

| Metric | Phase 4 locked | Phase 5 selected |
|---|---:|---:|
| Accuracy | 0.644184 | 0.691438 |
| Balanced accuracy | 0.197915 | 0.303420 |
| Macro precision | 0.196081 | 0.287087 |
| Macro recall | 0.197915 | 0.303420 |
| Macro F1 | 0.196975 | 0.292562 |
| Weighted F1 | 0.643927 | 0.712474 |
| Low recall | 0 | 0.234339 |
| Medium recall | 0 | 0.188552 |
| High recall | 0 | 0.033333 |

Phase 5 weighted precision is 0.736592, weighted recall 0.691438 and multiclass log loss 0.872265. Macro F1 increases by **0.095587** on this reused test. This is an observed aggregate metric improvement, not statistical significance or independent prospective validation.

| Recorded class | Precision | Recall | F1 | Support | Correct |
|---|---:|---:|---:|---:|---:|
| not observed | .879326 | .812659 | .844679 | 5,909 | 4,802 |
| very low | .223363 | .248217 | .235135 | 701 | 174 |
| low | .170897 | .234339 | .197652 | 431 | 101 |
| medium | .148148 | .188552 | .165926 | 297 | 56 |
| high | .013699 | .033333 | .019417 | 90 | 3 |

**87 of 90 high observations are still missed.** High precision is only **1.37%**, with 216 false high predictions. Despite improved macro F1 and some low/medium recall, high-class performance remains very weak. The outcome does not support reliable rare-high classification or operational alerts. No improvement is fabricated or forced, and no retuning followed these test results.

`reports/phase5_vs_phase4.csv` records the direct comparison. The Phase 5 final confusion matrix is separate from both old and error-audit matrices.

## Model interpretation

The selected forest supplies final-refit impurity-based feature importances, including the fitted missingness indicators. It assigned the largest impurity importances to **LONGITUDE, LATITUDE and day_of_year_sin**. `reports/phase5_feature_importance.csv` and its PNG show **model-associated predictive importance**, not causality. Correlated calendar/spatial features, their cardinality and observation-process missingness can bias impurity importance. No statement that temperature, season or location causes HAB is made.

If another fixed recipe were selected in a genuinely reproduced study, the script uses standardized class coefficients for Logistic Regression or training-only-winner validation permutation importance for HistGradientBoosting. Test data never supplies importance-based feature selection. Diagnostic permutations, if used, are not saved as research observations; the actual selected forest uses native importance.

## Complete reproducibility and artifact loading

The complete script is executed twice with the identical frozen plan:

```powershell
python scripts/phase5_model_improvement.py
python scripts/phase5_model_improvement.py
python -B -m unittest discover -s tests -v
```

This means 15 experiments plus final refit per run, not merely repeating predictions from one saved model. The second run must choose the same validation winner. Complete validation metrics/confusion matrices and final test metrics, row IDs, predicted labels and class probabilities are compared, with tolerance **1e-12** for floating values. The two completed runs selected the same winner: maximum metric difference **0.0**, maximum test probability difference **1.1102230246251565e-16** (CSV floating-point round trip within tolerance). Reproducibility does not depend on serialized model/PNG bytes or timing. Results are recorded in `reports/phase5_reproducibility.json` and separate run logs.

Each independent fixed reproduction has one final selected-model test pass; reexecution is not an opportunity to tune on test. Both runs reuse the same previously examined test and are not independent validation cohorts. Phase 5 does not refit or overwrite Phase 4's model.

The saved custom transformer must remain available when loading the new pipeline:

```python
import sys
import joblib
sys.path.insert(0, "scripts")
pipeline = joblib.load("models/phase5_best_model.joblib")
```

Dependencies and tested versions are those recorded in Phase 4 and Phase 5 metadata; no new scientific library installation is required. Source groups/calendars are implemented in the new Phase 5 helper only. Existing scientific files are unchanged.

## Phase 5-specific outputs

- Required reports: `phase5_baseline.json`, `phase5_error_analysis.csv`, `phase5_candidate_features.json`, `phase5_ablation_results.csv`, `phase5_vs_phase4.csv`, `phase5_feature_importance.csv`, `phase5_model_metadata.json`, `phase5_reproducibility.json`, all under `reports/`.
- Additional audit reports: `reports/phase5_experiment_plan.json`, `phase5_selection.json`, `phase5_test_predictions.csv`, `phase5_run1_log.txt`, `phase5_run2_log.txt`, and `phase5_test_results.txt`.
- Figures: `reports/figures/phase5_error_confusion_matrix.png`, `phase5_feature_importance.png`, `phase5_final_confusion_matrix.png`.
- Model: `models/phase5_best_model.joblib`.
- Code: `scripts/phase5_model_improvement.py`, `scripts/phase5_transformers.py`.
- Tests: `tests/test_phase5.py`.
- Documentation: this file.

## Final verification

**45 tests passed**: the 33 existing Phase 1–4 tests plus 12 Phase 5 tests. The complete suite checks baseline/source hashes, unchanged target definition, disjoint temporal splits, allowed feature groups, actual-date cyclic arithmetic, train-only imputation, absence of resampling, all 15 experiments, validation-only selection, saved test lineage, model loading and complete-run reproducibility. No synthetic test observations are generated. Full evidence is in `reports/phase5_test_results.txt`.

Both complete Phase 5 executions succeeded. They selected the same model, had zero metric difference, and matched probabilities within 1e-12. All 52 locked existing scientific files retained their hashes. Earlier-phase regression tests rebuild their own outputs as originally designed, with identical content; Phase 5 itself never rewrites those datasets or Phase 4 results.

## Limitations and stop boundary

Sparse/missing environmental data, retained QA cautions and upstream conversions, source-reported rather than independently verified abundance bins, site/time sampling dependence, rare high support and partial-year 2024 persist. The source scientific measurements and labels were not fabricated. Statistical model imputation is train-fitted only and never inserted into source data. No causal failure mechanism or causal importance interpretation is established. Aggregate improvements coexist with severe residual minority-class failures, especially high.

No future phase, hardware, ESP32, MQTT, sensor, buzzer/LED, dashboard, Git initialization, commit or push is implemented. Work stops at Phase 5.
