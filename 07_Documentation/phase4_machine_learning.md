# Phase 4 — recorded-abundance model development and evaluation

## Objective and scientific limitation

Train and compare classical classifiers for the existing Phase 3 `observed_abundance_category` target.

**The model predicts the source-recorded abundance category represented in the NOAA HABSOS data. It should not be interpreted as a validated future HAB-risk or early-warning model.**

No claims about future blooms, bloom onset, toxicity, human-health risk, causal environmental effects or operational early-warning performance follow from these scores. Target labels are copied from the verified Phase 3 files, not reconstructed or re-binned in Phase 4.

## Inputs and protection

Inputs are `data/ml/features.csv`, `target.csv`, `row_metadata.csv`, and `feature_metadata.json`. All 211,834 rows remain in these files. Only the 210,767 rows already marked eligible in Phase 3 enter supervised splits. The 1,067 ineligible labels are not filled. Existing NOAA source provenance and QA/eligibility policies remain as documented in Phases 1–3.

Before fitting, the script verifies Phase 3 CSV hashes against `reports/phase3_feature_report.json`, Phase 2 hash, aligned row identifiers, unique lineage keys, row counts, exact predictor columns, known target classes, target eligibility and source-date/year consistency. It fingerprints all files under data/raw, data/processed and data/ml, plus the Phase 3 feature-builder source, before and after. None are written by Phase 4. SOURCE_ROW_NUMBER is used only for joins/audit, never as a predictor.

Selected predictors, unchanged from Phase 3: **WATER_TEMP, SALINITY, SAMPLE_DEPTH, LATITUDE, LONGITUDE, observation_year, observation_month, day_of_year**. CATEGORY, CELLCOUNT, biological target-context fields, target/source quality flags, row IDs and site descriptions are not added to X.

## Temporal splits and class imbalance

Eligibility comes exclusively from the existing target file. Date boundaries use Phase 3 observation_year and are checked against original SAMPLE_DATE. There is no random train_test_split or cross-period row shuffling.

| Split | Dates | Eligible observations | not observed | very low | low | medium | high |
|---|---|---:|---:|---:|---:|---:|---:|
| Train | Through 2020 | 184,845 | 143,048 | 16,824 | 11,102 | 10,577 | 3,294 |
| Validation | 2021–2022 | 18,494 | 14,979 | 1,385 | 1,013 | 826 | 291 |
| Test | 2023–March 2024 | 7,428 | 5,909 | 701 | 431 | 297 | 90 |

All five classes occur in each main split. Exact observed minimum/maximum dates, years, class counts and hashes of split row IDs are saved in `reports/phase4_split_summary.json`. `reports/phase4_split_assignments.csv` lists every source-row assignment, including ineligible rows. Same-date observations cannot cross the fixed calendar boundaries.

The overall target imbalance is about 44.61:1 largest/smallest class. Accuracy alone is inadequate. No oversampling, undersampling, SMOTE, synthetic labels or altered validation/test class distributions are used. Balanced class-weight variants compute their weights internally from the labels passed to the current training fit, not from validation/test frequencies.

2024 is a partial year and contains only not observed and very low labels in the eligible test subset. Its fixed-five-class macro F1 breakdown assigns zero to absent classes and is explicitly labeled; it should not be treated as a full five-class annual performance estimate. The final report includes 2023 and 2024 descriptive breakdowns of the same test predictions.

## Preprocessing and missing values

Every estimator is inside a sklearn Pipeline containing `SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)`. Candidate imputers are fit on **training rows only**. Missingness indicators are created only for predictors with missingness in that fit. Logistic Regression additionally uses StandardScaler, fit after imputation on those same training rows. Tree models are not scaled. The majority baseline shares the imputer pipeline for a consistent input contract but ignores feature values.

The imputed medians are computational model inputs, not new scientific measurements. No imputed source/Phase 3 dataset is written. All original missing values and Phase 3 QA policies remain intact. Candidate imputer statistics are checked against independent training-only medians; the fitted final pipeline is audited against train+validation medians after selection. The final refit's larger fitting pool is explicitly authorized by this phase and still excludes test.

The input allowlist remains exactly eight source predictors. Consistent with the Phase 3 training-fold-only missing-indicator architecture, the imputer internally appends three indicators for missing temperature, salinity and depth, giving **11 internal transformed dimensions**. These are derived preprocessing signals, not new NOAA variables, IDs or target fields. Their names are recorded in model metadata and the coefficient table; no additional source predictor is automatically selected. This missingness representation can still reflect collection practices and is an interpretation limitation.

## Fixed candidate experiment and tuning policy

All recipes are declared in code before test predictions. Random seed is **42**. Numerical thread pools and Random Forest n_jobs are restricted to one for reproducibility.

| Candidate family | Fixed configuration | Controlled experiment |
|---|---|---|
| Majority baseline | DummyClassifier, most_frequent | Deterministic training-majority prediction |
| Logistic Regression | C=1, lbfgs, max_iter=1000; median/indicators/scaling | class_weight=None vs balanced |
| Random Forest | 80 trees, max_depth=18, min_samples_leaf=5, max_features=sqrt; median/indicators | class_weight=None vs balanced |
| HistGradientBoosting | 100 iterations, learning_rate=.1, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1; median/indicators | class_weight=None vs balanced |

Histogram boosting has early_stopping=False, so it does not create an internal random holdout. This is a small predeclared seven-recipe comparison (baseline plus six classifiers), not a massive hyperparameter search. Capacity settings are computational/model-design choices, not scientific HAB thresholds. No additional algorithm or post-test tuning is introduced.

Each recipe is fit on training data and evaluated on fixed 2021–2022 validation data. Training is repeated with an identical fresh clone and validation probabilities must agree to absolute tolerance **1e-12**. This checks actual training determinism without reevaluating the final test. Warnings, fit row hashes, imputer statistics, repeat differences, hyperparameters and timings are retained in model metadata/training logs. Times include repeated training and are not reproducibility targets.

## Model selection and final test protocol

Primary selection criterion is **highest validation macro F1**, with deterministic ties broken by validation macro recall and then model name. Balanced accuracy, weighted F1, per-class recall/F1 and confusion matrices are reported for context. The rare high class is inspected explicitly. No test metric participates in selection.

The selected recipe and all candidate settings are frozen to `reports/phase4_selection.json` with test_predictions_made=0 before final fitting/prediction. The selected pipeline is then fit from scratch on **203,339 train+validation observations**. A second identical refit is compared on validation inputs solely as a numerical reproducibility check; those in-sample predictions are not claimed as validation performance.

Only then is the selected refitted pipeline applied once to the 7,428 test observations. All final metrics, confusion matrices and year breakdowns reuse that one probability/prediction array. Other candidates are not evaluated on test. The complete test suite audits saved predictions and metrics rather than rerunning model evaluation on held-out data. Rerunning the full training script is an independent reproducibility execution with the same frozen experimental recipe, not permission to retune after inspecting test outcomes.

## Metrics and interpretability

Reports include accuracy, balanced accuracy, macro/weighted precision, recall and F1, per-class precision/recall/F1/support, confusion matrix and multiclass log loss. Undefined class precision/recall uses zero_division=0. Confusion-matrix order is **not observed, very low, low, medium, high**. Log-loss probabilities retain sklearn's model.classes_ column order, recorded explicitly. No binary ROC-AUC or binary HAB-risk score is computed.

The confusion-matrix figure shows both raw counts and row-normalized recalls, including the high class, with readable labels. Figures are saved as 300-dpi PNG plus vector SVG/PDF. Feature interpretation is **model-associated**, not causal: final-forest impurity importance if a forest wins; standardized coefficient summaries and class-specific coefficient table if logistic wins; otherwise validation permutation macro-F1 importance from the training-only selected pipeline before final refit. Any diagnostic permutations are transient model diagnostics, not new research observations. The exact method used is recorded in model metadata. Correlated spatial/calendar variables and missingness patterns can distort attribution.

## Observed results

Validation metrics, in the predeclared experiment order (all values below are fractions):

| Model | Accuracy | Balanced accuracy / macro recall | Macro precision | Macro F1 | Weighted F1 | High recall |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | .809938 | .200000 | .161988 | .178998 | .724887 | .000000 |
| Logistic, unweighted | .809884 | .199987 | .161986 | .178991 | .724860 | .000000 |
| Random Forest, unweighted | .809884 | .199987 | .161986 | .178991 | .724860 | .000000 |
| HistGradientBoosting, unweighted | .809235 | .199826 | .161961 | .178912 | .724539 | .000000 |
| **Logistic, balanced** | **.597545** | **.288093** | **.224465** | **.202731** | **.644857** | **.563574** |
| Random Forest, balanced | .788364 | .199606 | .203216 | .185304 | .717607 | .000000 |
| HistGradientBoosting, balanced | .763815 | .199872 | .199590 | .190108 | .710780 | .006873 |

**Balanced Logistic Regression was selected** because it had the highest validation macro F1. It sacrificed majority-class accuracy for more minority sensitivity on validation, including high-class recall .563574. Its high-class validation precision was only .041123 and F1 .076653, so even this apparent sensitivity came with many false positives. The baseline's higher validation accuracy is not evidence of useful minority-category classification.

Final refit uses 203,339 training+validation rows. Test results on 7,428 observations:

| Metric | Value |
|---|---:|
| Accuracy | .644184 |
| Balanced accuracy | .197915 |
| Macro precision | .196081 |
| Macro recall | .197915 |
| Macro F1 | .196975 |
| Weighted precision | .643692 |
| Weighted recall | .644184 |
| Weighted F1 | .643927 |
| Multiclass log loss | 1.230081 |

| Recorded class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| not observed | .786113 | .785581 | .785847 | 5,909 |
| very low | .194293 | .203994 | .199026 | 701 |
| low | .000000 | .000000 | .000000 | 431 |
| medium | .000000 | .000000 | .000000 | 297 |
| high | .000000 | .000000 | .000000 | 90 |

**The selected model failed to correctly classify any test low, medium or high observations. All 90 high-category test observations were missed.** This weak generalization must be reported rather than hidden behind the 64.42% aggregate accuracy. The model also made false high predictions on other categories; its confusion matrix preserves those errors. No model/configuration was changed in response to these test outcomes.

The descriptive 2023 subset accuracy is .603548, versus .888574 in partial-year 2024, whose recorded labels contain only not observed and very low. These are reaggregations of the same saved predictions, not additional model-selection passes. Differences do not establish why performance shifted. Missingness, sampling changes, source-label differences and temporal/spatial shift are limitations to investigate in a separately authorized study.

Interpretation for this selected Logistic Regression is the mean absolute standardized coefficient across its five class logits, with a separate signed class-by-feature coefficient table. Calendar day/month have relatively large associated coefficients, but they are correlated and do not establish a seasonal causal mechanism. The final pipeline is saved for reproducibility, **not for operational alerts**.

Every candidate and final refit repeated-fit probability comparison had maximum absolute difference **0.0**, within the 1e-12 tolerance. No fitting/convergence warnings were recorded. The held-out test evaluation occurred once after selection; software testing reuses its saved predictions.

## Outputs and reproduction

```powershell
python scripts/train_models.py
python -B -m unittest discover -s tests -v
```

The implementation uses the tested Python/scikit-learn environment recorded in `reports/model_metadata.json`; package pins are in `requirements_phase4.txt`. Matplotlib and missing plotting helpers were installed only under `.phase4_dependencies/`; the script adds that project-local directory to its import path and keeps its font cache there. Existing scientific libraries were not globally upgraded. For a fresh environment, install the pinned requirements in an isolated virtual environment before running the commands.

Primary outputs:

- `models/best_model.joblib`: only the selected final refitted pipeline; no intermediate model artifacts.
- `reports/model_metadata.json`: full recipe, software versions, audit, source hashes, validation/test results and limitations.
- `reports/model_comparison.csv`: all validation comparisons; test metric columns populated only for the selected recipe.
- `reports/validation_metrics.json`: every candidate's complete validation metrics.
- `reports/final_test_metrics.json`: selected final test results and year breakdowns.
- `reports/phase4_test_predictions.csv`: original source-row linkage, recorded/predicted categories and class probabilities.
- `reports/phase4_split_summary.json`, `phase4_split_assignments.csv`, `phase4_selection.json`: split/selection audit.
- `reports/feature_importance.csv` and `reports/figures/feature_importance.{png,svg,pdf}`.
- `reports/figures/confusion_matrix.{png,svg,pdf}`.
- `reports/phase4_training_log.txt`, `reports/phase4_test_results.txt`: execution/test evidence.

Model metrics/probabilities should repeat within 1e-12 on the documented software/thread configuration. Wall-clock times, serialized artifact bytes and figure metadata may differ. The suite verifies artifact loading, input integrity, feature/target contracts, disjoint temporal splits, training-only preprocessing, baseline calculation, actual repeated-fit audit, selection order, known predictions and figure/report existence, and includes all Phase 1–3 tests.

## Final verification

The training command completed and the complete suite passed: **33 tests** (1 Phase 1, 10 Phase 2, 10 Phase 3 and 12 Phase 4). Every candidate and final refit had a 0.0 maximum repeated-fit validation probability difference. The saved final pipeline loads successfully; split/source hashes, train-only preprocessing, predictor allowlist, selection freeze, saved test predictions, metrics and all figure formats passed their checks.

The final raw, Phase 2, Phase 3 dataset and Phase 3 builder-source hashes match their pre-training fingerprints. Existing earlier-phase tests rebuild their own outputs exactly as designed; content remains byte-identical. Plot title/layout and software-version metadata were clarified after training using saved metrics/artifacts only; model recipes, model parameters, split definitions and test predictions were not changed or reevaluated.

## Limitations and phase boundary

This remains an observational same-sample classification experiment. Collection/site/time shifts, strong imbalance, missing environmental measurements, retained source QA cautions/conversions, partial-year coverage and possible sampling-process proxies limit interpretation. Site overlap can remain despite chronological separation. No unseen-site, prospective, toxicity, health-risk, causal or deployment validation was performed. Results are not clinical/environmental safety decisions.

No Phase 5, edge/fog deployment, MQTT, ESP32, sensors, alerts, dashboard, hardware, Git initialization, commit or push is implemented.

## Implementation references

Implementation follows the official scikit-learn documentation for [Pipeline](https://scikit-learn.org/1.7/modules/generated/sklearn.pipeline.Pipeline.html), [HistGradientBoostingClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) and [permutation importance](https://scikit-learn.org/1.7/modules/generated/sklearn.inspection.permutation_importance.html). These are software references, not additional research datasets.
