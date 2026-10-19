# Karenia brevis HAB Analysis

*A reproducible machine-learning framework for source-recorded abundance analysis using NOAA HABSOS data.*

> **Research/software demonstration:** this project classifies contemporaneous recorded abundance categories. It is **not a validated future HAB forecasting or early-warning system**, toxicity detector or human-health risk assessment.

## 1. Project Overview

The project uses genuine NOAA Harmful Algal BloomS Observing System (HABSOS) observations to study and classify the source-recorded abundance category of *Karenia brevis*. It preserves dataset provenance, documents quality and missingness, compares classical classifiers using chronological splits, and exposes the selected saved model through a reusable inference API and Streamlit dashboard.

Phases 1–7 are represented in this organized sharing repository. Raw observations, processed files, model artifacts, locked experiment results and regression-test evidence are included. No synthetic research observations or future bloom labels are introduced.

## 2. Problem Statement

Historical coastal observations contain missing environmental measurements, unequal sampling effort and strongly imbalanced abundance categories. A classifier can achieve a high overall accuracy while failing to identify rare categories. The research problem is therefore to evaluate how well available environmental, spatial and calendar information distinguishes **recorded same-observation abundance categories**, with transparent temporal evaluation and explicit reporting of minority-class errors.

This study does not establish a future prediction horizon, toxin threshold or operational alert rule.

## 3. Objectives

- Preserve the official dataset, dictionary, original values and source-row lineage.
- Document data quality, missingness and a defensible source-recorded target.
- Prepare approved predictors without adding cell counts, labels or identifiers as model features.
- Compare classical models and class-weighted variants using validation-based selection.
- Report reproducible results, including rare-category failures.
- Provide saved-model inference and an informative software dashboard without modifying scientific datasets.

## 4. Dataset

**NOAA** is the U.S. National Oceanic and Atmospheric Administration. Its National Centers for Environmental Information (NCEI) archives the HABSOS physical and biological observations used here.

| Property | Verified project information |
|---|---|
| Source | NOAA NCEI, HABSOS accession **0120767**, version **8.8** |
| Original CSV | [habsos_20240430.csv](01_Dataset/Raw_Dataset/habsos_20240430.csv) |
| Original records | **211,834** |
| Recorded date coverage | **1953-08-19–2024-03-25** |
| Recorded states | AL, FL, MS, TX |
| Organism | *Karenia brevis* |
| Derived target | `observed_abundance_category`, copied from original `CATEGORY` under documented eligibility rules |
| Eligible supervised labels | **210,767**; 1,067 ineligible rows remain traceable |
| DOI | Not provided in the inspected official metadata |

The five unchanged abundance categories are:

- `not observed`
- `very low`
- `low`
- `medium`
- `high`

Labels describe source-recorded contemporaneous abundance bins. They are not reconstructed from invented thresholds or converted into future HAB-risk labels. Eligibility uses the existing organism, units, finite count and source count-quality policy; missing or unresolved labels are not fabricated.

See the [verified dataset documentation and citation](07_Documentation/dataset.md), [original dictionary](01_Dataset/Dataset_Documentation/HABSOS_Data_Dictionary.xlsx) and [target definition](07_Documentation/phase3_feature_engineering.md). pH, turbidity, TDS, dissolved oxygen and chlorophyll-a are **not available in this dataset**. Salinity is not a measured TDS value.

## 5. Methodology

```text
NOAA HABSOS Dataset
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
Chronological Train/Validation/Test Split
        ↓
ML Model Comparison
        ↓
Class-Balanced Training
        ↓
Validation-Based Model Selection
        ↓
Final Balanced Random Forest
        ↓
Inference
        ↓
Streamlit Dashboard
```

1. **Collection:** archive the official original CSV, supporting dictionary and verified source metadata.
2. **Preprocessing:** preserve source records and values while parsing dates and recording quality/missingness issues.
3. **Feature engineering:** prepare eight approved source predictors, recorded calendar components and traceable eligible labels; exclude target-derived fields.
4. **Temporal split:** use recorded years to prevent random mixing across the fixed evaluation periods.
5. **Model comparison:** compare Logistic Regression, Random Forest and HistGradientBoosting configurations.
6. **Class-balanced training:** evaluate class-weighted candidates within that comparison. Weights come from current fitting labels; no synthetic oversampling or altered test distribution is used. This is part of candidate fitting, not a separate post-test adjustment.
7. **Selection:** rank candidates on validation Macro F1, freeze the winner and refit on train+validation only.
8. **Inference and interface:** apply the saved fitted pipeline to validated individual inputs and display its unchanged outputs.

| Split | Recorded dates | Eligible observations |
|---|---|---:|
| Train | Through 2020-12-31 | 184,845 |
| Validation | 2021-01-01–2022-12-31 | 18,494 |
| Test | 2023-01-01–2024-03-25 | 7,428 |

Final refitting uses **203,339 train+validation rows**, excluding test. Details are preserved in the [Phase 4 methodology](07_Documentation/phase4_machine_learning.md) and [Phase 5 study](07_Documentation/phase5_model_improvement.md).

## 6. Features Used

The saved inference pipeline accepts the same **eight source predictors**:

| Group | Public input | Existing predictor | Unit / role |
|---|---|---|---|
| Environmental | temperature | `WATER_TEMP` | Degrees Celsius |
| Environmental | salinity | `SALINITY` | Parts per thousand (ppt) |
| Environmental | depth | `SAMPLE_DEPTH` | Meters |
| Spatial | latitude | `LATITUDE` | Decimal degrees |
| Spatial | longitude | `LONGITUDE` | Decimal degrees |
| Temporal | year | `observation_year` | Recorded calendar year |
| Temporal | month | `observation_month` | Recorded month |
| Temporal | day_of_year | `day_of_year` | Recorded calendar day |

The selected Phase 5 feature group actually implements these additional deterministic representations inside its saved pipeline:

| Derived feature | Definition | Unit |
|---|---|---|
| `month_sin`, `month_cos` | sin/cos of `2π(month − 1)/12` | Dimensionless |
| `day_of_year_sin`, `day_of_year_cos` | sin/cos of `2π(day_of_year − 1)/days_in_year` | Dimensionless |

`days_in_year` is 365 or 366 according to the recorded year's leap-year status. In this selected group, the classifier uses three environmental values, two coordinates and four cyclic features; absolute year/raw month/raw day are not direct classifier predictors. Three fitted missingness indicators give **12 internal model dimensions**. These are preprocessing representations, not additional NOAA measurements.

The [original transformer](02_Source_Code/Phase_5_Model_Improvement/phase5_transformers.py) contains the exact formulas. Source count, original category, quality flags and row IDs are not added to the eight-predictor input.

## 7. Machine Learning Algorithms

| Algorithm | What it does in this project | Balanced version |
|---|---|---|
| Logistic Regression | Learns multiclass linear decision relationships from imputed, standardized predictors. It supplies an interpretable classical comparison. | Uses `class_weight='balanced'` to give less frequent fitting labels greater influence. |
| Random Forest | Combines decision trees to represent nonlinear relationships among environmental, spatial and calendar inputs. | Uses balanced class weights within the fitted forest; this is the final selected family. |
| HistGradientBoosting | Sequentially builds histogram-based boosted trees to improve classification of patterns missed by earlier trees. | Uses balanced class weighting in the existing implementation. |

The **Majority Class baseline is a reference predictor, not one of the ML algorithms under comparison**: it always predicts the training-majority category. Its purpose is to expose how misleading accuracy can be under class imbalance.

All candidate preprocessing is fitted on training rows only. Pipelines contain a median imputer and missingness indicators; Logistic Regression additionally uses scaling. Phase 4 compared the majority reference and balanced/unweighted variants of the three families. Phase 5 used **15 fixed recipes** across four approved feature groups: balanced candidates across all groups, with unweighted controls for the cyclic group. It was not a full factorial weighting search.

## 8. Model Selection

The primary selection criterion is **validation Macro F1**, which weights the five category F1 scores equally. Phase 5 resolves exact ties using mean validation recall across low/medium/high, then experiment name.

**Final selected model: Balanced Random Forest.**

- Selected experiment: `D_environmental_temporal_spatial_cyclic__random_forest__balanced`.
- Configuration: 40 trees, maximum depth 14, minimum leaf size 8, balanced class weights, seed 42 and one job.
- Artifact: [phase5_best_model.joblib](03_Models/phase5_best_model.joblib).
- Metadata: [Phase 5 model metadata](03_Models/phase5_model_metadata.json).
- Locked Phase 4 model: [best_model.joblib](03_Models/best_model.joblib), a balanced Logistic Regression.

**The test set was not used for model selection or tuning.** Selection used validation data; the frozen winner was then refitted on train+validation and evaluated on test. However, Phase 5 reused the previously inspected Phase 4 test cohort. It is therefore **not a new independent holdout or prospective validation cohort**.

## 9. Experimental Results

The following values come from the existing locked metadata and [Phase 4/5 comparison](04_Reports/Phase_5/phase5_vs_phase4.csv). Macro F1 values are fractions; accuracy and recall values are percentages.

| Metric | Phase 4: balanced Logistic Regression | Phase 5: balanced Random Forest |
|---|---:|---:|
| Validation Macro F1 | 0.202731 | 0.252558 |
| Test Macro F1 | 0.196975 | 0.292562 |
| Test accuracy | 64.42% | 69.14% |
| Test balanced accuracy | 19.79% | 30.34% |

| Phase 5 test category | Recall |
|---|---:|
| Low | 23.43% |
| Medium | 18.86% |
| High | 3.33% |

**High-class performance remains weak: only 3 of 90 high observations were correctly classified; 87 were missed.** High precision was only **1.37%**, with 216 false high predictions. Improved aggregate metrics do not establish a reliable rare-high detector or operational warning capability. No significance claim is made from this reused test comparison.

Existing figures: [Phase 5 confusion matrix](04_Reports/Phase_5/figures/phase5_final_confusion_matrix.png) and [feature importance](04_Reports/Phase_5/figures/phase5_feature_importance.png). Importance is model-associated, not causal.

## 10. Prediction / Inference

The [Phase 6 API](02_Source_Code/Phase_6_Inference/inference.py) validates a single supplied observation and applies the unchanged saved pipeline:

```text
User Input
    ↓
Input Validation
    ↓
Saved Preprocessing Pipeline
    ├─ Feature Transformation: approved cyclic calendar representation
    └─ Fitted Imputation: saved medians and missingness indicators
    ↓
Balanced Random Forest
    ↓
Predicted Recorded Abundance Category
    ↓
Class Probabilities for all five categories
```

**Actual saved order:** cyclic transformation precedes imputation. No transformation or imputation statistics are refitted during prediction.

Required context is latitude, longitude, year, month and day_of_year. Temperature, salinity and depth are optional: None/omitted inputs remain missing until the saved fitted imputer handles them. Numeric inputs must be finite; coordinates and integer calendar fields must be valid and month/day_of_year must agree for the recorded year. The API rejects invalid inputs before prediction.

Returned values include predicted category, correctly mapped class probabilities, input validation status, missing environmental fields and model/version metadata. Probability mapping follows the saved model's class order rather than assuming a display order. Inputs and predictions are not appended to the scientific dataset.

> The prediction represents the source-recorded abundance category and is not a validated future HAB-risk or early-warning prediction.

The [CLI](02_Source_Code/Phase_6_Inference/run_inference.py) exposes the same behavior. See [inference documentation](07_Documentation/phase6_inference.md) for its schema and limitations.

## 11. Streamlit Dashboard

The [dashboard](06_Dashboard/app.py) uses the existing inference API, with no duplicated model training or manual preprocessing.

| Page | Contents |
|---|---|
| Overview | Dataset scope, selected model, phases, categories and visible limitations |
| Prediction | Eight supported inputs, genuine-row example, validation, actual prediction and all class probabilities |
| Historical Data | Recorded category/year/month counts, environmental distributions, missingness and historical coordinate display |
| Model Performance | Locked Phase 5 metrics, Phase 4/5 comparison and unchanged figures |
| Dataset Information | Official provenance, predictor units, coverage, missingness and unavailable variables |
| About / Limitations | Implemented phases, scientific interpretation, known weaknesses and unimplemented hardware scope |

Historical charts do not impute missing measurements. The coordinate display uses an explicitly documented source-order subset of at most 4,000 actual points, not real-time HAB locations. User inputs remain session-only. Details: [Phase 7 dashboard documentation](07_Documentation/phase7_dashboard.md).

## 12. System Architecture

```text
NOAA HABSOS
     ↓
Raw Data
     ↓
Preprocessing
     ↓
Feature Engineering
     ↓
Temporal Split
     ↓
Model Comparison
     ↓
Validation-Based Selection
     ↓
Balanced Random Forest
     ↓
Saved Model
     ↓
Inference
     ↓
Streamlit Dashboard
```

The implemented architecture is a reproducible software workflow. No live edge/fog network, sensor feed, ESP32, MQTT or hardware alert subsystem is implemented.

## 13. Reproducibility

### Restore the runnable layout

This repository is an **organized archive**, not the original runnable folder layout. Code resolves `data/`, `scripts/`, `models/`, `reports/`, `docs/` and `tests/` relative to a common root. Running `streamlit run app.py` inside `06_Dashboard/` alone will not supply those paths.

The existing [inventory](PROJECT_INVENTORY.md) maps every copy to its original relative path and SHA-256. From this repository root, the commands below restore unchanged files into a separate `Reproduction_Workspace`, verify their hashes, install the original pinned dependencies, and show the existing inference/test/dashboard commands. They are **instructions only**; no restoration, installation, test run or retraining was performed for this README edit.

```powershell
@'
from pathlib import Path
import hashlib, shutil
package = Path.cwd()
work = package / "Reproduction_Workspace"
work.mkdir(exist_ok=True)
for line in (package / "PROJECT_INVENTORY.md").read_text(encoding="utf-8").splitlines():
    if not line.startswith("| `"):
        continue
    cells = [c.strip().strip("`") for c in line.split("|")[1:-1]]
    filename, folder, phase, purpose, original, relative, expected = cells
    source = package / folder / filename
    target = work / relative
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source) == expected, str(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert digest(target) == expected, str(target)
    else:
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
print("Restored original layout:", work)
'@ | python -B
Set-Location Reproduction_Workspace
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_phase4.txt -r requirements_phase7.txt
python scripts/run_inference.py --help
python scripts/run_inference.py --temperature 26.0 --salinity 36.0 --depth 0.5 --latitude 29.1058 --longitude -79.5015 --year 1980 --month 12 --day-of-year 346
python -B -m unittest tests.test_dashboard -v
python -B -m unittest discover -s tests -v
streamlit run app.py
```

Use the documented Python 3.10 scientific environment with scikit-learn 1.7.2; dependency pins are in [requirements_phase4.txt](02_Source_Code/Phase_4_Baseline_ML/requirements_phase4.txt) and [requirements_phase7.txt](06_Dashboard/requirements_phase7.txt).

The CLI example is genuine source row 18, recorded 1980-12-11. Its recorded category is `not observed` while the model predicts `very low`; it demonstrates inference, not a correct classification. The dashboard requires no retraining.

### Training scripts: optional research reproduction

The original entry points are [train_models.py](02_Source_Code/Phase_4_Baseline_ML/train_models.py) and [phase5_model_improvement.py](02_Source_Code/Phase_5_Model_Improvement/phase5_model_improvement.py). They write experiment outputs and enforce frozen manifests. **Do not run them in this archive or over existing locked results.** The Phase 5 script explicitly requires the balanced Logistic Regression baseline, and can stop if a newly retrained Phase 4 baseline differs. Training commands are therefore not presented as a safe one-command reproduction over this locked package. Consult the preserved [Phase 4](07_Documentation/phase4_machine_learning.md) and [Phase 5](07_Documentation/phase5_model_improvement.md) protocols before planning a separate experiment.

The earlier-phase regression tests can rebuild their output copies byte-identically in the reproduction workspace. Preserved provenance JSON may contain original absolute paths; those are historical records, not a reason to edit scientific artifacts.

## 14. Testing

Verified existing log counts are shown with their scope to avoid double-counting:

| Phase / evidence | Verified tests | Scope |
|---|---:|---|
| [Phase 4 log](05_Test_Results/phase4_test_results.txt) | 33 | Cumulative Phase 1–4 suite |
| [Phase 5 log](05_Test_Results/phase5_test_results.txt) | 45 | Cumulative Phase 1–5 suite |
| [Phase 6 log](05_Test_Results/phase6_dedicated_test_results.txt) | 15 | Dedicated inference tests; cumulative total became 60 |
| [Phase 7 log](05_Test_Results/phase7_dashboard_test_results.txt) | 16 | Dedicated dashboard tests |
| [Complete project log](05_Test_Results/phase7_test_results.txt) | **76** | Complete Phase 1–7 suite: 45 + 15 + 16 |

The [test sources](02_Source_Code/Tests/) verify raw/data/model hashes, source-row lineage, target eligibility, temporal splits, fitted preprocessing, saved results, class-probability mapping, missing/invalid input handling, repeatability and dashboard rendering/interactions. These are preserved passing logs, not tests newly rerun during this documentation update.

## 15. Project Structure

```text
Karenia brevis HAB Analysis - Project/
├── 01_Dataset/
│   ├── Raw_Dataset/habsos_20240430.csv
│   ├── Processed_Dataset/
│   │   ├── habsos_cleaned.csv
│   │   └── ML_Ready/                 # features, targets and lineage
│   └── Dataset_Documentation/
├── 02_Source_Code/
│   ├── Phase_2_Preprocessing/
│   ├── Phase_3_Feature_Engineering/
│   ├── Phase_4_Baseline_ML/
│   ├── Phase_5_Model_Improvement/
│   ├── Phase_6_Inference/
│   ├── Phase_7_Dashboard/
│   └── Tests/
├── 03_Models/
│   ├── best_model.joblib
│   └── phase5_best_model.joblib
├── 04_Reports/                       # Phase_2 through Phase_7; figures
├── 05_Test_Results/
├── 06_Dashboard/
│   ├── app.py
│   ├── dashboard.py
│   └── requirements_phase7.txt
├── 07_Documentation/
├── 08_Paper/                         # empty locally; no paper source
├── 09_Presentation/                  # empty locally; no presentation
├── .gitignore
├── README.md
├── PROJECT_INVENTORY.md
└── GOOGLE_DRIVE_UPLOAD_GUIDE.md
```

The pre-existing local `New folder` remains untouched. Git does not represent empty directories, so the empty paper/presentation directories may not appear after cloning. Several copies intentionally appear in multiple archive sections; the [inventory](PROJECT_INVENTORY.md) identifies their original paths. [Manual Drive-upload guidance](GOOGLE_DRIVE_UPLOAD_GUIDE.md) is also preserved.

## 16. Limitations

- **Severe class imbalance:** eligible labels have an approximately 44.61:1 largest/smallest class ratio; accuracy alone is inadequate.
- **Weak rare-class performance:** low and medium recall remain limited and high recall is only 3.33%.
- **Observational dataset:** historical collection practices, missing measurements, source conversions/cautions and uneven effort affect interpretation.
- **Recorded-abundance target:** labels are source-reported contemporaneous categories, not independently validated toxicity, harm or future bloom outcomes.
- **No validated future forecasting:** no future horizon, prospective bloom-onset target or unseen-site generalization has been established.
- **No validated operational early-warning capability:** the model is unsuitable for dependable HAB alerts or health-risk decisions.
- **Evaluation dependence:** Phase 5 reused the Phase 4 test cohort; chronological splitting does not guarantee spatial independence.
- **Coverage and interpretation:** 2024 is partial, coordinate datum is unspecified in the inspected dictionary, probabilities are not calibrated confidence, and feature importance is not causal.
- **Unavailable measurements:** pH, turbidity, TDS, dissolved oxygen and chlorophyll-a are absent. No fake replacement values or silent cross-source combinations are used.

## 17. Future Work

The following are possibilities, **not implemented results**:

- Obtain independent prospective evaluation data and investigate temporal/spatial shift.
- Study rare-category errors and assess uncertainty/calibration under an appropriate validation design.
- Evaluate documented quality-policy sensitivity and defensible sampling/site dependence.
- Consider additional official variables only after explicit source selection, alignment and scientific justification.
- Define and validate a genuine future-outcome target before making any forecasting or early-warning claim.
- Consider separately authorized hardware/edge/fog demonstrations; Phase 8 remains unimplemented.

## 18. Research References

Only sources already cited in the preserved project documentation are listed here.

1. NOAA National Centers for Environmental Information (2014). *Physical and biological data collected along the Texas, Mississippi, Alabama, and Florida coasts in the Gulf of America as part of the Harmful Algal BloomS Observing System from 1953-08-19 to 2024-03-25 (NCEI Accession 0120767).* Full original CSV `habsos_20240430.csv`, version 8.8. NOAA National Centers for Environmental Information. Dataset. https://www.ncei.noaa.gov/archive/accession/0120767. Accessed 2026-10-14.
2. **NOAA HABSOS supporting dictionary**, archived with accession 0120767 version 8.8: [official dictionary](https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/Support%20Documents/HABSOS_Data_Dictionary.xlsx) and [unchanged local copy](01_Dataset/Dataset_Documentation/HABSOS_Data_Dictionary.xlsx).
3. **scikit-learn implementation references** from the [Phase 4 documentation](07_Documentation/phase4_machine_learning.md): [Pipeline](https://scikit-learn.org/1.7/modules/generated/sklearn.pipeline.Pipeline.html), [HistGradientBoostingClassifier](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html), and [permutation importance](https://scikit-learn.org/1.7/modules/generated/sklearn.inspection.permutation_importance.html).
4. **Streamlit implementation references** from the [Phase 7 documentation](07_Documentation/phase7_dashboard.md): [AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) and [number_input](https://docs.streamlit.io/develop/api-reference/widgets/st.number_input).

Software documentation is an implementation reference, not additional scientific data or evidence of HAB forecasting capability. No journal paper, author, DOI or research finding has been invented.

## 19. Authors

Research author names and affiliations are **not specified in the existing project documentation**. They are therefore not inferred from a local account name, Git identity or repository ownership. Contributor-supplied authorship details are needed before listing authors.
