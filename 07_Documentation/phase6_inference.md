# Phase 6 — single-observation inference

## Objective and model

Return a **Predicted recorded abundance category** for one recorded observation using the unchanged `models/phase5_best_model.joblib`. This is contemporaneous source-category classification, **not a validated future HAB early-warning system**, confirmed HAB, toxic bloom or human-health risk assessment. No model fitting, tuning or new research observations occur in Phase 6.

The Phase 5 winner is `D_environmental_temporal_spatial_cyclic__random_forest__balanced`: 40 trees, depth 14, minimum leaf 8, balanced class weights, seed 42, one job. Artifact SHA-256: `a4809c0f2549b51344749393a0ec5bf2e2db6e10b5b42f866bde8be270fcc734`.

## Input schema

The Python API accepts one mapping, with these public names only:

| Public field | Saved source predictor | Unit / validation | Missing policy |
|---|---|---|---|
| temperature | WATER_TEMP | finite number, degrees C | None or omitted allowed |
| salinity | SALINITY | finite number, ppt | None or omitted allowed |
| depth | SAMPLE_DEPTH | finite number, meters | None or omitted allowed |
| latitude | LATITUDE | decimal degrees, -90..90 | required |
| longitude | LONGITUDE | decimal degrees, -180..180 | required |
| year | observation_year | integer, 1..9999 Python calendar domain | required |
| month | observation_month | integer, 1..12 | required |
| day_of_year | day_of_year | integer, 1..365/366 depending on leap year | required |

Calendar components must describe the same recorded date: month must match year/day_of_year. Unknown keys, missing context, strings, booleans, NaN, infinity, fractional calendar fields and invalid coordinates are rejected with ValueError. CLI options parse numeric text before applying these same rules. Calendar bounds are representational validity rules, not scientific forecast horizons. No invented environmental cutoffs, clipping or unit conversions are applied. Validation PASS means schema validity; it does not certify measurement quality or scientific applicability. Values outside the historical study domain are not validated for generalization.

Missing environmental values are represented as NaN in the transient model input only. API callers use None or omit a field; CLI callers omit the corresponding option. The unchanged saved pipeline performs deterministic cyclic calendar engineering, then its **already fitted median imputer with three missingness indicators**, then Random Forest prediction. The classifier uses nine engineered columns plus three indicators. Absolute year determines leap-year calendar arithmetic and is not directly a classifier predictor. No scaler is present in the selected forest. No imputation statistics are fitted or recomputed during inference. Computational replacements are not measurements and are never written to datasets.

The eight-field API has no QA-code inputs. Callers must respect the established source units and Phase 3 QA policy: source no-data/unresolved scientific measurements must be supplied as missing, not as valid readings. Inference cannot independently certify their quality.

## Reusable API and output

From project root, `from scripts.inference import predict_observation` imports the API without running training scripts. Pass a dictionary using the public names above. The custom serialized `phase5_transformers` module remains available under scripts; inference resolves it from the project path even if launched from another directory. Scientific dependencies use existing Phase 4/5 versions; no new dependencies are installed.

The result is JSON-serializable: terminology, predicted_class, class_probabilities, input_validation (status, missing fields and policy), model (relative artifact, SHA-256, identity, inference version, training/runtime versions), scientific_limitation. No timestamp or random field affects repeatability. The local trusted artifact hash is verified when first loaded in a process; the loaded pipeline is then cached. Restart the process after intentional artifact replacement; altered artifact bytes are rejected against Phase 5 metadata on loading. A matching scikit-learn version is required. Only trusted local project artifacts should be loaded with joblib.

Output order is `not observed`, `very low`, `low`, `medium`, `high`. The artifact's actual probability order is `high`, `low`, `medium`, `not observed`, `very low`. Inference maps each probability using **model.classes_**, then renders the display order. Values must be finite, between zero and one, and sum to one within 1e-12. Probabilities are model scores, not validated toxicity probabilities or calibrated health-risk estimates.

## Genuine-row demonstration

The command below copies the complete eight-value combination from `data/ml/features.csv`, SOURCE_ROW_NUMBER **18** (recorded 1980-12-11). Its actual source category is `not observed`, while the model predicts `very low`; the two leading probabilities are very close. This demonstrates inference behavior, not a correct classification or independent performance claim.

```powershell
python scripts/run_inference.py --help
python scripts/run_inference.py --temperature 26.0 --salinity 36.0 --depth 0.5 --latitude 29.1058 --longitude -79.5015 --year 1980 --month 12 --day-of-year 346
```

Actual saved-model output, rounded for display:

```text
Predicted recorded abundance category: very low
Class probabilities:
not observed: 0.322065
very low: 0.322502
low: 0.129463
medium: 0.190006
high: 0.035964
Input validation: PASS
```

Add `--json` to print full precision probabilities and all metadata. Omitting environmental options applies only the saved imputer; this does not create a new research row. The CLI/API write no files.

## Integrity, compatibility and reproduction

`reports/phase6_integrity_baseline.json` captures the 74 pre-existing scientific/source/documentation/test files before implementation. One necessary compatibility change is explicitly recorded with before/after hashes: `scripts/phase5_model_improvement.py:snapshot` now hashes the existing frozen manifest's paths when that manifest exists. Its former folder scan incorrectly classified any new later-phase file as a change to Phase 1–4. All 52 original lock entries are still hashed and checked; missing or modified original files still fail. First-run behavior when no lock exists is retained. No model, dataset, metadata, selection, metric, prediction or figure is changed. New `tests/__init__.py` enables the requested dotted unittest command. Existing test assertions and results are unchanged.

Run from the project root with the Phase 4/5 Python environment:

```powershell
python -B -m unittest tests.test_inference -v
python -B -m unittest discover -s tests -v
```

Tests use actual complete and incomplete source rows; invalid-input probes are transient validation checks and never dataset observations. They check schemas, calendar consistency, output mapping against the saved pipeline, missing handling without fitting, exact repeated results, dataset inventories, hashes and absence of unauthorized new phase files. Dedicated and complete logs are under `reports/phase6_*test_results.txt`; verified counts and real-row full output appear in `reports/phase6_inference_report.json`.

The existing Phase 2/3 regression tests rebuild their own outputs as previously implemented; these outputs must remain byte-identical. Phase 6 inference itself does not rebuild anything. Integrity verification compares all prior content hashes, with only the documented inventory compatibility source change permitted.

Final verification: **15 dedicated tests passed; 60 complete project tests passed**. Separate CLI processes produced identical full JSON results. All 52 original Phase 5 lock entries match; 73 of the 74 pre-existing files remain byte-identical, with only the documented Phase 5 inventory source change. Raw data, all derived datasets, both saved models and all prior results remain unchanged.

## Limitations and boundary

Phase 5 missed 87 of 90 recorded high-category test observations and had only 1.37% high precision. Its previously reused test cohort is not new independent validation. Sampling bias, historical/environmental missingness, spatial/calendar dependence, source-reported labels and unsupported geographic or temporal extrapolation persist. pH, turbidity, TDS, oxygen and chlorophyll-a remain unavailable; no additional scientific predictors or sources are introduced.

Work stops at Phase 6. Phase 7 dashboard/UI and Phase 8 hardware, ESP32, MQTT, sensors and alerts are not implemented. No Git operations or retraining are performed.
