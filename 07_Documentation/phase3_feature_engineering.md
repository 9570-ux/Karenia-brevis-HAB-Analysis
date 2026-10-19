# Phase 3 — feature engineering and source-reported abundance target

## Objective and scientific boundary

Prepare reproducible features and a defensible source-reported target, with lineage and explicit leakage controls. No model is trained and no HAB prediction performance is claimed.

**No directly validated binary HAB-risk target was established from the available NOAA fields.**

The strongest supported formulation is **contemporaneous multiclass classification of NOAA's recorded Karenia brevis abundance CATEGORY**. This is not future bloom onset, toxin exposure, human/ecological harm, or an early-warning endpoint. The project still needs a separately justified future-outcome definition before it can claim early-warning prediction. The target meaning here is confidently established by NOAA's dictionary as recorded cell-count bins; no bin boundaries or binary conversion are inferred.

## Inputs, provenance and protection

Input: `data/processed/habsos_cleaned.csv`, 211,834 records and 37 fields. Raw NOAA HABSOS accession 0120767, version 8.8, source citation and units are preserved in `docs/dataset.md`. The Phase 2 documentation, quality report, actual values/distributions, scripts, tests and archived dictionary were inspected before implementation.

NOAA dictionary: `docs/HABSOS_Data_Dictionary.xlsx`, Sheet1 A1:C32. CATEGORY is defined in A11:C11; CELLCOUNT and its unit/QA in A12:C14; salinity/temperature/winds and their QA appear in A15:C26; missing-value conventions are in A30:A32. [Official archived dictionary](https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/Support%20Documents/HABSOS_Data_Dictionary.xlsx).

Input Phase 2 SHA-256: `c7747c215658d3155f8227a5668ddf97635cf50b90c01eee99e5e0696696b2ec`.

The feature builder checks this against the Phase 2 report, verifies the raw download hash, and hashes all raw files plus the Phase 2 CSV before and after. It opens those inputs only for reading. No raw/Phase 2 observation, column, byte, file name or row order is changed by Phase 3. Original fields remain available in the unchanged Phase 2 input and in the Phase 3 row-metadata copy.

## Target definition and eligibility

Derived field: `observed_abundance_category`, copied verbatim from the original CATEGORY. Its five values are `not observed`, `very low`, `low`, `medium`, and `high`. All 211,834 source organisms are Karenia/brevis and all cell-count unit values are cells/L. No thresholds are derived from temperature, salinity, cell counts or a model.

A row receives a nonblank target only when:

1. CATEGORY is a documented nonblank category.
2. GENUS/SPECIES exactly identify Karenia/brevis.
3. CELLCOUNT_UNIT is cells/L and CELLCOUNT is a finite recorded number.
4. CELLCOUNT_QA is 1 (source good) or 5 (source converted upstream).

The last item is an explicit **project label-eligibility policy**, not a NOAA-mandated filter. QA=5 is not relabeled as good; it remains distinguishable in original metadata and should receive sensitivity analysis later. QA=9 is documented no-data and blank/unknown QA supplies no documented label-quality assertion. Those records retain their original category in metadata but receive blank derived target and a reason. Missing categories are not reconstructed from cell count. Any unrecognized category stops the builder for review rather than inventing its meaning.

All rows remain in all three CSV outputs. **210,767 labels are eligible; 1,067 are ineligible**: 856 missing CATEGORY and 211 unresolved/no-data count QA (192 blank QA, 19 QA=9). Eligibility is not determined from predictor completeness. QA=4 environmental observations are retained; they do not silently remove rows or labels.

| Class | Original CATEGORY count | Eligible target count | Percentage of eligible |
|---|---:|---:|---:|
| not observed | 164,102 | 163,936 | 77.780677% |
| very low | 18,925 | 18,910 | 8.971993% |
| low | 12,567 | 12,546 | 5.952545% |
| medium | 11,705 | 11,700 | 5.551154% |
| high | 3,679 | 3,675 | 1.743632% |

Original missing CATEGORY count: 856. Largest/smallest eligible-class ratio: **44.608435:1**. No balancing, class weights, oversampling, undersampling or SMOTE is performed. Source category correctness is not independently certified; labels remain source-reported bins across agencies and decades.

## Feature selection and categories

Candidate universe: **37 Phase 2 columns + 3 derived calendar columns = 40**. **8 selected, 32 excluded**. A complete per-candidate decision table is in `reports/phase3_feature_report.json` and `data/ml/feature_metadata.json`, including source, category, type, unit, original/derived status, missing percentage, target relation, leakage risk, selection and reason.

| Category | Fields and treatment |
|---|---|
| Environmental | WATER_TEMP, SALINITY, SAMPLE_DEPTH selected; WIND_SPEED/WIND_DIR excluded |
| Spatial | LATITUDE/LONGITUDE selected; STATE_ID metadata only |
| Temporal | observation_year, observation_month, day_of_year selected; source/Phase 2 date/time fields metadata only |
| Biological | GENUS/SPECIES retained as target context, excluded from predictors |
| Quality | All QA and compound validation fields metadata/eligibility only |
| Target related | CATEGORY, CELLCOUNT and its unit/QA excluded from predictors |
| Administrative | OBJECTID, SOURCE_ROW_NUMBER, duplicate linkage, DESCRIPTION and empty columns excluded |

Environmental units remain **temperature deg. C, salinity ppt, depth meters**. There are no unit conversions or new physical cutoffs. Blank depth remains blank rather than receiving an invented near-surface value. No extremes are clipped.

Wind speed is excluded from the initial candidate matrix because only 1,992 observations contain it. Wind direction has only 1,029 values; degrees alone do not specify reference axis or meteorological from/to convention. **No cyclic wind representation or u/v components are created.** These are feature-design exclusions with explicit coverage/convention reasons, not scientific validity thresholds. Original winds, units and QA remain in row metadata.

## Spatial and temporal engineering

Latitude and longitude have zero blanks and finite values: latitude **24–30.7149**, longitude **-97.66535–-78.2789 decimal degrees**. They match the archived bounds documented in Phase 1. The builder checks valid angular domains and stops for review on invalid coordinates. It preserves coordinate strings and creates no regions, grids, geographic bins, clusters or invented site identifiers. Coordinate datum remains unspecified by the inspected dictionary; identical coordinates do not prove identical sampling protocols/sites.

Calendar fields derive **only from original SAMPLE_DATE**, parsed as `%Y/%m/%d`:

| Feature | Formula | Unit/range |
|---|---|---|
| observation_year | date.year | Calendar year, observed 1953–2024 |
| observation_month | date.month | Calendar month, 1–12 |
| day_of_year | date.timetuple().tm_yday | Calendar day, 1–366; leap years handled by calendar parser |

No future values, lags, rolling windows or future aggregation are used. No season is assigned because a relevant marine seasonal regime is not documented. UTC source time exists for some rows, but 49,920 values are blank/documented placeholders; time-of-day features are excluded from this initial calendar-only design. Dates and times remain in original metadata for review. No timestamp is invented.

## Leakage analysis

The table below covers all Phase 2 fields and all derived predictors; grouped rows share the same decision. The JSON candidate table has one separate record per field.

| Column | Target related | Leakage risk | Allowed as feature | Reason |
|---|---|---|---|---|
| CATEGORY | Yes | Direct label | No | Target copied from this field |
| CELLCOUNT | Yes | Direct abundance proxy | No | CATEGORY describes cell-count bins |
| CELLCOUNT_UNIT, CELLCOUNT_QA | Yes | Target measurement/eligibility context | No | Label-source metadata |
| GENUS, SPECIES | Yes | Target organism context | No | Defines target scope; constant here |
| INVALID_FIELDS, SUSPICIOUS_FIELDS, QA_NO_DATA_FIELDS | Yes | Can reveal count-related flags | No | Compound fields may include CELLCOUNT |
| OBJECTID, SOURCE_ROW_NUMBER, EXACT_DUPLICATE_OF_SOURCE_ROW | No | Identifier/order proxy | No | Traceability only |
| WATER_TEMP, SALINITY, SAMPLE_DEPTH | No direct label content | Sampling/measurement bias still possible | Yes | Documented same-sample environmental context |
| LATITUDE, LONGITUDE | No direct label content | Site/sampling correlation remains | Yes | Measured coordinates; evaluate spatial generalization separately |
| observation_year, observation_month, day_of_year | No direct label content | Time/sampling shift remains | Yes | Same-observation calendar context; chronological evaluation required |
| SALINITY_QA, WATER_TEMP_QA, WIND_DIR_QA, WIND_SPEED_QA | No direct label content | Capture/quality process proxy | No | Quality handling and metadata only |
| SALINITY_UNIT, WATER_TEMP_UNIT, WIND_DIR_UNIT, WIND_SPEED_UNIT | No | Measurement metadata | No | Unit validation only |
| WIND_SPEED | No | Sparse source coverage | No | 99.06% missing; no manufactured coverage |
| WIND_DIR | No | Sparse coverage and undocumented convention | No | No assumed wind vectors/cyclic interpretation |
| STATE_ID | No | Geography/collection program proxy | No | Initial matrix uses coordinates |
| DESCRIPTION | No | High-cardinality site/sampling proxy | No | Site-name identity and case variants unresolved |
| RAW_UNNAMED_3, RAW_UNNAMED_28 | No | None | No | Entirely empty |
| SAMPLE_DATE, SAMPLE_DATE_ISO | No | Redundant time context | No | Only explicitly derived calendar features selected |
| SAMPLE_TIME, SAMPLE_TIME_UTC_PARSED, SAMPLE_DATETIME_UTC, TIME_STATUS | No | Partial sampling-clock/process context | No | Initial calendar-only design; times remain metadata |

`SOURCE_ROW_NUMBER` appears in features.csv **only as a join key**, not a ninth predictor. Use **only** `feature_metadata.json: predictor_columns` when constructing X. Never concatenate row_metadata.csv or target.csv into predictors. TARGET_STATUS is an eligibility field, not a feature. CATEGORY, CELLCOUNT, identifiers and label-derived flags must not enter X through a later automatic column selector.

Removing known direct leakage does not prove that all observational sampling bias or spatial/temporal dependence has disappeared. Measurements are contemporaneous; their real-time availability and ordering relative to laboratory counts are not established. This representation supports a bounded classification study, not an established operational forecasting system.

## QA and missingness

| Source condition | Feature view | Metadata |
|---|---|---|
| QA=1 | Retain recorded measurement | good |
| QA=4 | Retain; no clipping | caution |
| QA=5 | Retain converted source value | source_converted, distinct from good |
| QA=9 | Missing even when original is populated | Original number, QA and no_data status retained |
| Blank measurement | Missing | unavailable (or no_data if QA=9) |
| Unknown QA/unit or invalid representation | Missing, not guessed | Explicit unresolved/invalid status; original retained |
| Depth with recorded value and no QA column | Retain recorded depth | qa_not_available; no assertion of source quality |

The only populated selected measurement masked by QA=9 in this dataset is one WATER_TEMP cell. Original Phase 2 cell values remain unchanged. This is source-defined missingness, not deletion or imputation.

| Selected feature | Missing percentage after source QA handling |
|---|---:|
| WATER_TEMP | 49.619513% |
| SALINITY | 48.970420% |
| SAMPLE_DEPTH | 2.597789% |
| LATITUDE, LONGITUDE | 0% |
| Three calendar features | 0% |

No means, medians, scaler parameters or other fitted transformations are computed for feature replacement. **Training-fold-only architecture for later work:** first form the proposed chronological folds and eligible supervised subset; then fit any imputer, missingness encoder, scaler, categorical encoder or feature selector on each training fold only; apply its frozen transformation to validation/test. A missing-aware estimator is another future option. No choice is trained or instantiated here. Full-data missingness/class counts are descriptive reporting, not fitted preprocessing.

## Split design — recommendation only

Coverage is **1953-08-19–2024-03-25** and sampling counts vary substantially by year. The report records class/eligibility counts for every observed year. Recommend a chronological, calendar-block study of later-time **contemporaneous category classification**, rather than a random record split:

| Proposed role | Observation dates |
|---|---|
| Training | Through 2020-12-31 |
| Validation | 2021-01-01–2022-12-31 |
| Test | 2023-01-01–2024-03-25 |

Coverage check against the actual source years: these proposed periods contain **184,845**, **18,494** and **7,428** eligible labels respectively, and all five classes occur in each. Their high-category counts are 3,294, 291 and 90. Ineligible rows remain traceable (875 in the proposed training period and 192 in the test period). These are counts under proposed date rules, not physically generated splits.

These are transparent **proposed experimental cutoffs**, not NOAA/scientific thresholds and not results of model optimization. Freeze them before any model development. No final split CSVs are created. Keep all samples from the same date in the same split, use rolling-origin folds inside training for later tuning, check class support and year/site shifts, and report incomplete 2024 separately from complete 2023. Do not shuffle observations into random train/test subsets without a separate justification.

Chronological separation does not prevent site overlap. An unseen-site claim would require independently justified site identification/grouping; no geographic bins are invented here. No forecast horizon, future label, event-level independence or prospective early-warning claim follows from this split.

## Outputs and reproducibility

All CSVs retain **211,834 aligned rows** in source order:

- `data/ml/features.csv`: SOURCE_ROW_NUMBER plus the 8 documented predictors.
- `data/ml/target.csv`: SOURCE_ROW_NUMBER, observed_abundance_category, TARGET_STATUS. Blank targets are explicitly ineligible; they are not a sixth class.
- `data/ml/row_metadata.csv`: all 37 Phase 2 fields, target status and five environmental feature-quality statuses.
- `data/ml/feature_metadata.json`: predictor allowlist, join-key contract, all 40 selection/exclusion decisions, target and missing-value policy.
- `reports/phase3_feature_report.json`: input/output hashes, target/source distributions, candidate/leakage decisions, QA/spatial profiles and proposed splitting strategy.
- `reports/phase3_run_summary.json`: command output summary.
- `reports/phase3_test_results.txt`: complete combined test output.

From project root:

```powershell
python scripts/build_features.py
python -B -m unittest discover -s tests -v
```

The builder uses Python's standard library only. Paths resolve relative to the project, no random operation or execution timestamp is serialized, and repeated builds produce identical feature/target/metadata/report hashes. The tests compare two builds, trace every row to its unchanged Phase 2 values, validate real-date calendar derivations, verify QA=4 retention and QA=9 masking, check label reproduction and predictor exclusions, and run existing Phase 1/2 checks. No synthetic test observations are generated. The existing Phase 2 test suite rebuilds its own outputs as originally designed; their content hashes remain unchanged.

## Final verification

The requested builder command completed successfully, followed by the complete suite: **21 tests passed** (1 Phase 1, 10 Phase 2, 10 Phase 3). Two feature builds produced byte-identical features, targets, row metadata and JSON metadata/reports. All 211,834 source rows were traced through the outputs and original fields compared without generating test observations. Raw and Phase 2 CSV hashes remained unchanged. Full results are saved in `reports/phase3_test_results.txt`.

## Limitations and stop boundary

pH, turbidity, TDS, dissolved oxygen and chlorophyll-a remain unavailable. Environmental missingness, sparse winds, missing times, unknown coordinate datum, source conversion/quality flags, category imbalance, historical collection differences and sampling dependence remain. Source-reported CATEGORY is not independently validated binary risk or future ground truth. No model accuracy, HAB prediction capability or deployment readiness is asserted.

Phase 4, model training, learned preprocessing, balancing, MQTT, ESP32, hardware, dashboards and adaptive sampling are outside this implementation. No Git initialization, repository creation, commit or push is performed.
