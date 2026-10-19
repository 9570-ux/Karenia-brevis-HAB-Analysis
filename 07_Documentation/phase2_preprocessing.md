# Phase 2 — NOAA HABSOS data cleaning and preprocessing

## Objective and scope

Create a deterministic, traceable processed copy of the genuine NOAA HABSOS archive without inventing observations, measurements, timestamps, thresholds, or labels. Phase 2 preserves source evidence and exposes documented data-quality problems. It does not certify scientific validity or implement training, scaling, real-time prediction, communications, dashboards, or hardware.

The existing Phase 1 documentation, original CSV, inspection report, download manifest, scripts, tests, and NOAA dictionary were inspected before writing this pipeline. No additional dataset is used.

## Reproduce

From the project root:

```powershell
python scripts/preprocess_dataset.py
```

Only Python's standard library is required. The script uses paths relative to its own project directory, not the terminal's current directory. It rebuilds processed CSV and reports directly from raw input; it never reads a previous processed dataset. Repeating it produces identical data and report bytes. There are no random operations, execution timestamps, or unordered sets serialized into outputs.

Run existing Phase 1 and new Phase 2 tests together:

```powershell
python -B -m unittest discover -s tests -v
```

The saved verification log is `reports/phase2_test_results.txt`. The Phase 2 suite executes the full pipeline twice, compares all three deterministic outputs, checks every original record/cell against the processed copy, and intercepts attempts to open raw files for writing. Tests use the actual NOAA observations, not synthetic fixtures.

## Input and raw protection

Input: `data/raw/habsos_20240430.csv`, NOAA NCEI HABSOS accession 0120767, version 8.8. Official citation and source coverage remain in `docs/dataset.md`.

Raw SHA-256: `0c561afc3239fc885afd161bbc0c3e9826969019f8266ffaade4adca7f700596`.

The script verifies the Phase 1 hash, file size and header before processing. Every file under `data/raw/`, including its placeholder, is hashed before and after. Raw files are opened only for reading. No raw file is renamed, rewritten, truncated, or deleted. Existing Phase 1 files are preserved apart from adding a documentation cross-reference to this completed phase.

## Lineage and outputs

```text
NOAA version 8.8 original CSV (read only)
  -> hash/schema validation and profiling
  -> documented parsing and quality rules
  -> data/processed/habsos_cleaned.csv
  -> reports/phase2_data_quality_report.json
     reports/phase2_record_issues.csv
```

Raw: **211,834 rows, 28 columns**. Processed: **211,834 rows, 37 columns**. No observations or source columns are removed. All original cell strings, including blank scientific values and source QA codes, remain equal and in source order. CSV quoting and line endings may differ in the processed file; the raw byte stream remains unchanged.

The only source-header changes in the processed copy are naming the two empty headers `RAW_UNNAMED_3` and `RAW_UNNAMED_28`. Both empty columns are retained. The raw headers remain empty. The report maps every processed source column back to its raw name and position.

Nine derived columns are appended:

| Column | Meaning |
|---|---|
| SOURCE_ROW_NUMBER | One-based CSV data-record position, excluding header; stable linkage even without OBJECTID |
| SAMPLE_DATE_ISO | Successfully parsed source date, YYYY-MM-DD; otherwise blank |
| SAMPLE_TIME_UTC_PARSED | Successfully parsed non-placeholder time, HH:MM; otherwise blank |
| SAMPLE_DATETIME_UTC | Combined known date and time, ISO 8601 with minute precision and Z; otherwise blank |
| TIME_STATUS | valid, missing, metadata_missing_placeholder, or invalid |
| INVALID_FIELDS | Semicolon-separated fields with invalid numeric/date/time representations |
| SUSPICIOUS_FIELDS | Fields flagged by source QA=4, QA=9 with a populated measurement, undocumented QA codes, or unexpected units |
| QA_NO_DATA_FIELDS | Measurement fields with source QA=9, regardless of whether a number is present |
| EXACT_DUPLICATE_OF_SOURCE_ROW | Earlier identical record's source-row number; blank if none |

These are lineage/validation fields, not additional research observations or future-risk labels.

## Profiling results

Dates span **1953-08-19 through 2024-03-25**. All 211,834 dates parse using the documented YYYY/MM/DD format. Time format is H:MM or HH:MM, 24-hour UTC per NOAA's dictionary. There are **161,914 usable recorded times**, **48,920 blank times**, and **1,000 documented missing-time placeholders**. Combined timestamps are absent for **49,920 records**. The earliest full timestamp is **1954-02-05T08:55Z** and the latest is **2024-03-25T14:40Z**; earliest date-only observations are retained without inventing times.

The machine-readable report provides every source column's type, total, blank count/percentage, documented missingness, exact distinct nonblank count, small categorical frequency tables, numeric extrema/median/zeros, parsing problems, whitespace count, and case/whitespace variants. Large categorical vocabularies, such as location descriptions, receive exact counts and variant groups rather than an enormous frequency dump.

## Duplicate handling

**0 exact duplicate extra records detected; 0 removed.** Exact means equality of all 28 parsed original CSV fields, including OBJECTID and unnamed columns, without trimming or case normalization. Any duplicates found on a future identical-schema input would be listed with complete original field values and both source-row positions and retained. Equal coordinates, organism, or sample dates alone do not justify deleting an observation. This phase does not assert that broader near-duplicate or repeated-sampling events are accidental duplicates.

## Missing-value strategy

All scientific missing values remain missing. No mean/median replacement, forward/backward fill, interpolation, random fill, numeric substitution for missing depth, or placeholder midnight is used. Original populated values marked QA=9 remain present as evidence, with an explicit no-data flag; downstream analysis must honor that flag.

| Variable | Blank cells | Blank percentage | Documented missing including QA=9 | Action and reason |
|---|---:|---:|---:|---|
| CELLCOUNT | 0 | 0% | 19 | Preserve original; flag source no-data QA |
| SALINITY | 103,736 | 48.970420% | 103,736 | Leave missing; no measured replacement |
| WATER_TEMP | 105,110 | 49.619041% | 105,111 | Leave missing; one populated value has QA=9 |
| SAMPLE_DEPTH | 5,503 | 2.597789% | 5,503 | Leave missing; near-surface is not an exact depth |
| WIND_DIR | 210,805 | 99.514242% | 210,805 | Leave missing; no measured replacement |
| WIND_SPEED | 209,842 | 99.059641% | 209,842 | Leave missing; no measured replacement |
| SAMPLE_TIME | 48,920 | approximately 23.09% | 49,920 | Preserve strings; do not parse placeholders as real times |

Every column, including categories, units, identifiers and empty columns, has its own missing count/percentage/action/reason in the JSON report. QA=9 semantic missingness is separately distinguished from literal empty cells, not double-counted. Null environmental values mean data not collected according to NOAA's dictionary. True recorded zeros are retained.

## Invalid and suspicious values

**0 unparseable/nonfinite numeric or date/time cells detected; 0 removed.** This is a representation-validation result, not a claim of zero scientific measurement errors. Numeric parsing requires finite numbers; QA codes, count values and IDs must have integer representations. Calendar/time parsing follows the documented formats. No arbitrary temperature, salinity, depth, wind or cell-count limit is introduced.

NOAA QA meanings are 1=good, 4=exceeds normals, 5=converted upstream by script, 9=no data. The report finds **667 QA=4 measurement cells across 666 records**, and **20 populated measurement cells with QA=9 across 20 records**. All are retained and flagged. Rules can overlap, so affected-record counts must not be summed as unique records without deduplication.

Units checked against the dictionary and Phase 1's verified source labels: cells/L, ppt, deg. C, deg, and mph. No unexpected/missing unit label for a populated measurement was detected. No unit conversions were performed. The dictionary's SALINTY spelling errors are documented in Phase 1; the CSV's actual SALINITY columns are used.

`reports/phase2_record_issues.csv` supplies each affected source-row number, column, rule, classification, and original value. JSON quality-rule entries give counts, source/reason and a link to this complete record-level listing. Missing placeholders are classified as missing rather than invalid timestamps. Source QA=4 is classified as suspicious rather than definitive invalidity.

## Categorical and numerical handling

HAB categories are retained exactly: high, low, medium, not observed, very low; **856 records have no category**. Genus/species remain Karenia/brevis. No category thresholds, assignments, merges or spelling corrections are invented. Location descriptions contain case variants, for example `R. G. Kreusler Park` and `R. G. Kreusler park`. These are reported without assuming that text similarity proves site identity. No leading/trailing whitespace was detected. Raw strings remain unchanged.

For each numerical environmental variable, the report includes minimum, maximum, median, zeros, missingness and suspicious counts. These summaries describe all finite recorded numbers, including QA-flagged entries; they are not statistics of a certified valid-only subset. No clipping, winsorization, normalization or scaling occurs. Example medians are salinity 33 ppt and temperature 25.4 °C. Measurement validity and future training/evaluation splits must be decided before ML scaling.

## Scientific rules and sources

Primary local source: `docs/HABSOS_Data_Dictionary.xlsx`, Sheet1 A1:C32, downloaded unchanged in Phase 1. Online source: https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/Support%20Documents/HABSOS_Data_Dictionary.xlsx. The missing-time convention is specifically in A32.

| Rule | Affected fields/records | Reason |
|---|---|---|
| Preserve original cells, zeros, missingness and all rows | All 211,834 records | Research provenance and explicit project requirements |
| Report source QA=4 without removing measurements | CELLCOUNT/SALINITY/WATER_TEMP/WIND fields as flagged; 667 cells, 666 records | NOAA dictionary: exceeds normals |
| Report QA=9 separately from blanks | Five measurement QA fields; full counts in JSON | NOAA dictionary: no data |
| Flag populated measurements with QA=9 | 19 CELLCOUNT cells, 1 WATER_TEMP cell | Recorded number does not override NOAA's no-data flag |
| Do not parse noon/midnight placeholders as observations | SAMPLE_TIME; 1,000 records | NOAA dictionary: samples submitted without a time |
| Keep blank depth blank | SAMPLE_DEPTH; 5,503 records | Dictionary's near-surface assumption supplies no numeric depth |
| Preserve categories and location strings | All source records | No documented authority for category reassignment or site merging |
| Report only representational invalidity without scientific limits | Numeric/date/time fields; 0 invalid cells | Source format definitions; no unsupported physical cutoffs |

No ambiguous scientific correction was applied. Unresolved validity, site-identity and repeated-observation questions are preserved for review rather than guessed.

## Limitations and stop boundary

pH, turbidity, TDS, dissolved oxygen and chlorophyll-a: **Not available in this dataset.** Salinity is not substituted for measured TDS. Only Karenia brevis is represented. Environmental missingness, missing/placeholder times, sparse winds, QA issues, irregular sampling and location case variants remain. Coordinate datum and numeric definitions of QA thresholds are not supplied in the inspected dictionary. Blank CELLCOUNT_QA entries remain unresolved. CATEGORY is contemporaneous, not a future outcome. The dataset is not claimed ready for ML or deployment.

Verification: **11 tests passed** (1 existing Phase 1 test and 10 Phase 2 tests). The suite verified two complete pipeline runs and byte-identical processed CSV, JSON report and record-issue CSV, including deterministic missing-value handling. A final fresh rebuild was also executed using the documented preprocessing command.

No existing file was deleted, no Git commit/push was made, and no Phase 3 work was implemented. The processed file and reports are repeatably regenerated; the original raw dataset and Phase 1 artifacts are preserved.
