# A Reproducible Machine Learning Framework for Karenia brevis Abundance Analysis Using NOAA HABSOS Data

## Overview and research objective

This package preserves the completed Phase 1–7 research/software demonstration. Its objective is reproducible analysis and classification of the **source-recorded contemporaneous abundance category of Karenia brevis** using genuine NOAA HABSOS data. It is **not a validated future HAB early-warning system**, confirmed toxicity detector or health-risk assessment.

Original source: `C:\Users\mahad\OneDrive\Desktop\Fog and Edge Computing`. All copied files retain their original bytes. Packaging does not train models, change results or create scientific observations. Only README.md, PROJECT_INVENTORY.md and GOOGLE_DRIVE_UPLOAD_GUIDE.md are new packaging documents.

## Dataset description and provenance

NOAA National Centers for Environmental Information, HABSOS accession 0120767, version 8.8. Original CSV: habsos_20240430.csv; 211,834 observations, 29,951,499 bytes, recorded 1953-08-19 through 2024-03-25, with recorded states AL, FL, MS and TX. Original values, missingness and row order remain unchanged.

[Official metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0120767) · [Original archived CSV](https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/habsos_20240430.csv)

DOI: not provided in the inspected official metadata. The exact official dataset title and supplied citation are copied in 01_Dataset/Dataset_Documentation/dataset.md and 07_Documentation/dataset.md. No DOI or citation has been invented.

Raw SHA-256: `0c561afc3239fc885afd161bbc0c3e9826969019f8266ffaade4adca7f700596`.

Approved source predictors are temperature (degrees C), salinity (ppt), depth (meters), latitude/longitude (decimal degrees), recorded year, month and day_of_year. pH, turbidity, TDS, dissolved oxygen and chlorophyll-a are **not available in this dataset**. Salinity is not a measured TDS value.

## Phase 1–7 and software architecture

| Phase | Preserved implementation |
|---|---|
| 1 | Official NOAA dataset collection, metadata and dictionary |
| 2 | Lossless preprocessing and quality/missingness reports |
| 3 | Approved features, recorded target eligibility and source-row lineage |
| 4 | Fixed chronological baseline machine-learning comparison |
| 5 | Controlled model improvement and error analysis |
| 6 | Reusable validated single-observation inference |
| 7 | Streamlit dashboard using the unchanged Phase 6 API |

Architecture: original NOAA CSV → lossless processed context → approved predictors/eligible recorded labels → validation-selected saved pipeline → Phase 6 inference → Phase 7 dashboard. Edge/fog hardware and live sensor integration are not implemented.

## Machine-learning methodology and temporal splits

Labels are not observed, very low, low, medium and high, copied from recorded CATEGORY under the documented eligibility policy. No new abundance thresholds or future-risk target are invented. There are 210,767 eligible labels and 1,067 ineligible rows retained for traceability.

| Split | Recorded dates | Eligible observations |
|---|---|---:|
| Train | Through 2020-12-31 | 184,845 |
| Validation | 2021-01-01–2022-12-31 | 18,494 |
| Test | 2023-01-01–2024-03-25 | 7,428 |

No random cross-time split or synthetic resampling is used. Phase 5 compares 15 fixed recipes and selects by validation macro F1 with the documented minority-recall/name tie rules. Final fitting uses 203,339 train+validation rows and excludes test. Phase 5 reuses the already inspected Phase 4 test cohort; it is not independent prospective validation.

## Final model and main existing results

Selected experiment: `D_environmental_temporal_spatial_cyclic__random_forest__balanced`. Balanced Random Forest: 40 trees, max_depth 14, min_samples_leaf 8, seed 42, one job. The saved pipeline accepts the original eight predictors, performs deterministic cyclic calendar engineering and applies its already fitted median imputer plus missingness indicators. Computational imputation is not new scientific measurement. Absolute year provides leap-year context rather than a direct classifier feature in the selected group.

Phase 5 model: 03_Models/phase5_best_model.joblib. Locked Phase 4 baseline: 03_Models/best_model.joblib. Original model metadata accompanies both. The Phase 5 custom transformer phase5_transformers.py must remain importable. Model SHA-256: `a4809c0f2549b51344749393a0ec5bf2e2db6e10b5b42f866bde8be270fcc734`.

| Locked Phase 5 test metric | Value |
|---|---:|
| Accuracy | 69.1438% |
| Macro F1 | 29.2562% |
| Balanced accuracy | 30.3420% |
| High recall | 3.3333% |
| High precision | 1.3699% |

Only 3 of 90 high observations were classified correctly; 87 were missed. Rare **high** performance remains weak. Phase 4 macro F1 was 19.6975%, compared with Phase 5 29.2562% on the reused cohort. These existing results are not new experiment runs, statistical-significance claims or evidence of operational alert reliability.

## Dashboard information

The local Streamlit interface provides Overview, Prediction, Historical Data, Model Performance, Dataset Information and About/Limitations. It delegates directly to scripts/inference.py. Missing optional environmental values use only the saved fitted imputer. User inputs remain in session memory and are never appended to scientific datasets. Historical plots use real records, including documented display-only location subsampling.

Correct output wording: **Predicted Recorded Abundance Category**. Estimated class probabilities are not future bloom probabilities, calibrated confidence or health-risk scores.

## Folder structure

- 01_Dataset/Raw_Dataset: original NOAA CSV and existing integrity placeholder.
- 01_Dataset/Processed_Dataset: unchanged Phase 2 CSV; ML_Ready contains existing Phase 3 predictors, targets, metadata and lineage.
- 01_Dataset/Dataset_Documentation: verified metadata/dictionary and collection/inspection scripts.
- 02_Source_Code: phase-specific code, original test suite and dependency pins.
- 03_Models: both saved models and original metadata.
- 04_Reports: Phase 2–7 reports, audits, predictions, logs and unchanged figures.
- 05_Test_Results: existing dedicated and complete-suite test evidence.
- 06_Dashboard: original dashboard files, usage documentation and dashboard reports.
- 07_Documentation: all existing documentation and provenance metadata.
- 08_Paper: empty; no existing paper source found.
- 09_Presentation: empty; no existing presentation found.
- New folder: pre-existing destination directory, preserved untouched.

Some files intentionally appear in multiple requested sections, including dashboard files, documentation, model metadata and test logs. They are identical archive copies, not additional observations or alternative results. Existing confusion-matrix/feature-importance PDFs are report figures, not a paper or slide deck.

## Reproducibility commands: restore the original layout first

**This organized archive is not directly runnable from the phase folders or 06_Dashboard.** Original code resolves scripts/data/models/reports/tests relative to a common project root. Do not rewrite the archived code or its historical manifests.

The optional commands below reconstruct the exact original relative layout inside a separate Reproduction_Workspace within this package. They are instructions only: no reconstruction, dependency installation, model training or test rerun was performed while packaging.

Open PowerShell in this package root:

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

The example is the genuine existing source row 18, recorded 1980-12-11; actual category not observed, model output very low. This demonstrates inference behavior, not a correct classification. Use the documented Python 3.10/scikit-learn 1.7.2 environment and original pinned requirements. Installed third-party libraries and caches are intentionally omitted.

Existing Phase 2/3 tests may rebuild their output copies byte-identically in the reproduction workspace. Training scripts are archived for research reproducibility; running the dashboard requires no retraining. Preserved provenance files may reference historical absolute source paths.

## Testing information and archive verification

Existing logs record **16 dedicated Phase 7 tests passed** and **76 complete project tests passed**. These are copied historical results, not tests rerun during packaging. The original dashboard report records successful launch/health verification and its unavailable browser screenshot verification.

All 119 copied file hashes match their sources. The entire source inventory of 1758 files, including excluded libraries/caches, was hashed before and after copying and remains unchanged. PROJECT_INVENTORY.md lists every copy with original path, purpose and SHA-256.

## Important limitations and future work

Historical missing measurements/times, QA cautions/conversions, strong imbalance, source-reported labels, uneven collection, spatial/temporal dependence and unspecified coordinate datum remain. Partial-year 2024 is not a complete annual sample. Geographic generalization, prospective bloom onset, toxicity and health risk are not validated.

Future work requires separate authorization and defensible study design, potentially independent official variables/sources and prospective evaluation. No new sources were combined here. No Phase 8 hardware, ESP32, sensors, LED/buzzer or MQTT implementation is included. No paper, presentation, references or scientific observations were fabricated. No Git/GitHub operations or cloud upload occurred.
