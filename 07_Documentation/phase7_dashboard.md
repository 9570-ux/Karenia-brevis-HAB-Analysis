# Phase 7 — NOAA HABSOS software dashboard

## Purpose and scientific boundary

A local Streamlit research interface for the saved Phase 5 model and unchanged Phase 6 inference API. The displayed **Predicted Recorded Abundance Category** means contemporaneous, source-recorded Karenia brevis abundance category. It is not a confirmed HAB, toxin result, health-risk assessment or future bloom forecast. No training, model tuning, synthetic observations, source merging or scientific data writes occur.

The current model predicts source-recorded abundance categories. It is not a validated future HAB-risk or early-warning model. Its rare high-abundance performance remains weak: high recall 3.33%, only 3 of 90 high observations classified correctly, 87 missed; high precision 1.37%. The Phase 4/5 test cohort was reused and is not independent prospective validation.

## Dependencies and launch

Use the already tested Phase 4/5 Python environment (Python 3.10, scikit-learn 1.7.2 and versions in reports/phase5_model_metadata.json). Phase 7 uses installed Streamlit 1.57.0, Altair 6.1.0 and pandas 2.3.3; it requires no library upgrades. Its dedicated direct dependencies are pinned in requirements_phase7.txt. For a fresh isolated environment, install the existing scientific dependencies first, then:

```powershell
cd 'C:\Users\mahad\OneDrive\Desktop\Fog and Edge Computing'
python -m pip install -r requirements_phase7.txt
streamlit run app.py
```

Equivalent if the executable is absent from PATH: `python -m streamlit run app.py`. The app runs locally; no publishing, Git or external scientific-data downloads are part of Phase 7. Streamlit session state holds submitted inputs/results in memory only. Read-only report/data loaders use st.cache_data; Phase 6 already caches its loaded fitted pipeline. No user input is appended to a file. Historical geographic plotting needs no map tile server or API key.

## Files and architecture

- app.py: six Streamlit pages, form, charts and existing figure display.
- dashboard.py: read-only artifact/data adapters and a direct call to scripts.inference.predict_observation.
- requirements_phase7.txt: dashboard dependencies only; prior requirement files are unchanged.
- tests/test_dashboard.py: model/source integrity, direct API delegation, historical provenance and automated Streamlit UI tests.
- reports/phase7_integrity_baseline.json: pre-change SHA-256 snapshot of all 83 prior files, plus the single necessary test-allowlist compatibility change.
- reports/phase7_dashboard_report.json: final verification evidence and limitations.
- reports/phase7_dashboard_test_results.txt, phase7_test_results.txt, phase7_launch_log.txt: dedicated/full test and server launch evidence.

## Pages

**Overview:** archive record count (211,834), classes, locked selected-model metrics, model/inference/dashboard phases and visible scientific limitations.

**Prediction:** exactly latitude, longitude, year, month and day_of_year as required context; optional temperature (degrees C), salinity (ppt), depth (meters). All fields initially blank. The optional “Load genuine NOAA example” button loads SOURCE_ROW_NUMBER 18 from the existing Phase 6 report, recorded 1980-12-11; this is an actual eight-value combination, not invented defaults. The recorded label is not observed and the saved model returns very low; this example does not establish model accuracy. Inputs are user submissions, not new research observations.

“Run Prediction” calls the Phase 6 function directly. That function validates before model prediction: exact schema, finite numeric inputs, valid angular coordinates, integer/consistent calendar components with leap-year day limits. The dashboard does not implement different calendar/coordinate rules, preprocess values or load the model itself. Blank optional inputs reach Phase 6 as None; only the unchanged saved fitted imputer replaces them inside the pipeline. PASS means schema validation only, not measurement-quality certification. The caller remains responsible for source QA and correct units.

Results preserve the API's class and probability mapping: not observed, very low, low, medium, high. The page shows a neutral-color horizontal probability chart, exact numeric table, validation status, actual missing environmental fields, artifact/model identity and submitted input/model metadata. Probabilities are estimated category scores, not calibrated confidence or future HAB/health probabilities. Results are labeled as belonging to the last submitted form; a rejected new submission clears the old result.

**Historical Data:** reads the existing processed NOAA context, Phase 3 predictors and target; verifies aligned source row numbers and one-to-one joins. Original CATEGORY supplies source-category charts, including recorded categories on rows ineligible for supervised modeling; missing categories remain separately counted. Existing Phase 3 QA-approved environmental views supply distributions and missingness, so the one populated source temperature marked no-data is not treated as valid. No historical values are imputed.

Year and state filters change the display subset only. Charts show source categories, observations by recorded year/month, and temperature/salinity/depth histograms with units and available/missing counts. Each histogram excludes only its own missing values; all available values are aggregated into 25 display bins without clipping, so browser charts do not send hundreds of thousands of measurements. Bins and zero counts are visualization aggregates, not added measurements. Sampling counts are not prevalence estimates; partial-year 2024 and uneven collection matter.

The historical location plot uses actual LATITUDE/LONGITUDE with Mercator display projection and no invented coordinates or model features. To keep rendering small it selects every ceil(N/4000)-th coordinate-complete source row after filtering, preserving source order, at most 4,000 locations. Display count/stride are visible; repeated locations overlap. It is a geographic coordinate plot without external basemap tiles and is not a real-time HAB map. No dataset sampling/deletion occurs on disk.

**Model Performance:** reads the existing locked reports, displaying Phase 5 accuracy 69.14%, macro F1 29.26%, balanced accuracy 30.34%, the Phase 4/5 comparison table/chart, and unchanged Phase 5 confusion-matrix/feature-importance PNGs. Native impurity importance is model-associated, not causal. No metrics or test predictions are recalculated.

**Dataset Information:** verified NOAA accession/version, raw size/record count/date coverage, local raw/processed/ML paths, predictor names/units, existing feature missingness, classes, unavailable variables, quality limitations, official metadata/download links and original source documentation/citation. DOI remains not provided in the inspected metadata; none is invented. pH, turbidity, TDS, dissolved oxygen and chlorophyll-a remain unavailable.

**About / Limitations:** Phases 1–7, observational/source-label and evaluation limitations, session-only inputs and Phase 8 status. Phase 8 hardware integration is not part of this dashboard implementation.

## Integrity compatibility

Phase 6's test_no_new_dataset_or_dashboard_hardware_files previously prohibited all later dashboard additions. Its test-only allowlist is minimally extended to the exact authorized new Phase 7 files. No assertions about old data/model hashes are removed and no inference code changes. This is the only existing-file modification, recorded with before/after SHA-256 in the Phase 7 baseline. The Phase 6 API/CLI, model artifacts, datasets, previous reports/figures/documentation and other existing tests remain unchanged. The Phase 5 frozen lock behavior already supports later-phase files.

The existing Phase 2/3 tests still rebuild their own outputs byte-identically as previously designed; the dashboard never invokes those builders. Final checks compare all prior file hashes and allow only the documented test-source change. No scientific dataset/model file may be added or modified.

## Verification and reproducibility

```powershell
python -B -m unittest tests.test_dashboard -v
python -B -m unittest discover -s tests -v
streamlit run app.py
```

Tests use real saved rows and Streamlit AppTest, without manual browser interaction. They exercise all six pages, empty/invalid form rejection, genuine-row prediction equality against Phase 6, missing environmental inputs, repeatability, exact class mapping, historical lineage and dataset/model/API hashes. A local server launch and health check are separately recorded; test/launch results are reported honestly in reports/phase7_dashboard_report.json.

Final verification: **16 Phase 7 tests passed; all 76 project tests passed** (the existing 60 plus 16). All 83 prior files were checked: 82 remain byte-identical and only the documented Phase 6 test allowlist changed. Raw/processed/ML data, Phase 4/5 models/results and Phase 6 inference code/CLI/reports retain their hashes. The Streamlit server launched at 127.0.0.1:8501 and its health endpoint returned `ok`; localhost socket access required sandbox escalation. The verification server session was stopped after checking. Launch it again with `streamlit run app.py` for use. Browser screenshot verification was unavailable because CUA failed to initialize (failed to write kernel assets: system cannot find the path); page rendering and interactions were verified using AppTest, not claimed as visual browser verification. AppTest's expected bare-mode ScriptRunContext warnings do not represent test failures.

Implementation references: [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest) and [number_input](https://docs.streamlit.io/develop/api-reference/widgets/st.number_input). Installed API signatures were also inspected to avoid assuming features from a different release.

No Phase 8 ESP32, sensors, LED, buzzer, MQTT broker/publisher/subscriber or hardware threshold logic is created. Work stops after Phase 7.
