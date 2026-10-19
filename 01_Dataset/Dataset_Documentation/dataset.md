# NOAA HABSOS — Phase 1 dataset source preparation

Project: **AI-Enabled Edge-Fog Based Marine Harmful Algal Bloom (HAB) Risk Prediction and Early Warning System**.

Completed on 2026-10-14: source and metadata investigation, original dataset collection, read-only inspection, and integrity test. No scientific preprocessing, imputation, source combination, ML training, MQTT, dashboard, adaptive sampling, or hardware implementation was performed. `data/processed/` contains only its directory placeholder.

## Project directory

Confirmed using terminal inspection: `C:\Users\mahad\OneDrive\Desktop\Fog and Edge Computing`. The root contained the Phase 1 folders created during this task; no existing application files or project AGENTS.md were found. This is the directory supplied for the new project.

## Official source and citation

Organization: NOAA National Centers for Environmental Information (NCEI), U.S. Department of Commerce.

Official dataset name: **Physical and biological data collected along the Texas, Mississippi, Alabama, and Florida coasts in the Gulf of America as part of the Harmful Algal BloomS Observing System from 1953-08-19 to 2024-03-25 (NCEI Accession 0120767)**.

[NOAA metadata and provided dataset citation](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0120767). Accession **0120767**, version **8.8**, published 2024-05-10.

DOI: **not provided in the inspected official metadata**. The accession identifier is not a DOI. No DOI has been invented.

Citation, populated from NOAA's provided citation with the downloaded scope and access date:

NOAA National Centers for Environmental Information (2014). *Physical and biological data collected along the Texas, Mississippi, Alabama, and Florida coasts in the Gulf of America as part of the Harmful Algal BloomS Observing System from 1953-08-19 to 2024-03-25 (NCEI Accession 0120767).* Full original CSV `habsos_20240430.csv`, version 8.8. NOAA National Centers for Environmental Information. Dataset. https://www.ncei.noaa.gov/archive/accession/0120767. Accessed 2026-10-14.

This is NOAA's dataset citation, not a journal-paper citation. NOAA metadata notes the official title was updated in 2025 for a geographic naming change; the version 8.8 data were published in 2024.

## Exact download and integrity

- [Verified original NOAA HTTPS CSV](https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/habsos_20240430.csv)
- Official FTP location: `ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0069/0120767/8.8/data/0-data/habsos_20240430.csv`
- Local raw file: `data/raw/habsos_20240430.csv`
- Size: **29,951,499 bytes** (29.951499 MB; approximately 28.564 MiB).
- Records: **211,834**, excluding the header. This matches NOAA's accession-detail observation count.
- Header: **28 columns**, including **26 named columns and two unnamed, entirely empty columns** at positions 3 and 28. These original columns remain unchanged.
- SHA-256: `0c561afc3239fc885afd161bbc0c3e9826969019f8266ffaade4adca7f700596`

NOAA's archive README identifies `data/0-data/` as unmodified originator data. Metadata and the supporting dictionary were inspected before observation download. The complete file was transferred through verified HTTP byte ranges after single-transfer timeouts. Each response's range and ETag, final byte length, and CSV readability were checked before moving the assembled original bytes into `data/raw/`. No CSV fields, header, encoding, line endings, values, or row order were rewritten. SHA-256 was recorded at download and checked during inspection and testing; it is a locally recorded integrity hash, not a NOAA-published checksum.

Provenance details: `docs/download_manifest.json`. Full inspection results: `docs/inspection_report.json`. Commands and resolved transfer problems: `docs/phase1_commands.md`.

## Coverage verified from the raw file

- Dates: **1953-08-19 through 2024-03-25**, matching accession metadata.
- States: **AL, FL, MS, TX**.
- Latitude: **24 to 30.7149 decimal degrees**.
- Longitude: **-97.66535 to -78.2789 decimal degrees**.
- Marine region: Texas, Mississippi, Alabama, and Florida coastal waters, the Florida shelf, and eastern Florida's North Atlantic coast.
- Organism: all observed genus/species values are **Karenia brevis**. This is not an all-species marine HAB dataset.

The dictionary specifies UTC sample times. Date and time strings are retained as recorded. Coverage does not imply continuous or uniformly spaced observations at every location.

## Available fields, units, and types

Units and definitions were verified from the original [NOAA data dictionary](https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/Support%20Documents/HABSOS_Data_Dictionary.xlsx), preserved locally as `docs/HABSOS_Data_Dictionary.xlsx`, and unit strings checked in the raw CSV. The dictionary has spelling errors `SALINTY_UNIT` and `SALINTY_QA`; the actual CSV uses `SALINITY_UNIT` and `SALINITY_QA`.

| CSV fields | Meaning and units | Inspection type |
|---|---|---|
| STATE_ID, DESCRIPTION | State abbreviation and sampling location description | Text; DESCRIPTION contains some numeric-looking location labels |
| LATITUDE, LONGITUDE | Location, decimal degrees | Float |
| SAMPLE_DATE | Sampling date, yyyy/mm/dd | Recorded text |
| SAMPLE_TIME | Sampling time, 24-hour UTC | Recorded text |
| SAMPLE_DEPTH | Depth below surface, meters | Float |
| GENUS, SPECIES | Identified HAB organism | Text |
| CATEGORY | Cell-count category: not observed, very low, low, medium, high | Text |
| CELLCOUNT | Organism abundance, cells/L | Integer |
| CELLCOUNT_UNIT | Observed value: cells/L | Text |
| CELLCOUNT_QA | Cell-count quality flag | Integer |
| SALINITY | Salinity, parts per thousand | Float |
| SALINITY_UNIT | Observed value: ppt | Text |
| SALINITY_QA | Salinity quality flag | Integer |
| WATER_TEMP | Water temperature, degrees Celsius | Float |
| WATER_TEMP_UNIT | Observed value: deg. C | Text |
| WATER_TEMP_QA | Temperature quality flag | Integer |
| WIND_DIR | Wind direction, degrees | Float |
| WIND_DIR_UNIT | Observed value: deg | Text |
| WIND_DIR_QA | Wind-direction quality flag | Integer |
| WIND_SPEED | Wind speed, miles per hour | Float |
| WIND_SPEED_UNIT | Observed value: mph | Text |
| WIND_SPEED_QA | Wind-speed quality flag | Integer |
| OBJECTID | Source database sample identifier | Integer |
| Unnamed columns 3 and 28 | Entirely empty original columns | Empty |

These are CSV-inferred types, not a claim that the CSV stores typed database values. The inspector labels columns by position to report duplicate blank headers without renaming the raw file. NOAA's companion map layer includes QA_COMMENT and lacks SAMPLE_TIME; its schema was not silently substituted for the archive CSV.

### Requested project variables

| Variable | Availability |
|---|---|
| Temperature | Available: WATER_TEMP, °C |
| pH | **Not available in this dataset.** |
| Turbidity | **Not available in this dataset.** |
| Salinity | Available: SALINITY, ppt |
| TDS | **Not available in this dataset.** Salinity is not a measured TDS value. |
| Dissolved oxygen | **Not available in this dataset.** |
| Chlorophyll-a | **Not available in this dataset.** |
| Algal/HAB observations | Available: CELLCOUNT, GENUS, SPECIES, CATEGORY |

Other environmental/context variables: sampling depth, wind direction, wind speed, dates, times, and coordinates. Metadata keywords such as biomass and brevetoxin do not establish additional measurement columns in this CSV.

## Missingness, quality, and recorded ranges

The dictionary says null environmental values mean data were not collected. Quality flags: **1 = good; 4 = exceeds normals; 5 = value converted by script; 9 = no data**. The source dictionary's converted-value flag describes upstream source processing, not project-created values.

| Measurement | Blank cells | QA=9 records | Finite recorded range |
|---|---:|---:|---|
| CELLCOUNT | 0 | 19 | 0–388,400,000 cells/L |
| SALINITY | 103,736 | 103,736 | 0–86 ppt |
| WATER_TEMP | 105,110 | 105,111 | 4–39.9 °C |
| WIND_DIR | 210,805 | 210,805 | 22.5–360 degrees |
| WIND_SPEED | 209,842 | 209,842 | 0–34.5 mph |
| SAMPLE_DEPTH | 5,503 | No depth-QA column | 0–600 meters |

Ranges include all finite recorded values and have **not** been QA-filtered, outlier-corrected, or declared scientifically valid. A populated cell may still have QA=9: cell count and temperature demonstrate why blank counts alone are insufficient. Source zeros are preserved and are not automatically missing.

Other blanks: SAMPLE_TIME **48,920**, CATEGORY **856**, DESCRIPTION **23**, CELLCOUNT_QA **192**. Both unnamed columns have **211,834** blank cells each. Complete per-column missing counts, QA categories, numeric ranges, and categorical values appear in the inspection report.

The dictionary says a null depth is assumed very near the surface; this project leaves it blank and does not assign a numeric depth. Times `12:00` or `0:00` denote submissions without a time. The file contains **1,000** `12:00` placeholders and zero `0:00` or `00:00` values, in addition to the 48,920 blank times. No invented times are assigned. The coordinate datum and QA threshold definitions are not specified by the inspected dictionary and remain limitations.

## Validation and limitations

The read-only test passed: **1 test, OK**. It checks that the genuine raw CSV exists, matches the download manifest's filename, size, and SHA-256, can be parsed, has records and columns, and retains the same hash after reading.

Large environmental gaps and extremely sparse wind observations prevent treating this as a complete sensor-feature dataset. CATEGORY is a contemporaneous cell-count bin, not an independently measured future-risk label. Sampling balance, duplicate observations, QA handling, date validation, missing-time handling, and a future prediction horizon require explicit decisions before later preprocessing or modeling. No rows were removed and no values were filled.

Additional data sources may be needed for unavailable variables. Selecting, matching, or combining NOAA, EPA, Kaggle, or other datasets remains a **future decision**. Only this single NOAA archive was imported. The statements above describe the preserved Phase 1 result. Phase 2 was subsequently authorized; its lossless preprocessing rules, reports, tests and limitations are documented in [phase2_preprocessing.md](phase2_preprocessing.md). The Phase 1 raw file remains unchanged.
