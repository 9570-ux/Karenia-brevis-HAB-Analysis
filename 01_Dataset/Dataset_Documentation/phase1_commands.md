# Phase 1 commands

Working directory: `C:\Users\mahad\OneDrive\Desktop\Fog and Edge Computing`.

## Project inspection

```powershell
Get-Location
Get-ChildItem -Force
rg --files -g AGENTS.md -g '*.py' -g '*.md'
python --version
```

The initial sandbox terminal failed at startup. An approved execution route worked; later ordinary local commands also worked. No AGENTS.md was found in the project search. Root inspection showed only the Phase 1 directories created during this task.

## Source discovery and verification

Official metadata was inspected through the web tool before observation download:

- https://habsos.noaa.gov/about
- https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:0120767
- https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncei:Cell_Counts
- https://gis.ngdc.noaa.gov/arcgis/rest/services/EnvironmentalMonitoring/HABSOSViewBase/MapServer/0

NOAA's archive web links returned redirects/pages without usable data links. Its map-layer metadata provided the FTP archive. Directory listings identified version 8.8, its originator CSV, README, and supporting data dictionary. These commands were used, substituting the observed child paths as each listing was read:

```powershell
curl.exe --silent --show-error --max-time 20 --list-only 'ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0069/0120767/'
curl.exe --silent --show-error --max-time 20 --list-only 'ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0069/0120767/8.8/'
curl.exe --silent --show-error --max-time 20 'ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0069/0120767/8.8/NCEI-Readme.txt'
curl.exe --silent --show-error --max-time 20 --list-only 'ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0069/0120767/8.8/data/'
curl.exe --silent --show-error --max-time 20 --list-only 'ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0069/0120767/8.8/data/0-data/'
curl.exe --silent --show-error --max-time 20 --list-only 'ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0069/0120767/8.8/data/0-data/Support%20Documents/'
curl.exe --silent --show-error --max-time 20 --list-only 'ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0069/0120767/8.8/about/'
curl.exe --fail --silent --show-error --max-time 30 --output 'docs/HABSOS_Data_Dictionary.xlsx' 'ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0069/0120767/8.8/data/0-data/Support%20Documents/HABSOS_Data_Dictionary.xlsx'
curl.exe --silent --show-error --max-time 25 'ftp://ftp-oceans.ncei.noaa.gov/nodc/archive/arc0069/0120767/8.8/data/0-data/'
curl.exe --head --silent --show-error --max-time 20 'https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/habsos_20240430.csv'
curl.exe --compressed --fail --silent --show-error --max-time 40 --range 0-262143 --output 'docs/range_probe.bin' --write-out 'status=%{http_code} bytes=%{size_download} speed=%{speed_download}' 'https://www.ncei.noaa.gov/data/oceans/archive/arc0069/0120767/8.8/data/0-data/habsos_20240430.csv'
```

The dictionary was extracted read-only using the bundled Python runtime:

```powershell
& 'C:\Users\mahad\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -c "import openpyxl; w=openpyxl.load_workbook('docs/HABSOS_Data_Dictionary.xlsx',read_only=True,data_only=True); [(print('SHEET',s.title),[print(r) for r in s.iter_rows(values_only=True)]) for s in w]"
```

Full-file FTP and HTTPS curl attempts timed out. The incomplete transfers were kept outside the accepted raw dataset and were not used for research. Unsuccessful HTTPS transport probes returned 404 for `/data/oceans/nodc/archive/` and `/pub/data/nodc/archive/`; the FTP hostname's HTTPS service timed out. The working `/data/oceans/archive/` HTTPS path returned HTTP 200 and Content-Length 29,951,499. A range probe returned HTTP 206 with the expected 262,144 bytes.

## Reproducible download and inspection

```powershell
python -B scripts/download_dataset.py
python -B scripts/inspect_dataset.py > docs/inspection_report.json
python -B -m unittest discover -s tests -v
```

The downloader validates each byte range, ETag, assembled length, and CSV readability. It creates a SHA-256 provenance manifest and refuses to overwrite an existing raw file. It does not rewrite CSV data or combine sources.

The first test run failed as expected because no accepted raw CSV existed. The inspector also initially rejected the archive's unnamed headers; this was corrected in the inspector without changing the CSV. Final results are recorded in dataset.md.
