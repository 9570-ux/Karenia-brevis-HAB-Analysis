"""Deterministic, lossless NOAA HABSOS Phase 2 profiling. Standard library only."""
import csv
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = {'LATITUDE', 'LONGITUDE', 'SAMPLE_DEPTH', 'CELLCOUNT', 'SALINITY',
           'WATER_TEMP', 'WIND_DIR', 'WIND_SPEED', 'OBJECTID', 'CELLCOUNT_QA',
           'SALINITY_QA', 'WATER_TEMP_QA', 'WIND_DIR_QA', 'WIND_SPEED_QA'}
ENV = ['SAMPLE_DEPTH', 'CELLCOUNT', 'SALINITY', 'WATER_TEMP', 'WIND_DIR', 'WIND_SPEED']
UNITS = {'SAMPLE_DEPTH': 'm', 'CELLCOUNT': 'cells/L', 'SALINITY': 'ppt',
         'WATER_TEMP': 'deg. C', 'WIND_DIR': 'deg', 'WIND_SPEED': 'mph',
         'LATITUDE': 'decimal degrees', 'LONGITUDE': 'decimal degrees'}
MEANINGS = {'SAMPLE_DEPTH': 'Depth below surface', 'CELLCOUNT': 'Karenia brevis abundance',
            'SALINITY': 'Sample salinity', 'WATER_TEMP': 'Water temperature',
            'WIND_DIR': 'Wind direction', 'WIND_SPEED': 'Wind speed',
            'LATITUDE': 'Sample latitude', 'LONGITUDE': 'Sample longitude',
            'STATE_ID': 'State postal abbreviation', 'DESCRIPTION': 'Sample location description',
            'SAMPLE_DATE': 'Sampling date', 'SAMPLE_TIME': 'Sampling time in UTC',
            'GENUS': 'HAB organism genus', 'SPECIES': 'HAB organism species',
            'CATEGORY': 'Source cell-count bin', 'OBJECTID': 'Source database identifier'}
ADDED = ['SOURCE_ROW_NUMBER', 'SAMPLE_DATE_ISO', 'SAMPLE_TIME_UTC_PARSED',
         'SAMPLE_DATETIME_UTC', 'TIME_STATUS', 'INVALID_FIELDS', 'SUSPICIOUS_FIELDS',
         'QA_NO_DATA_FIELDS', 'EXACT_DUPLICATE_OF_SOURCE_ROW']
SOURCE = 'docs/HABSOS_Data_Dictionary.xlsx, Sheet1 A1:C32; docs/dataset.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_snapshot(root=ROOT):
    return {p.relative_to(root).as_posix(): sha(p)
            for p in sorted((root / 'data/raw').rglob('*')) if p.is_file()}


def parse_date(value):
    value = value.strip()
    if not value:
        return '', 'missing'
    if not re.fullmatch(r'\d{4}/\d{2}/\d{2}', value):
        return '', 'invalid'
    try:
        return datetime.strptime(value, '%Y/%m/%d').date().isoformat(), 'valid'
    except ValueError:
        return '', 'invalid'


def parse_time(value):
    value = value.strip()
    if not value:
        return '', 'missing'
    if not re.fullmatch(r'\d{1,2}:\d{2}', value):
        return '', 'invalid'
    try:
        parsed = datetime.strptime(value, '%H:%M').time()
    except ValueError:
        return '', 'invalid'
    if (parsed.hour, parsed.minute) in ((0, 0), (12, 0)):
        return '', 'metadata_missing_placeholder'
    return parsed.strftime('%H:%M'), 'valid'


def run(root=ROOT):
    root = Path(root).resolve()
    before = raw_snapshot(root)
    manifest = json.loads((root / 'docs/download_manifest.json').read_text(encoding='utf-8'))
    raw = root / 'data/raw' / manifest['filename']
    if raw.name != 'habsos_20240430.csv' or sha(raw) != manifest['sha256']:
        raise ValueError('Raw provenance mismatch; stop without processing')
    if raw.stat().st_size != manifest['size_bytes']:
        raise ValueError('Raw size differs from Phase 1 manifest')
    out = root / 'data/processed/habsos_cleaned.csv'
    report_path = root / 'reports/phase2_data_quality_report.json'
    events_path = root / 'reports/phase2_record_issues.csv'
    for path in (out, report_path, events_path):
        if (root / 'data/raw') in path.resolve().parents:
            raise ValueError('Output cannot be under data/raw')
        path.parent.mkdir(parents=True, exist_ok=True)
    issues = Counter()
    affected = defaultdict(set)
    dates, timestamps = [], []
    time_statuses = Counter()
    date_statuses = Counter()
    seen = {}
    duplicates = []
    qa_counts = defaultdict(Counter)
    values = defaultdict(list)
    suspicious_counts = Counter()
    semantic_missing = Counter()
    with raw.open(encoding='utf-8-sig', newline='') as source, out.open('w', encoding='utf-8', newline='') as dest, events_path.open('w', encoding='utf-8', newline='') as event_file:
        reader = csv.reader(source, strict=True)
        header = next(reader)
        expected = json.loads((root / 'docs/inspection_report.json').read_text(encoding='utf-8-sig'))['column_names']
        if header != expected:
            raise ValueError('Header differs from inspected Phase 1 schema')
        names = [name or f'RAW_UNNAMED_{i}' for i, name in enumerate(header, 1)]
        if len(set(names + ADDED)) != len(names + ADDED):
            raise ValueError('Output field collision')
        index = {name: i for i, name in enumerate(names)}
        frequency = [Counter() for _ in names]
        whitespace = Counter()
        invalid_counts = Counter()
        writer = csv.writer(dest, lineterminator='\n')
        writer.writerow(names + ADDED)
        event_writer = csv.writer(event_file, lineterminator='\n')
        event_writer.writerow(['SOURCE_ROW_NUMBER', 'COLUMN', 'RULE', 'CLASSIFICATION', 'ORIGINAL_VALUE'])

        def event(row_number, column, rule, classification, value):
            issues[rule] += 1
            affected[rule].add(row_number)
            event_writer.writerow([row_number, column, rule, classification, value])

        rows = 0
        for row_number, row in enumerate(reader, 1):
            if len(row) != len(header):
                raise ValueError(f'Record {row_number}: malformed field count; no silent removal')
            rows = row_number
            d = dict(zip(names, row))
            invalid, suspicious, no_data = set(), set(), []
            for name, value, counter in zip(names, row, frequency):
                counter[value] += 1
                stripped = value.strip()
                if value != stripped:
                    whitespace[name] += 1
                    event(row_number, name, 'surrounding_whitespace', 'formatting', value)
                if name in NUMERIC and stripped:
                    try:
                        number = float(stripped)
                        if not math.isfinite(number):
                            raise ValueError('Nonfinite')
                        if (name.endswith('_QA') or name in ('OBJECTID', 'CELLCOUNT')) and not number.is_integer():
                            raise ValueError('Expected integer')
                        values[name].append(number)
                    except ValueError:
                        invalid.add(name)
                        invalid_counts[name] += 1
                        event(row_number, name, 'unparseable_or_nonfinite_numeric', 'invalid_representation', value)
            date, date_status = parse_date(d['SAMPLE_DATE'])
            parsed_time, time_status = parse_time(d['SAMPLE_TIME'])
            date_statuses[date_status] += 1
            time_statuses[time_status] += 1
            if date:
                dates.append(date)
            for name, status in [('SAMPLE_DATE', date_status), ('SAMPLE_TIME', time_status)]:
                if status == 'invalid':
                    invalid.add(name)
                    invalid_counts[name] += 1
                    event(row_number, name, 'unparseable_date_or_time', 'invalid_representation', d[name])
                elif status == 'metadata_missing_placeholder':
                    event(row_number, name, 'metadata_missing_time_placeholder', 'missing', d[name])
            timestamp = f'{date}T{parsed_time}Z' if date and parsed_time else ''
            if timestamp:
                timestamps.append(timestamp)
            for field in ENV:
                qa_name = field + '_QA'
                qa = d.get(qa_name, '').strip()
                if not d[field].strip() or qa == '9':
                    semantic_missing[field] += 1
                if qa_name in d:
                    qa_counts[field][qa or '(blank)'] += 1
                if qa == '9':
                    no_data.append(field)
                    if d[field].strip():
                        suspicious.add(field)
                        event(row_number, field, 'populated_measurement_with_qa9', 'suspicious', d[field])
                elif qa == '4':
                    suspicious.add(field)
                    event(row_number, field, 'noaa_qa4_exceeds_normals', 'suspicious', d[field])
                if qa and qa not in ('1', '4', '5', '9'):
                    suspicious.add(qa_name)
                    event(row_number, qa_name, 'undocumented_qa_code', 'suspicious', qa)
                unit_name = field + '_UNIT'
                if unit_name in d and d[field].strip() and d[unit_name].strip() != UNITS[field]:
                    suspicious.add(field)
                    event(row_number, unit_name, 'unexpected_or_missing_unit_for_measurement', 'suspicious', d[unit_name])
            suspicious_counts.update(suspicious)
            if time_status in ('missing', 'metadata_missing_placeholder'):
                semantic_missing['SAMPLE_TIME'] += 1
            key = tuple(row)
            duplicate_of = seen.get(key, '')
            if duplicate_of:
                duplicates.append({'source_row': row_number, 'duplicate_of_source_row': duplicate_of,
                                   'original_values': row})
            else:
                seen[key] = row_number
            writer.writerow(row + [row_number, date, parsed_time, timestamp, time_status,
                                   ';'.join(sorted(invalid)), ';'.join(sorted(suspicious)),
                                   ';'.join(no_data), duplicate_of])
    after = raw_snapshot(root)
    if before != after:
        raise RuntimeError('Raw directory content changed during preprocessing')
    profiles = []
    for i, name in enumerate(names):
        counter = frequency[i]
        blank = sum(count for value, count in counter.items() if not value.strip())
        numeric = values[name]
        normalization_groups = defaultdict(set)
        for value in counter:
            if value.strip():
                normalization_groups[value.strip().casefold()].add(value)
        variants = [sorted(group) for group in normalization_groups.values() if len(group) > 1]
        profile = {'name': name, 'raw_name': header[i], 'position': i + 1, 'total': rows,
                   'meaning': MEANINGS.get(name, 'QA flag' if name.endswith('_QA') else 'Unit label' if name.endswith('_UNIT') else 'Empty source column'),
                   'unit': UNITS.get(name), 'data_type': 'integer' if name in NUMERIC and all(v.is_integer() for v in numeric) else 'float' if name in NUMERIC else 'text',
                   'missing': blank, 'missing_percentage': round(blank * 100 / rows, 6),
                   'documented_missing_including_qa_or_placeholder': semantic_missing[name] if name in ENV + ['SAMPLE_TIME'] else blank,
                   'documented_missing_percentage': round((semantic_missing[name] if name in ENV + ['SAMPLE_TIME'] else blank) * 100 / rows, 6),
                   'unique_nonblank_count': sum(bool(v.strip()) for v in counter),
                   'categorical_values': dict(sorted(counter.items())) if name not in NUMERIC and len(counter) <= 100 else None,
                   'minimum': min(numeric) if numeric else None, 'maximum': max(numeric) if numeric else None,
                   'median': median(numeric) if numeric else None, 'zeros': sum(v == 0 for v in numeric),
                   'suspicious_values': suspicious_counts[name], 'invalid_values': invalid_counts[name],
                   'surrounding_whitespace': whitespace[name], 'case_or_whitespace_variants': variants,
                   'action': 'retain original values and blanks; no imputation',
                   'reason': 'Preserve source observations; no justified replacement measurement'}
        profiles.append(profile)
    report = {'phase': 2, 'raw': {'filename': raw.relative_to(root).as_posix(), 'rows': rows,
               'columns': len(header), 'column_names': header, 'sha256': manifest['sha256'], 'size_bytes': raw.stat().st_size},
              'processed': {'filename': out.relative_to(root).as_posix(), 'rows': rows, 'columns': len(names + ADDED),
                            'column_names': names + ADDED, 'sha256': sha(out), 'size_bytes': out.stat().st_size},
              'raw_directory_unchanged': before == after, 'raw_directory_hashes': before,
              'metadata_source': SOURCE, 'dictionary_sha256': sha(root / 'docs/HABSOS_Data_Dictionary.xlsx'),
              'duplicates': {'definition': 'All 28 original CSV fields equal, including OBJECTID; no normalization',
                             'extra_records': len(duplicates), 'removed': 0, 'records': duplicates,
                             'decision': 'Retain all records; repetition alone does not prove accidental collection/ingestion'},
              'invalid_values': {'detected_cells': sum(invalid_counts.values()), 'by_column': dict(sorted(invalid_counts.items())),
                                 'removed_cells': 0, 'removed_records': 0,
                                 'definition': 'Invalid representation only: malformed numeric/date/time. No invented scientific thresholds.'},
              'date_time': {'date_format': '%Y/%m/%d', 'time_format': '%H:%M (one/two digit input hour)', 'timezone': 'UTC per NOAA dictionary',
                            'earliest_date': min(dates) if dates else None, 'latest_date': max(dates) if dates else None,
                            'earliest_complete_timestamp': min(timestamps) if timestamps else None,
                            'latest_complete_timestamp': max(timestamps) if timestamps else None,
                            'date_status_counts': dict(sorted(date_statuses.items())), 'time_status_counts': dict(sorted(time_statuses.items())),
                            'missing_combined_timestamp': rows - len(timestamps)},
              'columns': profiles, 'qa_counts': {k: dict(sorted(v.items())) for k, v in sorted(qa_counts.items())},
              'quality_rules': [{'rule': rule, 'source_or_reason': SOURCE if rule.startswith(('noaa_', 'populated_', 'metadata_', 'unexpected_', 'undocumented_')) else 'Format validation, not a scientific threshold',
                                 'affected_cells': count, 'affected_records': len(affected[rule]),
                                 'record_locations': events_path.relative_to(root).as_posix()} for rule, count in sorted(issues.items())],
              'record_issues_file': events_path.relative_to(root).as_posix(),
              'transformations': ['Retain all original rows in source order and all original cell strings',
                                  'Name empty headers RAW_UNNAMED_3 and RAW_UNNAMED_28; no source columns removed',
                                  'Append source record number, parsed date/time, documented missing-time status, quality flags and duplicate linkage',
                                  'Serialize processed CSV as UTF-8 with LF newlines; raw bytes untouched'],
              'columns_removed': [], 'columns_added': ADDED,
              'unavailable_variables': ['pH', 'turbidity', 'TDS', 'dissolved oxygen', 'chlorophyll-a'],
              'limitations': ['QA=9 is missing even when a numeric value is populated; original number is retained and flagged',
                              'No scientific outlier thresholds applied; QA=4 provides the source-defined suspicious flag',
                              'Blank QA flags are unresolved; observed range/median includes all finite recorded values regardless of QA',
                              'CATEGORY is a same-observation cell-count bin, not a future-risk outcome',
                              'Irregular sampling, environmental missingness and sparse wind measurements remain',
                              'No inference, merging, scaling, clipping, imputation, or model training',
                              'Unusual unflagged measurements remain unassessed, not automatically valid']}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    result = run()
    print(json.dumps({key: result[key] for key in ('raw', 'processed', 'duplicates', 'invalid_values', 'date_time')}, indent=2))
