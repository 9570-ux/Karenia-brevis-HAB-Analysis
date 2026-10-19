"""Phase 3: source-reported abundance labels and leakage-controlled candidates.

No training, learned preprocessing, forecast labels, or scientific thresholds.
"""
import csv
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['not observed', 'very low', 'low', 'medium', 'high']
TARGET = 'observed_abundance_category'
SELECTED = ['WATER_TEMP', 'SALINITY', 'SAMPLE_DEPTH', 'LATITUDE', 'LONGITUDE',
            'observation_year', 'observation_month', 'day_of_year']
TEMPORAL = ['observation_year', 'observation_month', 'day_of_year']
ENV = ['WATER_TEMP', 'SALINITY', 'SAMPLE_DEPTH', 'WIND_SPEED', 'WIND_DIR']
UNITS = {'WATER_TEMP': 'deg. C', 'SALINITY': 'ppt', 'SAMPLE_DEPTH': 'm',
         'WIND_SPEED': 'mph', 'WIND_DIR': 'deg', 'LATITUDE': 'decimal degrees',
         'LONGITUDE': 'decimal degrees'}
SOURCE = 'docs/HABSOS_Data_Dictionary.xlsx, Sheet1 A1:C32; docs/dataset.md; docs/phase2_preprocessing.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot(root=ROOT):
    paths = sorted((root / 'data/raw').rglob('*')) + [root / 'data/processed/habsos_cleaned.csv']
    return {p.relative_to(root).as_posix(): sha(p) for p in paths if p.is_file()}


def measurement(row, field):
    """Return recorded value or missing, and a documented quality status."""
    value = row[field]
    qa = row.get(field + '_QA', '').strip()
    if qa == '9':
        return '', 'no_data'
    if not value.strip():
        return '', 'unavailable'
    try:
        if not math.isfinite(float(value)):
            return '', 'invalid_representation'
    except ValueError:
        return '', 'invalid_representation'
    if field + '_UNIT' in row and row[field + '_UNIT'].strip() != UNITS[field]:
        return '', 'unit_unresolved'
    if field == 'SAMPLE_DEPTH':
        return value, 'qa_not_available'
    if qa == '1':
        return value, 'good'
    if qa == '4':
        return value, 'caution'
    if qa == '5':
        return value, 'source_converted'
    return '', 'qa_unresolved'


def label(row):
    category = row['CATEGORY']
    if category and category not in CLASSES:
        raise ValueError('Ambiguous undocumented CATEGORY: stop target generation')
    reasons = []
    if not category:
        reasons.append('category_missing')
    if (row['GENUS'], row['SPECIES']) != ('Karenia', 'brevis'):
        reasons.append('organism_unresolved')
    if row['CELLCOUNT_UNIT'] != 'cells/L':
        reasons.append('cellcount_unit_unresolved')
    if row['CELLCOUNT_QA'] not in ('1', '5'):
        reasons.append('cellcount_quality_unresolved_or_no_data')
    try:
        if not math.isfinite(float(row['CELLCOUNT'])):
            reasons.append('cellcount_unparseable')
    except ValueError:
        reasons.append('cellcount_unparseable')
    return ('', ';'.join(reasons)) if reasons else (category, 'eligible')


def role(name):
    if name in ENV:
        return 'environmental'
    if name in ('LATITUDE', 'LONGITUDE', 'STATE_ID'):
        return 'spatial'
    if name in TEMPORAL or 'TIME' in name or 'DATE' in name:
        return 'temporal'
    if name in ('GENUS', 'SPECIES'):
        return 'biological'
    if name == 'CATEGORY' or name.startswith('CELLCOUNT'):
        return 'target_related'
    if name.endswith('_QA') or name in ('INVALID_FIELDS', 'SUSPICIOUS_FIELDS', 'QA_NO_DATA_FIELDS'):
        return 'quality'
    return 'administrative_or_metadata'


def exclusion(name):
    if name in ('CATEGORY', 'GENUS', 'SPECIES') or name.startswith('CELLCOUNT'):
        return 'Target definition/context or direct abundance proxy; prevent label leakage'
    if name in ('INVALID_FIELDS', 'SUSPICIOUS_FIELDS', 'QA_NO_DATA_FIELDS'):
        return 'Compound quality fields may disclose CELLCOUNT information; metadata only'
    if name in ('OBJECTID', 'SOURCE_ROW_NUMBER', 'EXACT_DUPLICATE_OF_SOURCE_ROW'):
        return 'Identifier/row linkage; never a predictor'
    if name.endswith('_QA'):
        return 'Quality/capture process metadata, used only for quality handling'
    if name.endswith('_UNIT'):
        return 'Measurement metadata; units checked, no predictive encoding'
    if name == 'WIND_DIR':
        return 'Only 1,029 recorded values; direction origin/from-to convention undocumented; no cyclic/vector conversion'
    if name == 'WIND_SPEED':
        return 'Only 1,992 recorded values; exclude from initial candidate matrix rather than manufacture wind coverage'
    if name == 'DESCRIPTION':
        return 'High-cardinality site description with unresolved spelling variants; no site-identity encoding'
    if name == 'STATE_ID':
        return 'Retain as geography/sampling metadata; initial matrix uses measured coordinates without invented regions'
    if name.startswith('RAW_UNNAMED'):
        return 'Empty source column'
    return 'Date/time or derived metadata retained for lineage; selected temporal fields derive only from SAMPLE_DATE'


def run(root=ROOT):
    root = Path(root).resolve()
    before = protected_snapshot(root)
    phase2 = json.loads((root / 'reports/phase2_data_quality_report.json').read_text(encoding='utf-8'))
    source = root / 'data/processed/habsos_cleaned.csv'
    if sha(source) != phase2['processed']['sha256']:
        raise ValueError('Phase 2 input differs from verified report')
    manifest = json.loads((root / 'docs/download_manifest.json').read_text(encoding='utf-8'))
    if sha(root / 'data/raw' / manifest['filename']) != manifest['sha256']:
        raise ValueError('Raw integrity mismatch')
    ml = root / 'data/ml'
    ml.mkdir(parents=True, exist_ok=True)
    (root / 'reports').mkdir(exist_ok=True)
    paths = [ml / 'features.csv', ml / 'target.csv', ml / 'row_metadata.csv']
    for p in paths:
        if p.is_symlink() or (root / 'data/raw') in p.resolve().parents or (root / 'data/processed') in p.resolve().parents:
            raise ValueError('Unsafe output path')
    counts, raw_categories, target_status, missing = Counter(), Counter(), Counter(), Counter()
    source_missing, input_frequencies = Counter(), defaultdict(Counter)
    qa_status, years = defaultdict(Counter), defaultdict(Counter)
    spatial_values = defaultdict(list)
    eligible_total = 0
    row_count = 0
    with source.open(encoding='utf-8', newline='') as src, paths[0].open('w', encoding='utf-8', newline='') as fx, paths[1].open('w', encoding='utf-8', newline='') as fy, paths[2].open('w', encoding='utf-8', newline='') as fm:
        reader = csv.DictReader(src)
        header = reader.fieldnames
        if header != phase2['processed']['column_names']:
            raise ValueError('Phase 2 schema changed')
        wx, wy, wm = [csv.writer(f, lineterminator='\n') for f in (fx, fy, fm)]
        wx.writerow(['SOURCE_ROW_NUMBER'] + SELECTED)
        wy.writerow(['SOURCE_ROW_NUMBER', TARGET, 'TARGET_STATUS'])
        wm.writerow(header + ['TARGET_STATUS'] + [f + '_FEATURE_QUALITY' for f in ENV])
        for row_count, row in enumerate(reader, 1):
            if None in row or row['SOURCE_ROW_NUMBER'] != str(row_count):
                raise ValueError('Malformed record or lineage mismatch')
            for key, value in row.items():
                input_frequencies[key][value] += 1
                source_missing[key] += int(not value.strip())
            date = datetime.strptime(row['SAMPLE_DATE'], '%Y/%m/%d').date()
            calendar = dict(zip(TEMPORAL, (date.year, date.month, date.timetuple().tm_yday)))
            engineered = {}
            statuses = {}
            for field in ENV:
                value, status = measurement(row, field)
                engineered[field] = value
                statuses[field] = status
                qa_status[field][status] += 1
            for field, low, high in [('LATITUDE', -90, 90), ('LONGITUDE', -180, 180)]:
                value = row[field]
                number = float(value)
                if not math.isfinite(number) or not low <= number <= high:
                    raise ValueError('Coordinate representation/domain invalid; stop for review')
                engineered[field] = value
                spatial_values[field].append(number)
            engineered.update(calendar)
            target, status = label(row)
            raw_categories[row['CATEGORY'] or '(missing)'] += 1
            target_status[status] += 1
            if target:
                counts[target] += 1
                eligible_total += 1
            years[str(date.year)][target or '(ineligible)'] += 1
            for field in SELECTED:
                missing[field] += int(engineered[field] == '')
            wx.writerow([row['SOURCE_ROW_NUMBER']] + [engineered[f] for f in SELECTED])
            wy.writerow([row['SOURCE_ROW_NUMBER'], target, status])
            wm.writerow([row[f] for f in header] + [status] + [statuses[f] for f in ENV])
    if row_count != phase2['processed']['rows']:
        raise ValueError('Unexpected input record count')
    after = protected_snapshot(root)
    if before != after:
        raise RuntimeError('Protected input content changed')
    profiles = {p['name']: p for p in phase2['columns']}
    candidates = []
    for name in header + TEMPORAL:
        selected = name in SELECTED
        target_related = name in ('CATEGORY', 'GENUS', 'SPECIES') or name.startswith('CELLCOUNT') or name in ('INVALID_FIELDS', 'SUSPICIOUS_FIELDS', 'QA_NO_DATA_FIELDS')
        original_missing = source_missing[name] if name in header else 0
        candidates.append({'feature_name': name, 'source_column': 'SAMPLE_DATE' if name in TEMPORAL else name,
                           'category': role(name), 'type': 'integer' if name in TEMPORAL else profiles.get(name, {}).get('data_type', 'metadata/text'),
                           'unit': UNITS.get(name, {'observation_year': 'calendar year', 'observation_month': 'calendar month', 'day_of_year': 'calendar day index'}.get(name)),
                           'origin': 'derived' if name in TEMPORAL else 'Phase 2 input',
                           'original_missing_percentage': round(original_missing * 100 / row_count, 6),
                           'missing_percentage': round((missing[name] if selected else original_missing) * 100 / row_count, 6),
                           'target_related': target_related, 'leakage_risk': 'direct_or_target_context' if target_related else 'identifier' if name in ('OBJECTID', 'SOURCE_ROW_NUMBER', 'EXACT_DUPLICATE_OF_SOURCE_ROW') else 'sampling_or_quality_proxy' if role(name) in ('quality', 'administrative_or_metadata') else 'no_direct_target_content; sampling/spatiotemporal correlation remains',
                           'selected': selected, 'allowed_as_feature': selected,
                           'reason': 'Documented environmental/context measurement; QA-aware missingness retained' if selected and name in ENV else 'Measured coordinate; no invented geographic bins' if selected and name in ('LATITUDE', 'LONGITUDE') else 'Calendar-only derivation from the same observation date; no future information' if selected else exclusion(name)})
    distribution = {c: {'count': counts[c], 'percentage_of_eligible': round(counts[c] * 100 / eligible_total, 6)} for c in CLASSES}
    definition = {'name': TARGET, 'formulation': 'Multiclass source-reported contemporaneous Karenia brevis abundance category',
                  'binary_risk_status': 'No directly validated binary HAB-risk target was established from the available NOAA fields.',
                  'source_columns': ['CATEGORY', 'GENUS', 'SPECIES', 'CELLCOUNT', 'CELLCOUNT_UNIT', 'CELLCOUNT_QA'],
                  'rule': 'Copy CATEGORY verbatim when it is documented, organism is Karenia/brevis, cell-count unit is cells/L, count parses finitely and CELLCOUNT_QA is 1 or 5; otherwise blank target with explicit status. No numerical bin thresholds are inferred.',
                  'quality_policy': 'QA=1 source good; QA=5 upstream converted, retained with caution in metadata; QA=9 and undocumented/missing QA excluded from supervised-label eligibility only. This is a project eligibility policy, not a NOAA-mandated filter.',
                  'source': SOURCE, 'eligible_observations': eligible_total, 'ineligible_observations': row_count - eligible_total,
                  'exclusion_status_counts': dict(sorted(target_status.items())), 'raw_category_distribution': dict(sorted(raw_categories.items())),
                  'class_distribution': distribution, 'imbalance_ratio_largest_to_smallest': max(counts.values()) / min(counts.values()),
                  'forecast_horizon': None, 'claims_not_supported': ['binary harm/toxicity risk', 'future bloom onset', 'early warning', 'validated forecast performance']}
    report = {'phase': 3, 'input': {'filename': source.relative_to(root).as_posix(), 'rows': row_count, 'columns': len(header), 'sha256': sha(source)},
              'protected_inputs_unchanged': before == after, 'protected_input_hashes': before,
              'outputs': {p.name: {'filename': p.relative_to(root).as_posix(), 'rows': row_count, 'sha256': sha(p), 'size_bytes': p.stat().st_size} for p in paths},
              'output_rows': row_count, 'candidate_feature_count': len(candidates), 'selected_feature_count': len(SELECTED),
              'excluded_feature_count': len(candidates) - len(SELECTED), 'predictor_columns': SELECTED,
              'lineage_columns_not_predictors': ['SOURCE_ROW_NUMBER'], 'target_definition': definition,
              'candidate_features': candidates, 'excluded_fields': [p['feature_name'] for p in candidates if not p['selected']],
              'temporal_features': {'source': 'SAMPLE_DATE', 'formulas': {'observation_year': 'date.year', 'observation_month': 'date.month (1..12)', 'day_of_year': 'date.timetuple().tm_yday (1..366)'},
                                    'withheld': {'season': 'No documented marine seasonal regime; no arbitrary season assignments', 'time_of_day': '49,920 blank/placeholder times; unnecessary for initial calendar-only representation'}},
              'spatial_features': {key: {'unit': 'decimal degrees', 'missing': source_missing[key], 'min': min(v), 'max': max(v), 'domain_check': 'finite valid latitude/longitude angular domain; no bins'} for key, v in spatial_values.items()},
              'environmental_features': {'selected': [f for f in SELECTED if f in ENV], 'excluded': ['WIND_SPEED', 'WIND_DIR'], 'unit_conversions': [], 'quality_status_counts': {f: dict(sorted(c.items())) for f, c in sorted(qa_status.items())}},
              'qa_handling': 'QA metadata never predictors. 1=good; 4=caution retained; 5=upstream converted retained and distinguished from good; 9=missing in feature view even if populated in original. Unknown QA/unit masks feature, original preserved in row_metadata.',
              'identifier_handling': 'SOURCE_ROW_NUMBER is alignment metadata, not a predictor. OBJECTID and duplicate linkage only in row_metadata.',
              'source_field_distributions': {f: dict(sorted(c.items())) for f, c in sorted(input_frequencies.items()) if f in ('CATEGORY', 'GENUS', 'SPECIES') or f.endswith(('_QA', '_UNIT'))},
              'source_environmental_profiles': [p for p in phase2['columns'] if p['name'] in ENV + ['CELLCOUNT']],
              'missing_strategy': 'No fitting or filling now. Preserve NA. Later select missing-aware estimator or fit imputer/missing indicators/scaler only inside each training fold; never use validation/test data for fitted preprocessing. Drop SOURCE_ROW_NUMBER before X.',
              'wind_decision': 'No cyclic or u/v features: only degree units are documented, not reference axis or meteorological from/to convention; coverage also sparse.',
              'split_strategy': {'status': 'proposal only; no split files or fitted transformers', 'task': 'Temporal generalization of contemporaneous abundance classification, not early-warning forecasting',
                                 'proposal': {'train': '1953-08-19 through 2020-12-31', 'validation': '2021-01-01 through 2022-12-31', 'test': '2023-01-01 through 2024-03-25'},
                                 'reason': 'Chronological ordering matches later-time evaluation and observed coverage; calendar cutoffs are explicit design choices, not scientific thresholds.',
                                 'safeguards': ['Keep same-date samples together; no random record split', 'Fit every learned preprocessing step on training folds only', 'Rolling-origin folds within training period for later tuning', 'Freeze cutoffs before model development; audit class support and site/year shift', 'Report 2024 partial-year results separately from complete 2023', 'Coordinate/site overlap can remain; future unseen-site evaluation needs documented site grouping'],
                                 'eligible_class_counts_by_year': {y: dict(sorted(c.items())) for y, c in sorted(years.items())}},
              'limitations': ['Same-sample CATEGORY is a documented reported bin, not independently validated harm or future-risk ground truth', 'No CATEGORY thresholds reconstructed; source labeling may vary across agencies/decades', 'Retained QA=4 and QA=5 values require sensitivity review; depth has no QA field', 'Observed spatial/temporal context can encode sampling bias despite removal of direct label fields', 'pH, turbidity, TDS, dissolved oxygen and chlorophyll-a not available', 'Missingness and sparse winds remain; no deployability/accuracy claim'],
              'metadata_source': SOURCE}
    metadata = {'predictor_columns': SELECTED, 'lineage_columns_not_predictors': ['SOURCE_ROW_NUMBER'],
                'join_key': 'SOURCE_ROW_NUMBER', 'candidate_features': candidates, 'target_definition': definition,
                'source_sha256': sha(source), 'missing_strategy': report['missing_strategy']}
    (ml / 'feature_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    (root / 'reports/phase3_feature_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    result = run()
    print(json.dumps({k: result[k] for k in ('output_rows', 'candidate_feature_count', 'selected_feature_count', 'excluded_feature_count', 'target_definition')}, indent=2))
