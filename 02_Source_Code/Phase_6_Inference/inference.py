"""Read-only single-observation inference with the frozen Phase 5 pipeline."""
import calendar
from collections.abc import Mapping
from datetime import date, timedelta
from functools import lru_cache
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
# The serialized transformer uses this original top-level module name.
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))
import phase5_transformers  # noqa: E402,F401

ARTIFACT = ROOT / 'models/phase5_best_model.joblib'
FIELDS = dict(zip(
    ('temperature', 'salinity', 'depth', 'latitude', 'longitude', 'year', 'month', 'day_of_year'),
    phase5_transformers.ORIGINAL))
CLASSES = ('not observed', 'very low', 'low', 'medium', 'high')
OPTIONAL = ('temperature', 'salinity', 'depth')
TERM = 'Predicted recorded abundance category'
VERSION = 'phase6-inference-1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_model():
    """Load only the local, hash-verified project artifact; never fit a model."""
    metadata = json.loads((ROOT / 'reports/phase5_model_metadata.json').read_text(encoding='utf-8'))
    if sha256(ARTIFACT) != metadata['model_sha256']:
        raise RuntimeError('Phase 5 model SHA-256 differs from locked metadata')
    model = joblib.load(ARTIFACT)
    if list(model.feature_names_in_) != list(FIELDS.values()):
        raise RuntimeError('Saved model source schema differs from the approved eight fields')
    if len(model.classes_) != len(CLASSES) or set(model.classes_) != set(CLASSES):
        raise RuntimeError('Saved model class labels differ from the approved labels')
    if sklearn.__version__ != metadata['versions']['sklearn']:
        raise RuntimeError('Use the scikit-learn version recorded in Phase 5 metadata')
    return model, metadata


def validate_observation(observation):
    """Return ordered model input and missing fields; no numerical imputation."""
    if not isinstance(observation, Mapping):
        raise ValueError('Observation must be a mapping of named fields')
    unknown = set(observation) - set(FIELDS)
    absent = set(FIELDS) - set(OPTIONAL) - set(observation)
    if unknown or absent:
        raise ValueError(f'Invalid schema: unknown={sorted(unknown, key=str)}, missing={sorted(absent)}')
    values, missing = {}, []
    for field, source in FIELDS.items():
        value = observation.get(field)
        if value is None:
            if field not in OPTIONAL:
                raise ValueError(f'{field} is required and cannot be missing')
            values[source] = np.nan
            missing.append(field)
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f'{field} must be a finite number (or None for an environmental field)')
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f'{field} must be a finite number') from exc
        if not math.isfinite(number):
            raise ValueError(f'{field} must be finite; use None for missing environmental values')
        if field in ('year', 'month', 'day_of_year') and not number.is_integer():
            raise ValueError(f'{field} must be an integer')
        values[source] = int(number) if field in ('year', 'month', 'day_of_year') else number
    year, month, day = (values[FIELDS[f]] for f in ('year', 'month', 'day_of_year'))
    if not 1 <= year <= 9999:
        raise ValueError('year must be in the Python calendar range 1..9999')
    if not 1 <= month <= 12:
        raise ValueError('month must be 1..12')
    if not 1 <= day <= (366 if calendar.isleap(year) else 365):
        raise ValueError('day_of_year is invalid for the recorded year')
    if (date(year, 1, 1) + timedelta(days=day - 1)).month != month:
        raise ValueError('month and day_of_year describe different recorded dates')
    if not -90 <= values['LATITUDE'] <= 90 or not -180 <= values['LONGITUDE'] <= 180:
        raise ValueError('latitude must be -90..90 and longitude -180..180 decimal degrees')
    return pd.DataFrame([values], columns=list(FIELDS.values())), missing


def predict_observation(observation):
    """Predict one recorded observation. None/omitted environmental fields stay missing.

    Input uses the eight public names in FIELDS. Coordinates and calendar fields
    are required. This function writes no files and never calls fit.
    """
    frame, missing = validate_observation(observation)
    model, metadata = load_model()
    with threadpool_limits(limits=1):
        probabilities = model.predict_proba(frame)[0]
        predicted = str(model.predict(frame)[0])
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.isclose(probabilities.sum(), 1, atol=1e-12, rtol=0)):
        raise RuntimeError('Model returned invalid class probabilities')
    mapping = dict(zip(model.classes_, probabilities))
    return {
        'terminology': TERM,
        'predicted_class': predicted,
        'class_probabilities': {label: float(mapping[label]) for label in CLASSES},
        'input_validation': {'status': 'PASS', 'missing_environmental_fields': missing,
                             'missing_policy': 'Saved fitted median imputer and missingness indicators only'},
        'model': {'artifact': ARTIFACT.relative_to(ROOT).as_posix(),
                  'sha256': metadata['model_sha256'], 'identity': metadata['selected_experiment'],
                  'inference_version': VERSION, 'training_versions': metadata['versions'],
                  'runtime_versions': {'python': sys.version.split()[0], 'sklearn': sklearn.__version__,
                                       'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__}},
        'scientific_limitation': metadata['scientific_limitation'],
    }
