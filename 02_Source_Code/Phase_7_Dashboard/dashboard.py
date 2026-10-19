"""Read-only dashboard adapters. Predictions belong exclusively to Phase 6."""
import json
from pathlib import Path

import pandas as pd
from scripts import inference

ROOT = Path(__file__).resolve().parent
CLASSES = inference.CLASSES
PAGES = ('Overview', 'Prediction', 'Historical Data', 'Model Performance',
         'Dataset Information', 'About / Limitations')
LIMITATION = ('The current model predicts source-recorded abundance categories. '
              'It is not a validated future HAB-risk or early-warning model.')
HIGH_WARNING = ('Important limitation: performance for the rare high-abundance class is weak. '
                'This dashboard is a research demonstration and should not be used as an '
                'operational HAB early-warning or health-risk system.')
PROBABILITY_NOTE = ("The probabilities represent the model's estimated class probabilities for the supplied input. "
                    'They should not be interpreted as probability of future HAB occurrence or human-health risk.')
ENV_UNITS = {'WATER_TEMP': ('Temperature', 'degrees C'),
             'SALINITY': ('Salinity', 'ppt'), 'SAMPLE_DEPTH': ('Depth', 'm')}


def predict_for_dashboard(observation):
    """Return the Phase 6 result verbatim; its validator runs before prediction."""
    return inference.predict_observation(observation)


def load_artifacts():
    return {
        'phase5': json.loads((ROOT / 'reports/phase5_model_metadata.json').read_text(encoding='utf-8')),
        'phase4': json.loads((ROOT / 'reports/final_test_metrics.json').read_text(encoding='utf-8')),
        'inspection': json.loads((ROOT / 'docs/inspection_report.json').read_text(encoding='utf-8')),
        'comparison': pd.read_csv(ROOT / 'reports/phase5_vs_phase4.csv'),
        'importance': pd.read_csv(ROOT / 'reports/phase5_feature_importance.csv'),
    }


def load_history():
    """Join recorded context with existing QA-approved features; never fill blanks."""
    context = pd.read_csv(ROOT / 'data/processed/habsos_cleaned.csv',
                          usecols=['SOURCE_ROW_NUMBER','STATE_ID','SAMPLE_DATE_ISO','CATEGORY',
                                   'LATITUDE','LONGITUDE'])
    features = pd.read_csv(ROOT / 'data/ml/features.csv')
    target = pd.read_csv(ROOT / 'data/ml/target.csv')
    if not context.SOURCE_ROW_NUMBER.equals(features.SOURCE_ROW_NUMBER) or not context.SOURCE_ROW_NUMBER.equals(target.SOURCE_ROW_NUMBER):
        raise ValueError('Existing source/feature/target row lineage differs')
    context = context.merge(features.drop(columns=['LATITUDE','LONGITUDE']),
                            on='SOURCE_ROW_NUMBER', validate='one_to_one')
    context = context.merge(target, on='SOURCE_ROW_NUMBER', validate='one_to_one')
    context['recorded_date'] = pd.to_datetime(context.SAMPLE_DATE_ISO, format='%Y-%m-%d', errors='raise')
    return context


def genuine_example():
    """Return the already documented Phase 6 genuine row, not invented defaults."""
    report = json.loads((ROOT / 'reports/phase6_inference_report.json').read_text(encoding='utf-8'))
    return report['example']


def filter_history(frame, years, states):
    return frame.loc[frame.observation_year.between(*years) & frame.STATE_ID.isin(states)].copy()


def category_counts(frame):
    # Original recorded category, including rows ineligible for model evaluation.
    counts = frame.CATEGORY.value_counts().reindex(CLASSES, fill_value=0)
    return counts.rename_axis('Recorded category').reset_index(name='Observations')


def map_subset(frame, limit=4000):
    """Every ceil(N/limit)-th valid source row, in source order, for display only."""
    points = frame.dropna(subset=['LATITUDE','LONGITUDE'])
    stride = max(1, (len(points) + limit - 1) // limit)
    return points.iloc[::stride].copy(), stride
