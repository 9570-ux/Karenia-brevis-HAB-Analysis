"""Serializable Phase 5-only, unfitted calendar/feature-group transformer."""
import calendar
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ORIGINAL = ['WATER_TEMP', 'SALINITY', 'SAMPLE_DEPTH', 'LATITUDE', 'LONGITUDE',
            'observation_year', 'observation_month', 'day_of_year']
ENV = ['WATER_TEMP', 'SALINITY', 'SAMPLE_DEPTH']
SPACE = ['LATITUDE', 'LONGITUDE']
TIME = ['observation_year', 'observation_month', 'day_of_year']
CYCLIC = ['month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']
GROUPS = {'A_original8': ORIGINAL, 'B_temporal_spatial': SPACE + TIME,
          'C_environmental': ENV, 'D_environmental_temporal_spatial_cyclic': ENV + SPACE + CYCLIC}


class Phase5FeatureGroup(TransformerMixin, BaseEstimator):
    def __init__(self, feature_group='A_original8'):
        self.feature_group = feature_group

    def fit(self, X, y=None):
        if self.feature_group not in GROUPS or list(X.columns) != ORIGINAL:
            raise ValueError('Unsupported Phase 5 feature schema/group')
        self.feature_names_in_ = np.asarray(ORIGINAL, dtype=object)
        self.n_features_in_ = len(ORIGINAL)
        return self

    def transform(self, X):
        if list(X.columns) != ORIGINAL:
            raise ValueError('Input predictors differ from approved source allowlist')
        if self.feature_group != 'D_environmental_temporal_spatial_cyclic':
            return X.loc[:, GROUPS[self.feature_group]].copy()
        result = X.loc[:, ENV + SPACE].copy()
        days = np.asarray([366 if calendar.isleap(int(year)) else 365 for year in X.observation_year])
        month_angle = 2 * np.pi * (X.observation_month.to_numpy() - 1) / 12
        day_angle = 2 * np.pi * (X.day_of_year.to_numpy() - 1) / days
        result['month_sin'], result['month_cos'] = np.sin(month_angle), np.cos(month_angle)
        result['day_of_year_sin'], result['day_of_year_cos'] = np.sin(day_angle), np.cos(day_angle)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(GROUPS[self.feature_group], dtype=object)
