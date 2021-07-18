import numpy as np
from scipy.stats import pearsonr

from brainscore.metrics import Score, Metric


class SignificantCorrelation(Metric):
    def __init__(self, x_coord, significance_threshold=0.05, ignore_nans=False):
        super(SignificantCorrelation, self).__init__()
        self.x_coord = x_coord
        self.significance_threshold = significance_threshold
        self.ignore_nans = ignore_nans

    def __call__(self, source, target):
        source_significant = self.is_significant(source)
        target_significant = self.is_significant(target)
        score = source_significant == target_significant
        return Score([score], coords={'aggregation': ['center']}, dims=['aggregation'])

    def is_significant(self, assembly):
        x = assembly[self.x_coord].values
        y = assembly.values
        if self.ignore_nans:
            nan = np.isnan(x) | np.isnan(y)
            x = x[~nan]
            y = y[~nan]
        r, p = pearsonr(x, y)
        return p < self.significance_threshold
