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
        source_significant_direction = self.significant_direction(source)
        target_significant_direction = self.significant_direction(target)
        score = source_significant_direction == target_significant_direction
        return Score([score], coords={'aggregation': ['center']}, dims=['aggregation'])

    def significant_direction(self, assembly):
        """
        Tests whether and in which direction the assembly's values are significantly correlated with the
        assembly's coordinate's values (`assembly[self.x_coord]`)
        :return: +1 if the correlation is significantly positive, -1 if the correlation is significantly negative,
          False otherwise
        """
        x = assembly[self.x_coord].values
        y = assembly.values
        if self.ignore_nans:
            nan = np.isnan(x) | np.isnan(y)
            x = x[~nan]
            y = y[~nan]
        r, p = pearsonr(x, y)
        if p >= self.significance_threshold:
            return False
        # at this point, we know the correlation is significant
        if r > 0:
            return +1
        else:
            return -1
