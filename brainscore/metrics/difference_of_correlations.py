import numpy as np
from scipy.stats import pearsonr

from brainscore.metrics import Metric, Score


class DifferenceOfCorrelations(Metric):
    def __init__(self, correlation_variable):
        """
        :param correlation_variable: the variable that assembly's values are correlated with
        """
        self._correlation_variable = correlation_variable

    def __call__(self, assembly1, assembly2):
        """
        Computes difference of correlations score based on the similarity of two correlation values.
        :param assembly1: A single-dimensional assembly with at least one coordinate `self._correlation_variable`,
            as set in the `__init__`, to correlate the assembly's values with
        :param assembly2: Same format as assembly1
        :return: a :class:`~brainscore.metrics.Score` of how similar the correlations are between the two assemblies
        """
        correlation1, p1 = pearsonr(assembly1[self._correlation_variable], assembly1.values)
        correlation2, p2 = pearsonr(assembly2[self._correlation_variable], assembly2.values)
        score = 1 - np.abs(correlation1 - correlation2)
        return Score(score)
