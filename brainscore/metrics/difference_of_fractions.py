import numpy as np

from brainscore.metrics import Metric, Score


class DifferenceOfFractions(Metric):
    def __init__(self, chance_performance, maximum_performance):
        """
        :param chance_performance: used for normalization
        :param maximum_performance: used for normalization
        """
        self._maximum_performance = maximum_performance
        self._chance_performance = chance_performance

    def __call__(self, assembly1, assembly2):
        """
        Computes difference of fractions score based on the similarity in delta performance of two assemblies.
        :param assembly1: An assembly with two values, indexable by
            `performance='perturbed'` and `performance='unperturbed'` (baseline)
        :param assembly2: Same format as assembly1
        :return: a :class:`~brainscore.metrics.Score` of how similar the drop in performance is between the two
            assemblies
        """
        frac1 = self.normalized_fraction(assembly1)
        frac2 = self.normalized_fraction(assembly2)
        score = 1 - np.abs(frac1 - frac2)
        # add meta
        score.attrs['assembly1'] = assembly1
        score.attrs['assembly2'] = assembly2
        return Score(score)

    def normalized_fraction(self, assembly):
        assembly = type(assembly)(assembly)  # re-index to make sure we can access all coordinates
        fraction_performance = assembly.sel(performance='perturbed').squeeze() / \
                               assembly.sel(performance='unperturbed').squeeze()
        normalized_fraction = (fraction_performance - self._chance_performance) / \
                              (self._maximum_performance - self._chance_performance)
        return normalized_fraction
