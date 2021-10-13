import numpy as np

from brainscore.metrics import Metric, Score


class DifferenceOfFractions(Metric):
    def __call__(self, assembly1, assembly2):
        """
        Computes difference of fractions score based on the similarity in delta performance of two assemblies.
        :param assembly1: An assembly with two values, indexable by
            `performance='perturbed'` and `performance='unperturbed'` (baseline) as well as
            two `.attrs` `'chance_performance'` and `'maximum_performance'` to use for normalization
        :param assembly2: Same format as assembly1
        :return: a :class:`~brainscore.metrics.Score` of how similar the drop in performance is between the two
            assemblies
        """
        frac1 = self.normalized_fraction(assembly1)
        frac2 = self.normalized_fraction(assembly2)
        score = 1 - np.abs(frac1 - frac2)
        return Score(score)

    def normalized_fraction(self, assembly):
        assembly = type(assembly)(assembly)  # re-index to make sure we can access all coordinates
        fraction_performance = assembly.sel(performance='perturbed').squeeze() / \
                               assembly.sel(performance='unperturbed').squeeze()
        chance_performance = assembly.attrs['chance_performance']
        max_performance = assembly.attrs['maximum_performance']
        normalized_fraction = (fraction_performance - chance_performance) / (max_performance - chance_performance)
        return normalized_fraction
