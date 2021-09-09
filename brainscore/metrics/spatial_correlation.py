import numpy as np
from brainscore.metrics import Score, Metric


def _aggregate(scores):
    '''
    Aggregates list of values into Score object
    :param scores: list of values assumed to be scores
    :return: Score object | where score['center'] = mean(scores) and score['error'] = std(scores)
    '''
    center = np.median(scores)
    error = np.median(np.absolute(scores - np.median(scores)))  # MAD
    aggregate_score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))
    aggregate_score.attrs[Score.RAW_VALUES_KEY] = scores

    return aggregate_score


class SpatialCorrelationSimilarity(Metric):
    '''
    Computes the similarity of two given distributions using a given similarity_function. The similarity_function is
    applied to each bin which in turn are created based on a given bin size and the independent variable of the
    distributions
    '''

    def __init__(self, similarity_function, bin_size_mm) -> object:
        '''
        :param similarity_function: similarity_function to be applied to each bin
        :param bin_size_mm: size per bin in mm | one fixed size, utilize Score.RAW_VALUES_KEY to change weighting
        '''
        self.similarity_function = similarity_function
        self.bin_size = bin_size_mm

    def __call__(self, target_statistic, candidate_statistic):
        '''
        :param target_statistic: list of 2 lists, [0] distances -> binning over this, [1] correlation per distance value
        :param candidate_statistic: list of 2 lists, [0] distances -> binning over this, [1] correlation per distance value
        '''
        self.target_statistic = target_statistic
        self.candidate_statistic = candidate_statistic
        self._bin_min = np.min(self.target_statistic.distances)
        self._bin_max = np.max(self.target_statistic.distances)

        bin_scores = []
        for in_bin_t, in_bin_c, enough_data in self._bin_masks():
            if enough_data:
                bin_scores.append(self.similarity_function(self.target_statistic.values[in_bin_t],
                                                           self.candidate_statistic.values[in_bin_c]))

        return _aggregate(bin_scores)

    def _bin_masks(self):
        '''
        Generator: Yields masks indexing which elements are within each bin.
        :yield: mask(target, current_bin), mask(candidate, current_bin), enough data in the bins to do further computations
        '''
        for lower_bound_mm in np.linspace(self._bin_min, self._bin_max,
                                          int(self._bin_max * (1 / self.bin_size) + 1) * 2):
            t = np.where(np.logical_and(self.target_statistic.distances >= lower_bound_mm,
                                        self.target_statistic.distances < lower_bound_mm + self.bin_size))[0]
            c = np.where(np.logical_and(self.candidate_statistic.distances >= lower_bound_mm,
                                        self.candidate_statistic.distances < lower_bound_mm + self.bin_size))[0]
            enough_data = t.size > 0 and c.size > 0  # random threshold

            yield t, c, enough_data
