from brainscore.metrics.ceiling import Ceiling, Score
import numpy as np


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


class InterIndividualStatisticsCeiling(Ceiling):
    '''
    Cross-validation-like, animal-wise computation of ceiling
    '''

    def __init__(self, metric):
        '''
        :param metric: used to compute the ceiling
        '''
        self.metric = metric

    def __call__(self, target_statistic):
        '''
        Applies given metric to dataset, comparing data from one animal to all remaining animals, i.e.:
        For each animal: metric({dataset\animal_i}, animal_i)
        :param target_statistic: list of n lists, every second element expected to contain data from another animal
        :return: ceiling
        '''
        scores = []
        for held_out in range(0, len(target_statistic) - 1, 2):
            score = self.metric(np.hstack(target_statistic[:held_out] + target_statistic[held_out + 2:]),
                                np.hstack(target_statistic[held_out:held_out + 2]))
            scores.append(score.data[0])
        return _aggregate(scores)
