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
        self._metric = metric

    def __call__(self, statistic):
        '''
        Applies given metric to dataset, comparing data from one animal to all remaining animals, i.e.:
        For each animal: metric({dataset\animal_i}, animal_i): cross validation like
        :param statistic: xarray structure with values & and corresponding meta information: distances, source
        :return: ceiling
        '''
        assert len(set(statistic.source.data)) > 1, 'your stats contain less than 2 animals'
        self.statistic = statistic

        scores = []
        for monkey in set(self.statistic.source.data):
            score = self._score_monkey(monkey)
            scores.append(score)

        return _aggregate(scores)

    def _score_monkey(self, monkey):
        monkey_pool = self.statistic.where(self.statistic.source != monkey, drop=True)
        held_out_monkey = self.statistic.sel(source=monkey)

        score = self._metric(monkey_pool, held_out_monkey)
        return score.data[0]
