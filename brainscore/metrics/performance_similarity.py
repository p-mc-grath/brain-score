from brainscore.metrics import Score, Metric
from brainio.assemblies import DataAssembly
import numpy as np
import itertools
from scipy.stats import fisher_exact


class PerformanceSimilarity(Metric):
    def __call__(self, candidate: DataAssembly, target: DataAssembly):
        '''
        We used Fisher’s exact tests to determine the significance of microstimulation on
        behavioral performance22. This test, performed on the contingency table of correct
        and incorrect responses for same- and different-identity trials per object category
        (depending on the experiment) and per microstimulation condition, returns the
        probability of erroneously assuming differences between columns.
        Unlike the chi-squared test, Fisher’s exact test works with small, sparse or
        unbalanced data as encountered when the performance reaches 100%. Since we only
        performed one test per trial type (i.e., same- or different-iden- tity), multiple
        comparison corrections were neither required nor used. Fisher’s exact test is
        nonparametric and does not assume normality or equal variances

        :param target, candidate:
            values: list of contingency matrices; rows: Hit count, Miss count; columns: not stimulatedm sitmulated
            dims:   condition_level_0 : {same_id, different_id}
                    statistics  : {Hits, Misses}
                    stimulation : {0, 300}
            coords: object_name  : list of strings, category
        :return score: average over RAW values
            RAW: 1 if significance direction in the biological data matches that in the model data, else 0
        '''
        alpha = 0.05
        same_effect, significances = [], []
        for object_name, condition in itertools.product(set(target.object_name.values),
                                                        set(target.condition_level_0.values)):

            significance, change_direction = [], []
            for data in [target, candidate]:
                contingency_matrix = data.sel(object_name=object_name, condition_level_0=condition).data
                significance.append(fisher_exact(contingency_matrix)[1])
                change_direction.append(_compute_effect_direction(contingency_matrix))

            matching_significant_effect = (significance[0] < alpha) == (significance[1] < alpha)
            matching_effect_direction = change_direction[0] == change_direction[1]
            if matching_significant_effect and significance[0] > alpha:
                effect = matching_effect_direction  # in case not significant, direction does not matter
            else:
                effect = matching_significant_effect and matching_effect_direction
            same_effect.append(effect)
            significances += [(object_name, condition, {'target': significance[0], 'candidate': significance[1]})]

        center = np.mean(same_effect)
        error = np.std(same_effect)
        score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))
        score.attrs[Score.RAW_VALUES_KEY] = same_effect
        score.attrs['significances'] = significances
        return score


def _compute_effect_direction(contingency_matrix):
    accuracy_nonstimulated, accuracy_stimulated = _compute_accuracies(contingency_matrix)
    if accuracy_nonstimulated - accuracy_stimulated > 0:
        return 1
    elif accuracy_nonstimulated - accuracy_stimulated < 0:
        return -1
    else:
        return 0


def _compute_accuracies(contingency_matrix):
    hits, hits_stimulated = contingency_matrix[0][0], contingency_matrix[0][1]
    misses, misses_stimulated = contingency_matrix[1][0], contingency_matrix[1][1]
    accuracy_nonstimulated = hits / (hits + misses)
    accuracy_stimulated = hits_stimulated / (hits_stimulated + misses_stimulated)
    return accuracy_nonstimulated, accuracy_stimulated
