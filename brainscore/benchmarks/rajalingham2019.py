import itertools
import logging
import numpy as np
import xarray as xr
import pandas as pd
import warnings
from numpy.random import RandomState
from pandas import DataFrame
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr
from tqdm import tqdm
from xarray import DataArray

import brainscore
from model_tools.brain_transformation.tissue.neural_perturbation import MuscimolInjection
from brainio.assemblies import merge_data_arrays, walk_coords, DataAssembly, array_is_element
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.behavior_differences import DeficitPredictionTask, DeficitPredictionObject
from brainscore.metrics.difference_of_correlations import DifferenceOfCorrelations
from brainscore.metrics.image_level_behavior import _o2
from brainscore.metrics.inter_individual_stats_ceiling import InterIndividualStatisticsCeiling
from brainscore.metrics.significant_match import SignificantCorrelation
from brainscore.metrics.spatial_correlation import SpatialCorrelationSimilarity
from brainscore.metrics.transformations import CrossValidation
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname, LazyLoad
from packaging.rajalingham2019 import collect_assembly

TASK_LOOKUP = {
    'dog': 'Dog',
    # 'face0': '',
    # 'table4': '',
    'bear': 'Bear',
    # 'apple': '',
    'elephant': 'Elephant',
    'airplane3': 'Plane',
    # 'turtle': '',
    # 'car_alfa': '',
    'chair0': 'Chair'
}

BIBTEX = """@article{RAJALINGHAM2019493,
                title = {Reversible Inactivation of Different Millimeter-Scale Regions of Primate IT Results in Different Patterns of Core Object Recognition Deficits},
                journal = {Neuron},
                volume = {102},
                number = {2},
                pages = {493-505.e5},
                year = {2019},
                issn = {0896-6273},
                doi = {https://doi.org/10.1016/j.neuron.2019.02.001},
                url = {https://www.sciencedirect.com/science/article/pii/S0896627319301102},
                author = {Rishi Rajalingham and James J. DiCarlo},
                keywords = {object recognition, neural perturbation, inactivation, vision, primate, inferior temporal cortex},
                abstract = {Extensive research suggests that the inferior temporal (IT) population supports visual object recognition behavior. However, causal evidence for this hypothesis has been equivocal, particularly beyond the specific case of face-selective subregions of IT. Here, we directly tested this hypothesis by pharmacologically inactivating individual, millimeter-scale subregions of IT while monkeys performed several core object recognition subtasks, interleaved trial-by trial. First, we observed that IT inactivation resulted in reliable contralateral-biased subtask-selective behavioral deficits. Moreover, inactivating different IT subregions resulted in different patterns of subtask deficits, predicted by each subregion’s neuronal object discriminability. Finally, the similarity between different inactivation effects was tightly related to the anatomical distance between corresponding inactivation sites. Taken together, these results provide direct evidence that the IT cortex causally supports general core object recognition and that the underlying IT coding dimensions are topographically organized.}
                }"""

# "Each inactivation session began with a single focal microinjection of 1ml of muscimol
# (5mg/mL, Sigma Aldrich) at a slow rate (100nl/min) via a 30-gauge stainless-steel cannula at
# the targeted site in ventral IT."
MUSCIMOL_PARAMETERS = {
    'amount_microliter': 1
}


class _Rajalingham2019(BenchmarkBase):
    def __init__(self, identifier, metric, ceiling_func=None, num_sites=9):
        self._target_assembly = collect_assembly()
        self._training_stimuli = brainscore.get_stimulus_set('dicarlo.hvm')
        self._training_stimuli['image_label'] = self._training_stimuli['object_name']
        # use only those images where it's the same object (label)
        self._training_stimuli = self._training_stimuli[self._training_stimuli['object_name'].isin(
            self._target_assembly.stimulus_set['object_name'])]
        self._test_stimuli = self._target_assembly.stimulus_set

        self._num_sites = num_sites
        self._metric = metric
        self._logger = logging.getLogger(fullname(self))
        super(_Rajalingham2019, self).__init__(
            identifier=identifier,
            ceiling_func=ceiling_func,
            version=1, parent='IT',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        # TODO: Both animals were previously trained on other images of other objects, and were proficient in
        #  discriminating among over 35 arbitrarily sampled basic-level object categories
        training_stimuli = self._training_stimuli
        candidate.start_task(task=BrainModel.Task.probabilities, fitting_stimuli=training_stimuli)

        # "[...] inactivation sessions were interleaved over days with control behavioral sessions.
        # Thus, each inactivation experiment consisted of three behavioral sessions:
        # the baseline or pre-control session (1 day prior to injection),
        # the inactivation session,
        # and the recovery or post-control session (2 days after injection)"
        # --> we here front-load one control session and then run many inactivation sessions
        # control
        unperturbed_behavior = self._perform_task_unperturbed(candidate)

        # silencing sessions
        behaviors = [unperturbed_behavior]
        # "We varied the location of microinjections to randomly sample the ventral surface of IT
        # (from approximately + 8mm AP to approx + 20mm AP)."
        # stay between [0, 10] since that is the extent of the tissue

        for site, injection_location in enumerate(self._sample_injection_locations()):
            perturbation_parameters = {**MUSCIMOL_PARAMETERS,
                                       **{'location': injection_location}}
            perturbed_behavior = self._perform_task_perturbed(candidate,
                                                              perturbation_parameters=perturbation_parameters,
                                                              site_number=site)
            behaviors.append(perturbed_behavior)

        behaviors = merge_data_arrays(behaviors)
        behaviors = self.align_task_names(behaviors)

        score = self._metric(behaviors, self._target_assembly)
        return score

    def _perform_task_unperturbed(self, candidate: BrainModel):
        candidate.perturb(perturbation=None, target='IT')  # reset
        behavior = candidate.look_at(self._test_stimuli, number_of_trials=None)
        behavior = behavior.expand_dims('injected')
        behavior['injected'] = [False]

        return behavior

    def _perform_task_perturbed(self, candidate: BrainModel, perturbation_parameters, site_number):
        candidate.perturb(perturbation=None, target='IT')  # reset
        candidate.perturb(perturbation=BrainModel.Perturbation.muscimol,
                          target='IT',
                          perturbation_parameters=perturbation_parameters)
        behavior = candidate.look_at(self._test_stimuli)

        behavior = behavior.expand_dims('injected').expand_dims('site')
        behavior['injected'] = [True]
        behavior['site_iteration'] = 'site', [site_number]
        behavior['site_x'] = 'site', [perturbation_parameters['location'][0]]
        behavior['site_y'] = 'site', [perturbation_parameters['location'][1]]
        behavior['site_z'] = 'site', [0]
        behavior = type(behavior)(behavior)  # make sure site and injected are indexed

        return behavior

    def _sample_injection_locations(self):
        border_area = MuscimolInjection()._cov * 1  # TODO
        injection_locations = np.random.rand(self._num_sites * 10) * (10 - border_area)
        injection_locations = injection_locations[injection_locations > border_area]
        injection_locations = injection_locations[:self._num_sites * 2]
        injection_locations = injection_locations.reshape((self._num_sites, 2))
        return injection_locations

    @staticmethod
    def align_task_names(behaviors):
        behaviors = type(behaviors)(behaviors.values, coords={
            coord: (dims, values if coord not in ['object_name', 'truth', 'image_label', 'choice']
            else [TASK_LOOKUP[name] if name in TASK_LOOKUP else name for name in behaviors[coord].values])
            for coord, dims, values in walk_coords(behaviors)},
                                    dims=behaviors.dims)
        return behaviors

    @staticmethod
    def sample_grid_points(low, high, num_x, num_y):
        assert len(low) == len(high) == 2
        grid_x, grid_y = np.meshgrid(np.linspace(low[0], high[0], num_x),
                                     np.linspace(low[1], high[1], num_y))
        return np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)  # , np.zeros(num_x * num_y) for empty z dimension

    def sample_points(self, low, high, num):
        assert len(low) == len(high) == 2
        random_state = RandomState(0)
        points_x = random_state.uniform(low=low[0], high=high[0], size=num)
        points_y = random_state.uniform(low=low[1], high=high[1], size=num)
        return np.stack((points_x, points_y), axis=1)


def Rajalingham2019DeficitsSignificant():
    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-deficits_significant',
                            metric=SpatialCharacterizationMetric(SignificantCorrelation(x_coord='distance')))


def Rajalingham2019SpatialDeficits():
    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-spatial_deficit_similarity',
                            metric=SpatialCharacterizationMetric(
                                DifferenceOfCorrelations(correlation_variable='distance')))


def Rajalingham2019DeficitPredictionTask():
    metric = DeficitPredictionTask()
    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-deficit_prediction_task',
                            # num_sites=100,  # TODO
                            metric=metric,
                            ceiling_func=None,  # TODO
                            )


def Rajalingham2019DeficitPredictionObject():
    metric = DeficitPredictionObject()
    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-deficit_prediction_object',
                            # num_sites=100,  # TODO
                            metric=metric)


def Rajalingham2019DeficitsSignificant():
    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-deficits_significant',
                            metric=SpatialCharacterizationMetric())


def Rajalingham2019SpatialDeficits():
    return _Rajalingham2019(identifier='dicarlo.Rajalingham2019-spatial_deficit_similarity',
                            metric=DifferenceOfCorrelations(correlation_variable='distance'))


def DicarloRajalingham2019SpatialDeficitsQuantified():
    def inv_ks_similarity(p, q):
        '''
        Inverted ks similarity -> resulting in a score within [0,1], 1 being a perfect match
        '''
        import scipy.stats
        return 1 - scipy.stats.ks_2samp(p, q)[0]

    similarity_metric = SpatialCorrelationSimilarity(similarity_function=inv_ks_similarity,
                                                     bin_size_mm=.8)  # arbitrary bin size
    metric = SpatialCharacterizationMetric()
    metric._similarity_metric = similarity_metric
    benchmark = _Rajalingham2019(identifier='dicarlo.Rajalingham2019.IT-spatial_deficit_similarity_quantified',
                                 metric=metric)

    # TODO really messy solution | only works after benchmark metric has been called once
    benchmark._ceiling_func = lambda: InterIndividualStatisticsCeiling(similarity_metric)(
        benchmark._metric._similarity_metric.target_statistic)
    return benchmark


class SpatialCharacterizationMetric:
    def __init__(self, similarity_metric):
        self._similarity_metric = similarity_metric

    def __call__(self, behaviors, target):
        dprime_assembly_all = self.characterize(behaviors)
        dprime_assembly = self.subselect_tasks(dprime_assembly_all, target)
        candidate_assembly = dprime_assembly.transpose('injected', 'site', 'task')  # match target assembly shape
        candidate_statistic = self.compute_response_deficit_distance_candidate(candidate_assembly)
        target_statistic = self.compute_response_deficit_distance_target(target)

        score = self._similarity_metric(candidate_statistic, target_statistic)
        score.attrs['candidate_behaviors'] = behaviors
        score.attrs['candidate_assembly'] = candidate_assembly
        return score

    def compute_response_deficit_distance_target(self, target_assembly):
        dprime_assembly = target_assembly.mean('bootstrap')

        statistics_list = []
        for monkey in set(dprime_assembly.monkey.data):
            # for array in set(dprime_assembly.array.data):
            sub_assembly = dprime_assembly.sel(monkey=monkey)  # , array=array)
            distances, correlations = self._compute_response_deficit_distance(sub_assembly)
            mask = np.triu_indices(sub_assembly.site.size)

            statistics_list.append(
                self.to_xarray(correlations[mask], distances[mask], source=monkey))  # , array=array))

        return xr.concat(statistics_list, dim='meta')

    def compute_response_deficit_distance_candidate(self, dprime_assembly):
        distances, correlations = self._compute_response_deficit_distance(dprime_assembly)
        mask = np.triu_indices(dprime_assembly.site.size)

        return self.to_xarray(correlations[mask], distances[mask])

    def _compute_response_deficit_distance(self, dprime_assembly):
        '''
        :param dprime_assembly: assembly of behavioral performance
        :return: square matrices with correlation and distance values; each matrix elem == value between site_i, site_j
        '''
        distances = self.pairwise_distances(dprime_assembly)

        behavioral_differences = self.compute_differences(dprime_assembly)
        # dealing with nan values while correlating; not np.ma.corrcoef: https://github.com/numpy/numpy/issues/15601
        correlations = DataFrame(behavioral_differences.data).T.corr().values

        return distances, correlations

    @staticmethod
    def to_xarray(correlations, distances, source='model', array=None):
        '''
        :param values: list of data values
        :param distances: list of distance values, each distance value has to correspond to one data value
        :param source: name of monkey
        :param array: name of recording array
        '''
        xarray_statistic = DataArray(
            data=correlations,
            dims=["meta"],
            coords={
                'meta': pd.MultiIndex.from_product([distances, [source], [array]],
                                                   names=('distances', 'source', 'array'))
            }
        )

        return xarray_statistic

    @staticmethod
    def pairwise_distances(dprime_assembly):
        locations = np.stack([dprime_assembly.site.site_x.data,
                              dprime_assembly.site.site_y.data,
                              dprime_assembly.site.site_z.data]).T

        return squareform(pdist(locations, metric='euclidean'))

    @property
    def ceiling(self):
        split1, split2 = self._target_assembly.sel(split=0), self._target_assembly.sel(split=1)
        split1_diffs = split1.sel(silenced=False) - split1.sel(silenced=True)
        split2_diffs = split2.sel(silenced=False) - split2.sel(silenced=True)
        split_correlation, p = pearsonr(split1_diffs.values.flatten(), split2_diffs.values.flatten())
        return Score([split_correlation], coords={'aggregation': ['center']}, dims=['aggregation'])

    @staticmethod
    def characterize(assembly):
        """ compute per-task performance from `presentation x choice` assembly """
        # xarray can't do multi-dimensional grouping, do things manually
        o2s = []
        adjacent_values = assembly['injected'].values, assembly['site'].values
        # TODO: this takes 2min (4.5 in debug)
        for injected, site in tqdm(itertools.product(*adjacent_values), desc='characterize',
                                   total=np.prod([len(values) for values in adjacent_values])):
            current_assembly = assembly.sel(injected=injected, site=site)
            o2 = _o2(current_assembly)
            o2 = o2.expand_dims('injected').expand_dims('site')
            o2['injected'] = [injected]
            for (coord, _, _), value in zip(walk_coords(assembly['site']), site):
                o2[coord] = 'site', [value]
            o2 = DataAssembly(o2)  # ensure multi-index on site
            o2s.append(o2)
        o2s = merge_data_arrays(o2s)  # this only takes ~1s, ok
        return o2s

    @staticmethod
    def subselect_tasks(assembly, reference_assembly):
        tasks_left, tasks_right = reference_assembly['task_left'].values, reference_assembly['task_right'].values
        task_values = [assembly.sel(task_left=task_left, task_right=task_right).values
                       for task_left, task_right in zip(tasks_left, tasks_right)]
        task_values = type(assembly)(task_values, coords=
        {**{
            'task_number': ('task', reference_assembly['task_number'].values),
            'task_left': ('task', tasks_left),
            'task_right': ('task', tasks_right),
        }, **{coord: (dims, values) for coord, dims, values in walk_coords(assembly)
              if not any(array_is_element(dims, dim) for dim in ['task_left', 'task_right'])}
         }, dims=['task'] + [dim for dim in assembly.dims if
                             dim not in ['task_left', 'task_right']])
        return task_values

    @classmethod
    def apply_site(cls, source_assembly, site_target_assembly):
        site_target_assembly = site_target_assembly.squeeze('site')
        np.testing.assert_array_equal(source_assembly.sortby('task_number')['task_left'].values,
                                      site_target_assembly.sortby('task_number')['task_left'].values)
        np.testing.assert_array_equal(source_assembly.sortby('task_number')['task_right'].values,
                                      site_target_assembly.sortby('task_number')['task_right'].values)

        # filter non-nan task measurements from target
        nonnan_tasks = site_target_assembly['task'][~site_target_assembly.isnull()]
        if len(nonnan_tasks) < len(site_target_assembly):
            warnings.warn(f"Ignoring tasks {site_target_assembly['task'][~site_target_assembly.isnull()].values}")
        site_target_assembly = site_target_assembly.sel(task=nonnan_tasks)
        source_assembly = source_assembly.sel(task=nonnan_tasks.values)

        # try to predict from model
        task_split = CrossValidation(split_coord='task_number', stratification_coord=None,
                                     kfold=True, splits=len(site_target_assembly['task']))
        task_scores = task_split(source_assembly, site_target_assembly, apply=cls.apply_task)
        task_scores = task_scores.raw
        correlation, p = pearsonr(task_scores.sel(type='source'), task_scores.sel(type='target'))
        score = Score([correlation, p], coords={'statistic': ['r', 'p']}, dims=['statistic'])
        score.attrs['predictions'] = task_scores.sel(type='source')
        score.attrs['target'] = task_scores.sel(type='target')
        return score

    @staticmethod
    def apply_task(source_train, target_train, source_test, target_test):
        """
        finds the best-matching site in the source train assembly to predict the task effects in the test target.
        :param source_train: source assembly for mapping with t tasks and n sites
        :param target_train: target assembly for mapping with t tasks
        :param source_test: source assembly for testing with 1 task and n sites
        :param target_test: target assembly for testing with 1 task
        :return: a pair
        """
        # deal with xarray bug
        source_train, source_test = deal_with_xarray_bug(source_train), deal_with_xarray_bug(source_test)
        # map: find site in assembly1 that best matches mapping tasks
        correlations = {}
        for site in source_train['site'].values:
            source_site = source_train.sel(site=site)
            np.testing.assert_array_equal(source_site['task'].values, target_train['task'].values)
            correlation, p = pearsonr(source_site, target_train)
            correlations[site] = correlation
        best_site = [site for site, correlation in correlations.items() if correlation == max(correlations.values())]
        best_site = best_site[0]  # choose first one if there are multiple
        # test: predictivity of held-out task.
        # We can only collect the single prediction here and then correlate in outside loop
        source_test = source_test.sel(site=best_site)
        np.testing.assert_array_equal(source_test['task'].values, target_test['task'].values)
        pair = type(target_test)([source_test.values[0], target_test.values[0]],
                                 coords={  # 'task': source_test['task'].values,
                                     'type': ['source', 'target']},
                                 dims=['type'])  # , 'task'
        return pair

    @staticmethod
    def compute_differences(behaviors):
        """
        :param behaviors: an assembly with a dimension `injected` and values `[True, False]`
        :return: the difference between these two conditions (injected - control)
        """
        return behaviors.sel(injected=True) - behaviors.sel(injected=False)


def deal_with_xarray_bug(assembly):
    if hasattr(assembly, 'site_level_0'):
        return type(assembly)(assembly.values, coords={
            coord: (dim, values) for coord, dim, values in walk_coords(assembly) if coord != 'site_level_0'},
                              dims=assembly.dims)
