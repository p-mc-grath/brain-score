import itertools
from warnings import warn
import sys
from pathlib import Path

# TODO figure out why the below is necessary: else cant import StoreSourceAssembly from dir_caus.model -> not actucally the problem
import direct_causality
from direct_causality import data
from direct_causality import model

import numpy as np
import scipy.io
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr
from pandas import DataFrame
from tqdm import tqdm

import brainscore
from brainscore.utils import LazyLoad
from brainscore.metrics.spatial_correlation import SpatialCorrelationSimilarity
from brainscore.metrics import Score
from brainscore.metrics.image_level_behavior import _o2
from brainscore.metrics.transformations import CrossValidation
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.majajhong2015 import DicarloMajajHong2015ITSpatialCorrelation
from brainscore.model_interface import BrainModel
from brainio_base.assemblies import merge_data_arrays, walk_coords, array_is_element, DataAssembly

pairwise_distance = DicarloMajajHong2015ITSpatialCorrelation.pairwise_distances
inv_ks_similarity = DicarloMajajHong2015ITSpatialCorrelation.inv_ks_similarity

BIBTEX = '''@article{RAJALINGHAM2019493,
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
            abstract = {Summary
            Extensive research suggests that the inferior temporal (IT) population supports visual object recognition behavior. However, causal evidence for this hypothesis has been equivocal, particularly beyond the specific case of face-selective subregions of IT. Here, we directly tested this hypothesis by pharmacologically inactivating individual, millimeter-scale subregions of IT while monkeys performed several core object recognition subtasks, interleaved trial-by trial. First, we observed that IT inactivation resulted in reliable contralateral-biased subtask-selective behavioral deficits. Moreover, inactivating different IT subregions resulted in different patterns of subtask deficits, predicted by each subregion’s neuronal object discriminability. Finally, the similarity between different inactivation effects was tightly related to the anatomical distance between corresponding inactivation sites. Taken together, these results provide direct evidence that the IT cortex causally supports general core object recognition and that the underlying IT coding dimensions are topographically organized.}
            }'''

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


class DicarloRajalingham2019SpatialDeficits(BenchmarkBase):

    def __init__(self):
        super().__init__(identifier='dicarlo.Rajalingham2019.IT-spatial_deficit',
                         ceiling_func=lambda: None,
                         version=0.1,
                         parent='IT',
                         bibtex=BIBTEX)
        self._target_assembly = self._load_assembly()
        self._stimulus_set = self._target_assembly.stimulus_set
        self._target_statistic = LazyLoad(self.compute_response_deficit_distance_target)
        self._score = SpatialCorrelationSimilarity(similarity_function=inv_ks_similarity,
                                                   bin_size_mm=.1)  # .1 mm is an arbitrary choice

        self.perturbation = {'type': BrainModel.Perturbation.muscimol,
                             'target': 'IT',
                             'perturbation_parameters': {'amount_microliter': 1,
                                                         'location': None}}

    def __call__(self, candidate: BrainModel):
        candidate_assembly = self.create_dprime_assembly(candidate)
        candidate_statistic = self.compute_response_deficit_distance_candidate(candidate_assembly)

        score = self._score(self._target_statistic, candidate_statistic)
        score.attrs['target_statistic'] = self._target_statistic
        score.attrs['candidate_statistic'] = candidate_statistic

        return score

    def create_dprime_assembly(self, candidate: BrainModel):
        training_stimuli = self._stimulus_set  # TODO: true?
        candidate.start_task(task=BrainModel.Task.probabilities, fitting_stimuli=training_stimuli)
        unperturbed_behavior = self.perform_task(candidate, perturbation=None)

        behaviors = [unperturbed_behavior]
        injection_locations = self.sample_grid_points([2, 2], [8, 8], num_x=4, num_y=4)
        for site, injection_location in enumerate(injection_locations):
            perturbation = self.perturbation
            perturbation['perturbation_parameters']['location'] = injection_location
            perturbation['site_number'] = site

            perturbed_behavior = self.perform_task(candidate, perturbation=perturbation)
            behaviors.append(perturbed_behavior)

        behaviors = merge_data_arrays(behaviors)
        behaviors = self.align_task_names(behaviors)

        dprime_assembly_all = self.characterize(behaviors)
        dprime_assembly = self.subselect_tasks(dprime_assembly_all, self._target_assembly)

        return dprime_assembly.transpose('injected', 'site', 'task')  # match target assembly shape

    def perform_task(self, candidate: BrainModel, perturbation):
        if perturbation is None:
            return self._perform_task_unperturbed(candidate)
        else:
            return self._perform_task_perturbed(candidate, perturbation)

    def _perform_task_unperturbed(self, candidate: BrainModel):
        candidate.perturb(perturbation=None, target='IT')  # reset
        behavior = candidate.look_at(self._stimulus_set, number_of_trials=None)
        behavior = behavior.expand_dims('injected')
        behavior['injected'] = [False]

        return behavior

    def _perform_task_perturbed(self, candidate: BrainModel, perturbation):
        candidate.perturb(perturbation=None, target='IT')  # reset
        candidate.perturb(perturbation=perturbation['type'],
                          target=perturbation['target'],
                          perturbation_parameters=perturbation['perturbation_parameters'])
        behavior = candidate.look_at(self._stimulus_set)

        behavior = behavior.expand_dims('injected').expand_dims('site')
        behavior['injected'] = [True]
        behavior['site_iteration'] = 'site', [perturbation['site_number']]
        behavior['site_x'] = 'site', [perturbation['perturbation_parameters']['location'][0]]
        behavior['site_y'] = 'site', [perturbation['perturbation_parameters']['location'][1]]
        behavior['site_z'] = 'site', [0]  # [perturbation['perturbation_parameters']['location'][2]]
        behavior = type(behavior)(behavior)  # make sure site and injected are indexed

        return behavior

    def compute_response_deficit_distance_target(self):
        dprime_assembly = self._target_assembly.mean('bootstrap',
                                                     skipna=True)  # obviously skipna no effect

        mask = np.full((dprime_assembly.site.size, dprime_assembly.site.size), False)
        for i in range(len(mask)):
            if i < 10:
                mask[i, i:10] = True  # monkey 1, task 1, upper triangle
            elif i < 17:
                mask[i, i:17] = True  # monkey 2, task 1, upper triangle
            else:
                mask[i, i:] = True  # monkey 2, task 2, upper triangle

        return self._compute_response_deficit_distance(dprime_assembly, mask)

    def compute_response_deficit_distance_candidate(self, dprime_assembly):
        mask = np.triu_indices(dprime_assembly.site.size)

        return self._compute_response_deficit_distance(dprime_assembly, mask)

    def _compute_response_deficit_distance(self, dprime_assembly, mask):
        distances = self.pairwise_distances(dprime_assembly)

        behavioral_differences = self.compute_differences(dprime_assembly)
        # dealing with nan values while correlating; not np.ma.corrcoef: https://github.com/numpy/numpy/issues/15601
        correlations = DataFrame(behavioral_differences.data).T.corr().values

        statistic = np.array([distances[mask], correlations[mask]])

        return statistic

    @staticmethod
    def pairwise_distances(dprime_assembly):
        locations = np.stack([dprime_assembly.site.site_x.data,
                              dprime_assembly.site.site_y.data,
                              dprime_assembly.site.site_z.data]).T

        return squareform(pdist(locations, metric='euclidean'))

    @staticmethod
    def align_task_names(behaviors):
        behaviors = type(behaviors)(behaviors.values, coords={
            coord: (dims, values if coord not in ['object_name', 'truth', 'image_label', 'choice']
            else [TASK_LOOKUP[name] if name in TASK_LOOKUP else name for name in behaviors[coord].values])
            for coord, dims, values in walk_coords(behaviors)},
                                    dims=behaviors.dims)
        return behaviors

    def _load_assembly(self, contra_hemisphere=True):
        """
        :param contra_hemisphere: whether to only select data and associated stimuli
            where the target object was contralateral to the injection hemisphere
        """
        path = Path(__file__).parent / 'Rajalingham2019_data_summary.mat'
        data = scipy.io.loadmat(path)['data_summary']
        struct = {d[0]: v for d, v in zip(data.dtype.descr, data[0, 0])}
        tasks = [v[0] for v in struct['O2_task_names'][:, 0]]
        tasks_left, tasks_right = zip(*[task.split(' vs. ') for task in tasks])
        k1 = {d[0]: v for d, v in zip(struct['k1'].dtype.descr, struct['k1'][0, 0])}

        class missing_dict(dict):
            def __missing__(self, key):
                return key

        dim_replace = missing_dict({'sub_metric': 'hemisphere', 'nboot': 'bootstrap', 'exp': 'site',
                                    'niter': 'trial_split', 'subj': 'subject'})
        condition_replace = {'ctrl': 'saline', 'inj': 'muscimol'}
        dims = [dim_replace[v[0]] for v in k1['dim_labels'][0]]
        subjects = [v[0] for v in k1['subjs'][0]]
        conditions, subjects = zip(*[subject.split('_') for subject in subjects])
        metrics = [v[0] for v in k1['metrics'][0]]
        assembly = DataAssembly([k1['D0'], k1['D1']],
                                coords={
                                    'injected': [True, False],
                                    'injection': (dim_replace['subj'], [condition_replace[c] for c in conditions]),
                                    'subject_id': (dim_replace['subj'], list(subjects)),
                                    'metric': metrics,
                                    dim_replace['sub_metric']: ['all', 'ipsi', 'contra'],
                                    # autofill
                                    dim_replace['niter']: np.arange(k1['D0'].shape[3]),
                                    dim_replace['k']: np.arange(k1['D0'].shape[4]),
                                    dim_replace['nboot']: np.arange(k1['D0'].shape[5]),
                                    'site_number': ('site', np.arange(k1['D0'].shape[6])),
                                    'site_iteration': ('site', np.arange(k1['D0'].shape[6])),
                                    'experiment': ('site', np.arange(k1['D0'].shape[6])),
                                    'task_number': ('task', np.arange(k1['D0'].shape[7])),
                                    'task_left': ('task', list(tasks_left)),
                                    'task_right': ('task', list(tasks_right)),
                                },
                                dims=['injected'] + dims)
        assembly['monkey'] = 'site', ['M' if site <= 9 else 'P' for site in assembly['site_number'].values]
        assembly = assembly.squeeze('k').squeeze('trial_split')
        if contra_hemisphere:
            assembly = assembly.sel(hemisphere='contra')
        assembly = assembly.sel(metric='o2_dp')
        assembly = assembly[{'subject': [injection == 'muscimol' for injection in assembly['injection'].values]}]
        assembly = assembly.squeeze('subject')  # squeeze single-element subject dimension since data are pooled already

        # add site locations
        path = Path(__file__).parent / 'xray_3d.mat'
        site_locations = scipy.io.loadmat(path)['MX']
        assembly['site_x'] = 'site', site_locations[:, 0] / 1000  # scaling micro m to mm
        assembly['site_y'] = 'site', site_locations[:, 1] / 1000
        assembly['site_z'] = 'site', site_locations[:, 2] / 1000
        assembly = DataAssembly(assembly)  # reindex

        # load stimulus_set subsampled from hvm
        stimulus_set_meta = scipy.io.loadmat('/braintree/home/msch/rr_share_topo/topoDCNN/dat/metaparams.mat')
        stimulus_set_ids = stimulus_set_meta['id']
        stimulus_set_ids = [i for i in stimulus_set_ids if len(set(i)) > 1]  # filter empty ids
        stimulus_set = brainscore.get_stimulus_set('dicarlo.hvm')
        stimulus_set = stimulus_set[stimulus_set['image_id'].isin(stimulus_set_ids)]
        stimulus_set = stimulus_set[stimulus_set['object_name'].isin(TASK_LOOKUP)]
        stimulus_set['image_label'] = stimulus_set['truth'] = stimulus_set['object_name']  # 10 labels at this point
        stimulus_set.identifier = 'dicarlo.hvm_10'
        if contra_hemisphere:
            stimulus_set = stimulus_set[(stimulus_set['rxz'] > 0) & (stimulus_set['variation'] == 6)]
        assembly.attrs['stimulus_set'] = stimulus_set
        assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier
        return assembly

    @staticmethod
    def _rearrange_sites_tasks(data, tasks_per_site, number_of_sites):
        assert data.shape[-1] == tasks_per_site * number_of_sites
        return np.reshape(data, list(data.shape[:-1]) + [tasks_per_site, number_of_sites], order='F')

    @property
    def ceiling(self):
        split1, split2 = self._target_assembly.sel(split=0), self._target_assembly.sel(split=1)
        split1_diffs = split1.sel(silenced=False) - split1.sel(silenced=True)
        split2_diffs = split2.sel(silenced=False) - split2.sel(silenced=True)
        split_correlation, p = pearsonr(split1_diffs.values.flatten(), split2_diffs.values.flatten())
        return Score([split_correlation], coords={'aggregation': ['center']}, dims=['aggregation'])

    @staticmethod
    def sample_grid_points(low, high, num_x, num_y):
        assert len(low) == len(high) == 2
        grid_x, grid_y = np.meshgrid(np.linspace(low[0], high[0], num_x),
                                     np.linspace(low[1], high[1], num_y))
        return np.stack((grid_x.flatten(), grid_y.flatten()), axis=1)  # , np.zeros(num_x * num_y) for empty z dimension

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
            warn(f"Ignoring tasks {site_target_assembly['task'][~site_target_assembly.isnull()].values}")
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
