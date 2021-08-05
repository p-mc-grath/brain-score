import itertools
import logging

import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from brainio.assemblies import merge_data_arrays, walk_coords, array_is_element, DataAssembly
from brainscore.metrics import Metric, Score
from brainscore.metrics.image_level_behavior import _o2
from brainscore.metrics.transformations import TestOnlyCrossValidationSingle, CrossValidation
from brainscore.utils import fullname


class BehaviorDifferences(Metric):
    def __init__(self):
        super(BehaviorDifferences, self).__init__()
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, assembly1, assembly2):
        """
        :param assembly1: a tuple with the first element representing the control behavior in the format of
            `presentation: p, choice: c` and the second element representing inactivations behaviors in
            `presentation: p, choice: c, site: n`
        :param assembly2: a processed assembly in the format of `injected :2, task: c * (c-1), site: m`
        :return: a Score
        """
        # process assembly1 TODO: move characterization to benchmark
        assembly1_characterized = self.characterize(assembly1)
        assembly1_tasks = self.subselect_tasks(assembly1_characterized, assembly2)
        assembly1_differences = self.compute_differences(assembly1_tasks)
        assembly2_differences = self.compute_differences(assembly2)
        assembly2_differences = assembly2_differences.mean('bootstrap')

        # compare
        site_split = TestOnlyCrossValidationSingle(  # instantiate on-the-fly to control the kfolds for 1 test site each
            split_coord='site', stratification_coord=None, kfold=True, splits=len(assembly2['site']))
        score = site_split(assembly2_differences,
                           apply=lambda site_assembly: self.apply_site(assembly1_differences, site_assembly))
        return score

    def characterize(self, assembly):
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

    def subselect_tasks(self, assembly, reference_assembly):
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

    def apply_site(self, source_assembly, site_target_assembly):
        site_target_assembly = site_target_assembly.squeeze('site')
        np.testing.assert_array_equal(source_assembly.sortby('task_number')['task_left'].values,
                                      site_target_assembly.sortby('task_number')['task_left'].values)
        np.testing.assert_array_equal(source_assembly.sortby('task_number')['task_right'].values,
                                      site_target_assembly.sortby('task_number')['task_right'].values)

        # filter non-nan task measurements from target
        nonnan_tasks = site_target_assembly['task'][~site_target_assembly.isnull()]
        if len(nonnan_tasks) < len(site_target_assembly):
            self._logger.warning(
                f"Ignoring tasks {site_target_assembly['task'][~site_target_assembly.isnull()].values}")
        site_target_assembly = site_target_assembly.sel(task=nonnan_tasks)
        source_assembly = source_assembly.sel(task=nonnan_tasks.values)

        # try to predict from model
        task_split = CrossValidation(split_coord='task_number', stratification_coord=None,
                                     kfold=True, splits=len(site_target_assembly['task']))
        task_scores = task_split(source_assembly, site_target_assembly, apply=self.apply_task)
        task_scores = task_scores.raw
        correlation, p = pearsonr(task_scores.sel(type='source'), task_scores.sel(type='target'))
        score = Score([correlation, p], coords={'statistic': ['r', 'p']}, dims=['statistic'])
        score.attrs['predictions'] = task_scores.sel(type='source')
        score.attrs['target'] = task_scores.sel(type='target')
        return score

    def apply_task(self, source_train, target_train, source_test, target_test):
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

    def compute_differences(self, behaviors):
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
