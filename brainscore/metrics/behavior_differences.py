import itertools

import numpy as np
from tqdm import tqdm

from brainio_base.assemblies import merge_data_arrays, walk_coords, array_is_element
from brainscore.metrics import Metric
from brainscore.metrics.image_level_behavior import I2n

from brainscore.metrics.transformations import TestOnlyCrossValidationSingle, CrossValidation


class BehaviorDifferences(Metric):
    def __init__(self):
        self.i2n = I2n()
        self.site_split = TestOnlyCrossValidationSingle()
        self.task_split = CrossValidation()

    def __call__(self, assembly1, assembly2):
        """
        :param assembly1: a raw assembly in the format of `silenced: 2, presentation: p, choice: c, site: n`
        :param assembly2: a processed assembly in the format of `silenced :2, task: c * (c-1), site: m`
        :return: a Score
        """

        # process assembly1
        assembly1_characterized = self.characterize(assembly1)
        assembly1_tasks = self.subselect_tasks(assembly1, assembly2)
        assembly1_tasks = [assembly1_characterized.sel(task_left=task_left, task_right=task_right).values for
                           task_left, task_right in zip(assembly2['task_left'].values, assembly2['task_right'].values)]
        assembly1_differences = self.compute_differences(assembly1_tasks)
        assembly2_differences = self.compute_differences(assembly2)

        # compare
        site_scores = self.site_split(assembly2_differences,
                                      apply=lambda site_assembly: self.apply_site(assembly1_differences, site_assembly))
        score = site_scores.mean('site')
        return score

    def characterize(self, assembly):
        """ compute per-task performance from `presentation x choice` assembly """
        # xarray can't do multi-dimensional grouping, do things manually
        o2s = []
        adjacent_values = assembly['silenced'].values, assembly['site'].values  # TODO: this takes 4.5 min, way too long
        for silenced, site in tqdm(itertools.product(*adjacent_values), desc='characterize',
                                   total=np.prod([len(values) for values in adjacent_values])):
            current_assembly = assembly.sel(silenced=silenced, site=site)
            source_response_matrix = self.i2n.target_distractor_scores(current_assembly)
            i2 = self.i2n.dprimes(source_response_matrix)
            o2 = i2.groupby('truth').mean('presentation')
            o2 = o2.rename({'truth': 'task_left', 'choice': 'task_right'})
            o2 = o2.expand_dims('silenced').expand_dims('site')
            o2['silenced'] = [silenced]
            for (coord, _, _), value in zip(walk_coords(assembly['site']), site):
                o2[coord] = 'site', [value]
            o2s.append(o2)
        o2s = merge_data_arrays(o2s)  # this only takes ~1s, ok
        # a = assembly.sel(silenced=False, site=0).squeeze('site')  # TODO: unfold
        # source_response_matrix = self.i2n.target_distractor_scores(a)
        # i2 = self.i2n.dprimes(source_response_matrix)
        # o2 = i2.groupby('truth').mean('presentation')
        # o2 = o2.rename({'truth': 'task_left', 'choice': 'task_right'})
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
        task_scores = self.task_split(source_assembly, site_target_assembly, apply=self.apply_task)
        return task_scores

    def apply_task(self, source_train, target_train, source_test, target_test):
        """
        finds the best-matching site in the source train assembly to predict the task effects in the test target.
        :param source_train: source assembly for mapping with t tasks and n sites
        :param target_train: target assembly for mapping with t tasks
        :param source_test: source assembly for testing with 1 task and n sites
        :param target_test: target assembly for testing with 1 task
        :return: a Score
        """
        # map: find site in assembly1 that best matches mapping tasks
        correlations = correlate(source_train, target_train)
        best_site = correlations.argmax('site')
        # test: predictivity of held-out task
        source_test = source_test.sel(site=best_site)
        score = target_test - source_test
        return score

    def compute_differences(self, behaviors):
        """
        :param behaviors: an assembly with a dimension `silenced` and values `[True, False]`
        :return: the difference between these two conditions
        """
        return behaviors.sel(silenced=False) - behaviors.sel(silenced=True)
