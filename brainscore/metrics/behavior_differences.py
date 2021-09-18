import itertools
import logging
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

from brainio.assemblies import merge_data_arrays, walk_coords, array_is_element, DataAssembly
from brainscore.metrics import Metric, Score
from brainscore.metrics.image_level_behavior import _o2
from brainscore.metrics.regression import linear_regression
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
        sites = assembly2_differences['site_iteration'].values
        tasks = assembly2_differences['task_number'].values
        prediction_pairs = []
        for target_test_site, target_test_task in tqdm(
                itertools.product(sites, tasks), desc='site+task kfold', total=len(sites) * len(tasks)):
            # test assembly is 1 task, 1 site
            target_test = assembly2_differences[{
                'site': [site == target_test_site for site in assembly2_differences['site_iteration'].values],
                'task': [task == target_test_task for task in assembly2_differences['task_number'].values]}]
            if len(target_test) < 1:
                continue  # not all tasks were run on all sites
            # train are the other tasks on the same site
            target_train = assembly2_differences[{
                'site': [site == target_test_site for site in assembly2_differences['site_iteration'].values],
                'task': [task != target_test_task for task in assembly2_differences['task_number'].values]}]
            # source test assembly is same 1 task, all sites
            source_test = assembly1_differences[{
                'task': [task == target_test_task for task in assembly1_differences['task_number'].values]}]
            # source train assembly are other tasks, all sites
            source_train = assembly1_differences[{
                'task': [task != target_test_task for task in assembly1_differences['task_number'].values]}]

            # filter non-nan task measurements from target
            nonnan_tasks = target_train['task'][~target_train.squeeze('site').isnull()].values
            target_train = target_train.sel(task=nonnan_tasks)
            source_train = source_train.sel(task=nonnan_tasks)

            # map: regress from source to target
            regression = linear_regression(
                xarray_kwargs=dict(expected_dims=('task', 'site'),
                                   neuroid_dim='site',
                                   neuroid_coord='site_iteration',
                                   stimulus_coord='task'))
            regression.fit(source_train, target_train)
            # test: predictivity of held-out task
            # We can only collect the single prediction here and then correlate in outside loop
            prediction_test = regression.predict(source_test)
            np.testing.assert_array_equal(prediction_test['task'].values, prediction_test['task'].values)
            np.testing.assert_array_equal(prediction_test.shape, target_test.shape)
            pair = type(target_test)([prediction_test, target_test],
                                     coords={**{'type': ['source', 'target']},
                                             **{coord: (dims, values) for coord, dims, values in
                                                walk_coords(target_test)}},
                                     dims=('type',) + target_test.dims)
            prediction_pairs.append(pair)

        prediction_pairs = merge_data_arrays(prediction_pairs)

        source = prediction_pairs.sel(type='source')
        target = prediction_pairs.sel(type='target')
        source_vector = source.values.flatten()
        target_vector = target.values.flatten()
        not_nan = ~np.isnan(target_vector)  # not every task is part of every site split
        source_vector, target_vector = source_vector[not_nan], target_vector[not_nan]
        correlation, p = pearsonr(source_vector, target_vector)
        score = Score([correlation, p], coords={'statistic': ['r', 'p']}, dims=['statistic'])
        score.attrs['predictions'] = source
        score.attrs['target'] = target
        return score

    def characterize(self, assembly):
        """ compute per-task performance from `presentation x choice` assembly """
        # xarray can't do multi-dimensional grouping, do things manually
        o2s = []
        adjacent_values = assembly['injected'].values, assembly['site'].values
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
