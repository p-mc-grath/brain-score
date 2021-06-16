import logging

import numpy as np
import scipy.io
from pathlib import Path
from scipy.stats import pearsonr

import brainscore
from brainio_base.assemblies import DataAssembly, merge_data_arrays, walk_coords
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.behavior_differences import BehaviorDifferences
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname

BIBTEX = """@article(...,
url=https://www.sciencedirect.com/science/article/pii/S0896627319301102
)
"""

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


class Rajalingham2019(BenchmarkBase):
    def __init__(self):
        self._target_assembly = self._load_assembly()
        self._training_stimuli = brainscore.get_stimulus_set('dicarlo.hvm')
        self._training_stimuli['image_label'] = self._training_stimuli['object_name']
        # use only those images where it's the same object (label)
        self._training_stimuli = self._training_stimuli[self._training_stimuli['object_name'].isin(
            self._target_assembly.stimulus_set['object_name'])]
        self._similarity_metric = BehaviorDifferences()
        self._logger = logging.getLogger(fullname(self))
        super(Rajalingham2019, self).__init__(
            identifier='dicarlo.Rajalingham2019-deficits',
            ceiling_func=None,
            version=1, parent='IT',
            bibtex=BIBTEX)

    def __call__(self, candidate: BrainModel):
        # approach:
        # 1. have model inactivate many different sites in IT
        # 2. benchmark metric searches over those sites to find the ones that maximally correspond to 7/8 tasks
        # 3. test generalization to 8th task (measuring distance), cross-validate

        stimulus_set = self._target_assembly.stimulus_set
        # Training
        # TODO: Both animals were previously trained on other images of other objects, and were proficient in
        #  discriminating among over 35 arbitrarily sampled basic-level object categories
        training_stimuli = stimulus_set  # self._training_stimuli
        # stimulus_set = repeat_trials(number_of_trials=10) TODO

        # "[...] inactivation sessions were interleaved over days with control behavioral sessions.
        # Thus, each inactivation experiment consisted of three behavioral sessions:
        # the baseline or pre-control session (1 day prior to injection),
        # the inactivation session,
        # and the recovery or post-control session (2 days after injection)"
        # --> we here front-load one control session and then run many inactivation sessions

        # control
        candidate.perturb(perturbation=None, target='IT')  # reset
        candidate.start_task(task=BrainModel.Task.probabilities, fitting_stimuli=training_stimuli)
        control_behavior = candidate.look_at(stimulus_set, number_of_trials=None)
        control_behavior = control_behavior.expand_dims('injected')
        control_behavior['injected'] = [False]

        # silencing sessions
        behaviors = [control_behavior]
        # "We varied the location of microinjections to randomly sample the ventral surface of IT
        # (from approximately + 8mm AP to approx + 20mm AP)."
        # stay between [0, 10] since that is the extent of the tissue
        injection_locations = sample_grid_points([2, 2], [8, 8], num_x=3, num_y=3)
        # injection_locations = sample_grid_points([2, 2], [8, 8], num_x=10, num_y=10)
        for site, injection_location in enumerate(injection_locations):
            candidate.perturb(perturbation=None, target='IT')  # reset
            self._logger.debug(f"Perturbing at {injection_location}")
            candidate.perturb(perturbation=BrainModel.Perturbation.muscimol,
                              target='IT', perturbation_parameters={
                    # "Each inactivation session began with a single focal microinjection of 1ml of muscimol
                    # (5mg/mL, Sigma Aldrich) at a slow rate (100nl/min) via a 30-gauge stainless-steel cannula at
                    # the targeted site in ventral IT."
                    'amount_microliter': 1,
                    'location': injection_location,
                })
            behavior = candidate.look_at(stimulus_set)  # TODO the whole stimulus_set each session?
            behavior = behavior.expand_dims('injected').expand_dims('site')
            behavior['injected'] = [True]
            behavior['site_iteration'] = 'site', [site]
            behavior['site_x'] = 'site', [injection_location[0]]
            behavior['site_y'] = 'site', [injection_location[1]]
            behavior = type(behavior)(behavior)  # make sure site and injected are indexed
            behaviors.append(behavior)
        behaviors = merge_data_arrays(behaviors)

        # align naming: from stimulus_set object name to assembly task
        # unfortunately setting `['object_name'] = ...` directly fails due to MultiIndex, so we'll re-create.
        behaviors = self.align_task_names(behaviors)

        # TODO:
        # Our choice of five objects resulted in ten possible pairwise object discrimination subtasks
        # (see Figure 1A for complete list). To accumulate enough trials to precisely measure performance
        # for each subtask within a single behavioral session (i.e., a single experimental day),
        # we sub-selected six of these ten subtasks for most experiments
        # TODO: how to figure out which of the 6 tasks in the data is which?

        # score
        # behaviors = behaviors.unstack('presentation').stack(presentation=['image_id', 'run'])
        score = self._similarity_metric(behaviors, self._target_assembly)
        # score = ceil(score, self.ceiling)
        return score

    def align_task_names(self, behaviors):
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
        assembly['site_x'] = 'site', site_locations[:, 0]
        assembly['site_y'] = 'site', site_locations[:, 1]
        assembly['site_z'] = 'site', site_locations[:, 2]
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

    def _rearrange_sites_tasks(self, data, tasks_per_site, number_of_sites):
        assert data.shape[-1] == tasks_per_site * number_of_sites
        return np.reshape(data, list(data.shape[:-1]) + [tasks_per_site, number_of_sites], order='F')

    @property
    def ceiling(self):
        split1, split2 = self._target_assembly.sel(split=0), self._target_assembly.sel(split=1)
        split1_diffs = split1.sel(silenced=False) - split1.sel(silenced=True)
        split2_diffs = split2.sel(silenced=False) - split2.sel(silenced=True)
        split_correlation, p = pearsonr(split1_diffs.values.flatten(), split2_diffs.values.flatten())
        return Score([split_correlation], coords={'aggregation': ['center']}, dims=['aggregation'])


def sample_grid_points(low, high, num_x, num_y):
    assert len(low) == len(high) == 2
    grid_x, grid_y = np.meshgrid(np.linspace(low[0], high[0], num_x),
                                 np.linspace(low[1], high[1], num_y))
    return np.array(list(zip(grid_x.flatten(), grid_y.flatten())))
    # TODO: could we also do
    # return np.dstack((grid_x, grid_y))


class Kar2020:
    pass


class Bohn2020:
    pass
