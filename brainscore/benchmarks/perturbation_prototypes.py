import functools
import pickle
from pathlib import Path

import numpy as np
from numpy.random.mtrand import RandomState
from scipy.stats import pearsonr

import brainscore
from brainio_base.assemblies import DataAssembly, merge_data_arrays
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.behavior_differences import BehaviorDifferences
from brainscore.model_interface import BrainModel
import scipy.io


class Rajalingham2019(BenchmarkBase):
    def __init__(self):
        self._target_assembly = self._load_assembly()
        self._similarity_metric = BehaviorDifferences()
        super(Rajalingham2019, self).__init__(
            identifier='dicarlo.Rajalingham2019',
            ceiling_func=None,
            version=1, parent='IT',
            paper_link='https://www.sciencedirect.com/science/article/pii/S0896627319301102')

    def __call__(self, candidate: BrainModel):
        # approach:
        # 1. have model inactivate many different sites in IT
        # 2. benchmark metric searches over those sites to find the ones that maximally correspond to 7/8 tasks
        # 3. test generalization to 8th task (measuring distance), cross-validate

        stimulus_set = self._target_assembly.stimulus_set
        # Training
        # TODO: Both animals were previously trained on other images of other objects, and were proficient in
        #  discriminating among over 35 arbitrarily sampled basic-level object categories
        training_stimuli = stimulus_set  # TODO: what were monkeys trained on? all of hvm?
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
        control_behavior = candidate.look_at(stimulus_set)
        control_behavior = control_behavior.expand_dims('silenced')
        control_behavior['silenced'] = [False]

        # silencing sessions
        random_state = RandomState(0)
        behaviors = [control_behavior]
        for site in range(10):
            # "We varied the location of microinjections to randomly sample the ventral surface of IT
            # (from approximately + 8mm AP to approx + 20mm AP)."
            injection_location = random_state.uniform(low=[8, 8], high=[20, 20])  # TODO: uniform or gaussian?
            candidate.perturb(perturbation=BrainModel.Perturbation.muscimol,
                              target='IT', perturbation_parameters={
                    # "Each inactivation session began with a single focal microinjection of 1ml of muscimol
                    # (5mg/mL, Sigma Aldrich) at a slow rate (100nl/min) via a 30-gauge stainless-steel cannula at
                    # the targeted site in ventral IT."
                    'amount_microliter': 1,
                    'location': injection_location,
                })
            candidate.start_task(task=BrainModel.Task.probabilities, fitting_stimuli=training_stimuli)
            behavior = candidate.look_at(stimulus_set)  # TODO the whole stimulus_set each session?
            behavior = behavior.expand_dims('silenced').expand_dims('site')
            behavior['silenced'] = [True]
            behavior['site_iteration'] = 'site', [site]
            behavior['site_x'] = 'site', [injection_location[0]]
            behavior['site_y'] = 'site', [injection_location[1]]
            behaviors.append(behavior)
        behaviors = merge_data_arrays(behaviors)

        # TODO:
        # Our choice of five objects resulted in ten possible pairwise object discrimination subtasks
        # (see Figure 1A for complete list). To accumulate enough trials to precisely measure performance
        # for each subtask within a single behavioral session (i.e., a single experimental day),
        # we sub-selected six of these ten subtasks for most experiments
        # TODO: how to figure out which of the 6 tasks in the data is which?

        # score
        # behaviors = behaviors.unstack('presentation').stack(presentation=['image_id', 'run'])
        target_assembly = self._target_assembly.sel(split=0)
        score = self._similarity_metric(behaviors, target_assembly)
        score = ceil(score, self.ceiling)
        return score

    def _load_assembly(self):
        directory = Path('/braintree/home/msch/rr_share_topo/topoDCNN/')
        with open(directory / f'dat/HVM8_1/rajalingham2018_k2_all.pkl', 'rb') as f:
            exp = pickle.load(f, encoding='latin1')
        exp = exp['all_sites']
        tasks, sites = 6, 11
        rearrange = functools.partial(self._rearrange_sites_tasks, tasks_per_site=tasks, number_of_sites=sites)
        exp['d0'], exp['d1'] = rearrange(exp['d0']), rearrange(exp['d1'])
        assembly = DataAssembly([exp['d0'], exp['d1']], coords={
            'silenced': [False, True],
            'split': np.arange(exp['d0'].shape[1]),
            'bootstrap': np.arange(exp['d0'].shape[0]),
            'task_number': ('task', np.arange(tasks)),
            # TODO: actually assign these.
            #  paper tasks are elephant-v-bear, dog-v-bear, dog-v-elephant, plane-v-dog, chair-v-dog, chair-v-plane
            'task_left': ('task', ['elephant', 'dog', 'dog', 'airplane3', 'chair0', 'chair0']),
            'task_right': ('task', ['bear', 'bear', 'elephant', 'dog', 'dog', 'airplane3']),
            'site': np.arange(sites)},
                                dims=['silenced', 'bootstrap', 'split', 'task', 'site'])
        # additional tasks for experiment 2 only are plane-v-bear, plane-v-elephant, chair-v-bear, chair-v-elephant
        assembly = assembly.mean('bootstrap')

        # TODO: load stimulus_set subsampled from hvm
        stimulus_set_ids = scipy.io.loadmat(directory / 'dat/metaparams.mat')
        stimulus_set_ids = stimulus_set_ids['id']
        stimulus_set_ids = [i for i in stimulus_set_ids if len(set(i)) > 1]  # filter empty ids
        stimulus_set = brainscore.get_stimulus_set('dicarlo.hvm')
        stimulus_set = stimulus_set[stimulus_set['image_id'].isin(stimulus_set_ids)]
        stimulus_set['image_label'] = stimulus_set['truth'] = stimulus_set['object_name']  # 10 labels at this point
        stimulus_set.identifier = 'dicarlo.hvm_10'
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


class Kar2020:
    pass


class Bohn2020:
    pass
