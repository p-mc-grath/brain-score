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


class Rajalingham2019(BenchmarkBase):
    def __init__(self):
        self._stimulus_set = brainscore.get_stimulus_set('dicarlo.hvm')
        self._stimulus_set['image_label'] = self._stimulus_set['truth'] = self._stimulus_set['category_name']
        self._target_assembly = self._load_assembly()
        self._similarity_metric = BehaviorDifferences()
        super(Rajalingham2019, self).__init__(
            identifier='dicarlo.Rajalingham2019',
            ceiling_func=None,
            version=1, parent='IT',
            paper_link='https://www.sciencedirect.com/science/article/pii/S0896627319301102')

    def __call__(self, candidate: BrainModel):
        stimulus_set = self._stimulus_set
        training_stimuli = stimulus_set  # TODO: is training the entire stimulus set?
        # stimulus_set = repeat_trials(number_of_trials=10) TODO

        # run silencing sessions
        random_state = RandomState(0)
        behaviors = []
        for run in range(10):  # TODO: how many runs?
            # "[...] inactivation sessions were interleaved over days with control behavioral sessions.
            # Thus, each inactivation experiment consisted of three behavioral sessions:
            # the baseline or pre-control session (1 day prior to injection),
            # the inactivation session,
            # and the recovery or post-control session (2 days after injection)"
            for day in ['pre-control', 'inactivation', 'post-control']:
                if day == 'inactivation':
                    # "We varied the location of microinjections to randomly sample the ventral surface of IT
                    # (from approximately + 8mm AP to approx + 20mm AP)."
                    # TODO: is this specified via xyz? If yes, what are yz?
                    injection_location = random_state.uniform(low=[8, 8], high=[20, 20])  # TODO: uniform or gaussian?
                    candidate.perturb(perturbation=BrainModel.Perturbation.muscimol,
                                      target='IT', perturbation_parameters={
                            # "Each inactivation session began with a single focal microinjection of 1ml of muscimol
                            # (5mg/mL, Sigma Aldrich) at a slow rate (100nl/min) via a 30-gauge stainless-steel cannula at
                            # the targeted site in ventral IT."
                            'amount_microliter': 1,
                            'location': injection_location,
                        })
                else:  # non-inactivation day
                    candidate.perturb(perturbation=None, target='IT')  # reset
                candidate.start_task(task=BrainModel.Task.probabilities, fitting_stimuli=training_stimuli)
                behavior = candidate.look_at(stimulus_set)  # TODO the whole stimulus_set each session?
                # TODO: are the runs averaged somehow or just treated as different trials?
                # behavior = behavior.expand_dims('run').expand_dims('day')
                # behavior['run'], behavior['day'] = [run], [day]
                behavior = behavior.expand_dims('day')
                behavior['run'] = ('presentation', [run] * len(behavior['presentation']))
                behavior['day'] = [day]
                behaviors.append(behavior)
        behaviors = merge_data_arrays(behaviors)

        # score
        # behaviors = behaviors.unstack('presentation').stack(presentation=['image_id', 'run'])
        score = self._similarity_metric(behaviors, self._target_assembly)
        return score

    def _load_assembly(self):
        directory = Path('/braintree/home/msch/rr_share_topo/topoDCNN/')
        with open(directory / f'dat/HVM8_1/rajalingham2018_k2_all.pkl', 'rb') as f:
            exp = pickle.load(f, encoding='latin1')
        exp = exp['all_sites']
        tasks, sites = 6, 11
        rearrange = functools.partial(self._rearrange_sites_tasks, tasks_per_site=tasks, number_of_sites=sites)
        exp['d0'], exp['d1'] = rearrange(exp['d0']), rearrange(exp['d1'])
        return DataAssembly([exp['d0'], exp['d1']], coords={
            'silenced': [False, True],
            'split': np.arange(exp['d0'].shape[1]),
            'bootstrap': np.arange(exp['d0'].shape[0]),
            'task': np.arange(tasks),
            'site': np.arange(sites)},
                            dims=['silenced', 'bootstrap', 'split', 'task', 'site'])

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
