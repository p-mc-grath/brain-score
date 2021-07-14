import logging

import numpy as np
from scipy.stats import pearsonr

import brainscore
from brainio.assemblies import merge_data_arrays, walk_coords
from brainio_packaging.rajalingham2019 import collect_assembly
from brainscore.benchmarks import BenchmarkBase
from brainscore.metrics import Score
from brainscore.metrics.behavior_differences import BehaviorDifferences
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname

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

    def __init__(self):
        self._target_assembly = collect_assembly()
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
            bibtex=Rajalingham2019.BIBTEX)

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
                    'amount_µl': 1,
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
    return np.dstack((grid_x, grid_y)).reshape(-1, 2)
