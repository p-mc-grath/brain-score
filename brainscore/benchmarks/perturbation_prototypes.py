import logging
from pathlib import Path

import numpy as np
import scipy.io
from scipy.optimize import fsolve
from scipy.stats import pearsonr
from tqdm import tqdm
from xarray import DataArray

import brainscore
from brainio_base.assemblies import DataAssembly, merge_data_arrays, walk_coords
from brainio_packaging.afraz2006 import train_test_stimuli, collect_assembly
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
    return np.dstack((grid_x, grid_y)).reshape(-1, 2)


class Kar2020:
    pass


class Bohn2020:
    pass


class Afraz2006(BenchmarkBase):
    BIBTEX = """@article{Afraz2006,
                abstract = {The inferior temporal cortex (IT) of primates is thought to be the final visual area in the ventral stream of cortical areas responsible for object recognition. Consistent with this hypothesis, single IT neurons respond selectively to highly complex visual stimuli such as faces. However, a direct causal link between the activity of face-selective neurons and face perception has not been demonstrated. In the present study of macaque monkeys, we artificially activated small clusters of IT neurons by means of electrical microstimulation while the monkeys performed a categorization task, judging whether noisy visual images belonged to 'face' or 'non-face' categories. Here we show that microstimulation of face-selective sites, but not other sites, strongly biased the monkeys' decisions towards the face category. The magnitude of the effect depended upon the degree of face selectivity of the stimulation site, the size of the stimulated cluster of face-selective neurons, and the exact timing of microstimulation. Our results establish a causal relationship between the activity of face-selective neurons and face perception.},
                author = {Afraz, Seyed Reza and Kiani, Roozbeh and Esteky, Hossein},
                doi = {10.1038/nature04982},
                file = {:C\:/Users/Martin/AppData/Local/Mendeley Ltd./Mendeley Desktop/Downloaded/Afraz, Kiani, Esteky - 2006 - Microstimulation of inferotemporal cortex influences face categorization.pdf:pdf},
                isbn = {1476-4687 (Electronic) 0028-0836 (Linking)},
                issn = {14764687},
                journal = {Nature},
                month = {aug},
                number = {7103},
                pages = {692--695},
                pmid = {16878143},
                publisher = {Nature Publishing Group},
                title = {{Microstimulation of inferotemporal cortex influences face categorization}},
                url = {http://www.nature.com/articles/nature04982},
                volume = {442},
                year = {2006}
                }"""

    def __init__(self):
        self._logger = logging.getLogger(fullname(self))
        self._assembly, self._fitting_stimuli = self._load_assembly()
        self._metric = None  # TODO
        super(Afraz2006, self).__init__(
            identifier='esteky.Afraz2006-selective_psychometric_shift',
            ceiling_func=None,
            version=1, parent='IT',
            bibtex=Afraz2006.BIBTEX)

    def _load_assembly(self):
        assembly = collect_assembly()
        # stimuli
        train_stimuli, test_stimuli = train_test_stimuli()
        assembly.attrs['stimulus_set'] = test_stimuli
        return assembly, train_stimuli

    def __call__(self, candidate: BrainModel):
        # determine face-selectivity
        candidate.start_recording('IT', time_bins=[(50, 100)])
        recordings = candidate.look_at(self._assembly.stimulus_set)
        face_selectivities = self.determine_face_selectivity(recordings)

        # "We trained two adult macaque monkeys to perform a face/non-face categorization task
        # upon viewing single images from one or the other category that were systematically degraded
        # by varying amounts of visual signal."
        # train on face/non-face categorization task
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=self._fitting_stimuli)
        nonstimulated_behavior = candidate.look_at(self._assembly.stimulus_set)

        # "Altogether, we assessed stimulus selectivity at 348 recording sites in 86 electrode penetrations in
        # two monkeys (46 and 40 in monkeys FR and KH, respectively).
        # We conducted microstimulation experiments at 31 face-selective sites and 55 non-selective sites,
        # while the monkey performed the object categorization task.
        # Selectivity for faces was defined as having a d' value > 1."
        # We here stimulate all sites that we have recordings in since we only compare the overall trend effects
        stimulation_locations = np.stack((recordings['tissue_x'], recordings['tissue_y'])).T.tolist()[:10]
        candidate_behaviors = []
        for site, location in enumerate(tqdm(stimulation_locations, desc='stimulation locations')):
            candidate.perturb(perturbation=None, target='IT')  # reset
            self._logger.debug(f"Stimulating at {location}")
            candidate.perturb(perturbation=BrainModel.Perturbation.microstimulation,
                              target='IT', perturbation_parameters={
                    # "Microstimulation consisted of bipolar current pulses of 50mA delivered at 200 Hz (refs 19, 20).
                    # The stimulation pulses were biphasic, with the cathodal pulse leading. Each pulse was 0.2 ms in
                    # duration with 0.1 ms between the cathodal and anodal phase. [...] Stimulating pulses were
                    # delivered for 50 ms in one of three time periods following onset of the visual stimulus:
                    # 0–50 ms, 50–100 ms or 100–150 ms."
                    # We here focus on the 100-150ms condition.
                    'current_pulse_mA': 50,
                    'pulse_type': 'biphasic',
                    'pulse_rate_Hz': 200,
                    'pulse_duration_ms': 0.2,
                    'pulse_interval_ms': 0.1,
                    'stimulation_onset_ms': 100,
                    'stimulation_duration_ms': 50,
                    'location': location,
                })
            behavior = candidate.look_at(self._assembly.stimulus_set)
            behavior = behavior.expand_dims('site')
            behavior['site_iteration'] = 'site', [site]
            behavior['site_x'] = 'site', [location[0]]
            behavior['site_y'] = 'site', [location[1]]
            behavior = type(behavior)(behavior)  # make sure site is indexed
            candidate_behaviors.append(behavior)
        candidate_behaviors = merge_data_arrays(candidate_behaviors)

        psychometric_shifts = self.characterize_psychometric_shifts(candidate_behaviors, nonstimulated_behavior)
        self.attach_face_selectivities(psychometric_shifts, face_selectivities[:subselect])
        score = self._metric(psychometric_shifts, self._assembly)
        # TODO: ceiling normalize
        return score

    def characterize_psychometric_shifts(self, behaviors, nonstimulated_behavior):
        nonstimulated_curve = self.grouped_face_responses(nonstimulated_behavior)
        nonstimulated_logistic = self.fit_logistic(x=nonstimulated_curve['label_signal_level'],
                                                   y=nonstimulated_curve.values)
        nonstimulated_signal_midpoint = self.logistic_midpoint(nonstimulated_logistic)

        psychometric_shifts = []
        for site_iteration in behaviors['site_iteration'].values:
            # index instead of `.sel` to preserve all site coords
            behavior = behaviors[{'site': [site == site_iteration for site in behaviors['site_iteration'].values]}]
            site_coords = {coord: (dims, values) for coord, dims, values in walk_coords(behavior['site'])}
            behavior = behavior.squeeze('site')
            psychometric_curve = self.grouped_face_responses(behavior)
            site_logistic = self.fit_logistic(x=psychometric_curve['label_signal_level'],
                                              y=psychometric_curve.values)
            site_midpoint = self.logistic_midpoint(site_logistic)
            psychometric_shift = nonstimulated_signal_midpoint - site_midpoint
            psychometric_shift = DataAssembly([psychometric_shift], coords=site_coords, dims=['site'])
            psychometric_shifts.append(psychometric_shift)
        psychometric_shifts = merge_data_arrays(psychometric_shifts)
        return psychometric_shifts

    def attach_face_selectivities(self, psychometric_shifts, face_selectivities):
        assert len(psychometric_shifts) == len(face_selectivities)
        # assume same ordering
        psychometric_shifts['face_selectivity'] = 'site', face_selectivities.values

    def determine_face_selectivity(self, recordings):
        assert (recordings >= 0).all()
        # A d' value of zero indicates indistinguishable responses to faces and non-faces.
        # Increasingly positive d' values indicate progressively better selectivity for faces.
        # Selectivity for faces was defined as having a d' value > 1.
        result = []
        for neuroid_id in tqdm(recordings['neuroid_id'].values, desc='neuron face dprime'):
            neuron = recordings.sel(neuroid_id=neuroid_id)
            neuron = neuron.squeeze()
            face_mean, face_variance = self.multiunit_stats(neuron.sel(image_label='face'))
            nonface_mean, nonface_variance = self.multiunit_stats(neuron.sel(image_label='nonface'))
            # face selectivity based on "more positive" firing
            dprime = (face_mean - nonface_mean) / np.sqrt((face_variance + nonface_variance) / 2)
            result.append(dprime)
        result = DataArray(result, coords={'neuroid_id': recordings['neuroid_id'].values}, dims=['neuroid_id'])
        return result

    def multiunit_stats(self, neuron):
        mean, var = np.mean(neuron.values), np.var(neuron.values)
        return mean, var

    def grouped_face_responses(self, behavior):
        np.testing.assert_array_equal(behavior['choice'], ['face', 'nonface'])
        behavior['choose_face'] = 'presentation', behavior.argmax('choice')
        face_responses = DataAssembly(behavior.argmax('choice'), coords={
            coord: (dims, values) for coord, dims, values in walk_coords(behavior['presentation'])},
                                      dims=['presentation'])
        face_responses = 1 - face_responses  # invert so that nonface (0) to face (1)
        grouped_face_responses = face_responses.groupby('label_signal_level').mean()
        return grouped_face_responses

    def fit_logistic(self, x, y):
        params, pcov = scipy.optimize.curve_fit(logistic, x, y)
        return params

    def logistic_midpoint(self, logistic_params, midpoint=0.5, initial_guess=0):
        func = lambda x: logistic(x, *logistic_params) - midpoint
        solution = fsolve(func, initial_guess)[0]
        assert np.isclose(logistic(solution, *logistic_params), midpoint)
        return solution


def logistic(x, a, b):
    return 1 / (1 + np.exp(-(a + b * x)))
