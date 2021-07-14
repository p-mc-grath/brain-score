import logging

import numpy as np
from numpy.random import RandomState
from tqdm import tqdm
from xarray import DataArray

from brainio.assemblies import merge_data_arrays, DataAssembly, walk_coords
from brainio_packaging.afraz2015 import muscimol_delta_overall_accuracy, collect_stimuli
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks.afraz2006 import mean_var
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname


class Afraz2015MuscimolDeltaAccuracy(BenchmarkBase):
    BIBTEX = """@article {Afraz6730,
                author = {Afraz, Arash and Boyden, Edward S. and DiCarlo, James J.},
                title = {Optogenetic and pharmacological suppression of spatial clusters of face neurons reveal their causal role in face gender discrimination},
                volume = {112},
                number = {21},
                pages = {6730--6735},
                year = {2015},
                doi = {10.1073/pnas.1423328112},
                publisher = {National Academy of Sciences},
                abstract = {There exist subregions of the primate brain that contain neurons that respond more to images of faces over other objects. These subregions are thought to support face-detection and discrimination behaviors. Although the role of these areas in telling faces from other objects is supported by direct evidence, their causal role in distinguishing faces from each other lacks direct experimental evidence. Using optogenetics, here we reveal their causal role in face-discrimination behavior and provide a mechanistic explanation for the process. This study is the first documentation of behavioral effects of optogenetic intervention in primate object-recognition behavior. The methods developed here facilitate the usage of the technical advantages of optogenetics for future studies of high-level vision.Neurons that respond more to images of faces over nonface objects were identified in the inferior temporal (IT) cortex of primates three decades ago. Although it is hypothesized that perceptual discrimination between faces depends on the neural activity of IT subregions enriched with {\textquotedblleft}face neurons,{\textquotedblright} such a causal link has not been directly established. Here, using optogenetic and pharmacological methods, we reversibly suppressed the neural activity in small subregions of IT cortex of macaque monkeys performing a facial gender-discrimination task. Each type of intervention independently demonstrated that suppression of IT subregions enriched in face neurons induced a contralateral deficit in face gender-discrimination behavior. The same neural suppression of other IT subregions produced no detectable change in behavior. These results establish a causal link between the neural activity in IT face neuron subregions and face gender-discrimination behavior. Also, the demonstration that brief neural suppression of specific spatial subregions of IT induces behavioral effects opens the door for applying the technical advantages of optogenetics to a systematic attack on the causal relationship between IT cortex and high-level visual perception.},
                issn = {0027-8424},
                URL = {https://www.pnas.org/content/112/21/6730},
                eprint = {https://www.pnas.org/content/112/21/6730.full.pdf},
                journal = {Proceedings of the National Academy of Sciences}
            }"""

    def __init__(self):
        self._logger = logging.getLogger(fullname(self))
        self._assembly, self._fitting_stimuli, self._selectivity_stimuli = self._load_assembly()
        self._metric = None  # TODO
        super(Afraz2015MuscimolDeltaAccuracy, self).__init__(
            identifier='esteky.Afraz2015-selective_psychometric_shift',
            ceiling_func=None,
            version=1, parent='IT',
            bibtex=self.BIBTEX)

    def _load_assembly(self):
        assembly = muscimol_delta_overall_accuracy()
        # stimuli
        # TODO: separate train/test
        # TODO All images (60) -- used for testing
        stimuli = collect_stimuli()
        gender_stimuli = stimuli[stimuli['category'].isin(['male', 'female'])]
        selectivity_stimuli = stimuli[stimuli['category'].isin(['object', 'face'])]
        gender_stimuli['image_label'] = gender_stimuli['category']
        assembly.attrs['stimulus_set'] = gender_stimuli.sample(n=60)
        return assembly, gender_stimuli, selectivity_stimuli

    def __call__(self, candidate: BrainModel):
        # "In six experimental sessions, we stereotactically targeted the center of the high-FD neural cluster
        # (in both monkeys) for muscimol microinjection (SI Methods). We used microinjectrodes (30) to record the
        # neural activity before muscimol injection, to measure the FD [face-detection index] of each targeted IT
        # subregion, and to precisely execute small microinjections along the entire cortical thickness."
        # "Before virus injection we recorded extensively from the lower bank of STS and the ventral surface of CIT
        # cortex in the 2- to 9-mm range anterior to the interaural line and located a large (∼3 × 4 mm) cluster of
        # face-selective units (defined as FD d′ > 1)."

        # record to determine face-selectivity
        candidate.start_recording('IT', time_bins=[(50, 100)])
        recordings = candidate.look_at(self._selectivity_stimuli)

        # "Face-detector sites, summarizes data shown in A (n = 6 microinjections);
        # other IT sites, micro- injections away from high-FD subregions of IT (n = 6)"
        # We here randomly sub-select the recordings to match the number of stimulation sites in the experiment, based
        # on the assumption that we can compare trend effects even with a random sample.
        random_state = RandomState(seed=1)
        num_face_detector_sites = 6
        num_nonface_detector_sites = 6
        face_selectivity_threshold = 1  # "face-selective units (defined as FD d′ > 1)" (SI Electrophysiology)
        face_detector_sites, nonface_detector_sites = [], []
        while len(face_detector_sites) < num_face_detector_sites:
            neuroid_id = random_state.choice(recordings['neuroid_id'].values)
            selectivity = determine_selectivity(recordings[{'neuroid': [
                nid == neuroid_id for nid in recordings['neuroid_id'].values]}])
            if selectivity > face_selectivity_threshold:
                face_detector_sites.append(neuroid_id)
        while len(nonface_detector_sites) < num_nonface_detector_sites:
            neuroid_id = random_state.choice(recordings['neuroid_id'].values)
            selectivity = determine_selectivity(recordings[{'neuroid': [
                nid == neuroid_id for nid in recordings['neuroid_id'].values]}])
            if selectivity <= face_selectivity_threshold:
                nonface_detector_sites.append(neuroid_id)
        recordings = recordings[{'neuroid': [neuroid_id in (face_detector_sites + nonface_detector_sites)
                                             for neuroid_id in recordings['neuroid_id'].values]}]

        # "In practice, we first trained the animals on a fixed set of 400 images (200 males and 200 females)."
        candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli=self._fitting_stimuli)
        unperturbed_behavior = candidate.look_at(self._assembly.stimulus_set)

        # This benchmark ignores the parafoveal presentation of images.
        stimulation_locations = np.stack((recordings['tissue_x'], recordings['tissue_y'])).T.tolist()
        candidate_behaviors = []
        for site, location in enumerate(tqdm(stimulation_locations, desc='stimulation locations')):
            candidate.perturb(perturbation=None, target='IT')  # reset
            self._logger.debug(f"Injecting at {location}")
            candidate.perturb(perturbation=BrainModel.Perturbation.muscimol,
                              target='IT', perturbation_parameters={
                    # "1 μL of muscimol (5 mg/mL) was injected at 0.1 μL/min rate"
                    'amount_µl': 1,
                    'mg_per_ml': 5,
                    'rate_µl_per_min': 0.1,
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

        # accuracies
        accuracies = self.characterize_delta_accuracies(unperturbed_behavior=unperturbed_behavior,
                                                        perturbed_behaviors=candidate_behaviors)

        # face selectivities
        selectivities = determine_selectivity(recordings)
        self.attach_selectivity(accuracies, selectivities)

        # compare
        score = self._metric(accuracies, self._assembly)
        # TODO: ceiling normalize
        return score

    def characterize_delta_accuracies(self, unperturbed_behavior, perturbed_behaviors):
        unperturbed_accuracy = self.accuracy(unperturbed_behavior)

        accuracy_deltas = []
        for site_iteration in perturbed_behaviors['site_iteration'].values:
            # index instead of `.sel` to preserve all site coords
            behavior = perturbed_behaviors[{'site': [site == site_iteration for site
                                                     in perturbed_behaviors['site_iteration'].values]}]
            site_coords = {coord: (dims, values) for coord, dims, values in walk_coords(behavior['site'])}
            behavior = behavior.squeeze('site', drop=True)

            site_accuracy = self.accuracy(behavior)
            accuracy_delta = site_accuracy - unperturbed_accuracy

            accuracy_delta = accuracy_delta.expand_dims('site')
            for coord, (dims, values) in site_coords.items():
                accuracy_delta[coord] = dims, values
            accuracy_deltas.append(DataAssembly(accuracy_delta))
        accuracy_deltas = merge_data_arrays(accuracy_deltas)
        return accuracy_deltas

    def accuracy(self, behavior):
        labels = behavior['image_label']
        correct_choice_index = [behavior['choice'].values.tolist().index(label) for label in labels]
        behavior = behavior.transpose('presentation', 'choice')  # we're operating on numpy array directly below
        accuracies = behavior.values[np.arange(len(behavior)), correct_choice_index]
        accuracies = DataAssembly(accuracies,
                                  coords={coord: (dims, values) for coord, dims, values
                                          in walk_coords(behavior['presentation'])},
                                  dims=['presentation'])
        return accuracies.mean('presentation')

    def attach_selectivity(self, accuracies, selectivities):
        assert len(accuracies) == len(selectivities)
        # assume same ordering
        accuracies['face_selectivity'] = 'site', selectivities.values  # TODO: should this be d'? fig. 4


def determine_selectivity(recordings):
    assert (recordings >= 0).all()
    # A d' value of zero indicates indistinguishable responses to faces and non-faces.
    # Increasingly positive d' values indicate progressively better selectivity for faces.
    # Selectivity for faces was defined as having a d' value > 1.
    result = []
    iterator = recordings['neuroid_id'].values
    if len(iterator) > 1:
        iterator = tqdm(iterator, desc='neuron face dprime')
    for neuroid_id in iterator:
        neuron = recordings.sel(neuroid_id=neuroid_id)
        neuron = neuron.squeeze()
        face_mean, face_variance = mean_var(neuron.sel(category='face'))
        nonface_mean, nonface_variance = mean_var(neuron.sel(category='object'))
        # face selectivity based on "more positive" firing
        dprime = (face_mean - nonface_mean) / np.sqrt((face_variance + nonface_variance) / 2)
        result.append(dprime)
    result = DataArray(result, coords={'neuroid_id': recordings['neuroid_id'].values}, dims=['neuroid_id'])
    return result
