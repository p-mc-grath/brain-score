from brainscore.utils import LazyLoad
from xarray import DataArray
import numpy as np
import pandas as pd
import xarray as xr
from brainscore.metrics.inter_individual_stats_ceiling import InterIndividualStatisticsCeiling
from brainscore.benchmarks import BenchmarkBase
from brainscore.model_interface import BrainModel

# TODO create metrics
from brainscore.metrics.performance_measures import percent_correct
from brainscore.metrics.performance_similarity import performance_similarity

BIBTEX = '''@article{article,
            author = {Moeller, Sebastian and Crapse, Trinity and Chang, Le and Tsao, Doris},
            year = {2017},
            month = {03},
            pages = {},
            title = {The effect of face patch microstimulation on perception of faces and objects},
            volume = {20},
            journal = {Nature Neuroscience},
            doi = {10.1038/nn.4527}
            }'''


class _Moeller2017(BenchmarkBase):

    def __init__(self, stimulus_class, perturbation_location,
                 identifier,
                 metric=performance_similarity,
                 performance_measure=percent_correct):
        '''
        Perform a same vs different identity judgement task on the given dataset
            with and without Microstimulation in the specified location.
            Compute the behavioral performance on the given performance measure and
            compare it to the equivalent data retrieved from primate experiments.

        For each decision, only identities from the same object category are being compared.
        Within each dataset, the number of instances per category is equalized. As is the number of different
        representations (faces: expressions, object: viewing angles) per instance.

        :param stimulus_class: string: ['faces', 'objects', 'non_face_objects_eliciting_FP_response_plus_faces',
                                        'abstract_faces', 'abstract_houses']
        :param perturbation_location: string: ['within_facepatch', 'outside_facepatch']
        :param identifier: string: benchmark id
        :param metric: i: performances along multiple dimensions of 2 instances | o: Score object, evaluating similarity
        :param performance_measure: taking behavioral data, returns performance w.r.t. each dimension
        '''
        super().__init__(
            identifier=identifier,
            ceiling_func=lambda: InterIndividualStatisticsCeiling(metric)(self._target_assembly),
            version=1, parent='IT',
            bibtex=BIBTEX)

        self._stimulus_class = stimulus_class
        self._perturbation_location = perturbation_location
        self._perturbations = [{'type': None,
                                'perturbation_parameters': {'current_pulse_mA': 0}},
                               {'type': BrainModel.Perturbation.microstimulation,
                                'perturbation_parameters': {'current_pulse_mA': 100,
                                                            'stimulation_duration_ms': 200,
                                                            'pulse_rate_Hz': 150,
                                                            'location': LazyLoad(self._perturbation_coordinates)}},
                               {'type': BrainModel.Perturbation.microstimulation,
                                'perturbation_parameters': {'current_pulse_mA': 300,
                                                            'stimulation_duration_ms': 200,
                                                            'pulse_rate_Hz': 150,
                                                            'location': LazyLoad(self._perturbation_coordinates)}}]

        self._metric = metric()
        self._performance_measure = performance_measure()
        self._target_assembly = self._collect_target_assembly()
        self._stimulus_set = self._target_assembly.stimulus_set
        self._training_assembly = self._collect_train_assembly()

        self._seed = 123

    def __call__(self, candidate: BrainModel):
        '''
        Score model on chosen identification task
        :param candidate: BrainModel
        :return: Score.data = score per experiment
                 Score.raw =  score per category
                 Score.performance = performance aggregated per experiment
                 Score.raw_performance = performance per category
        '''
        self._perturbation_coordinates = self._compute_perturbation_coordinates(candidate)
        self._decoder = self._get_decoder(candidate)

        candidate.start_recording(recording_target='IT', time_bins=[(70, 170)])
        candidate_performance = []
        for perturbation in self._perturbations:
            behavior = self._perform_task(candidate, perturbation=perturbation)
            performance = self._compute_performance(behavior)
            candidate_performance.append(performance)

        # TODO use martins merge instead of concat
        candidate_performance = xr.concat(candidate_performance, dim='meta')  # merge_data_arrays(candidate_performance)

        score = self._metric(candidate_performance, self._target_assembly)
        return score

    def _perform_task(self, candidate: BrainModel, perturbation):
        '''
        Perturb model and compute behavior w.r.t. task
        :param candidate: BrainModel
        :perturbation dict with keys: type, perturbation_parameters
        :return: DataArray: values = choice, dims = [truth, current_pulse_mA, condition, object_name]
        '''
        candidate.perturb(perturbation=None, target='IT')  # reset
        candidate.perturb(perturbation=perturbation['type'], target='IT',
                          perturbation_parameters=perturbation['perturbation_parameters'])

        IT_recordings = candidate.look_at(self._stimulus_set)

        behavior = self._compute_behavior(IT_recordings)
        behavior = self._add_perturbation_info(behavior, perturbation)
        return behavior

    def _compute_perturbation_coordinates(self, candidate):
        '''
        TODO Compute stimulation site coordinates
        :param candidate: BrainModel
        :return (x, y) coordinates for perturbation according to self._perturbation_location
        '''
        face_selectivity_map = compute_face_selectivity(candidate)
        patch = find_FP(face_selectivity_map)
        if self._perturbation_location == 'within_facepatch':
            # find middle
            x, y = find_middle(patch)
        elif self._perturbation_location == 'outside_facepatch':
            # rand sample with certain distance
            x, y = sample_outside(patch)
        else:
            raise KeyError

        return (x, y)

    def _compute_behavior(self, IT_recordings):
        '''
        TODO xarray rng compatibility
        TODO xarray indexing compatibility
        TODO xarray lin model compatibility
        Compute behavior of given IT recordings in a identity matching task,
            i.e. given two images of the same category judge if they depict an object of same or different identity
        :param IT_recordings: DataArray:
            values: IT activation vectors
            dims:   object_name : category
                    object_ID   : object identity
                    image_ID    : object + view angle identity
        :return: behaviors DataArray
            values: choice
            dims:   truth       : ground truth
                    object_name : category
        '''

        def _sample_recordings():
            rng = np.random.defaul_rng(seed=self._seed)
            number_of_samples = 1000

            recording_pool_one = IT_recordings.sel(object_name=category)
            recordings_image_one = rng.choice(recording_pool_one, number_of_samples)  # TODO
            if condition == 'same_id':
                recordings_image_two = rng.choice(recording_pool_one, number_of_samples)
                not_same_recording = recordings_image_one != recordings_image_two
                recordings_image_one = recordings_image_one[not_same_recording]  # TODO
                recordings_image_two = recordings_image_two[not_same_recording]  # TODO
            elif condition == 'different_id':
                recording_pool_two = IT_recordings.where(IT_recordings.object_name != category, drop=True)
                recordings_image_two = rng.choice(recording_pool_two, number_of_samples)

            return zip(recordings_image_one, recordings_image_two)

        behavior_data = []
        for category in set(IT_recordings.object_name):
            for condition in ['same_id', 'different_id']:
                for recording_image_one, recording_image_two in _sample_recordings():
                    choice = self._decoder(recording_image_one, recording_image_two)  # TODO
                    behavior_data.append((choice, condition, category))

        behaviors = self._behavior_to_dataarray(behavior_data)
        return behaviors

    def _compute_performance(self, behavior):
        '''
        Given performance measure and behavior, compute performance w.r.t. current_pulse_mA, condition, object name
        :param behavior: DataArray: values = choice, dims = [truth, current_pulse_mA, condition, object name]
        :return: DataArray: values = performances, dims = [current_pulse_mA, condition, object_name]
        '''
        performance_data = []
        for object_name in set(behavior.object_name):
            for current_pulse_mA in set(behavior.current_pulse_mA):
                for condition in set(behavior.condition):
                    performance = self._performance_measure(behavior.sel(current_pulse_mA=current_pulse_mA,
                                                                         condition=condition,
                                                                         object_name=object_name))
                    performance_data.append((performance, object_name, condition, current_pulse_mA))

        performances = self._performance_to_xarray(performance_data)
        return performances

    def _collect_target_assembly(self):
        '''
        TODO make sure assembly.stimulus_set corresponds to target_stimulus_class
        TODO full list of objects
        Load Data from path + subselect as specified by Experiment

        :return: DataAssembly
        '''
        STIMULUS_CLASS_DICT = {
            'faces': ['faces'],  # 32; 6 expression each
            'objects': [],  # TODO 28; 3 viewing angles each
            'non_face_objects_eliciting_FP_response_plus_faces': ['apples_bw', 'citrus_fruits_bw', 'teapots_bw',
                                                                  'clocks_bw', 'faces_bw'],  # 15; 3/cats
            'abstract_faces': ['face_cartoons', 'face_linedrawings', 'face_mooneys', 'face_silhouettes'],  # 16; 4/cat
            'abstract_houses': ['house_cartoons', 'house_linedrawings', 'house_silhouettes', 'face_mooneys',
                                'face_mooneys_inverted']  # 20; 4/cat
        }

        assembly = self._load_assembly()

        target_stimulus_class = STIMULUS_CLASS_DICT[self._stimulus_class]
        target_assembly = assembly.multisel(object_name=target_stimulus_class)  # TODO

        return target_assembly

    def _get_decoder(self, candidate):
        '''
        TODO find linear model
        TODO sample recordings
        :return:
        '''
        candidate.start_recording()

        decoder = None  # TODO
        recordings = candidate.look_at(self._training_assembly.stimulus_set)
        stimulus_set = None  # TODO
        truth = None  # TODO
        decoder.fit(stimulus_set, truth)
        return decoder

    @staticmethod
    def _load_assembly():
        '''
        TODO Load DataArray from path
        TODO package all data
        TODO get stimulus set
        make sure it has a monkey dimension for ceiling computation
        :return: DataArray
        '''
        assembly = None  # TODO
        return assembly

    @staticmethod
    def _collect_train_assembly():
        '''
        TODO load assembly from path
        :return: Assembly of training stimuli
        '''
        assembly = None  # TODO
        return assembly

    @staticmethod
    def _add_perturbation_info(behavior, perturbation):
        '''
        Adds current_pulse_mA information to behavior
        :param behavior: DataArray
        :param perturbation: dict within dict with perturbation['perturbation_parameters']['current_pulse_mA'] = int
        :return: Behavior with added information on stimulus current_pulse_mA
        '''
        behavior = behavior.expand_dims('current_pulse_mA')
        behavior['current_pulse_mA'] = perturbation['perturbation_parameters']['current_pulse_mA']
        behavior = type(behavior)(behavior)  # make sure site and injected are indexed
        return behavior

    @staticmethod
    def _behavior_to_dataarray(behavior_data):
        # TODO change to brainio dataarray
        choices = truths = categories = []
        for choice, truth, category in behavior_data:
            choices.append(choice)
            truths.append(truth)
            categories.append(category)

        xarray_behavior = DataArray(
            data=choices,
            dims=["meta"],
            coords={
                'meta': pd.MultiIndex.from_product([truths, categories],
                                                   names=('truths', 'categories'))
            }
        )

        return xarray_behavior

    @staticmethod
    def _performance_to_xarray(performance_data):
        '''
        TODO change to brainio DataArray
        Create DataArray from list of tuples
        :param performance_data: list of tuples; ea/ containing 3 values: (performance, condition, current_pulse_mA)
        :return: DataArray containing all input info
        '''
        performances = object_names = conditions = currents = []
        for performance, object_name, condition, current in performance_data:
            performances.append(performance)
            object_names.append(object_name)
            conditions.append(condition)
            currents.append(current)

        xarray_performance = DataArray(
            data=performances,
            dims=["meta"],
            coords={
                'meta': pd.MultiIndex.from_product([object_names, conditions, currents],
                                                   names=('object_name', 'condition', 'current_pulse_mA'))
            }
        )

        return xarray_performance


def Moeller2017Exp1():
    '''
    EXP1:
    Stimulate face patch during face identification
    '''
    return _Moeller2017(stimulus_class='faces',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_1')


def Moeller2017Exp2():
    '''
    # TODO this does not work, returning 2 benchmarks -> helper benchmark? -? very unclean
    EXP2:
    i: Stimulate outside of the face patch during face identification
    ii: Stimulate face patch during object identification

    '''
    a = {'Experiment_2i': {'stimulus_class': 'faces',
                           'stimulation_location': 'outside_facepatch'},
         'Experiment_2ii': {'stimulus_class': 'objects',
                            'stimulation_location': 'within_facepatch'}}
    exp2i = _Moeller2017(experiment='Experiment_2i')
    exp2ii = _Moeller2017(experiment='Experiment_2ii')
    exp2 = (exp2i, exp2ii)
    return exp2


def Moeller2017Exp3():
    '''
    EXP3:
    Stimulate face patch during face & non-face object eliciting patch response identification
    '''
    return _Moeller2017(stimulus_class='non_face_objects_eliciting_FP_response_plus_faces',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_3')


def Moeller2017Exp4a():
    '''
    EXP4a:
    Stimulate face patch during abstract face identification
    '''
    return _Moeller2017(stimulus_class='abstract_faces',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_4a')


def Moeller2017Exp4b():
    '''
    EXP4b:
    Stimulate face patch during face & abstract houses identification
    '''
    return _Moeller2017(stimulus_class='abstract_houses',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_4b')
