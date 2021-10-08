from brainscore.utils import LazyLoad
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from brainscore.metrics.inter_individual_stats_ceiling import InterIndividualStatisticsCeiling
from brainscore.benchmarks import BenchmarkBase
from brainscore.model_interface import BrainModel
from brainscore.metrics.accuracy import Accuracy
from brainscore.metrics.performance_similarity import PerformanceSimilarity  # TODO
from brainio.assemblies import merge_data_arrays, DataArray

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

STIMULATION_PARAMETERS = {
    'type': [None, BrainModel.Perturbation.microstimulation, BrainModel.Perturbation.microstimulation],
    'current_pulse_mA': [0, 100, 300],
    'pulse_rate_Hz': 150,
    'pulse_duration_ms': 0.2,
    'pulse_interval_ms': 0.1,
    'stimulation_duration_ms': 200
}

DPRIME_THRESHOLD_SELECTIVITY = .66  # equiv to .5 in monkey
DPRIME_THRESHOLD_FACE_PATCH = .85  # equiv to .65 in monkey


class _Moeller2017(BenchmarkBase):

    def __init__(self, stimulus_class, perturbation_location,
                 identifier,
                 metric=PerformanceSimilarity,
                 performance_measure=Accuracy):
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

        self._metric = metric()
        self._performance_measure = performance_measure()
        self._perturbations = self._set_up_perturbations(perturbation_location)
        self._stimulus_class = stimulus_class

        self._target_assembly = self._collect_target_assembly()
        self._stimulus_set = self._target_assembly.stimulus_set
        self._stimulus_set_FP = self._target_assembly.stimulus_set_FP
        self._stimulus_set_training = self._target_assembly.stimulus_set_training

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
        self._compute_perturbation_coordinates(candidate)
        self._set_up_decoder(candidate)

        candidate.start_recording(recording_target='IT', time_bins=[(70, 170)])
        candidate_performance = []
        for perturbation in self._perturbations:
            behavior = self._perform_task(candidate, perturbation=perturbation)
            performance = self._compute_performance(behavior)
            candidate_performance.append(performance)
        candidate_performance = merge_data_arrays(candidate_performance)

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
        behavior['current_pulse_mA'] = behavior.dims[0], perturbation['perturbation_parameters']['current_pulse_mA']
        return behavior

    def _compute_behavior(self, IT_recordings):
        '''
        TODO actually sample vs. exhaustively run all conditions
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
            rng = np.random.default_rng(seed=self._seed)
            recording_size = len(IT_recordings[0].values)

            category_pool = IT_recordings.sel(object_name=object_name)
            random_indeces = rng.integers(0, len(category_pool), (samples, 2))

            sampled_recordings = np.full((samples, recording_size * 2), np.nan)
            for i, (random_idx_same, random_idx_different) in enumerate(random_indeces):
                # 'same_id': object_id ==, image_id!=
                image_one_same = category_pool[random_idx_same]
                sampled_recordings[i, :recording_size] = image_one_same.values
                sampled_recordings[i, recording_size:] = rng.choice(category_pool.where(
                    category_pool.object_id == image_one_same.object_id and
                    category_pool.image_id != image_one_same.image_id))

                # 'different_id': object_id !=
                image_one_diff = category_pool[random_idx_different]
                sampled_recordings[i + samples, :recording_size] = image_one_diff.values
                sampled_recordings[i + samples, recording_size:] = rng.choice(category_pool.where(
                    category_pool.object_id != image_one_diff.object_id))

            conditions = ['same_id'] * samples + ['different_id'] * samples
            return sampled_recordings, conditions

        samples = 500
        behavior_data = []
        for object_name in set(IT_recordings.object_name):
            recordings, conditions = _sample_recordings()
            choices = self._decoder.predict(recordings)
            behavior = DataArray(data=choices, dims='condition',
                                 coords={'condition': conditions,
                                         'object_name': ('condition', [object_name] * samples * 2)})
            behavior_data.append(behavior)

        behaviors = merge_data_arrays(behavior_data)
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
                    performance_array = DataArray(data=performance, dims='condition',
                                                  coords={'condition': condition,
                                                          'object_name': ('condition', object_name),
                                                          'current_pulse_mA': ('condition', current_pulse_mA)})
                    performance_data.append(performance_array)

        performances = merge_data_arrays(performance_data)
        return performances

    def _compute_perturbation_coordinates(self, candidate):
        '''
        Save stimulation coordinates (x,y) to self._perturbation_coordinates
        :param candidate: BrainModel
        '''
        candidate.start_recording('IT', time_bins=[(50, 100)])
        recordings = candidate.look_at(self._stimulus_set_FP)  # vs _training_assembly

        # 1 smooth spatial pattern by smoothing activity with gaussian kernel 1mm
        # 2 voxelize
        recordings_voxelized = self._spatial_smoothing(recordings, fwhm=1, res=.5)

        # 3. compute face selectivity
        face_selectivities_voxel = self._determine_face_selectivity(recordings_voxelized)

        # 4. Determine location
        if self._perturbation_location == 'within_facepatch':
            x, y = self._get_purity_center(face_selectivities_voxel)
        elif self._perturbation_location == 'outside_facepatch':
            x, y = self._sample_outside_FP(face_selectivities_voxel)
        else:
            raise KeyError

        self._perturbation_coordinates = (x, y)

    def _set_up_perturbations(self, perturbation_location):
        '''
        :param: perturbation_location: string ['within_facepatch','outside_facepatch']
        :return: list of dict, each containing parameters for one perturbation
        '''
        self._perturbation_location = perturbation_location

        perturbation_list = []
        for stimulation, current in zip(STIMULATION_PARAMETERS['type'], STIMULATION_PARAMETERS['current_pulse_mA']):
            perturbation_dict = {'type': stimulation,
                                 'perturbation_parameters': {
                                     'current_pulse_mA': current,
                                     'stimulation_duration_ms': STIMULATION_PARAMETERS['stimulation_duration_ms'],
                                     'location': LazyLoad(self._perturbation_coordinates)
                                 }}

            perturbation_list.append(perturbation_dict)
        return perturbation_list

    def _collect_target_assembly(self):
        '''
        TODO make sure assembly.stimulus_set corresponds to target_stimulus_class
        TODO stimulus set FP
        TODO stimulus set training decoder
        TODO full list of objects
        TODO make sure assembly has source attribute --> ceiling??
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
        # TODO add object_ID & image_ID coords; image_ID = expression or viewing angle
        return target_assembly

    def _set_up_decoder(self, candidate):
        '''
        Fit a linear regression between the recordings of the training stimuli and the ground truth
        Assign to self._decoder
        TODO sample recordings
        '''
        candidate.start_recording()
        recordings = candidate.look_at(self._stimulus_set_training)
        stimulus_set = None
        truth = None

        self._decoder = LinearRegression()
        self._decoder.fit(stimulus_set, truth)

    def _spatial_smoothing(self, assembly, fwhm=1., res=0.5):
        """
        Adapted from Hyo's 2020 paper code
        Do spatial filtering for each voxel
            Args:
                fwmh: FWMH for the gaussian kernel. default is 3mm.
        """

        def _get_grid_coord():
            """
            Adapted from Hyo's 2020 paper code
            Returns coordinates of grid with width of 0.5
            """
            x = x.reshape(len(x), 1)
            y = y.reshape(len(y), 1)
            xmin, xmax = np.floor(np.min(x)), np.ceil(np.max(x))
            ymin, ymax = np.floor(np.min(y)), np.ceil(np.max(y))
            grids = np.array(np.meshgrid(np.arange(xmin, xmax, res), np.arange(ymin, ymax, res)))
            gridx = grids[0].flatten().reshape(-1, 1)
            gridy = grids[1].flatten().reshape(-1, 1)
            return gridx, gridy

        # Get voxel coordinates
        x, y = assembly.neuroid.tissue_x.values, assembly.neuroid.tissue_y.values
        gridx, gridy = _get_grid_coord()

        # compute sigma from fwmh
        sigma = fwhm / np.sqrt(8. * np.log(2))

        # define gaussian kernel
        d_square = (x - gridx.T) ** 2 + (y - gridy.T) ** 2
        gf = 1. / (2 * np.pi * sigma ** 2) * np.exp(- d_square / (2 * sigma ** 2))

        features_smoothed = DataArray(
            data=np.dot(assembly.values, gf),
            dims=['category_name', 'neuroid_id'],
            coords={'category_name': assembly.category_name.values,
                    'neuroid_id': assembly.neuroid_id.values,
                    'voxel_x': gridx.squeeze(),  # TODO
                    'voxel_y': gridy.squeeze()}
        )
        return features_smoothed,

    @staticmethod
    def _get_purity_center(selectivity_assembly,
                           radius=1):
        '''
        Adapted from Hyo's 2020 paper code
        Computes the cell of the selectivity map with the highest purity
        Inputs:
            radius (scalar): radius in mm of the circle in which to consider units
        '''

        def get_purity(center_x, center_y):
            '''
            Evaluates purity at a given center position, radius, and corresponding selectivity values
            '''
            passing_indices = np.where(np.sqrt(np.square(x - center_x) + np.square(y - center_y)) < radius)[0]
            return 100. * np.sum(selectivity_assembly.values[passing_indices]) / passing_indices.shape[0]

        x, y = selectivity_assembly.voxel_x.values, selectivity_assembly.voxel_y.values
        purity = np.array(list(map(get_purity, x, y)))
        highest_purity_idx = np.argmax(purity)

        center_x, center_y = x[highest_purity_idx], y[highest_purity_idx]
        return center_x, center_y

    @staticmethod
    def _determine_face_selectivity(recordings):
        def mean_var(neuron):
            mean, var = np.mean(neuron.values), np.var(neuron.values)
            return mean, var

        assert (recordings >= 0).all()

        selectivities = []
        for neuroid_id in tqdm(recordings['neuroid_id'].values, desc='neuron face dprime'):
            neuron = recordings.sel(neuroid_id=neuroid_id)
            neuron = neuron.squeeze()
            face_mean, face_variance = mean_var(neuron.sel(category_name='Faces'))  # image_label='face'))
            nonface_mean, nonface_variance = mean_var(
                neuron.where(neuron.category_name != 'Faces', drop=True))  # sel(image_label='nonface'))
            dprime = (face_mean - nonface_mean) / np.sqrt((face_variance + nonface_variance) / 2)
            selectivities.append(dprime)

        selectivity_array = recordings.copy()  # DataArray(result, coords={'neuroid_id': recordings['neuroid_id'].values}, dims=['neuroid_id'])
        selectivity_array.data = selectivities
        return selectivity_array

    def _sample_outside_FP(self, selectivity_assembly, radius=2):
        '''
        Sample one voxel outside of face patch
        1. make a list of voxels where neither the voxel nor its close neighbors are in a face patch
        2. randomly sample from list
        :param selectivity_assembly:
        :param radius: determining the neighborhood size in mm of each voxel that cannot be selective
        :return: x, y location of voxel outside FP
        '''
        not_selective_voxels = selectivity_assembly[selectivity_assembly.values < DPRIME_THRESHOLD_SELECTIVITY]
        voxels = []
        for voxel in not_selective_voxels:
            inside_radius = np.where(np.sqrt(np.square(not_selective_voxels.voxel_x.values - voxel.voxel_x.values) +
                                             np.square(not_selective_voxels.voxel_y.values - voxel.voxel_y.values))
                                     < radius)[0]
            if np.all(not_selective_voxels[inside_radius].values < DPRIME_THRESHOLD_FACE_PATCH):
                voxels.append(voxel)

        rng = np.random.default_rng(seed=self._seed)
        voxel = rng.choice(voxels)
        x, y = voxel.voxel_x, voxel.voxel_y
        return x, y

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
    TODO  very unclean
    EXP2:
    i: Stimulate outside of the face patch during face identification
    ii: Stimulate face patch during object identification
    '''

    class _Moeller2017Exp2(BenchmarkBase):
        def __init__(self):
            super().__init__(
                identifier='dicarlo.Moeller2017-Experiment_2', ceiling_func=None, version=1, parent='IT', bibtex=BIBTEX)
            self.benchmark1 = _Moeller2017(stimulus_class='faces', perturbation_location='outside_facepatch',
                                           identifier='dicarlo.Moeller2017-Experiment_2i')
            self.benchmark2 = _Moeller2017(stimulus_class='objects', perturbation_location='within_facepatch',
                                           identifier='dicarlo.Moeller2017-Experiment_2ii')

        def __call__(self, candidate):
            return self.benchmark1(candidate), self.benchmark2(candidate)

    return _Moeller2017Exp2()


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
