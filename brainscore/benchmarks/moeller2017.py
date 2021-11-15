import itertools
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression

from brainscore.utils import LazyLoad
from brainscore.metrics import Metric
from brainscore.benchmarks import BenchmarkBase
from brainscore.model_interface import BrainModel
from brainscore.metrics.accuracy import Accuracy
from brainscore.metrics.performance_similarity import PerformanceSimilarity  # TODO
from brainio.assemblies import merge_data_arrays, DataArray, DataAssembly
from packaging.moeller2017.moeller2017 import collect_target_assembly

# TODO within Face patch means within AM right now

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
    'current_pulse_mA': [0, 300],
    'pulse_rate_Hz': 150,
    'pulse_duration_ms': 0.2,
    'pulse_interval_ms': 0.1,
    'stimulation_duration_ms': 200
}

DPRIME_THRESHOLD_SELECTIVITY = .66  # equivalent to .5 in monkey, see Lee et al. 2020
DPRIME_THRESHOLD_FACE_PATCH = .85  # equivalent to .65 in monkey, see Lee et al. 2020


class _Moeller2017(BenchmarkBase):

    def __init__(self, stimulus_class: str, perturbation_location: str, identifier: str,
                 metric: Metric, performance_measure):
        '''
        Perform a same vs different identity judgement task on the given dataset
            with and without Microstimulation in the specified location.
            Compute the behavioral performance on the given performance measure and
            compare it to the equivalent data retrieved from primate experiments.

        For each decision, only identities from the same object category are being compared.
        Within each dataset, the number of instances per category is equalized. As is the number of different
        representations (faces: expressions, object: viewing angles) per instance.

        :param stimulus_class: one of: ['Faces', 'Objects', 'Eliciting_Face_Response', 'Abstract_Faces', 'Abstract_Houses']
        :param perturbation_location: one of: ['within_facepatch', 'outside_facepatch']
        :param identifier: benchmark id
        :param metric: in: performances along multiple dimensions of 2 instances | out: Score object, evaluating similarity
        :param performance_measure: taking behavioral data, returns performance w.r.t. each dimension
        '''
        super().__init__(
            identifier=identifier,
            ceiling_func=lambda: None,
            version=1, parent='IT',
            bibtex=BIBTEX)

        self._metric = metric
        self._performance_measure = performance_measure
        self._perturbations = self._set_up_perturbations(perturbation_location)
        self._stimulus_class = stimulus_class

        self._target_assembly = collect_target_assembly(stimulus_class=stimulus_class,
                                                        perturbation_location=perturbation_location)
        self._stimulus_set = self._target_assembly.stimulus_set
        self._stimulus_set_face_patch = self._target_assembly.stimulus_set_face_patch
        self._training_stimuli = self._target_assembly.training_stimuli

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
        self._compute_perturbation_coordinates(candidate)  # TODO move to modeltools
        decoder = self._set_up_decoder(candidate)

        candidate.start_recording(recording_target='IT', time_bins=[(70, 170)])
        candidate_performance = []
        for perturbation in self._perturbations:
            behavior = self._perform_task(candidate, perturbation=perturbation,
                                          decoder=decoder)  # TODO move to modeltools
            performance = self._compute_performance(behavior)
            candidate_performance.append(performance)
        candidate_performance = merge_data_arrays(candidate_performance)

        score = self._metric(candidate_performance, self._target_assembly)
        return score

    def _perform_task(self, candidate: BrainModel, perturbation: dict, decoder):
        '''
        Perturb model and compute behavior w.r.t. task
        :param candidate: BrainModel
        :perturbation keys: type, perturbation_parameters
        :return: DataAssembly: values = choice, dims = [truth, current_pulse_mA, condition, object_name]
        '''
        candidate.perturb(perturbation=None, target='IT')  # reset
        candidate.perturb(perturbation=perturbation['type'], target='IT',
                          perturbation_parameters=perturbation['perturbation_parameters'])

        IT_recordings = candidate.look_at(self._stimulus_set)

        behavior = self._compute_behavior(IT_recordings, decoder)
        behavior['current_pulse_mA'] = behavior.dims[0], \
                                       [perturbation['perturbation_parameters']['current_pulse_mA']] * behavior.shape[0]
        behavior = type(behavior)(behavior)  # make sure current_pulse_mA is indexed
        return behavior

    def _compute_behavior(self, IT_recordings: DataArray, decoder):
        '''
        Compute behavior of given IT recordings in a identity matching task,
            i.e. given two images of the same category judge if they depict an object of same or different identity
        :param IT_recordings:
            values: IT activation vectors
            dims:   object_name : list of strings, category
            coords: object_ID   : list of strings, object identity
                    image_ID    : list of strings, object + view angle identity
        :return: behaviors DataAssembly
            values: choice
            dims:   condition   : list of strings, ['same_id == 1, 'different_id'==0],
            coords: truth       : list of int/bool, 'same_id'==1, 'different_id'==0,
                    object_name : list of strings, category
        '''
        samples = 500  # TODO why 500
        behavior_data = []
        for object_name in set(IT_recordings.object_name.values):
            recordings, conditions = self._sample_recordings(IT_recordings.sel(object_name=object_name),
                                                             samples=samples)
            choices = (decoder.predict(recordings) > .5).astype(int)
            behavior = DataArray(data=choices, dims='condition',
                                 coords={'condition': conditions,
                                         'truth': ('condition', (np.array(conditions) == 'same_id').astype(int)),
                                         'object_name': ('condition', [object_name] * samples * 2)})
            behavior_data.append(behavior)

        behaviors = self._merge_behaviors(behavior_data)
        return behaviors

    def _compute_performance(self, behaviors: DataAssembly):
        '''
        Given performance measure and behavior, compute performance w.r.t. current_pulse_mA, condition, object name
        :param behaviors:
            values: choice          : 'same_id'==1, 'different_id'==0
            dims:   condition       : list of strings, ['same_id', 'different_id]
            coords: truth           : 'same_id'==1, 'different_id'==0
                    object name     : list of strings, category
                    current_pulse_mA: float
        :return: DataAssembly:
            values: performances    : accuracy values
            dims:   condition       : list of strings, ['same_id', 'different_id]
            coords: object_name     : list of strings, category
                    current_pulse_mA: float
        '''
        performance_data = []
        for object_name, current_pulse_mA, truth in itertools.product(set(behaviors.object_name.values),
                                                                      set(behaviors.current_pulse_mA.values),
                                                                      set(behaviors.truth.values)):
            behavior = behaviors.sel(current_pulse_mA=current_pulse_mA, truth=truth, object_name=object_name)
            performance = self._performance_measure(behavior, [truth] * len(behavior))
            performance_array = DataAssembly(data=[performance.data[0]], dims='condition',
                                             coords={'condition': [behavior.condition_level_0.values[0]],
                                                     # because need DataAssembly for .sel above
                                                     'object_name': ('condition', [object_name]),
                                                     'current_pulse_mA': ('condition', [current_pulse_mA])})
            performance_data.append(performance_array)

        performances = merge_data_arrays(performance_data)
        return performances

    def _set_up_perturbations(self, perturbation_location: str):
        '''
        Create a list of dictionaries, each containing the parameters for one perturbation
        :param: perturbation_location: one of ['within_facepatch','outside_facepatch']
        :return: list of dict, each containing parameters for one perturbation
        '''
        self._perturbation_location = perturbation_location

        perturbation_list = []
        for stimulation, current in zip(STIMULATION_PARAMETERS['type'], STIMULATION_PARAMETERS['current_pulse_mA']):
            perturbation_dict = {'type': stimulation,
                                 'perturbation_parameters': {
                                     'current_pulse_mA': current,
                                     'stimulation_duration_ms': STIMULATION_PARAMETERS['stimulation_duration_ms'],
                                     'location': LazyLoad(lambda: self._perturbation_coordinates)
                                 }}

            perturbation_list.append(perturbation_dict)
        return perturbation_list

    def _set_up_decoder(self, candidate: BrainModel):
        '''
        Fit a logistic regression between the recordings of the training stimuli and the ground truth
        :return: trained linear regressor
        '''
        candidate.start_recording(recording_target='IT', time_bins=[(70, 170)])
        recordings = candidate.look_at(self._training_stimuli)
        samples = 500  # TODO why 500

        stimulus_set, truth = [], []
        for object_name in set(recordings.object_name.values):
            recordings_category, conditions = self._sample_recordings(recordings.sel(object_name=object_name),
                                                                      samples=samples)
            stimulus_set.append(recordings_category)
            truth += list((np.array(conditions) == 'same_id').astype(int))
        stimulus_set = np.vstack(stimulus_set)

        return LogisticRegression(random_state=self._seed,
                                  solver='liblinear').fit(stimulus_set, truth)

    def _sample_recordings(self, category_pool: DataArray, samples=500):  # TODO why 500
        '''
        Create an array of randomly sampled recordings, each line is one task, i.e. two recordings which are to be
        judged same vs. different ID

        From Online Methods: Visual Stimuli & Behavioral Tasks section:
        " By presenting faces of the same identity with six different expressions in image set 1
        and objects of the same identity at three different view angles in image set 2, we
        ensured that even in the match condition we presented two different images"
        -> for Faces & Objects, the same condition excludes the same image as stimulus. For
            all other Experiments, same means the exact same !note: decoder only trained on Faces

        :param category_pool: Model IT recordings, assumed be from one category only
        :param samples: int: number of samples
        :return: array (samples x length of two concatenated recordings), each line contains two recordings
                 ground truth, for each line in array, specifying if the two recordings belong to the same/different ID
        '''
        # TODO object and image_id disappear with subselection
        rng = np.random.default_rng(seed=self._seed)
        recording_size = category_pool.shape[0]
        random_indeces = rng.integers(0, category_pool.shape[1], (samples, 2))

        sampled_recordings = np.full((samples * 2, recording_size * 2), np.nan)
        for i, (random_idx_same, random_idx_different) in tqdm(enumerate(random_indeces),
                                                               desc='decoder training recordings'):
            # condition 'same_id': object_id same between recording one and two, image_id different between recording one and two
            image_one_same = category_pool[:, random_idx_same]
            same_image_id, same_object_id = image_one_same.presentation.values.item()  # TODO weird xarray behavior, diappearing dimension after selection
            sampled_recordings[i, :recording_size] = image_one_same.values
            if self._stimulus_class in ['Faces', 'Objects']:  # not same image requirement only in Experiment 1&2
                sampled_recordings[i, recording_size:] = rng.choice(category_pool.where(
                    (category_pool.object_id == same_object_id) &
                    (category_pool.image_id != same_image_id), drop=True).T)
            else:
                sampled_recordings[i, recording_size:] = image_one_same.values
                # <=> rng.choice(category_pool.where((category_pool.object_id == same_object_id), drop=True).T)
                # because there is only one image per object_id

            # condition 'different_id': object_id different between recording one and two
            image_one_diff = category_pool[:, random_idx_different]
            diff_object_id = image_one_diff.presentation.values.item()[
                1]  # TODO weird xarray behavior, diappearing dimension after selection
            sampled_recordings[i + samples, :recording_size] = image_one_diff.values
            sampled_recordings[i + samples, recording_size:] = rng.choice(category_pool.where(
                category_pool.object_id != diff_object_id, drop=True).T)

        conditions = ['same_id'] * samples + ['different_id'] * samples
        return sampled_recordings, conditions

    def _compute_perturbation_coordinates(self, candidate: BrainModel):
        '''
        Save stimulation coordinates (x,y) to self._perturbation_coordinates
        :param candidate: BrainModel
        '''
        candidate.start_recording('IT', time_bins=[(50, 100)])
        recordings = candidate.look_at(self._stimulus_set_face_patch)  # vs _training_assembly

        # 1 smooth spatial pattern by smoothing activity with gaussian kernel 1mm
        # 2 voxelize
        recordings_voxelized = self._spatial_smoothing(recordings, fwhm=1,
                                                       res=.5)  # move to modeltools candidate.start_recording('IT', technique='fMRI', ...)

        # 3. compute face selectivity
        face_selectivities_voxel = self._determine_face_selectivity(recordings_voxelized)

        # 4. Determine location
        if self._perturbation_location == 'within_facepatch':
            x, y = self._get_purity_center(face_selectivities_voxel)
        elif self._perturbation_location == 'outside_facepatch':
            x, y = self._sample_outside_face_patch(face_selectivities_voxel)
        else:
            raise KeyError

        self._perturbation_coordinates = (x, y)

    @staticmethod
    def _spatial_smoothing(assembly: DataArray, fwhm=1., res=0.5):
        """
        Adapted from Lee et al. 2020 code, not public
        Applies 2D Gaussian to tissue activations. Aggregates Voxels from tissue.
        :param assembly,
            values: Activations
            dims:   neuroid_id
                        coords: tissue_x: x coordinates
                                tissue_y: y coordinates
                    presentations
                        coords: category_name
        :param fwmh: FWMH for the gaussian kernel. default is 1mm.
        :param res: int, resolution of voxels in mm
        """

        def _get_grid_coord(x, y):
            """
            Adapted from Lee et al. 2020 code, not public
            Returns coordinates of grid with width of 0.5
            """
            xmin, xmax = np.floor(np.min(x)), np.ceil(np.max(x))
            ymin, ymax = np.floor(np.min(y)), np.ceil(np.max(y))
            grids = np.array(np.meshgrid(np.arange(xmin, xmax, res), np.arange(ymin, ymax, res)))
            gridx = grids[0].flatten().reshape(-1, 1)
            gridy = grids[1].flatten().reshape(-1, 1)
            return gridx, gridy

        # Get voxel coordinates
        x, y = assembly.neuroid.tissue_x.values, assembly.neuroid.tissue_y.values
        x = x.reshape(len(x), 1)
        y = y.reshape(len(y), 1)
        gridx, gridy = _get_grid_coord(x, y)

        # compute sigma from fwmh
        sigma = fwhm / np.sqrt(8. * np.log(2))

        # define gaussian kernel
        d_square = (x - gridx.T) ** 2 + (y - gridy.T) ** 2
        gf = 1. / (2 * np.pi * sigma ** 2) * np.exp(- d_square / (2 * sigma ** 2))

        features_smoothed = DataAssembly(
            data=np.dot(assembly.values.T, gf),
            dims=['category_name', 'voxel_id'],
            coords={'category_name': assembly.category_name.values,
                    'voxel_id': np.arange(gf.shape[1]),
                    'voxel_x': ('voxel_id', gridx.squeeze()),
                    'voxel_y': ('voxel_id', gridy.squeeze())}
        )
        return features_smoothed

    @staticmethod
    def _get_purity_center(selectivity_assembly: DataAssembly, radius=1):
        '''
        Adapted from Lee et al. 2020 code, not public
        Computes the voxel of the selectivity map with the highest purity
        :param: selectivity_assembly:
            dims: 'neuroid_id'
                    coords:
                        voxel_x: voxel coordinate
                        voxel_y: voxel coordinate
                  'category_name'
        :param: radius (scalar): radius in mm of the circle in which to consider units
        :return: (int,int) location of highest purity
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
    def _determine_face_selectivity(recordings: DataAssembly):
        '''
        Determines face selectivity of each neuroid
        :param recordings: DataAssembly
        :return: DataAssembly, same as recordings where activations have been replaced with dprime values
        '''

        def mean_var(neuron):
            mean, var = np.mean(neuron.values), np.var(neuron.values)
            return mean, var

        assert (recordings >= 0).all(), 'selectivities must be positive'

        selectivities = []
        for voxel_id in tqdm(recordings['voxel_id'].values, desc='neuron face dprime'):
            voxel = recordings.sel(voxel_id=voxel_id)
            voxel = voxel.squeeze()
            face_mean, face_variance = mean_var(voxel.sel(category_name='Faces'))  # image_label='face'))
            nonface_mean, nonface_variance = mean_var(
                voxel.where(voxel.category_name != 'Faces', drop=True))  # sel(image_label='nonface'))
            dprime = (face_mean - nonface_mean) / np.sqrt((face_variance + nonface_variance) / 2)
            selectivities.append(dprime)

        selectivity_array = DataAssembly(
            data=selectivities,
            dims=['voxel_id'],
            coords={'voxel_id': recordings.voxel_id.values,
                    'voxel_x': ('voxel_id', recordings.voxel_x.values),
                    'voxel_y': ('voxel_id', recordings.voxel_y.values)})
        return selectivity_array

    def _sample_outside_face_patch(self, selectivity_assembly: DataAssembly, radius=2):
        '''
        # TODO exclude borders from being sampled?
        Sample one voxel outside of face patch
        1. make a list of voxels where neither the voxel nor its close neighbors are in a face patch
        2. randomly sample from list
        :param selectivity_assembly:
        :param radius: determining the neighborhood size in mm of each voxel that cannot be selective
        :return: x, y location of voxel outside face_patch
        '''
        not_selective_voxels = selectivity_assembly[selectivity_assembly.values < DPRIME_THRESHOLD_SELECTIVITY]
        voxels = []
        for voxel in not_selective_voxels:
            voxel_x, voxel_y = voxel.voxel_id.item()[1:]  # because iteration removes coords somehow
            inside_radius = np.where(np.sqrt(np.square(not_selective_voxels.voxel_x.values - voxel_x) +
                                             np.square(not_selective_voxels.voxel_y.values - voxel_y)) < radius)[0]
            if np.all(not_selective_voxels[inside_radius].values < DPRIME_THRESHOLD_FACE_PATCH):
                voxels.append(voxel)  # TODO safety if there is tissue where this does not hold

        rng = np.random.default_rng(seed=self._seed)
        random_idx = rng.integers(0, len(voxels))  # choice removes meta data i.e. location
        x, y = voxels[random_idx].voxel_id.item()[1:]
        return x, y

    @staticmethod
    def _merge_behaviors(arrays):
        '''
        Hack because from brainio.assemblies import merge_data_arrays gives
        an index duplicate error
        :param arrays:
        :return:
        '''
        if len(arrays) > 1:
            data = np.concatenate([e.data for e in arrays])
            conditions = np.concatenate([e.condition.data for e in arrays])
            truths = np.concatenate([e.truth.data for e in arrays])
            object_names = np.concatenate([e.object_name.data for e in arrays])
            return DataAssembly(data=data, dims='condition',
                                coords={'condition': conditions, 'truth': ('condition', truths),
                                        'object_name': ('condition', object_names)})
        else:
            return DataAssembly(arrays[0])


def Moeller2017Experiment1():
    '''
    Stimulate face patch during face identification
    32 identities; 6 expressions each
    '''
    return _Moeller2017(stimulus_class='Faces',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_1',
                        metric=PerformanceSimilarity(),
                        performance_measure=Accuracy())


def Moeller2017Experiment2():
    '''
    TODO  very unclean
    i: Stimulate outside of the face patch during face identification
    ii: Stimulate face patch during object identification
    28 Objects; 3 viewing angles each
    '''

    class _Moeller2017Experiment2(BenchmarkBase):
        def __init__(self):
            super().__init__(
                identifier='dicarlo.Moeller2017-Experiment_2', ceiling_func=None, version=1, parent='IT', bibtex=BIBTEX)
            self.benchmark1 = _Moeller2017(stimulus_class='Faces', perturbation_location='outside_facepatch',
                                           identifier='dicarlo.Moeller2017-Experiment_2i',
                                           metric=PerformanceSimilarity(), performance_measure=Accuracy())
            self.benchmark2 = _Moeller2017(stimulus_class='Objects', perturbation_location='within_facepatch',
                                           identifier='dicarlo.Moeller2017-Experiment_2ii',
                                           metric=PerformanceSimilarity(), performance_measure=Accuracy())

        def __call__(self, candidate):
            import copy
            candidate_copy = copy.copy(candidate)
            return self.benchmark1(candidate), self.benchmark2(candidate_copy)

    return _Moeller2017Experiment2()


def Moeller2017Experiment3():
    '''
    Stimulate face patch during face & non-face object eliciting patch response identification
    15 black & white round objects + faces; 3 exemplars per category  (apples, citrus, teapots, alarmclocks, faces)
    '''
    return _Moeller2017(stimulus_class='Eliciting_Face_Response',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_3',
                        metric=PerformanceSimilarity(),
                        performance_measure=Accuracy())


def Moeller2017Experiment4a():
    '''
    Stimulate face patch during abstract face identification
    16 Face Abtractions; 4 per category (Line Drawings, Silhouettes, Cartoons, Mooney Faces)
    '''
    return _Moeller2017(stimulus_class='Abstract_Faces',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_4a',
                        metric=PerformanceSimilarity(),
                        performance_measure=Accuracy())


def Moeller2017Experiment4b():
    '''
    Stimulate face patch during face & abstract houses identification
    20 Abstract Houses & Faces; 4 per category (House Line Drawings, House Cartoons, House Silhouettes, Mooney Faces, Mooney Faces up-side-down)
    '''
    return _Moeller2017(stimulus_class='Abstract_Houses',
                        perturbation_location='within_facepatch',
                        identifier='dicarlo.Moeller2017-Experiment_4b',
                        metric=PerformanceSimilarity(),
                        performance_measure=Accuracy())
