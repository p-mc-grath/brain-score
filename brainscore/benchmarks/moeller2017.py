from os import listdir
from pathlib import Path
import itertools
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import brainscore
from brainscore.utils import LazyLoad
from brainscore.metrics import Metric
from brainscore.benchmarks import BenchmarkBase
from brainscore.model_interface import BrainModel
from brainscore.metrics.accuracy import Accuracy
from brainscore.metrics.performance_similarity import PerformanceSimilarity  # TODO
from brainio.assemblies import merge_data_arrays, DataArray
from brainio.stimuli import StimulusSet

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

        self._target_assembly = self._collect_target_assembly()
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
        :return: DataArray: values = choice, dims = [truth, current_pulse_mA, condition, object_name]
        '''
        candidate.perturb(perturbation=None, target='IT')  # reset
        candidate.perturb(perturbation=perturbation['type'], target='IT',
                          perturbation_parameters=perturbation['perturbation_parameters'])

        IT_recordings = candidate.look_at(self._stimulus_set)

        behavior = self._compute_behavior(IT_recordings, decoder)
        behavior['current_pulse_mA'] = behavior.dims[0], perturbation['perturbation_parameters']['current_pulse_mA']
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
        :return: behaviors DataArray
            values: choice
            dims:   truth       : list of int/bool, 'same_id'==1, 'different_id'==0,
            coords: condition   : list of strings, ['same_id == 1, 'different_id'==0]
                    object_name : list of strings, category
        '''
        samples = 500  # TODO why 500
        behavior_data = []
        for object_name in set(IT_recordings.object_name):
            recordings, conditions = self._sample_recordings(IT_recordings.sel(object_name=object_name),
                                                             samples=samples)
            choices = decoder.predict(recordings)
            behavior = DataArray(data=choices, dims='condition',
                                 coords={'truth': np.array(conditions) == 'same_id',
                                         'condition': ('truth', conditions),
                                         'object_name': ('truth', [object_name] * samples * 2)})
            behavior_data.append(behavior)

        behaviors = merge_data_arrays(behavior_data)
        return behaviors

    def _compute_performance(self, behavior: DataArray):
        '''
        Given performance measure and behavior, compute performance w.r.t. current_pulse_mA, condition, object name
        :param behavior:
            values: choice,
            dims:   truth           : 'same_id'==1, 'different_id'==0
            coords: condition       : list of strings, ['same_id', 'different_id]
                    object name     : list of strings, category
                    current_pulse_mA: float
        :return: DataArray:
            values: performances    : accuracy values
            dims:   condition       : list of strings, ['same_id', 'different_id]
            coords: object_name     : list of strings, category
                    current_pulse_mA: float
        '''
        performance_data = []
        for object_name, current_pulse_mA, condition in itertools.product(set(behavior.object_name),
                                                                          set(behavior.current_pulse_mA),
                                                                          set(behavior.condition)):
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
                                     'location': LazyLoad(self._perturbation_coordinates)
                                 }}

            perturbation_list.append(perturbation_dict)
        return perturbation_list

    def _set_up_decoder(self, candidate: BrainModel):
        '''
        Fit a linear regression between the recordings of the training stimuli and the ground truth
        :return: trained linear regressor
        '''
        candidate.start_recording()
        recordings = candidate.look_at(self._training_stimuli)
        samples = 500  # TODO why 500

        stimulus_set = truth = []
        for object_name in set(recordings.object_name):
            recordings, conditions = self._sample_recordings(recordings.sel(object_name=object_name),
                                                             samples=samples)
            stimulus_set.append(recordings)
            truth += np.array(conditions) == 'same_id'
        stimulus_set = np.vstack(stimulus_set)

        return LinearRegression().fit(stimulus_set, truth)

    def _sample_recordings(self, category_pool: DataArray, samples=500):  # TODO why 500
        '''
        Create an array of randomly sampled recordings, each line is one task, i.e. two recordings which are to be
        judged same vs. different ID
        :param category_pool: Model IT recordings, assumed be from one category only
        :param samples: int: number of samples
        :return: array (samples x length of two concatenated recordings), each line contains two recordings
                 ground truth, for each line in array, specifying if the two recordings belong to the same/different ID
        '''
        rng = np.random.default_rng(seed=self._seed)
        recording_size = len(category_pool[0].values)
        random_indeces = rng.integers(0, len(category_pool), (samples, 2))

        sampled_recordings = np.full((samples, recording_size * 2), np.nan)
        for i, (random_idx_same, random_idx_different) in enumerate(random_indeces):
            # condition 'same_id': object_id same between recording one and two, image_id different between recording one and two
            image_one_same = category_pool[random_idx_same]
            sampled_recordings[i, :recording_size] = image_one_same.values
            sampled_recordings[i, recording_size:] = rng.choice(category_pool.where(
                category_pool.object_id == image_one_same.object_id and
                category_pool.image_id != image_one_same.image_id))

            # condition 'different_id': object_id different between recording one and two
            image_one_diff = category_pool[random_idx_different]
            sampled_recordings[i + samples, :recording_size] = image_one_diff.values
            sampled_recordings[i + samples, recording_size:] = rng.choice(category_pool.where(
                category_pool.object_id != image_one_diff.object_id))

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

        def _get_grid_coord():
            """
            Adapted from Lee et al. 2020 code, not public
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
                    'voxel_x': ('neuroid_id', gridx.squeeze()),
                    'voxel_y': ('neuroid_id', gridy.squeeze())}
        )
        return features_smoothed,

    @staticmethod
    def _get_purity_center(selectivity_assembly: DataArray, radius=1):
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
    def _determine_face_selectivity(recordings: DataArray):
        '''
        Determines face selectivity of each neuroid
        :param recordings: DataArray
        :return: DataArray, same as recordings where activations have been replaced with dprime values
        '''

        def mean_var(neuron):
            mean, var = np.mean(neuron.values), np.var(neuron.values)
            return mean, var

        assert (recordings >= 0).all(), 'selectivities must be positive'

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

    def _sample_outside_face_patch(self, selectivity_assembly: DataArray, radius=2):
        '''
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
            inside_radius = np.where(np.sqrt(np.square(not_selective_voxels.voxel_x.values - voxel.voxel_x.values) +
                                             np.square(not_selective_voxels.voxel_y.values - voxel.voxel_y.values))
                                     < radius)[0]
            if np.all(not_selective_voxels[inside_radius].values < DPRIME_THRESHOLD_FACE_PATCH):
                voxels.append(voxel)

        rng = np.random.default_rng(seed=self._seed)
        voxel = rng.choice(voxels)
        x, y = voxel.voxel_x, voxel.voxel_y
        return x, y

    def _collect_target_assembly(self):
        '''
        Load Data from path + subselect as specified by Experiment

        :return: DataArray
        '''
        stimulus_set = self._load_stimulus_set()
        training_stimuli = self._load_training_stimuli()
        stimulus_set_face_patch = self._load_stimulus_set_face_patch()

        # make into dataarray
        data = self._load_target_data()
        target_assembly = DataArray(data=data['accuracies'], dims='condition',
                                    coords={'condition': data['condition'],  # same vs. diff
                                            'object_name': ('condition', data['object_name']),
                                            'current_pulse_mA': ('condition', data['current_pulse_mA'])},
                                    attrs={'stimulus_set': stimulus_set,
                                           'training_stimuli': training_stimuli,
                                           'stimulus_set_face_patch': stimulus_set_face_patch})

        return target_assembly

    def _load_target_data(self):
        # TODO path
        # TODO deal with data from multiple monkeys
        '''
        From the Results section:
        "We first stimulated in the most anterior face patch, AM, previously shown
        to contain a view-invariant representation of individual identity3."

        "We report maximums across sessions because
        effect size correlated with accuracy of targeting to the center of the
        face patch and varied across sessions, as discussed in detail below
        (Supplementary Tables 1 and 3 give detailed statistics for each
        session individually; Supplementary Fig. 1 and Supplementary
        Table 2 summarize the effects for each patch)."

        "We chose AM since it is the
        most anterior, high-level patch in the system, based on both functional
        and anatomical criteria3,23. Therefore, if any patch would be expected
        to code purely faces and not other objects, it would be AM."

        As our methods find the center of the model face patch perfectly, we are working with the values
        from the sessions with the maximum effects.
        From Lee et al. 2020 we make the assumption that out in silico face patch is equivalent to AM.

        :return: dictionary with keys:
                    accuracies          = list, accuracy computed w.r.t. rest of keys
                    condition           = list, {same, different}_id
                    current_pulse_mA    = list, {0, 300}
                    object_name         = list, category name
                    source              = list, monkey number
        '''
        # statistic for each dataset
        path = Path(__file__).parent.parent.parent.parent / 'moeller_stimuli' / 'SummaryMat.xlsx'  # TODO
        df = pd.read_excel(path)

        # select relevant lines
        df = df.loc[(df.Stimulus_Class == self._stimulus_class) &
                    (df.Perturbation_Location == self._perturbation_location) &
                    (df.Monkey == 1)]  # TODO

        # compute accuracies
        data = {'accuracies': [], 'condition': [], 'current_pulse_mA': [], 'object_name': [], 'source': []}
        for condition, stimulation in itertools.product(['Same', 'Different'], ['', '_MSC']):
            setup = condition + stimulation
            accuracy = df['Hit_' + setup] / (df['Hit_' + setup] + df['Miss_' + setup])
            data['accuracies'] += accuracy.to_list()
            data['condition'] += [condition.lower() + '_id'] * len(accuracy)
            data['current_pulse_mA'] += df.Current_Pulse_mA.to_list() if stimulation == '_MSC' else [0] * len(accuracy)
            data['object_name'] += df.Object_Names.to_list()
            data['source'] += df.Monkey.to_list()

        return data

    def _load_stimulus_set(self):
        # TODO path
        # TODO Faces JPEG, Abstract Houses tif, all other PNG
        '''
        Load stimuli as specified by the paper; relevant parameter: self._stimulus_class

        :return: StimulusSet object containing information about image path, object class and object identity
        '''
        path = Path(__file__).parent.parent.parent.parent / 'moeller_stimuli' / self._stimulus_class  # TODO
        image_ids = [self._stimulus_class + '/' + e for e in listdir(path)]
        object_names, object_ids = [], []
        for image_id in image_ids:
            object_ids.append(re.split(r"_", image_id)[1])
            if self._stimulus_class == 'Faces' or ''.join([c for c in object_ids[-1] if not c.isdigit()]) == 'r':
                object_names.append('face')
            else:
                object_names.append(''.join([c for c in object_ids[-1] if not c.isdigit()]))
        stimulus_set = StimulusSet({'image_id': image_ids, 'object_name': object_names, 'object_id': object_ids})
        return stimulus_set

    @staticmethod
    def _load_training_stimuli():
        # TODO path
        '''
        From Online Methods section. Behavioral training section: [...]
        "Next, we trained animals on the main task (32 faces, 6 exemplars each).
        Image selection was exactly as in the second training task except that in the sameidentity condition we
        drew the second cue from all six images of the selected
        identity. Both animals showed stable, good performance on this task across many
        sessions (Supplementary Fig. 11c,d).

        Finally, we presented stimulus sets consisting of non-face objects (either
        16, 19 or 28 objects; see Experiment 2). For this task, both animals immediately began performing at
        >70% correct, indicating that they could generalize
        the same/different identification task independent of the actual stimuli presented
        (Supplementary Fig. 11e,f). The same generalization was evident in
        the round object identification task (see Experiment 3 and Supplementary
        Fig. 11g,h; for the abstracted faces and houses from Experiment 4b,
        see Supplementary Fig. 11i,j)."

        From the excerpt above, I assume training only on faces; other stimuli just tested.
        I am ignoring the basic training for paradigm etc. before.

        :return: StimulusSet Object, same as 'Faces' used in Experiment 1
        '''
        stimulus_class = 'Faces'
        path = Path(__file__).parent.parent.parent.parent / 'moeller_stimuli' / stimulus_class  # TODO
        image_ids = [stimulus_class + '/' + e for e in listdir(path)]
        object_names, object_ids = [], []
        for image_id in image_ids:
            object_ids.append(re.split(r"_", image_id)[1])
            object_names.append('face')
        stimuli = StimulusSet({'image_id': image_ids, 'object_name': object_names, 'object_id': object_ids})
        return stimuli

    @staticmethod
    def _load_stimulus_set_face_patch():
        # TODO make sure images are labeled face
        # TODO remove bias in data towards non-faces
        '''
        From Online Methods, Face patch localization section:
        "Two male rhesus macaques were trained to maintain
        fixation on a small spot for juice reward. Monkeys were scanned in a 3T TIM Trio
        (Siemens) magnet equipped with an AC88 gradient insert while passively viewing images on a screen.
        MION contrast agent (8 mg/kg body weight, Feraheme, AMAG) was injected to improve signal to noise ratio. [...]"

        I assume just use some images for face patch localization

        :return: StimulusSet object, hvm images
        '''
        return brainscore.get_stimulus_set('dicarlo.hvm')


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
            return self.benchmark1(candidate), self.benchmark2(candidate)

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
