from xarray import DataArray

import brainscore
from brainscore.model_interface import BrainModel
from brainscore.benchmarks import BenchmarkBase
from brainscore.benchmarks._neural_common import NeuralBenchmark, average_repetition
from brainscore.metrics.ceiling import InternalConsistency, RDMConsistency
from brainscore.metrics.rdm import RDMCrossValidated
from brainscore.metrics.regression import CrossRegressedCorrelation, mask_regression, ScaledCrossRegressedCorrelation, \
    pls_regression, pearsonr_correlation
from brainscore.metrics.spatial_correlation import SpatialCorrelationSimilarity
from brainscore.metrics.inter_individual_stats_ceiling import InterIndividualStatisticsCeiling
from brainscore.utils import LazyLoad
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import squareform, pdist

VISUAL_DEGREES = 8
NUMBER_OF_TRIALS = 50
BIBTEX = """@article {Majaj13402,
            author = {Majaj, Najib J. and Hong, Ha and Solomon, Ethan A. and DiCarlo, James J.},
            title = {Simple Learned Weighted Sums of Inferior Temporal Neuronal Firing Rates Accurately Predict Human Core Object Recognition Performance},
            volume = {35},
            number = {39},
            pages = {13402--13418},
            year = {2015},
            doi = {10.1523/JNEUROSCI.5181-14.2015},
            publisher = {Society for Neuroscience},
            abstract = {To go beyond qualitative models of the biological substrate of object recognition, we ask: can a single ventral stream neuronal linking hypothesis quantitatively account for core object recognition performance over a broad range of tasks? We measured human performance in 64 object recognition tests using thousands of challenging images that explore shape similarity and identity preserving object variation. We then used multielectrode arrays to measure neuronal population responses to those same images in visual areas V4 and inferior temporal (IT) cortex of monkeys and simulated V1 population responses. We tested leading candidate linking hypotheses and control hypotheses, each postulating how ventral stream neuronal responses underlie object recognition behavior. Specifically, for each hypothesis, we computed the predicted performance on the 64 tests and compared it with the measured pattern of human performance. All tested hypotheses based on low- and mid-level visually evoked activity (pixels, V1, and V4) were very poor predictors of the human behavioral pattern. However, simple learned weighted sums of distributed average IT firing rates exactly predicted the behavioral pattern. More elaborate linking hypotheses relying on IT trial-by-trial correlational structure, finer IT temporal codes, or ones that strictly respect the known spatial substructures of IT ({\textquotedblleft}face patches{\textquotedblright}) did not improve predictive power. Although these results do not reject those more elaborate hypotheses, they suggest a simple, sufficient quantitative model: each object recognition task is learned from the spatially distributed mean firing rates (100 ms) of \~{}60,000 IT neurons and is executed as a simple weighted sum of those firing rates.SIGNIFICANCE STATEMENT We sought to go beyond qualitative models of visual object recognition and determine whether a single neuronal linking hypothesis can quantitatively account for core object recognition behavior. To achieve this, we designed a database of images for evaluating object recognition performance. We used multielectrode arrays to characterize hundreds of neurons in the visual ventral stream of nonhuman primates and measured the object recognition performance of \&gt;100 human observers. Remarkably, we found that simple learned weighted sums of firing rates of neurons in monkey inferior temporal (IT) cortex accurately predicted human performance. Although previous work led us to expect that IT would outperform V4, we were surprised by the quantitative precision with which simple IT-based linking hypotheses accounted for human behavior.},
            issn = {0270-6474},
            URL = {https://www.jneurosci.org/content/35/39/13402},
            eprint = {https://www.jneurosci.org/content/35/39/13402.full.pdf},
            journal = {Journal of Neuroscience}}"""


def _DicarloMajajHong2015Region(region, identifier_metric_suffix, similarity_metric, ceiler):
    assembly_repetition = LazyLoad(lambda region=region: load_assembly(average_repetitions=False, region=region))
    assembly = LazyLoad(lambda region=region: load_assembly(average_repetitions=True, region=region))
    return NeuralBenchmark(identifier=f'dicarlo.MajajHong2015.{region}-{identifier_metric_suffix}', version=3,
                           assembly=assembly, similarity_metric=similarity_metric,
                           visual_degrees=VISUAL_DEGREES, number_of_trials=NUMBER_OF_TRIALS,
                           ceiling_func=lambda: ceiler(assembly_repetition),
                           parent=region,
                           bibtex=BIBTEX)


def DicarloMajajHong2015V4PLS():
    return _DicarloMajajHong2015Region('V4', identifier_metric_suffix='pls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=pls_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITPLS():
    return _DicarloMajajHong2015Region('IT', identifier_metric_suffix='pls',
                                       similarity_metric=CrossRegressedCorrelation(
                                           regression=pls_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015V4Mask():
    return _DicarloMajajHong2015Region('V4', identifier_metric_suffix='mask',
                                       similarity_metric=ScaledCrossRegressedCorrelation(
                                           regression=mask_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015ITMask():
    return _DicarloMajajHong2015Region('IT', identifier_metric_suffix='mask',
                                       similarity_metric=ScaledCrossRegressedCorrelation(
                                           regression=mask_regression(), correlation=pearsonr_correlation(),
                                           crossvalidation_kwargs=dict(splits=2, stratification_coord='object_name')),
                                       ceiler=InternalConsistency())


def DicarloMajajHong2015V4RDM():
    return _DicarloMajajHong2015Region('V4', identifier_metric_suffix='rdm',
                                       similarity_metric=RDMCrossValidated(
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=RDMConsistency())


def DicarloMajajHong2015ITRDM():
    return _DicarloMajajHong2015Region('IT', identifier_metric_suffix='rdm',
                                       similarity_metric=RDMCrossValidated(
                                           crossvalidation_kwargs=dict(stratification_coord='object_name')),
                                       ceiler=RDMConsistency())


def load_assembly(average_repetitions, region, access='private'):
    assembly = brainscore.get_assembly(name=f'dicarlo.MajajHong2015.{access}')
    assembly = assembly.sel(region=region)
    assembly['region'] = 'neuroid', [region] * len(assembly['neuroid'])
    assembly = assembly.squeeze("time_bin")
    assembly.load()
    assembly = assembly.transpose('presentation', 'neuroid')
    if average_repetitions:
        assembly = average_repetition(assembly)
    return assembly


class DicarloMajajHong2015ITSpatialCorrelation(BenchmarkBase):

    def __init__(self):
        '''
        This benchmark compares the distribution of pairwise response correlation as a function of distance between the
            MajajHong2015 assembly and a candidate BrainModel
        Plots of this can be found in Figure 2 of the corresponding publication (see BIBTEX)
        '''
        super().__init__(identifier='dicarlo.MajajHong2015.IT-spatial_correlation',
                         ceiling_func=lambda: InterIndividualStatisticsCeiling(
                             SpatialCorrelationSimilarity(similarity_function=self.inv_ks_similarity, bin_size_mm=.1))(
                             LazyLoad(self.compute_global_tissue_statistic_target)),  # .1 mm is an arbitrary choice
                         version=0.1,
                         parent='IT',
                         bibtex=BIBTEX)

        assembly = brainscore.get_assembly('dicarlo.MajajHong2015').sel(region='IT')
        assembly = self.make_static(assembly)
        assembly = self.tissue_update(assembly)

        self._neuroid_reliability = InternalConsistency()(assembly.transpose('presentation', 'neuroid'))
        self._stimulus_set = assembly.stimulus_set
        self._target_assembly = average_repetition(assembly)
        self._score = SpatialCorrelationSimilarity(similarity_function=self.inv_ks_similarity,
                                                   bin_size_mm=.1)  # .1 mm is an arbitrary choice

        self.bootstrap_samples = 100_000
        self.num_sample_arrs = 10  # number of simulated Utah arrays sampled from candidate model tissue
        self._array_size_mm = (np.ptp(self._target_assembly.neuroid.tissue_x.data),  # physical size of Utah array in mm
                               np.ptp(self._target_assembly.neuroid.tissue_y.data))

    def __call__(self, candidate: BrainModel):
        '''
        This computes the statistics, i.e. the pairwise response correlation of candidate and target, respectively and
        computes a score based on the ks similarity of the two resulting distributions
        :param candidate: BrainModel
        :return: Score, i.e. average inverted ks similarity, for the pairwise response correlation compared to the MajajHong Assembly
        '''
        candidate.start_recording(recording_target='IT', time_bins=[(70, 170)])
        candidate_assembly = candidate.look_at(self._stimulus_set)
        candidate_assembly = self.make_static(candidate_assembly)
        candidate_statistic = self.sample_global_tissue_statistic(candidate_assembly)

        self._target_statistic = self.compute_global_tissue_statistic_target()

        score = self._score(self._target_statistic, candidate_statistic)
        score.attrs['target_statistic'] = self._target_statistic
        score.attrs['candidate_statistic'] = candidate_statistic

        return score

    def sample_global_tissue_statistic(self, candidate_assembly):
        '''
        Simulates placement of multiple arrays in tissue and computes repsonse correlation as a function of distance on
        each of them
        :param candidate_assembly: NeuroidAssembly
        :return: xr DataArray: values = correlations; coordinates: distances, source, array
        '''
        candidate_statistic_list = []
        bootstrap_samples_per_array = int(self.bootstrap_samples / self.num_sample_arrs)
        for i, window in enumerate(self.sample_array_locations(candidate_assembly.neuroid)):
            distances, correlations = self.sample_response_corr_vs_dist(candidate_assembly[window],
                                                                        bootstrap_samples_per_array)

            array_statistic = self.to_xarray(correlations, distances, array=str(i))
            candidate_statistic_list.append(array_statistic)

        candidate_statistic = xr.concat(candidate_statistic_list, dim='meta')
        return candidate_statistic

    def compute_global_tissue_statistic_target(self):
        '''
        :return: xr DataArray: values = correlations; coordinates: distances, source, array
        '''
        target_statistic_list = []
        for animal in sorted(list(set(self._target_assembly.neuroid.animal.data))):
            for arr in sorted(list(set(self._target_assembly.neuroid.arr.data))):
                sub_assembly = self._target_assembly.sel(animal=animal, arr=arr)
                bootstrap_samples_sub_assembly = int(self.bootstrap_samples * (sub_assembly.neuroid.size /
                                                                               self._target_assembly.neuroid.size))

                distances, correlations = self.sample_response_corr_vs_dist(sub_assembly,
                                                                            bootstrap_samples_sub_assembly,
                                                                            self._neuroid_reliability)

                sub_assembly_statistic = self.to_xarray(correlations, distances, source=animal, array=arr)
                target_statistic_list.append(sub_assembly_statistic)

        target_statistic = xr.concat(target_statistic_list, dim='meta')
        return target_statistic

    def sample_array_locations(self, neuroid, seed=0):
        '''
        Generator: Sample Utah array-like portions from artificial model tissue and generate masks
        :param neuroid: NeuroidAssembly.neuroid, has to contain tissue_x, tissue_y coords
        :param seed: random seed
        :return: list of masks in neuroid dimension of assembly, usage: assembly[mask] -> neuroids within one array
        '''
        bound_max_x, bound_max_y = np.max([neuroid.tissue_x.data, neuroid.tissue_y.data], axis=1) - self._array_size_mm
        rng = np.random.default_rng(seed=seed)

        lower_corner = np.column_stack((rng.choice(neuroid.tissue_x.data[neuroid.tissue_x.data <= bound_max_x],
                                                   size=self.num_sample_arrs),
                                        rng.choice(neuroid.tissue_y.data[neuroid.tissue_y.data <= bound_max_y],
                                                   size=self.num_sample_arrs)))
        upper_corner = lower_corner + self._array_size_mm

        # create index masks of neuroids within sample windows
        for i in range(self.num_sample_arrs):
            yield np.logical_and.reduce([neuroid.tissue_x.data <= upper_corner[i, 0],
                                         neuroid.tissue_x.data >= lower_corner[i, 0],
                                         neuroid.tissue_y.data <= upper_corner[i, 1],
                                         neuroid.tissue_y.data >= lower_corner[i, 1]])

    @classmethod
    def sample_response_corr_vs_dist(cls, assembly, num_samples, neuroid_reliability=None, seed=0):
        '''
        1. Samples random pairs from the assembly
        2. Computes distances for all pairs
        3. Computes the response correlation between items of each pair
        (4. Ceils the response correlations by ceiling each neuroid | if neuroid_reliability not None)
        :param assembly: NeuroidAssembly without stimulus repetitions
        :param num_samples: how many random pair you want to be sampled out of the data
        :param neuroid_reliability: if not None: expecting Score object containing reliability estimates of all neuroids
        :param seed: random seed
        :return: [distance, pairwise_correlation_of_neuroids], pairwise correlations can be ceiled
        '''
        rng = np.random.default_rng(seed=seed)
        neuroid_pairs = rng.integers(0, assembly.shape[0], (2, num_samples))

        pairwise_distances_all = cls.pairwise_distances(assembly)
        pairwise_distance_samples = pairwise_distances_all[(*neuroid_pairs,)]

        response_samples = assembly.data[neuroid_pairs]
        response_correlation_samples = cls.corrcoef_rowwise(*response_samples)

        if neuroid_reliability is not None:
            pairwise_neuroid_reliability_all = cls.create_pairwise_neuroid_reliability_mat(neuroid_reliability)
            pairwise_neuroid_reliability_samples = pairwise_neuroid_reliability_all[(*neuroid_pairs,)]

            response_correlation_samples = response_correlation_samples / pairwise_neuroid_reliability_samples

        # properly removing nan values
        pairwise_distance_samples = pairwise_distance_samples[~np.isnan(response_correlation_samples)]
        response_correlation_samples = response_correlation_samples[~np.isnan(response_correlation_samples)]

        return np.vstack((pairwise_distance_samples, response_correlation_samples))

    @staticmethod
    def to_xarray(correlations, distances, source='model', array=None):
        '''
        :param values: list of data values
        :param distances: list of distance values, each distance value has to correspond to one data value
        :param source: name of monkey
        :param array: name of recording array
        '''
        xarray_statistic = DataArray(
            data=correlations,
            dims=["meta"],
            coords={
                'meta': pd.MultiIndex.from_product([distances, [source], [array]],
                                                   names=('distances', 'source', 'array'))
            }
        )

        return xarray_statistic

    @staticmethod
    def corrcoef_rowwise(a, b):
        # https://stackoverflow.com/questions/41700840/correlation-of-2-time-dependent-multidimensional-signals-signal-vectors
        a_ma = a - a.mean(1)[:, None]
        b_mb = b - b.mean(1)[:, None]
        ssa = np.einsum('ij,ij->i', a_ma, a_ma)  # var A
        ssb = np.einsum('ij,ij->i', b_mb, b_mb)  # var B
        return np.einsum('ij,ij->i', a_ma, b_mb) / np.sqrt(ssa * ssb)  # cov/sqrt(varA*varB)

    @staticmethod
    def pairwise_distances(assembly):
        '''
        Convenience function creating a simple lookup table for pairwise distances
        :param assembly: NeuroidAssembly
        :return: square matrix where each entry is the distance between the neuroids at the corresponding indices
        '''
        locations = np.stack([assembly.neuroid.tissue_x.data, assembly.neuroid.tissue_y.data]).T

        return squareform(pdist(locations, metric='euclidean'))

    @staticmethod
    def create_pairwise_neuroid_reliability_mat(neuroid_reliability):
        '''
        Convenience function creating a simple lookup table for combined reliabilities of neuroid pairs
        :param neuroid_reliability: expects Score object where neuroid_reliability.raw holds [cross validation subset,
            reliability per neuroid]
        :return: square matrix where each entry_ij = sqrt(reliability_i * reliability_j)
        '''
        reliability_per_neuroid = np.mean(neuroid_reliability.raw.data, axis=0)
        c_mat = np.zeros((reliability_per_neuroid.size, reliability_per_neuroid.size))
        for i, ci in enumerate(reliability_per_neuroid):
            for j, cj in enumerate(reliability_per_neuroid):
                c_mat[i, j] = np.sqrt(ci * cj)

        return c_mat

    @staticmethod
    def tissue_update(assembly):
        '''
        Temporary functions: Obsolete when all saved assemblies updated such that x and y coordinates of each array electrode
        are stored in assembly.neuroid.tissue_{x,y}
        '''
        if not hasattr(assembly, 'tissue_x'):
            assembly['tissue_x'] = assembly['x']
            assembly['tissue_y'] = assembly['y']

        return assembly

    @staticmethod
    def make_static(assembly):
        if 'time_bin' in assembly.dims:
            assembly = assembly.squeeze('time_bin')
        if hasattr(assembly, "time_step"):
            assembly = assembly.squeeze("time_step")

        return assembly

    @staticmethod
    def inv_ks_similarity(p, q):
        '''
        Inverted ks similarity -> resulting in a score within [0,1], 1 being a perfect match
        '''
        import scipy.stats
        return 1 - scipy.stats.ks_2samp(p, q)[0]
