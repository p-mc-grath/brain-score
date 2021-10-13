import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from brainio.assemblies import DataAssembly
from brainio.stimuli import StimulusSet


def collect_stimuli():
    stimulus_set = pd.read_pickle(Path(__file__).parent / 'stimuli/metadata_pd.pkl')
    stimulus_set = stimulus_set.rename(columns={'id': 'image_id', 'obj': 'category', 'xloc': 'location'})
    stimulus_set['category'] = stimulus_set['category'].replace(
        {'mal': 'male', 'fem': 'female', 'fac': 'face', 'obj': 'object'})
    stimulus_set['location'] = stimulus_set['location'].replace(
        {'0': 'right', '1': 'left'})
    image_directory = Path(__file__).parent / 'stimuli/images'
    stimulus_set = StimulusSet(stimulus_set)
    stimulus_set.image_paths = {row.image_id: image_directory / f"{row.image_id}.png"
                                for _, row in stimulus_set.iterrows()}
    stimulus_set.identifier = 'Afraz2015'
    assert all(Path(stimulus_set.get_image(image_id)).is_file() for image_id in stimulus_set['image_id'])
    assert set(stimulus_set['category']) == {'male', 'female', 'object', 'face'}
    assert set(stimulus_set['location']) == {'right', 'left'}
    return stimulus_set

    # "In practice, we first trained the animals on a fixed set of 400 images (200 males and 200 females)."
    fitting_stimuli = ...
    # "Once trained, we tested the animals’ performance on freshly generated sets of 400 images to confirm that they
    # could generalize the learning to novel stimuli. As each monkey gained more experience with the task, we
    # occasionally created a new set of 400 images with slightly increased task difficulty to keep the animal’s
    # performance level below ceiling (typically between 85–95% correct; chance is 50%). For muscimol experiments,
    # because the test blocks were shorter, smaller image sets (200 images) were used."
    test_stimuli_optogenetics = ...
    test_stimuli_muscimol = ...


def collect_assembly():
    # all experiments with those images (opto, musc, musc2)
    # where musc2 is the double injection == checkmate

    # there are two values for contra/ipsi stimuli.
    # if i remember correctly, there are 4 values for opto and musc because arash tested these in
    # both in the face patch and outside
    # the musc2 was only done in the face patch

    path = Path(__file__).parent / 'data/afraz2015_data.mat'
    data = scipy.io.loadmat(path)['summary']
    struct = {d[0]: v for d, v in zip(data.dtype.descr, data[0, 0])}
    description = [desc[0] for desc in struct['desc'].squeeze()]
    parts = [re.match(r"(?P<intervention_type>opto|musc|musc2)_"
                      r"(?P<selectivity>face|nonface)_"
                      r"(?P<hemisphere>ipsi|contra)", desc)
             for desc in description]
    intervention_type = [part.group("intervention_type") for part in parts]
    selectivity = [part.group("selectivity") for part in parts]
    hemisphere = [part.group("hemisphere") for part in parts]

    assembly = DataAssembly([struct['D0'], struct['D1']],
                            coords={
                                'injected': [False, True],
                                'intervention_type': ('injection', intervention_type),
                                'selectivity': ('injection', selectivity),
                                'hemisphere': ('injection', hemisphere),
                                'site_number': ('site', np.arange(struct['D0'].shape[0])),
                                'site_iteration': ('site', np.arange(struct['D0'].shape[0])),
                            },
                            dims=['injected', 'site', 'injection'])
    return assembly


def collect_delta_overall_accuracy():
    """ fig 3A """
    # data extracted with https://apps.automeris.io/wpd/ on 2021-07-09, points manually selected
    data = pd.read_csv(Path(__file__).parent / 'fig3A.csv')
    errors = data.groupby(['visual_field', 'condition']).apply(
        lambda group: (group['value'][group['aggregation'] == 'positive_error'].values
                       - group['value'][group['aggregation'] == 'mean'].values)[0])
    data.loc[data['aggregation'] == 'positive_error', 'value'] = errors.values
    data.loc[data['aggregation'] == 'positive_error', 'aggregation'] = 'error'
    data.loc[data['aggregation'] == 'mean', 'aggregation'] = 'center'

    # package into xarray
    assembly = DataAssembly(data['value'], coords={
        'visual_field': ('measurement', data['visual_field']),
        'condition_description': ('measurement', data['condition']),
        'aggregation': ('measurement', data['aggregation']),
    }, dims=['measurement'])
    assembly = assembly.unstack()
    assembly['laser_on'] = ('condition_description', [condition == 'image_laser'
                                                      for condition in assembly['condition_description']])
    assembly = assembly.stack(condition=['condition_description'])
    assembly = DataAssembly(assembly)
    return assembly


def collect_site_deltas():
    """ fig 3C """
    # data extracted with https://apps.automeris.io/wpd/ on 2021-07-09, points manually selected
    # Note that there is a slight discrepancy in the correlation which is -0.46 in the paper,
    # but turns out to be -0.472 with the annotated dots
    data = pd.read_csv(Path(__file__).parent / 'fig3C.csv')

    # package into xarray
    assembly = DataAssembly(data['delta_accuracy'], coords={
        'monkey': ('site', data['monkey']),
        'face_detection_index_dprime': ('site', data['face_detection_index_dprime']),
    }, dims=['site'])
    return assembly


def collect_subregion_deltas():
    """ fig 3D """
    # data extracted with https://apps.automeris.io/wpd/ on 2021-07-09, points manually selected
    data = pd.read_csv(Path(__file__).parent / 'fig3D.csv')

    # package into xarray
    assembly = DataAssembly(data['delta_accuracy'], coords={
        'monkey': ('subregion', data['monkey']),
        'face_detection_index_dprime': ('subregion', data['face_detection_index_dprime']),
    }, dims=['subregion'])
    return assembly


def muscimol_delta_overall_accuracy():
    """ fig 5b + S4{b,c} """
    # data extracted with https://apps.automeris.io/wpd/ on 2021-07-09, points manually selected
    # collect face/other sites for monkey C from 4b, E from 4d, and saline for C from 5B
    data = pd.read_csv(Path(__file__).parent / 'fig5.csv')
    errors = data.groupby(['monkey', 'condition']).apply(
        lambda group: np.absolute(group['value'][group['aggregation'] == 'positive_error'].values
                                  - group['value'][group['aggregation'] == 'mean'].values)[0])
    data.loc[data['aggregation'] == 'positive_error', 'value'] = errors.values
    data.loc[data['aggregation'] == 'positive_error', 'aggregation'] = 'error'

    # package into xarray
    assembly = DataAssembly(data['value'], coords={
        'monkey': ('measurement', data['monkey']),
        'condition': ('measurement', data['condition']),
        'aggregation': ('measurement', data['aggregation']),
    }, dims=['measurement'])
    assembly = assembly.unstack()
    return assembly


def site_suppression_distribution():
    """ fig 2C """
    # data extracted with https://apps.automeris.io/wpd/ on 2021-08-09, points manually selected
    data = pd.read_csv(Path(__file__).parent / 'fig2C.csv')
    # histogram with count, get rid of digitization errors
    data['number_of_sites'] = data['number_of_sites'].astype(int)

    # package into xarray
    assembly = DataAssembly(data['number_of_sites'], coords={
        'spikes_deleted_percent': data['percent_spikes_deleted'],
    }, dims=['spikes_deleted_percent'])
    return assembly
