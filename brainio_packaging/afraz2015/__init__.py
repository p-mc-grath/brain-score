from pathlib import Path

import pandas as pd

from brainio_base.assemblies import DataAssembly


def collect_delta_overall_accuracy():
    """ fig 3A """
    # data extracted with https://apps.automeris.io/wpd/ on 2021-07-09, points manually selected
    data = pd.read_csv(Path(__file__).parent / 'fig3A.csv')
    errors = data.groupby(['visual_field', 'condition']).apply(  # apply)
        lambda group: (group['value'][group['aggregation'] == 'positive_error'].values
                       - group['value'][group['aggregation'] == 'mean'].values)[0])
    data.loc[data['aggregation'] == 'positive_error', 'value'] = errors.values
    data.loc[data['aggregation'] == 'positive_error', 'aggregation'] = 'error'

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


def collect_per_site_delta():
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
