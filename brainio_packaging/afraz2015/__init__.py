from pathlib import Path

import pandas as pd

from brainio_base.assemblies import DataAssembly


def collect_assembly():
    # fig 3A
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
