import pandas as pd
from pathlib import Path

from brainio.assemblies import DataAssembly


def collect_assembly():
    """
    fig 2C
    """
    data = pd.read_csv(Path(__file__).parent / '2C.csv')

    # package into xarray
    assembly = DataAssembly(data['synapse_probability'], coords={
        'distance_micrometer': ('synapse_probability', data['distance_micrometer']),
        'stimulation_current_microampere': ('synapse_probability', data['stimulation_current_microampere']),
    }, dims=['synapse_probability'])
    return assembly
