import pandas as pd
from pathlib import Path

from brainio.assemblies import DataAssembly


def collect_assembly():
    """
    fig 7B
    """
    data = pd.read_csv(Path(__file__).parent / 'fig7B.csv')

    # package into xarray
    assembly = DataAssembly(data['dF_F0'], coords={
        'distance_micrometer': ('dF_F0', data['distance_from_tip_micrometer']),
        'stimulation_current_microampere': ('dF_F0', data['stimulation_current_microampere']),
    }, dims=['dF_F0'])
    return assembly
