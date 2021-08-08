import pandas as pd
from pathlib import Path

from brainio.assemblies import DataAssembly


def collect_assembly():
    """
    fig S3
    """
    data = pd.read_csv(Path(__file__).parent / 'figS3.csv')

    # package into xarray
    assembly = DataAssembly(data['irradiance_mW_mm2'], coords={
        'distance_microns': ('irradiance', data['y']),
        'unit': ('irradiance', ['mW_mm2'] * len(data)),
    }, dims=['irradiance'])
    assembly.attrs['irradiance_at_fiber_tip'] = 200
    return assembly
