import xarray as xr
from pathlib import Path

from brainio_base.assemblies import BehavioralAssembly
from brainscore.metrics.behavior_differences import BehaviorDifferences


def test():
    assembly = xr.open_dataarray(Path(__file__).parent / 'hvm_inactivation_runs.nc')
    assembly = BehavioralAssembly(assembly)
    metric = BehaviorDifferences()
    score = metric(assembly, assembly)
    assert score.sel(aggregation='center') == 1
