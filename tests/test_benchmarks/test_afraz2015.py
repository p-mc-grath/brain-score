import xarray as xr

from brainio.assemblies import DataAssembly
from brainscore.benchmarks.afraz2015 import stack_multiindex


class TestStackMultiIndex:
    def test_2d(self):
        assembly = DataAssembly([[1, 2],
                                 [3, 4],
                                 [5, 6]],
                                coords={
                                    'dim_0': [1, 2, 3],
                                    'dim_1': ['a', 'b'],
                                }, dims=['dim_0', 'dim_1'])
        actual = stack_multiindex(assembly, new_dim='combined')

        expected = DataAssembly([1, 2, 3, 4, 5, 6],
                                coords={
                                    'dim_0': ('combined', [1, 1, 2, 2, 3, 3]),
                                    'dim_1': ('combined', ['a', 'b', 'a', 'b', 'a', 'b']),
                                }, dims=['combined'])
        xr.testing.assert_equal(actual, expected)
