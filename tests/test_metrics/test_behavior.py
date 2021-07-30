import itertools

import numpy as np
import os
from pathlib import Path

import pandas as pd
import xarray as xr
import pytest
import xarray
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainscore.benchmarks.rajalingham2018 import load_assembly
from brainscore.metrics.image_level_behavior import I2n, _o2


@pytest.mark.private_access
class TestI2N:
    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .253),
                                 ('resnet34', .37787),
                                 ('resnet18', .3638),
                             ])
    def test_model(self, model, expected_score):
        # assemblies
        objectome = load_assembly()
        probabilities = Path(__file__).parent / f'{model}-probabilities.nc'
        probabilities = BehavioralAssembly(xr.load_dataarray(probabilities))
        # metric
        i2n = I2n()
        score = i2n(probabilities, objectome)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"

    def test_ceiling(self):
        objectome = load_assembly()
        i2n = I2n()
        ceiling = i2n.ceiling(objectome)
        assert ceiling.sel(aggregation='center') == approx(.4786, abs=.0064)
        assert ceiling.sel(aggregation='error') == approx(.00537, abs=.0015)


class TestO2:
    def test(self):
        objects = ['dog', 'cat', 'chair']
        probabilities = BehavioralAssembly([
            # dog
            [1, 0, 0],
            [.9, .1, 0],
            [.5, .5, 0],
            [.3, .3, .4],
            [0, 1, 0],
            # cat
            [0, 1, 0],
            [.5, .5, 0],
            [.8, .2, 0],
            [.5, 0, .5],
            [0, .7, .3],
            # chair
            [0, 0, 1],
            [.05, .05, .9],
            [.5, 0, .5],
            [0, 0, 1],
            [0, .1, .9],
        ], coords={
            'image_id': ('presentation', [f"{object_name}_{i}" for object_name in objects for i in range(5)]),
            'object_name': ('presentation', [object_name for object_name in objects for _ in range(5)]),
            'truth': ('presentation', [object_name for object_name in objects for _ in range(5)]),
            'choice': objects,
        }, dims=['presentation', 'choice'])
        o2 = _o2(probabilities)
        print(o2)
        expected = BehavioralAssembly([[np.nan, .302, 2.291],
                                       [.302, np.nan, 2.516],
                                       [2.291, 2.516, np.nan],
                                       ], coords={'task_left': objects, 'task_right': objects},
                                      dims=['task_left', 'task_right'])
        np.testing.assert_array_equal(o2.shape, expected.shape)
        # ideally we would use xarray.testing.assert_allclose but it doesn't seem to allow for re-ordering
        for task_left, task_right in itertools.product(expected['task_left'].values, expected['task_right'].values):
            value_actual = o2.sel(task_left=task_left, task_right=task_right)
            value_expected = expected.sel(task_left=task_left, task_right=task_right)
            assert value_actual == approx(value_expected.values, abs=.0005, nan_ok=True)
