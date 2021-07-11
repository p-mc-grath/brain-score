import pytest

from brainscore.benchmarks.rajalingham2019 import Rajalingham2019


class TestRajalingham2019:
    def test_assembly(self):
        benchmark = Rajalingham2019()
        assembly = benchmark._target_assembly
        assert assembly is not None
        assert len(assembly['injected']) == 2
        assert set(assembly['injected'].values) == {True, False}
        assert len(assembly['split']) == 2
        assert len(assembly['bootstrap']) == 10
        assert len(assembly['task']) == 6
        assert len(assembly['site']) == 11

    def test_ceiling(self):
        benchmark = Rajalingham2019()
        ceiling = benchmark.ceiling
        assert ceiling.sel(aggregation='center') == pytest.approx(.53, abs=.01)
