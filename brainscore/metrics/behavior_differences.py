from brainscore.metrics import Metric
from brainscore.metrics.image_level_behavior import I2n


class BehaviorDifferences(Metric):
    def __init__(self):
        self.i2n = I2n()

    def __call__(self, assembly1, assembly2):
        # split into control and inactivation days, ignoring postcontrol
        assembly1 = assembly1.groupby('image_id').mean('presentation')
        control1, inactivation1 = assembly1.sel(day='pre-control').squeeze(), assembly1.sel(day='inactivation').squeeze()
        # control2, inactivation2 = assembly2.sel(day='pre-control'), assembly2.sel(day='inactivation')
        # compute aggregate metrics
        behavior_pre1, behavior_inactivation1 = self.i2n.dprimes(control1), self.i2n.dprimes(inactivation1)
        # behavior_pre2, behavior_inactivation2 = self.i2n.dprimes(control2), self.i2n.dprimes(inactivation2)
        behavioral_differences1 = self._compute_differences(behavior_pre1, behavior_inactivation1)
        # behavioral_differences2 = self._compute_differences(behavior_pre2, behavior_inactivation2)

        # compare
        score = self._correlate(behavioral_differences1, behavioral_differences2)
        return score

    def _split_days(self, behaviors):
        non_silenced = behaviors[{'day': [day != 'inactivation' for day in behaviors['day'].values]}]
        silenced = behaviors.sel(day='inactivation')
        return ...
    def _compute_differences(self, behaviors):
        return ...
