import numpy as np
from brainio_base.assemblies import DataAssembly


def collect_assembly():
    # visual inspection of Fig. 2 in https://www.sciencedirect.com/science/article/pii/S0165027002001437
    # x goes left-to-right, y goes bottom-to-top
    assembly = DataAssembly([
        # parasagittal
        [
            'no', 'no', 'no', 'less', 'less', 'less', 'less', 'no', 'no',
            'no', 'no', 'less', 'more', 'more', 'more', 'less', 'no', 'no',
            'no', 'less', 'more', 'more', 'complete', 'complete', 'more', 'no', 'no',
            'no', 'more', 'complete', 'complete', 'complete', 'complete', 'more', 'more', 'no',
            'no', 'less', 'more', 'complete', 'complete', 'more', 'more', 'no', 'no',
            'no', 'less', 'less', 'more', 'more', 'more', 'less', 'no', 'no',
            'no', 'less', 'less', 'more', 'more', 'more', 'less', 'no', 'no',
            'no', 'no', 'less', 'less', 'less', 'less', 'less', 'no', 'no',
        ],
        # horizontal
        [
            'no', 'no', 'no', 'less', 'less', 'less', 'no', 'no', 'no',
            'no', 'no', 'less', 'more', 'more', 'more', 'less', 'no', 'no',
            'no', 'no', 'more', 'more', 'complete', 'complete', 'more', 'less', 'no',
            'no', 'less', 'complete', 'complete', 'complete', 'complete', 'complete', 'less', 'less',
            'no', 'less', 'more', 'complete', 'complete', 'more', 'more', 'less', 'no',
            'no', 'less', 'more', 'more', 'more', 'more', 'more', 'no', 'no',
            'no', 'no', 'less', 'more', 'more', 'less', 'less', 'no', 'no',
            'no', 'no', 'less', 'less', 'less', 'less', 'less', 'less', 'no',
        ]
    ], name='suppression', coords={
        'plane': ['parasagittal', 'horizontal'],
        'x_mm': ('position', [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4] * 8),  # 9 elements on x
        'y_mm': ('position', np.repeat([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], 9)),  # 8 elements on y
    }, dims=['plane', 'position'])
    # same for all data packaged here
    assembly.attrs['time_post_injection_start_hours'] = 0.5
    assembly.attrs['time_post_injection_hours_end'] = 1.5
    assembly.attrs['injection_site_x_mm'] = 2
    assembly.attrs['injection_site_y_mm'] = 1.5
    return assembly


if __name__ == '__main__':
    collect_assembly()
