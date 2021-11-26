import brainscore
from brainscore.utils import LazyLoad
from brainscore.metrics import Score
from brainscore.benchmarks import BenchmarkBase
from brainscore.model_interface import BrainModel

BIBTEX = '''@article{Verhoef2012InferotemporalCS,
              title={Inferotemporal Cortex Subserves Three-Dimensional Structure Categorization},
              author={B. Verhoef and R. Vogels and P. Janssen},
              journal={Neuron},
              year={2012},
              volume={73},
              pages={171-182}
            }'''


class DicarloVerhoef2012MicroStimulation(BenchmarkBase):

    def __init__(self):
        pass

    def __call__(self, candidate: BrainModel):
        '''
        find 3d structure selective patch !binocular disparity not in model!!
            - the 3D-structure selectivity of the MUA using a passive
                fixation task in which the monkey viewed 100% stereo-coherent
                convex or concave stimuli positioned at one of three positions in
                dept
            - 900microm clusters, separated by 450 microm
            - disparity-defined 3D surfaces
            - selectivity simple 3d structures
            - invariance for size and position (in depth)
        stimulate 35microA 200Hz
        2x 3d structure categorization task; stimulated and not
        stimuli
        - 3 planes: near, mid, far (timuli were positioned either behind
            (Far), within (Fix), or in front of (Near) the fixation plane)
        microstim induced shift of psychometric funtion as f of proportion of dots added to random dot stereograms
        magically compare
        '''
        pass
