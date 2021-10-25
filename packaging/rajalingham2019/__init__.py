from pathlib import Path

import numpy as np
import scipy.io

import brainscore
from brainio.assemblies import DataAssembly

EXPECTED_CATEGORIES = ['dog', 'bear', 'elephant', 'airplane3', 'chair0']


def collect_assembly(contra_hemisphere=True):
    """
    :param contra_hemisphere: whether to only select data and associated stimuli
        where the target object was contralateral to the injection hemisphere
    """
    path = Path(__file__).parent / 'Rajalingham2019_data_summary.mat'
    data = scipy.io.loadmat(path)['data_summary']
    struct = {d[0]: v for d, v in zip(data.dtype.descr, data[0, 0])}
    tasks = [v[0] for v in struct['O2_task_names'][:, 0]]
    tasks_left, tasks_right = zip(*[task.split(' vs. ') for task in tasks])
    k1 = {d[0]: v for d, v in zip(struct['k1'].dtype.descr, struct['k1'][0, 0])}

    class missing_dict(dict):
        def __missing__(self, key):
            return key

    dim_replace = missing_dict({'sub_metric': 'hemisphere', 'nboot': 'bootstrap', 'exp': 'site',
                                'niter': 'trial_split', 'subj': 'subject'})
    condition_replace = {'ctrl': 'saline', 'inj': 'muscimol'}
    dims = [dim_replace[v[0]] for v in k1['dim_labels'][0]]
    subjects = [v[0] for v in k1['subjs'][0]]
    conditions, subjects = zip(*[subject.split('_') for subject in subjects])
    metrics = [v[0] for v in k1['metrics'][0]]
    assembly = DataAssembly([k1['D0'], k1['D1']],
                            coords={
                                'injected': [False, True],
                                'injection': (dim_replace['subj'], [condition_replace[c] for c in conditions]),
                                'subject_id': (dim_replace['subj'], list(subjects)),
                                'metric': metrics,
                                dim_replace['sub_metric']: ['all', 'ipsi', 'contra'],
                                # autofill
                                dim_replace['niter']: np.arange(k1['D0'].shape[3]),
                                dim_replace['k']: np.arange(k1['D0'].shape[4]),
                                dim_replace['nboot']: np.arange(k1['D0'].shape[5]),
                                'site_number': ('site', np.arange(k1['D0'].shape[6])),
                                'site_iteration': ('site', np.arange(k1['D0'].shape[6])),
                                'experiment': ('site', np.arange(k1['D0'].shape[6])),
                                'task_number': ('task', np.arange(k1['D0'].shape[7])),
                                'task_left': ('task', list(tasks_left)),
                                'task_right': ('task', list(tasks_right)),
                            },
                            dims=['injected'] + dims)
    assembly['monkey'] = 'site', ['M' if site <= 9 else 'P' for site in assembly['site_number'].values]
    assembly = assembly.squeeze('k').squeeze('trial_split')
    if contra_hemisphere:
        assembly = assembly.sel(hemisphere='contra')
    assembly = assembly.sel(metric='o2_dp')
    assembly = assembly[{'subject': [injection == 'muscimol' for injection in assembly['injection'].values]}]
    assembly = assembly.squeeze('subject')  # squeeze single-element subject dimension since data are pooled already

    # add site locations
    path = Path(__file__).parent / 'xray_3d.mat'
    site_locations = scipy.io.loadmat(path)['MX']
    assembly['site_x'] = 'site', site_locations[:, 0] / 1000  # scale micro to mili
    assembly['site_y'] = 'site', site_locations[:, 1] / 1000
    assembly['site_z'] = 'site', site_locations[:, 2] / 1000
    assembly = DataAssembly(assembly)  # reindex

    # load stimulus_set subsampled from hvm
    stimulus_set = collect_stimulus_set(contra_hemisphere=contra_hemisphere)
    assembly.attrs['stimulus_set'] = stimulus_set
    assembly.attrs['stimulus_set_identifier'] = stimulus_set.identifier
    return assembly


def collect_stimulus_set(contra_hemisphere=True):
    # load stimulus_set subsampled from hvm
    stimulus_set_meta = scipy.io.loadmat('/braintree/home/msch/rr_share_topo/topoDCNN/dat/metaparams.mat')
    stimulus_set_ids = stimulus_set_meta['id']
    stimulus_set_ids = [i for i in stimulus_set_ids if len(set(i)) > 1]  # filter empty ids
    stimulus_set = brainscore.get_stimulus_set('dicarlo.hvm')
    stimulus_set = stimulus_set[stimulus_set['image_id'].isin(stimulus_set_ids)]
    stimulus_set = stimulus_set[stimulus_set['object_name'].isin(EXPECTED_CATEGORIES)]
    stimulus_set['image_label'] = stimulus_set['truth'] = stimulus_set['object_name']  # 10 labels at this point
    stimulus_set.identifier = 'dicarlo.hvm_10'
    if contra_hemisphere:
        stimulus_set = stimulus_set[(stimulus_set['rxz'] > 0) & (stimulus_set['variation'] == 6)]
    return stimulus_set
