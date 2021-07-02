import copy
import math
import os

import numpy as np
from PIL import Image
from brainio_base.stimuli import StimulusSet
from numpy.random.mtrand import RandomState
from result_caching import store
from sklearn.model_selection import train_test_split
from tqdm import tqdm

source_face_directory = '/braintree/data2/active/common/labeled_faces_in_the_wild/'
source_imagenet_directory = '/braintree/data2/active/common/imagenet_raw/train'
source_imagenet_synsets = [  # from https://gist.github.com/fnielsen/4a5c94eaa6dcdf29b7a62d886f540372
    'n03376595',  # folding chair
    'n07753275',  # pineapple
    'n01514859',  # hen
    'n02321529',  # sea cucumber
    'n02974003',  # car wheel
    'n03792782',  # mountain bike
]

data_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../brainio_collection/brainio_contrib', 'data'))
face_directory = os.path.join(data_directory, 'faces')
nonface_directory = os.path.join(data_directory, 'nonfaces')


def get_images(test_size=.5, lam=0.0):
    face_paths = [os.path.join(face_directory, filename) for filename in os.listdir(face_directory)]
    nonface_paths = [os.path.join(nonface_directory, filename) for filename in os.listdir(nonface_directory)]
    face_paths, nonface_paths = list(sorted(face_paths)), list(sorted(nonface_paths))
    image_ids = [os.path.splitext(os.path.basename(path))[0] for path in face_paths + nonface_paths]
    stimuli = StimulusSet({'image_id': image_ids,
                           'image_label': ['face'] * len(face_paths) + ['nonface'] * len(nonface_paths)})
    stimuli.image_paths = dict(zip(image_ids, face_paths + nonface_paths))
    stimuli.name = 'faces_nonfaces'
    train_stimuli, test_stimuli = train_test_split(stimuli, test_size=test_size, random_state=42)
    noisy_train_stimuli, noisy_test_stimuli = make_noisy('train', train_stimuli, lam), make_noisy('test', test_stimuli)
    noisy_train_stimuli.name, noisy_test_stimuli.name = \
        stimuli.name + f'-train{1 - test_size:.2f}_noisy{lam:.2f}', stimuli.name + f'-test{test_size:.2f}_noisy'
    return noisy_train_stimuli, noisy_test_stimuli


@store(identifier_ignore=['stimuli'])
def make_noisy(identifier, stimuli, oversample_nonnoisy=False):
    noisy_stimuli = copy.deepcopy(stimuli)
    noisy_stimuli['noise_level'] = 0
    target_directory = os.path.join(data_directory, f"noisy-{identifier}-{oversample_nonnoisy}")
    os.makedirs(target_directory, exist_ok=True)
    random_state = RandomState(seed=20)
    for image_id in tqdm(stimuli['image_id'].values, desc='make noisy'):
        source_path = stimuli.get_image(image_id)
        image = Image.open(source_path)
        noise_level = random_state.uniform(0, 1) if not oversample_nonnoisy \
            else lopsided_sample(lam=oversample_nonnoisy)
        noisy_image = make_noisy_image(np.array(image), noise_level=noise_level, random_state=random_state)
        noisy_image = Image.fromarray(noisy_image)
        target_path = os.path.join(target_directory, os.path.basename(source_path))
        noisy_image.save(target_path)
        noisy_stimuli.image_paths[image_id] = target_path
        noisy_stimuli['noise_level'][noisy_stimuli['image_id'] == image_id] = noise_level
    return noisy_stimuli


def make_noisy_image(image, noise_level, random_state):
    shape = image.shape
    assert len(shape) == 2
    noise_mask = random_state.choice(a=[True, False], size=np.prod(shape), p=[noise_level, 1 - noise_level])
    noise = random_state.uniform(low=0, high=255, size=sum(noise_mask))
    image = image.reshape(-1)
    image[noise_mask] = noise
    image = image.reshape(shape)
    return image


def lopsided_sample(size=None, lam=.5):
    sample = np.random.poisson(lam=lam, size=size)
    sample = np.clip(sample, a_min=None, a_max=5)
    sample = (sample - 0) / (5 - 0)
    return sample


def sample_faces(num_images):
    # faces are not uniformly distributed
    directories = os.listdir(source_face_directory)
    num_directories = len(directories)
    images_per_directory = math.ceil(num_images / num_directories)
    paths = []
    random_state = RandomState(1)
    for directory in directories:
        if len(paths) == num_images:
            break
        directory_paths = os.listdir(os.path.join(source_face_directory, directory))
        paths += [os.path.join(source_face_directory, directory, filename)
                  for filename in random_state.choice(directory_paths, images_per_directory)]
    return paths


def sample_nonfaces(num_images):
    # uniform distribution of images
    paths = [os.path.join(source_imagenet_directory, synset, path) for synset in source_imagenet_synsets
             for path in os.listdir(os.path.join(source_imagenet_directory, synset))]
    random_state = RandomState(1)
    return random_state.choice(paths, num_images)


def collect_stimuli(num_images, face_nonface_ratio):
    num_faces = int(num_images * face_nonface_ratio)
    num_nonfaces = num_images - num_faces
    faces = sample_faces(num_faces)
    nonfaces = sample_nonfaces(num_nonfaces)

    os.makedirs(face_directory, exist_ok=True)
    os.makedirs(nonface_directory, exist_ok=True)
    for path in tqdm(faces, desc='faces'):
        collect_path(path, target_directory=face_directory)
    for path in tqdm(nonfaces, desc='nonfaces'):
        collect_path(path, target_directory=nonface_directory)


def collect_path(path, target_directory):
    image = Image.open(path)
    image = image.convert('L')
    image.save(os.path.join(target_directory, os.path.basename(path)))


if __name__ == '__main__':
    # from Afraz et al., Nature 2006
    paper_num_face_objects = 30
    paper_num_nonface_objects = 60
    ratio = paper_num_face_objects / (paper_num_face_objects + paper_num_nonface_objects)
    ratio = 1 / 2  # classifier will always choose nonface otherwise
    collect_stimuli(num_images=1000, face_nonface_ratio=ratio)
