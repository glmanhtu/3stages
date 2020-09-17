import json
import os

import requests
from requests.adapters import HTTPAdapter

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(dir_path)
with open(os.path.join(dir_path, 'resources', 'unbc', 'pretrained_models.json')) as f:
    pretrained = json.load(f)


def get_resource(path, create_parent_dir=False):
    """
    Get the abs path from the resource folder
    """
    abs_path = os.path.join(dir_path, 'resources', *path.split('/'))
    if create_parent_dir:
        parent_dir = os.path.dirname(abs_path)
        os.makedirs(parent_dir, exist_ok=True)
    return abs_path


def get_checkpoint_file_path(name):
    checkpoint_file = get_resource(os.path.join('checkpoints', name), create_parent_dir=True)
    if os.path.isfile(checkpoint_file):
        return checkpoint_file
    subject, mode = tuple(name.replace('.ckpt', '').split('_'))
    file_id = pretrained[subject][mode]
    file_url = 'https://drive.google.com/uc?export=download&id=' + file_id
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=10))
    r = s.get(file_url, allow_redirects=True)
    with open(checkpoint_file, 'wb') as cf:
        cf.write(r.content)
    return checkpoint_file


def get_cache_path(name):
    return get_resource(os.path.join('cache', name), create_parent_dir=True)
