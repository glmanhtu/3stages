import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(dir_path)


def get_resource(path, create_parent_dir=False):
    """
    Get the abs path from the resource folder
    """
    abs_path = os.path.join(dir_path, 'resources', *path.split('/'))
    if create_parent_dir:
        parent_dir = os.path.dirname(abs_path)
        os.makedirs(parent_dir, exist_ok=True)
    return abs_path


def get_checkpoint_path(name):
    return get_resource(os.path.join('checkpoints', name), create_parent_dir=True)


def get_cache_path(name):
    return get_resource(os.path.join('cache', name), create_parent_dir=True)
