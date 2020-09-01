import os

import torch
from torch.utils.data import DataLoader

from databases.cache_dataset import CacheDataset
from utils.constants import device
from utils.dl_utils import id_collate


def validate(data_loader: DataLoader, model_ft, get_labels_fn, reduce_fn):
    model_ft.eval()
    predict = []
    actual = []
    ids = []

    for batch_data in data_loader:
        # image_debug_utils.show_landmarks_batch(batch_data)
        inputs = batch_data['image'].to(device, dtype=torch.float, non_blocking=True)
        labels = get_labels_fn(batch_data)
        ids += batch_data['id']

        with torch.set_grad_enabled(False):
            # Get model outputs
            outputs = model_ft(inputs)
            # Make sure it works with output of model is a tuple or torch
            outputs = outputs if isinstance(outputs, tuple) else torch.split(outputs, 1, dim=1)
            labels = labels if isinstance(labels, tuple) else torch.split(labels, 1, dim=1)
            for idx, item in enumerate(outputs):
                if idx >= len(predict):
                    predict.append([])
                    actual.append([])
                predict[idx].append(reduce_fn(item).cpu())
                actual[idx].append(reduce_fn(labels[idx]).cpu())

    for idx in range(len(predict)):
        predict[idx] = torch.cat(predict[idx])
        actual[idx] = torch.cat(actual[idx])

    return predict, actual, ids


def default_reduce_fn(data):
    """
    Sometime, when the output of the model is a big tensor (like heatmap), then it is too expensive to keep it on memory
    So, this optional function will allow us to reduce it, so we can save it to memory safely
    """
    return data


def get_labels(batch_data):
    labels = batch_data['score']
    labels = labels.to(device, dtype=torch.float32, non_blocking=True)
    return labels.view(-1, 1)


def load_pretrained_model(checkpoint, model_ft, fold=None, val_data_loader=None, estimator_fn=None,
                          drop_unknown_layers=False,
                          get_labels_fn=get_labels, val_reduce_fn=default_reduce_fn, strict=True):
    if not os.path.isfile(checkpoint):
        return False, None, None
    pretrained_dict = torch.load(checkpoint, map_location=device)
    if drop_unknown_layers:
        model_state_dict = model_ft.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
    model_ft.load_state_dict(pretrained_dict, strict=strict)
    model_ft.to(device)

    model_ft.eval()
    if not val_data_loader:
        return True, None, None

    predict, actual, ids = validate(val_data_loader, model_ft, get_labels_fn, val_reduce_fn)
    mse, prediction = estimator_fn(0, 0., fold, predict, actual)
    return True, prediction, (predict, actual, ids)


def generate_cache_dataset(dataset, keys=None, cache_path=None, force_create=False, in_memory=False):
    cache_dataset = CacheDataset(cache_dir=cache_path, in_memory=in_memory)
    if cache_dataset.has_cache() and not force_create:
        return cache_dataset
    if force_create:
        cache_dataset.cleanup()
    data_loader = DataLoader(dataset, batch_size=35, shuffle=True, num_workers=15, collate_fn=id_collate)
    for records in data_loader:
        for idx, item in enumerate(records['id']):
            record = {'id': item}
            for key in keys:
                record = {**record, key: records[key][idx]}
            cache_dataset.add_record(record)
    return cache_dataset

