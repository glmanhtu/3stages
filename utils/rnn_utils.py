import os

import torch
from torch.utils.data import DataLoader

from databases.cache_dataset import CacheDataset
from utils.constants import device
from utils.dl_utils import extract_image_features, id_collate


def get_labels(batch_data):
    labels = batch_data['score'][:, -1]
    return labels


def validate(data_loader: DataLoader, model_ft, get_label_fn):
    model_ft.eval()
    predict = []
    actual = []
    ids = []

    for sequences in data_loader:
        inputs = sequences['image'].to(device, dtype=torch.float, non_blocking=True)
        labels = get_label_fn(sequences)
        for seq in sequences['id']:
            ids.append(seq)
        # zero the parameter gradients

        with torch.set_grad_enabled(False):
            # Get model outputs
            outputs = model_ft(inputs)
            predict.append(outputs.cpu())
            actual.append(labels.cpu())

    return predict, actual, ids


def load_pretrained_model(checkpoint, model_ft, val_data_loader=None, fold='', get_labels_fn=get_labels,
                          estimator_fn=None):
    if not os.path.isfile(checkpoint):
        return False, None, None
    model_ft.load_state_dict(torch.load(checkpoint, map_location=device))
    model_ft.eval()

    if not val_data_loader:
        return True, None, None

    # Validating
    predict, actual, ids = validate(val_data_loader, model_ft, get_labels_fn)
    prediction = estimator_fn(0, 0, fold, predict, actual)
    return True, prediction, (predict, actual, ids)


def generate_rnn_cache_dataset(dataset, dl_model, cache_path, force_create=False):
    cache_dataset = CacheDataset(cache_path)
    if cache_dataset.has_cache() and not force_create:
        return cache_dataset
    if force_create:
        cache_dataset.cleanup()
    data_loader = DataLoader(dataset, batch_size=10, num_workers=5, pin_memory=True, collate_fn=id_collate)
    for sequences in data_loader:
        inputs = sequences['image'].to(device, dtype=torch.float, non_blocking=True)
        features = extract_image_features(dl_model, inputs)
        for idx, item in enumerate(features):
            record = {'image': item.cpu(), 'score': sequences['score'][idx], 'id': sequences['id'][idx]}
            cache_dataset.add_record(record)
    return cache_dataset
