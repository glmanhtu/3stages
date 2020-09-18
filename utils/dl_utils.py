import os

import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate


def save_trained_model(model, checkpoint):
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(model.state_dict(), checkpoint)


def get_params_to_update(model_ft):
    # Gather the parameters to be optimized/updated in this run
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

    return params_to_update


def freeze_all_parameters(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def freeze_parameters(model, n_parameters):
    """
    Freeze the specific layers for fine tuning

    @param model: nn.Module object
    @param n_parameters: the number of parameters that need to be frozen
        if it is zero, fine tune the whole network
        if it is positive number then freeze first n parameters
        if it is negative, then freeze last n parameters
    """

    layers = []
    names = []
    for name, param in model.named_parameters():
        layers.append(param)
        names.append(name)

    if n_parameters == 0:
        return

    if n_parameters < 0:
        for param in layers[n_parameters:]:
            param.requires_grad = False
    else:
        for param in layers[:n_parameters]:
            param.requires_grad = False


def extract_image_features(dl_model, sequence_images):
    """
    Extracting the features from sequence of images
    @param dl_model: Deep learning model. The fine tuned vgg_faces to be precised
    @param sequence_images: batch of sequence images
    @return: batch of sequence features
    """
    if dl_model is None:
        return sequence_images

    dl_model.eval()

    with torch.set_grad_enabled(False):
        batch, seq, channels, im_h, im_w = tuple(sequence_images.shape)
        features = sequence_images.reshape((-1, channels, im_h, im_w))
        features = dl_model(features)
        features = features.reshape((batch, seq, -1))

    return features


def id_collate(batch):
    ids = []
    for _batch in batch:
        ids.append(_batch['id'])
    result = default_collate(batch)
    result['id'] = ids
    return result


def weights_init_xavie(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.zero_()


def get_list_aus(aus, locations):
    result = []
    for au in aus:
        au_id = int(au.replace('au_', ''))
        num_points = len([x for x in locations[au_id] if x is not None])
        for x in range(num_points):
            result.append('%s_%d' % (au, x))
    return result
