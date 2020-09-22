import logging

import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch.nn import L1Loss, MSELoss
from torch.utils.data import DataLoader
from torchvision import transforms

from databases.unbc import unbc_mcmaster
from databases.unbc.unbc_mcmaster_cnn import UNBCCNNDataset
from databases.unbc.unbc_mcmaster_rnn import UNBCRNNDataset
from network import rnn, inception_resnet
from preprocessing.image import FixedImageStandardization, AUCentralLocalisation, HeatMapGenerator, \
    GPAAlignment, CentralCrop
from utils import dl_utils, cnn_utils, rnn_utils, resource_utils, metric_utils
from utils.constants import device, AU_CENTRAL_POINTS
from utils.dl_utils import id_collate
from utils.resource_utils import get_cache_path
import numpy as np

torch.backends.cudnn.benchmark = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

aus = ['au_4', 'au_6', 'au_9']
total_aus = dl_utils.get_list_aus(aus, AU_CENTRAL_POINTS)

subjects = unbc_mcmaster.get_subjects()
hm_transforms = transforms.Compose([GPAAlignment(), CentralCrop(160),
                                    AUCentralLocalisation(AU_CENTRAL_POINTS),
                                    HeatMapGenerator(aus, input_shape=(160, 160), output_shape=(64, 64))])
init_transform = transforms.Compose([GPAAlignment(), CentralCrop(160)])
pre_processing = transforms.Compose([
    transforms.ToTensor(), FixedImageStandardization()
])

hm_conf = {
    'lr': 1e-5,
    'batch_size': 64,
    'n_layer_frozen': 0
}

cnn_conf = {
    'lr': 3e-4,
    'batch_size': 64,
    'n_layer_frozen': 270
}

rnn_conf = {
    'lr': 9e-6,
    'batch_size': 72,
    'n_layers': 2,
    'input_dim': 1792,
    'n_neurons': 744,
    'dropout': 0.5,
    'seq_length': 16
}


def get_labels(batch_data):
    labels = batch_data['heatmap']
    return labels.to(device, dtype=torch.float32, non_blocking=True)


def val_reduce_fn(data):
    out = data.reshape(data.size(0), -1).max(dim=-1).values
    return out


def cnn_estimator(epoch, train_loss, fold, predict, actual):
    """
    Estimate performance of the CNN model
    Save the estimated result to sacred observer

    @rtype: metrics estimated
    """

    metrics = metric_utils.regression_performance_analysis(actual[0], predict[0])

    logging.info('CNN Fold: {} | mse: {:.4f}, mae: {:.4f}'.format(fold, metrics[0], metrics[1]))
    return metrics[0], metrics


def hm_estimator(epoch, train_loss, fold, predict, actual):
    metrics = []
    mse_criterion, mae_criterion = MSELoss(), L1Loss()
    with torch.set_grad_enabled(False):
        for i, name in enumerate(total_aus):
            mse, mae = mse_criterion(predict[i], actual[i]), mae_criterion(predict[i], actual[i])
            metrics.append([mse.item(), mae.item()])

    metrics = np.array(metrics)
    metrics = tuple(metrics.mean(axis=0))

    logging.info('HM Fold: {} | avg_mse: {:.4f}, mae: {:.4f}'.format(fold, metrics[0], metrics[1]))

    return metrics[0], metrics


def rnn_estimator(epoch, train_loss, fold, predict, actual):
    """
    Estimate performance of the RNN model
    Save the estimated result to sacred observer

    @rtype: metrics estimated
    """

    predict = torch.cat([torch.flatten(x) for x in predict])
    actual = torch.cat([torch.flatten(x) for x in actual])
    metrics = metric_utils.regression_performance_analysis(actual, predict)

    logging.info('RNN Fold: {} | mse: {:.4f}, mae: {:.4f}'.format(fold, metrics[0], metrics[1]))
    return metrics[0], metrics


def hm_leave_one_out_validation(fold, conf, checkpoint):
    """
    Leave @fold out validation
    First stage of predicting facial action unit intensities
    Fine-tune VGG-Faces model with UNBC dataset for estimating AUs intensities
    The best model with lowest MSE will be saved into the given checkpoint file
    If this checkpoint file exists, then only validation operator will be performed

    @rtype: (DL model, metrics estimated, raw result)
    """

    lr, batch_size, n_layer_frozen = conf['lr'], conf['batch_size'], conf['n_layer_frozen']

    subjects_left = [x for x in subjects if x != fold]

    # Validation data will contain only the given subject, so we exclude the rest
    val_data = UNBCCNNDataset(excluded_subjects=subjects_left, init_transform=hm_transforms,
                              transform=pre_processing, apply_balancing=False, exclude_black_frames=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=10, pin_memory=True, collate_fn=id_collate)

    dl_model = inception_resnet.InceptionHeatMap(num_aus=len(total_aus), device=device)
    load_status, metrics, predicts = cnn_utils.load_pretrained_model(
        checkpoint, dl_model, fold, val_loader, get_labels_fn=get_labels, estimator_fn=hm_estimator,
        val_reduce_fn=val_reduce_fn)
    if not load_status:
        raise Exception('Unable to load checkpoint ' + checkpoint)

    return dl_model, metrics, predicts


def cnn_leave_one_out_validation(fold, conf, dl_model, checkpoint):
    """
    Leave @fold out validation
    Second stage of predicting PSPI score
    Freeze all encoding layers of the first stage
    The best model with lowest MSE will be saved into the given checkpoint file
    If this checkpoint file exists, then only validation operator will be performed

    @rtype: (DL model, metrics estimated, raw result)
    """

    lr, batch_size, n_layer_frozen = conf['lr'], conf['batch_size'], conf['n_layer_frozen']

    subjects_left = [x for x in subjects if x != fold]

    # Validation data will contain only the given subject, so we exclude the rest
    val_data = UNBCCNNDataset(excluded_subjects=subjects_left, transform=pre_processing, apply_balancing=False,
                              exclude_black_frames=True, init_transform=init_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=10, pin_memory=True, collate_fn=id_collate)

    dl_utils.freeze_parameters(dl_model, n_layer_frozen)
    load_status, metrics, predicts = cnn_utils.load_pretrained_model(checkpoint, dl_model, fold, val_loader,
                                                                     estimator_fn=cnn_estimator)
    if not load_status:
        raise Exception('Unable to load checkpoint ' + checkpoint)
    return dl_model, metrics, predicts


def rnn_leave_one_out_validation(fold, conf, dl_model, checkpoint):
    """
    Leave @fold out validation
    Last stage of estimating pain level of sequence, based on LSTM
    Train LSTM model with UNBC dataset, using output from block8 layer of CNN model, for pain intensity estimation
    The best model with lowest MSE will be saved into a checkpoint file
    If this checkpoint file exists, then only validation operator will be performed

    @rtype: (DL model, metrics estimated, raw result)
    """

    lr, batch_size, n_layers = conf['lr'], conf['batch_size'], conf['n_layers']
    input_dim, n_neurons, dropout = conf['input_dim'], conf['n_neurons'], conf['dropout']
    seq_length = conf['seq_length']
    subjects_left = [x for x in subjects if x != fold]

    # Validation data will contain only the given subject, so we exclude the rest
    val_data = UNBCRNNDataset(excluded_subjects=subjects_left, transform=pre_processing, apply_balancing=False,
                              init_transform=init_transform, exclude_black_frames=True, sequence_length=seq_length)

    # The encoding image sequences will take way too long, especially when we do it for multiple epochs
    # So, do it one time and save into local hard disk
    val_data = rnn_utils.generate_rnn_cache_dataset(val_data, dl_model, get_cache_path(name="%s_val" % fold))
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=10, pin_memory=True, collate_fn=id_collate)

    lstm_model = rnn.LSTM(input_dim, n_neurons, num_layers=n_layers, num_classes=1, dropout=dropout).to(device)
    load_status, metrics, predicts = rnn_utils.load_pretrained_model(checkpoint, lstm_model, val_loader, fold=fold,
                                                                     estimator_fn=rnn_estimator)
    if not load_status:
        raise Exception('Unable to load checkpoint ' + checkpoint)

    return lstm_model, metrics, predicts


def cross_validation():

    cnn_all_predicts, cnn_all_actuals = [], []
    rnn_all_predicts, rnn_all_actuals = [], []

    # Leave one subject out cross validation
    for idx, subject in enumerate(subjects):

        # First stage: Encoding with HeatMap
        s1_checkpoint = resource_utils.get_checkpoint_file_path(name='%s_hm.ckpt' % subject)
        hm_leave_one_out_validation(subject, hm_conf, s1_checkpoint)
        dl_model = InceptionResnetV1(classify=True, device=device, num_classes=1)
        load_status, _, _ = cnn_utils.load_pretrained_model(s1_checkpoint, dl_model, subject,
                                                            drop_unknown_layers=True, strict=False)
        if not load_status:
            raise Exception('Unable to load checkpoint ' + s1_checkpoint)

        # Second stage: Encoding with PSPI MSE
        s2_checkpoint = resource_utils.get_checkpoint_file_path(name='%s_cnn.ckpt' % subject)
        dl_model, metrics, predicts = cnn_leave_one_out_validation(subject, cnn_conf, dl_model, s2_checkpoint)
        predict, actual, labels = predicts

        cnn_all_predicts.append(predict[0].squeeze())
        cnn_all_actuals.append(actual[0].squeeze())

        # Third stage: LSTM training
        dl_model = inception_resnet.get_pretrained_facenet(classify=False, pretrained=None)
        load_status, _, _ = cnn_utils.load_pretrained_model(s2_checkpoint, dl_model)
        if not load_status:
            raise Exception('Unable to load checkpoint ' + s2_checkpoint)
        s3_checkpoint = resource_utils.get_checkpoint_file_path(name='%s_rnn.ckpt' % subject)
        dl_model, metrics, predicts = rnn_leave_one_out_validation(subject, rnn_conf, dl_model, s3_checkpoint)
        predict, actual, labels = predicts

        rnn_all_predicts.append(torch.cat([torch.flatten(x) for x in predict]))
        rnn_all_actuals.append(torch.cat([torch.flatten(x) for x in actual]))

    logging.info('--------------------------------------------------\n')

    actual, predict = torch.cat(cnn_all_actuals), torch.cat(cnn_all_predicts)
    mse_loss, mae, pcc = metric_utils.regression_performance_analysis(actual, predict)
    icc = metric_utils.calculate_icc(actual, predict)

    logging.info('Stage 1+2: mse: {:.2f}, mae: {:.2f}, pcc: {:.2f}, icc: {:.2f}'.format(mse_loss, mae, pcc, icc))

    actual, predict = torch.cat(rnn_all_actuals), torch.cat(rnn_all_predicts)
    mse_loss, mae, pcc = metric_utils.regression_performance_analysis(actual, predict)
    icc = metric_utils.calculate_icc(actual, predict)

    logging.info('Three stages: mse: {:.2f}, mae: {:.2f}, pcc: {:.2f}, icc: {:.2f}'.format(mse_loss, mae, pcc, icc))

    return mse_loss


if __name__ == '__main__':
    cross_validation()
