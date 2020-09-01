import math
import os

import numpy as np
import pandas as pd
import pingouin as pg
import torch
from scipy.stats import pearsonr
from torch.nn import MSELoss, L1Loss


def regression_performance_analysis(actual, predict):
    """
    Analyse the performance of the predicted result

    Args:
        actual: The actual labels
        predict: the predicted labels

    Returns:
        MSE, MAE, PCC
    """
    with torch.set_grad_enabled(False):
        mse_loss_ft, mae_loss_ft = MSELoss(), L1Loss()
        mse_loss, mae_loss = mse_loss_ft(predict, actual), mae_loss_ft(predict, actual)

    predict = predict.cpu().numpy().squeeze()
    actual = actual.cpu().numpy().squeeze()

    pcc = pearsonr(predict, actual)

    if math.isnan(pcc[0]):
        out_pcc = 0
    else:
        out_pcc = pcc[0]
    return mse_loss.item(), mae_loss.item(), out_pcc


def calculate_icc(actual, predict):
    actual = actual.numpy()
    predict = predict.numpy()
    ids = np.array([i for i in range(len(actual))])
    judge_origin = ['A' for _ in range(len(actual))]
    judge_model = ['B' for _ in range(len(predict))]
    ids = pd.Series(np.concatenate((ids, ids)))
    judges = pd.Series(judge_origin + judge_model)
    scores = pd.Series(np.concatenate((actual, predict)))
    data = pd.DataFrame({'ids': ids, 'judges': judges, 'scores': scores})
    icc = pg.intraclass_corr(data=data, targets='ids', raters='judges',
                             ratings='scores')
    return icc.ICC[2]


def check_env(key, value):
    if key in os.environ and os.environ[key] == value:
        return True
    return False

