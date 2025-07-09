import numpy as np
from sklearn.metrics import (
    mean_absolute_error,          # MAE
    mean_squared_error,           # MSE  (+ RMSE khi squared=False)
    mean_absolute_percentage_error   # MAPE
)

EPS = 1e-8

def mpe(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) / (y_true + EPS)) * 100

def rmspe(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + EPS)))) * 100

def calculate_smape(y_true, y_pred):
    epsilon = 1e-8
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return np.mean(numerator / denominator) * 100

def calculate_all_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("Mảng y_true / y_pred rỗng – không thể tính metric.")

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100      
    smape_val = calculate_smape(y_true, y_pred)                                

    return {
        "MAE"  : mae,
        "MSE"  : mse,
        "RMSE" : rmse,
        "MAPE" : mape,
        "sMAPE": smape_val,
        "MPE"  : mpe(y_true, y_pred),
        "RMSPE": rmspe(y_true, y_pred)
    }

