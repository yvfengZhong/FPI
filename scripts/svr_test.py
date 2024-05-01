import argparse
import os
import pickle
import sys
sys.path.append(".")
import time

import numpy as np
import pandas as pd

from utils.logging import Logger


def get_data(root_path):
    file_path = os.path.join(root_path, "dataset/CME/test_set.csv")
    data = pd.read_csv(file_path, header=None)
    data = data.iloc[:, 2:].values
    
    X_test = data[:, :-1]
    y_test = data[:, -1]
    
    print("test on {} samples.".format(len(X_test)))
    return X_test, y_test


def evaluate_metrics(y_test, y_pred):
    diff = y_test - y_pred
    diff = abs(diff)
    return diff.mean()


def MAE_evaluate_metrics(y_test, y_pred):
    mae = np.mean(np.abs(y_test - y_pred))

    return mae.item()


def RMSE_evaluate_metrics(y_test, y_pred):
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))

    return rmse.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CME-Solar_Wind svr')
    parser.add_argument('--exp_name', type=str, default='CS_svr')
    parser.add_argument('--log', type=str, default='checkpoints/cat_puma')
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))

    log_path = args.log
    os.makedirs(log_path, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_path, 'log_test.txt'))
    print(args)

    print('CME-Solar_Wind svr Testing')
    start = time.time()

    X_test, y_test = get_data(root_path)

    with open(os.path.join(log_path, 'scaler.pickle'), 'rb') as f:
        scaler_model = pickle.load(f)

    scaler = scaler_model
    X_test = scaler.transform(X_test)

    with open(os.path.join(log_path, 'clf_best.pickle'), 'rb') as f:
        clf_best_model = pickle.load(f)

    clf_best = clf_best_model
    y_pred = clf_best.predict(X_test)

    MAE_Error = MAE_evaluate_metrics(np.array(y_test), np.array(y_pred))
    print('%s%2.4f%s' % ('The prediction MAE is ', MAE_Error, ' hours.'))

    RMSE_Error = RMSE_evaluate_metrics(np.array(y_test), np.array(y_pred))
    print('%s%2.4f%s' % ('The prediction RMSE is ', RMSE_Error, ' hours.'))    