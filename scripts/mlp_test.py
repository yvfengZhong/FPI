#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append(".")
from utils.logging import Logger


class MLP(nn.Module):
    def __init__(self, in_features=12, hidden=512):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=hidden, out_channels=2048, kernel_size=1), nn.ReLU())
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=hidden, kernel_size=1)
        self.lnorm = nn.LayerNorm(hidden)
        self.fc = nn.Linear(hidden * in_features, 1)

    def forward(self, data):
        shape = data.shape
        data = data.unsqueeze(-1)
        data = self.net(data)
        out = self.conv1(data.transpose(1, 2))
        out = self.conv2(out).transpose(1, 2)
        out = self.lnorm(out + data)
        out = out.squeeze(-1).reshape(shape[0], 12 * 512)
        out = self.fc(out)
        return out


def get_net(in_features):
    net = MLP(in_features)
    return net


def mae(net, features, labels):
    preds = net(features)
    mae = torch.abs(labels - preds)
    mae = mae.mean().item()
    return mae


def test(test_features, in_features, pre_train):
    net = get_net(in_features).to(test_features.device)
    # 加载本地的模型参数
    model_weights = torch.load(os.path.join(pre_train, 'best_test_weights.pt'))
    net.load_state_dict(model_weights)
    with torch.no_grad():
        net.eval()
        preds = net(test_features).cpu().detach().numpy()

    return preds


def MAE_evaluate_metrics(y_test, y_pred):
    mae = np.mean(np.abs(y_test - y_pred))

    return mae.item()


def RMSE_evaluate_metrics(y_test, y_pred):
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))

    return rmse.item()


def save_weights(model, save_path):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, save_path)


print("test beginning!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 获得执行文件的绝对路径
root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))
log_root = os.path.join(root_path, 'checkpoints/mlp')
sys.stdout = Logger(os.path.join(log_root, 'log_test.txt'))

file_path = os.path.join(root_path, "dataset/CME/train_val_set.csv")
train_data = pd.read_csv(file_path, header=None) # 194
file_path = os.path.join(root_path, "dataset/CME/test_set.csv")
test_data = pd.read_csv(file_path, header=None) # 49

scaler = StandardScaler()
X_train = scaler.fit_transform(train_data.iloc[:, 2:-1].values)
X_test = scaler.transform(test_data.iloc[:, 2:-1].values)

train_labels = train_data.iloc[:, -1].values.reshape(-1, 1)
test_labels = test_data.iloc[:, -1].values.reshape(-1, 1)

train_features = torch.tensor(X_train, dtype=torch.float32, device=device)
test_features = torch.tensor(X_test, dtype=torch.float32, device=device)
train_labels = torch.tensor(train_labels, dtype=torch.float32, device=device)
test_labels = torch.tensor(test_labels, dtype=torch.float32, device=device)

in_features = train_features.shape[1]

y_pred = test(test_features, in_features, log_root)

# 传入的这两个都是ndarray
MAE_Error = MAE_evaluate_metrics(test_labels.cpu().detach().numpy(), y_pred)
print('%s%2.4f%s' % ('The prediction MAE is ', MAE_Error, ' hours.'))

RMSE_Error = RMSE_evaluate_metrics(test_labels.cpu().detach().numpy(), y_pred)
print('%s%2.4f%s' % ('The prediction RMSE is ', RMSE_Error, ' hours.'))    