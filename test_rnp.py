import torchvision
# -*- coding : utf-8 -*-
import pandas as pd
from dataloader_smanp import DatasetGP_test, data_load
from model_smanp import SpatialNeuralProcess, Criterion
from math import sqrt
import torch as torch
from train_configs import val_runner
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import time
from scipy.special import softmax
start = time.time()
import random
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_tasks = 1
batch_size = 1
x_size = 6
y_size = 1
z_size = 128
lr = 0.001
# num_context =997
num_hidden = 128
list1 = []
list = []
model_res = SpatialNeuralProcess(x_size=x_size, y_size=y_size, num_hidden=num_hidden)
model = model_res.to(device)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


list_test = []
list1_test = []

set_seed(42)
for xuhao in range(10):
    dataset = DatasetGP_test(n_tasks=n_tasks, xuhao=xuhao, batch_size=batch_size)
    testloader = DataLoader(dataset, batch_size=1, shuffle=False)
    state_dict = torch.load('./trained/checkpoint_{}.pth.tar'.format(xuhao))
    model.load_state_dict(state_dict=state_dict['model_res'])
    model.eval()
    criterion = Criterion()

    val_pred_y, val_var_y, val_target_id, val_target_y, val_loss, valid_r2 = val_runner(model, testloader, criterion)

    val_target_y = val_target_y.cpu().detach().numpy()
    val_pred_y = val_pred_y.cpu().detach().numpy()
    val_var_y = val_var_y.cpu().detach().numpy()
    val_target_id = val_target_id.cpu().detach().numpy()
    valid_mse = (np.sum((val_target_y - val_pred_y) ** 2)) / len(val_target_y)
    valid_rmse = sqrt(valid_mse)
    valid_mae = (np.sum(np.absolute(val_target_y - val_pred_y))) / len(val_target_y)
    valid_mape = (np.sum(np.absolute(val_target_y - val_pred_y) / val_target_y)) / len(val_target_y)
    corr = np.corrcoef(val_target_y, val_pred_y)
    C = (2 * corr[0, 1] * np.std(val_pred_y) * np.std(val_target_y)) / (np.var(val_target_y) + np.var(val_pred_y) + (val_target_y.mean() - val_pred_y.mean()) ** 2)

    prediction = pd.DataFrame(
        {"id": np.array(val_target_id), "res_true": np.array(val_target_y), "res_pred": np.array(val_pred_y),
         "res_cha": np.array(val_target_y) - np.array(val_pred_y), 'res_var_y': np.array(val_var_y)})
    prediction.to_csv('./trained/prediction_res_{}.csv'.format(xuhao), index=False)

    # list = [round(valid_mae, 4), round(valid_mse, 4), round(valid_rmse, 4), round(valid_r2.item(), 4), round(C, 4), round(np.mean(val_var_y), 4), val_loss]
    # list1.append(list)

    ##prediction=gnn+res
    train_data = pd.read_csv(r'./trained/train_out_{}.csv'.format(xuhao))
    valid_data = pd.read_csv(r'./trained/test_out_{}.csv'.format(xuhao))
    train_data = np.array(train_data)
    valid_data = np.array(valid_data)

    test_true = valid_data[:, 9]
    test_gnn_pred = valid_data[:, 10]
    test_cnp_res_pred = np.array(val_pred_y)
    test_pred = test_gnn_pred + test_cnp_res_pred

    test_gnn_r2 = 1 - ((np.sum((test_true - test_gnn_pred) ** 2)) / np.sum((test_true - test_gnn_pred.mean()) ** 2))
    test_gnn_mse = (np.sum((test_true - test_gnn_pred) ** 2)) / len(test_true)
    test_gnn_rmse = np.sqrt(test_gnn_mse)
    test_gnn_mae = (np.sum(np.absolute(test_true - test_gnn_pred))) / len(test_true)
    test_gnn_mape = (np.sum(np.absolute(test_true - test_gnn_pred) / test_true)) / len(test_true)
    gnn_corr = np.corrcoef(test_true, test_gnn_pred)
    test_gnn_C = (2 * gnn_corr[0, 1] * np.std(test_gnn_pred) * np.std(test_true)) / (np.var(test_true) + np.var(test_gnn_pred) + (test_true.mean() - test_gnn_pred.mean()) ** 2)

    test_r2 = 1 - ((np.sum((test_true - test_pred) ** 2)) / np.sum((test_true - test_pred.mean()) ** 2))
    test_mse = (np.sum((test_true - test_pred) ** 2)) / len(test_true)
    test_rmse = np.sqrt(test_mse)
    test_mae = (np.sum(np.absolute(test_true - test_pred))) / len(test_true)
    test_mape = (np.sum(np.absolute(test_true - test_pred) / test_true)) / len(test_true)
    # 计算对数似然
    log_likelihood = np.sum(np.log(softmax(test_pred)) * softmax(test_true) + np.log(1 - softmax(test_pred)) * (1 - softmax(test_true)))



    corr = np.corrcoef(test_true, test_pred)
    test_C = (2 * corr[0, 1] * np.std(test_pred) * np.std(test_true)) / (np.var(test_true) + np.var(test_pred) + (test_true.mean() - test_pred.mean()) ** 2)

    prediction = pd.DataFrame(
        {"id": np.array(val_target_id), "true": test_true,
         "test_gnn_pred": test_gnn_pred, "test_cnp_res_pred": test_cnp_res_pred,
         "test_pred": test_pred,
         "cha": test_true - test_pred, 'var_y': np.array(val_var_y)})
    prediction.to_csv('./trained/prediction_val_{}.csv'.format(xuhao), index=False)
    print('\n')

    print("ID:", xuhao, "valid_MAE:", round(valid_mae, 4), "valid_MSE:", round(valid_mse, 4), " valid_RMSE:", round(valid_rmse, 4),
          " valid_R-square:", round(valid_r2.item(), 4), "valid_MAPE:", round(valid_mape, 4), "average_var:", round(np.mean(val_var_y), 4), 'val_loss', val_loss)

    print("ID:", xuhao, "test_gnn_MAE:", round(test_gnn_mae, 4), "test_gnn_MSE:", round(test_gnn_mse, 4), " test_gnn_RMSE:", round(test_gnn_rmse, 4),
          " test_gnn_R-square:", round(test_gnn_r2.item(), 4), "test_gnn_mape:", round(test_gnn_mape, 4), )

    print("ID:", xuhao, "test_MAE:", round(test_mae, 4), "test_MSE:", round(test_mse, 4), " test_RMSE:", round(test_rmse, 4),
          " test_R-square:", round(test_r2.item(), 4), "t est_MAE:", round(test_mape, 4),
          "average_var:", round(np.mean(val_var_y), 4), 'val_loss', val_loss)

    list0 = [round(test_gnn_mae, 4), round(test_gnn_rmse, 4),  round(test_gnn_r2.item(), 4),round(test_gnn_mape, 4), round(test_gnn_mse, 4), ]
    list1.append(list0)

    list_test = [round(test_mae, 4), round(test_rmse, 4), round(test_r2.item(), 4), round(test_mape, 4), round(test_mse, 4), log_likelihood]
    list1_test.append(list_test)

# print('list:', list1)
# print('res_mean:', np.mean(list1, axis=0))
# print('res_mean:', np.std(list1, axis=0))
print('\n')
np.set_printoptions(precision=4)

print('test_gnn_mean:', np.mean(list1, axis=0))
print('test_gnn_std:', np.std(list1, axis=0))

print('test_mean:', np.mean(list1_test, axis=0))
print('test_std:', np.std(list1_test, axis=0))
