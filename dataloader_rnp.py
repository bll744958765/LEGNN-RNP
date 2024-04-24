# -*- coding : utf-8 -*-

"""
数据生成
"""

from torch.utils.data import Dataset
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

print(torch.version.cuda)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Taking simulation data as an example
def data_load(tt):
    data_name1 = {0: './trained/train_out_0.csv', 1: './trained/train_out_1.csv',
                  2: './trained/train_out_2.csv', 3: './trained/train_out_3.csv',
                  4: './trained/train_out_4.csv', 5: './trained/train_out_5.csv',
                  6: './trained/train_out_6.csv', 7: './trained/train_out_7.csv',
                  8: './trained/train_out_8.csv', 9: './trained/train_out_9.csv', }

    data_name2 = {0: './trained/test_out_0.csv', 1: './trained/test_out_1.csv',
                  2: './trained/test_out_2.csv', 3: './trained/test_out_3.csv',
                  4: './trained/test_out_4.csv', 5: './trained/test_out_5.csv',
                  6: './trained/test_out_6.csv', 7: './trained/test_out_7.csv',
                  8: './trained/test_out_8.csv', 9: './trained/test_out_9.csv', }
    train_data = pd.read_csv(data_name1[tt])
    valid_data = pd.read_csv(data_name2[tt])

    train_data = np.array(train_data)
    valid_data = np.array(valid_data)
    train_x = train_data[:, 1:9]
    train_y = train_data[:, (0, 11)]
    valid1_x = valid_data[:, 1:9]
    valid_y = valid_data[:, (0, 11)]

    # standardization


    y = torch.tensor(train_y).to(device).float()
    valid_y = torch.tensor(valid_y).to(device).float().squeeze(-1)
    x = torch.FloatTensor(train_x).to(device).float()
    valid_x = torch.FloatTensor(valid1_x).to(device).float()
    return x, y, valid_x, valid_y


def get_tensor_from_pd(dataframe_series):
    return torch.tensor(data=dataframe_series.values)

class DatasetGP(Dataset):
    def __init__(self, n_tasks, xuhao,
                 batch_size=2,
                 n_context_min=3,
                 n_context_max=3300,):
        super().__init__()
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.xuhao = xuhao
        self.n_context_min = n_context_min
        self.n_context_max = n_context_max

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, index):
        n_context = np.random.randint(self.n_context_min, self.n_context_max + 1)
        n_target = n_context + np.random.randint(3, 4129- n_context + 1)

        batch_context_x = []
        batch_context_y = []
        batch_target_x = []
        batch_target_y = []


        for _ in range(self.batch_size):
            x, y, _, _, = data_load(self.xuhao)
            context_x = x[0: n_context, :]
            context_y = y[0: n_context, :]

            target_x = x[0:n_target, :]
            target_y = y[0:n_target, :]

            batch_context_x.append(context_x)
            batch_context_y.append(context_y)
            batch_target_x.append(target_x)
            batch_target_y.append(target_y)

        batch_context_x = torch.stack(batch_context_x, dim=0)
        batch_context_y = torch.stack(batch_context_y, dim=0)
        batch_target_x = torch.stack(batch_target_x, dim=0)
        batch_target_y = torch.stack(batch_target_y, dim=0)

        return batch_context_x, batch_context_y, batch_target_x, batch_target_y


#test/valid set
class DatasetGP_test(Dataset):
    def __init__(self, n_tasks, xuhao, batch_size=1):
        super().__init__()
        self.n_tasks = n_tasks
        self.xuhao = xuhao
        self.batch_size = batch_size

    def __len__(self):
        return self.n_tasks

    def __getitem__(self, index):
        batch_context_x = []
        batch_context_y = []
        batch_target_x = []
        batch_target_y = []

        for _ in range(self.batch_size):
            x, y, valid_x, valid_y= data_load(self.xuhao)
            context_x = x
            context_y = y
            target_x = valid_x
            target_y = valid_y

            batch_context_x.append(context_x)
            batch_context_y.append(context_y)

            batch_target_x.append(target_x)
            batch_target_y.append(target_y)

        batch_context_x = torch.stack(batch_context_x, dim=0)
        batch_context_y = torch.stack(batch_context_y, dim=0)
        batch_target_x = torch.stack(batch_target_x, dim=0)
        batch_target_y = torch.stack(batch_target_y, dim=0)

        return batch_context_x, batch_context_y, batch_target_x, batch_target_y
