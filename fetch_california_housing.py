
# Load the TensorBoard notebook extension
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import warnings
from sklearn.model_selection import train_test_split
import datetime
warnings.filterwarnings("ignore")
from datetime import datetime
import argparse
from dataloader_rnp import DatasetGP, DatasetGP_test, data_load
from model_rnp import ResidualNeuralProcess, Criterion
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from train_configs import train_runner, val_runner
import random
from model import PEGCN, LossWrapper, GCN
import time
import pandas as pd

from math import sqrt
import torch
import torch.nn as nn

import torch.utils.data
import math

from sklearn.preprocessing import StandardScaler
start = time.perf_counter()
time.sleep(2)

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




def get_cali_housing_data():
    '''
    Download and process the California Housing Dataset

    Parameters:
    
    Return
    coords = spatial coordinates (lon/lat)
    x = features at location
    y = outcome variable
    '''


    cali_housing_ds = pd.read_csv('./fetch_california_housing.csv')
    cali_housing_ds = np.array(cali_housing_ds)
    id = cali_housing_ds[:, 0]
    coords = cali_housing_ds[:, 1:3]
    y = cali_housing_ds[:, 9]
    x = cali_housing_ds[:, 3:9]


    return torch.tensor(id), torch.tensor(coords), torch.tensor(x), torch.tensor(y)


class MyDataset(Dataset):
    def __init__(self, x, y, coords):
        self.features = x
        self.target = y
        self.coords = coords

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.as_tensor(self.features[idx]), torch.as_tensor(self.target[idx]), torch.as_tensor(self.coords[idx])



def train(args):
    # Get args
    for xuhao in range(args.n_xuhao):
        n_epoches = args.n_epochs_res
        n_tasks = 30
        batch_size1 = 1
        x_size = 6
        y_size = 1
        start_lr = args.lrrnp
        num_hidden = args.emb_dim
        dset = args.dset
        model_name = args.model_name
        random_state = args.random_state
        path = args.path
        train_size = args.train_size
        batched_training = args.batched_training
        batch_size = args.batch_size
        n_epochs = args.n_epochs
        train_crit = args.train_crit
        lr = args.lrgnn
        emb_dim = args.emb_dim
        k = args.k
        print_progress = args.print_progress

        # Set random seed
        np.random.seed(random_state)
        # set_seed(random_state)
        test_score = np.inf
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Access and process data
        if dset == "cali_housing":
            id, c, x, y = get_cali_housing_data()


        n = x.shape[0]
        n_train = np.round(n * train_size).astype(int)

        indices = np.arange(n)

        train_id, valid_id, _, valid_x, _, valid_y, idx_train, idx_valid = train_test_split(id, x, y, indices, test_size=(1 - train_size), random_state=random_state)
        _, test_id, _, _, _, _, _, idx_test = train_test_split(valid_id, valid_x, valid_y, idx_valid, test_size=0.3, random_state=random_state)

        train_x, test_x = x[idx_train], x[idx_test]
        train_y, test_y = y[idx_train], y[idx_test]
        train_c, test_c = c[idx_train], c[idx_test]

        scaler = StandardScaler()
        train_c=torch.FloatTensor(scaler.fit_transform(train_c))
        test_c = torch.FloatTensor(scaler.transform(test_c))
        train_x = torch.FloatTensor(scaler.fit_transform(train_x))
        test_x = torch.FloatTensor(scaler.transform(test_x))

        train_dataset, test_dataset = MyDataset(train_x, train_y, train_c), MyDataset(test_x, test_y, test_c)


        train_edge_index = False
        train_edge_weight = False
        test_edge_index = False
        test_edge_weight = False
        train_y_moran = False
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        # Make model
        if model_name == "gcn":
            model = GCN(num_features_in=train_x.shape[1], k=k,).to(device)
        if model_name == "pegcn":
            model = PEGCN(num_features_in=train_x.shape[1], k=k,  emb_dim=emb_dim).to(device)
        model = model.float()

        loss_wrapper = LossWrapper(model,loss=train_crit, k=k, batch_size=batch_size).to(device)
        optimizer = torch.optim.Adam(loss_wrapper.parameters(), lr=lr)
        score1 = nn.MSELoss()
        score2 = nn.L1Loss()

        # Tensorboard and logging
        test_ = dset + '-' + model_name + '-k' + str(k)
        if model_name == 'pegcn':
            test_ = test_ + '-emb' + str(emb_dim)

        test_ = test_ + "_mat-lam"
        if batched_training == True:
            test_ = test_ + "_bs" + str(batch_size) + "_ep" + str(n_epochs)
        else:
            test_ = test_ + "_bsn_ep" + str(n_epochs)

        # saved_file = "{}_{}{}-{}:{}:{}.{}".format(test_,
        saved_file = "{}_{}{}-{}-{}{}{}".format(test_,
                                                datetime.now().strftime("%h"),
                                                datetime.now().strftime("%d"),
                                                datetime.now().strftime("%H"),
                                                datetime.now().strftime("%M"),
                                                datetime.now().strftime("%S"),
                                                datetime.now().strftime("%f")
                                                )

        log_dir = path + "/trained/{}/log".format(saved_file)

        if not os.path.exists(path + "trained/{}/data".format(saved_file)):
            os.makedirs(path + "/trained/{}/data".format(saved_file))

        with open(path + "/trained/{}/train_notes.txt".format(saved_file), 'w') as f:
            # Include any experiment notes here:
            f.write("Experiment notes: .... \n\n")
            f.write("MODEL_DATA: {}\n".format(
                test_))
            f.write("BATCH_SIZE: {}\nLEARNING_RATE: {}\n".format(
                batch_size,
                lr))

        writer = SummaryWriter(log_dir)

        # Training loop
        it_counts = 0
        for epoch in range(n_epochs):
            for batch in train_loader:
                model.train()
                it_counts += 1
                x = batch[0].to(device).float()
                y = batch[1].to(device).float()
                c = batch[2].to(device).float()

                optimizer.zero_grad()


                loss = loss_wrapper(x, y, c, train_edge_index, train_edge_weight, train_y_moran)
                loss.backward()
                optimizer.step()
                # Eval

                model.eval()
                with torch.no_grad():

                    pred = model(torch.tensor(test_dataset.features).to(device), torch.tensor(test_dataset.coords).to(device), test_edge_index, test_edge_weight)
                test_score1 = score1(torch.tensor(test_dataset.target).reshape(-1).to(device), pred.reshape(-1))
                test_score2 = score2(torch.tensor(test_dataset.target).reshape(-1).to(device), pred.reshape(-1))

                if print_progress and epoch % 10 == 0:
                    print("Epoch [%d/%d] - Loss: %f - Test score (MSE): %f - Test score (MAE): %f" % (epoch, n_epochs, loss.item(), test_score1.item(), test_score2.item()))
                save_path = path + "/trained/{}/ckpts".format(saved_file)

                if test_score > test_score1:
                    test_score = test_score1
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(model.state_dict(), save_path + '/' + 'model.pth.tar')

                writer.add_scalar('Test score (MSE)', test_score1.item(), it_counts)
                writer.add_scalar('Test score (MAE)', test_score2.item(), it_counts)
                writer.add_scalar('Training loss', loss.item(), it_counts)

                writer.flush()

        model.load_state_dict(torch.load(save_path + '/' + 'model.pth.tar'))
        model.eval()

        train_pred = model(torch.tensor(train_dataset.features).to(device), torch.tensor(train_dataset.coords).to(device), train_edge_index, train_edge_weight)
        test_pred = model(torch.tensor(test_dataset.features).to(device), torch.tensor(test_dataset.coords).to(device), test_edge_index, test_edge_weight)

        test_score11 = score1(torch.tensor(test_dataset.target).reshape(-1).to(device), test_pred.reshape(-1))
        test_score22 = score2(torch.tensor(test_dataset.target).reshape(-1).to(device), test_pred.reshape(-1))
        print("Saved all models to {}".format(save_path))
        print("Epoch [%d/%d] - Loss: %f - Test score (MSE): %f - Test score (MAE): %f" % (epoch, n_epochs, loss.item(), test_score11.item(), test_score22.item()))

        train_res = torch.tensor(train_dataset.target).to(device) - train_pred.reshape(-1)
        test_res = torch.tensor(test_dataset.target).to(device) - test_pred.reshape(-1)

        train_out = torch.cat([torch.cat([torch.cat([torch.tensor(train_dataset.coords).to(device), torch.tensor(train_dataset.features).to(device)], dim=-1),
                                          torch.tensor(train_dataset.target).to(device).reshape(-1, 1).reshape(-1, 1)], dim=-1), train_pred], dim=-1)
        test_out = torch.cat([torch.cat([torch.cat([torch.tensor(test_dataset.coords).to(device), torch.tensor(test_dataset.features).to(device)], dim=-1),
                                         torch.tensor(test_dataset.target).to(device).reshape(-1, 1)], dim=-1), test_pred], dim=-1)

        train_out = torch.cat([train_out, train_res.reshape(-1, 1)], dim=-1)
        test_out = torch.cat([test_out, test_res.reshape(-1, 1)], dim=-1)

        train_out = train_out.detach().cpu().numpy()
        test_out = test_out.detach().cpu().numpy()


        train_out = np.concatenate((idx_train.reshape(-1, 1), train_out), axis=-1)
        test_out = np.concatenate((idx_test.reshape(-1, 1), test_out), axis=-1)

        header1 = "id,c1,c2,x1,x2, x3,x4,x5, x6,y,train_pred,train_res"
        header2 = "id,c1,c2,x1,x2, x3,x4,x5, x6,y,train_pred,test_res"



        np.savetxt((r'./trained/train_out_{}.csv'.format(xuhao)), train_out, delimiter=",", header=header1)
        np.savetxt((r'./trained/test_out_{}.csv'.format(xuhao)), test_out, delimiter=",", header=header2)
        print('GNN complete')

        ###residual processes

        trainset = DatasetGP(n_tasks=n_tasks, xuhao=xuhao, batch_size=batch_size1)
        testset = DatasetGP_test(n_tasks=n_tasks, xuhao=xuhao, batch_size=batch_size1)
        model_res = ResidualNeuralProcess(x_size=x_size, y_size=y_size, num_hidden=num_hidden).to(device)
        criterion = Criterion()
        optimizer = optim.Adam(model_res.parameters(), lr=start_lr)


        model_res.train()
        R_square = -(np.inf)
        epoch_train_loss = []

        for epoch in range(n_epoches):
            trainloader = DataLoader(trainset, shuffle=True)
            testloader = DataLoader(testset, batch_size=1, shuffle=True)

            mean_y, var_y, target_id, target_y, context_id, loss, kl_loss, train_r2 = train_runner(
                model_res, trainloader, criterion, optimizer)

            val_pred_y, val_var_y, val_target_id, val_target_y, val_loss, valid_r2 = val_runner(
                model_res, testloader, criterion)


            if R_square <= valid_r2:
                R_square = valid_r2
                train_mse = (torch.sum((target_y - mean_y) ** 2)) / len(target_y)
                train_rmse = sqrt(train_mse)
                train_mae = (torch.sum(torch.absolute(target_y - mean_y))) / len(target_y)
                train_r2 = 1 - ((torch.sum((target_y - mean_y) ** 2)) / torch.sum((target_y - target_y.mean()) ** 2))
                valid_mse = (torch.sum((val_target_y - val_pred_y) ** 2)) / len(val_target_y)
                valid_rmse = sqrt(valid_mse)
                valid_mae = (torch.sum(torch.absolute(val_target_y - val_pred_y))) / len(val_target_y)
                valid_r2 = 1 - ((torch.sum((val_target_y - val_pred_y) ** 2)) / torch.sum(
                    (val_target_y - val_target_y.mean()) ** 2))

                print(
                    "ID: {} \t Train Epoch: {} \t Lr:{:.4f},train loss: {:.4f},train_mae: {:.4f},train_mse: {:.4f}, train_rmse: {:.4f},train_r2: {:.4f}".format(
                        xuhao, epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], loss, train_mae, train_mse, train_rmse,
                        train_r2))
                print(
                    "ID: {} \t Valid Epoch: {} \t Lr:{:.4f},valid loss: {:.4f},valid_mae: {:.4f},valid_mse: {:.4f}, valid_rmse: {"
                    ":.4f},valid_r2: {:.4f}".
                    format(xuhao, epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], val_loss, valid_mae, valid_mse,
                           valid_rmse, valid_r2))

                torch.save({'model_res': model_res.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           os.path.join('trained', 'checkpoint_%d.pth.tar' % (xuhao)))

            if (epoch + 1) % 50 == 0:
                print("ID: {} \t Train Epoch: {} \t Lr:{:.4f},train loss: {:.4f},train_r2: {:.3f}".format(
                    xuhao, epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], loss, train_r2))
                print(
                    "ID: {} \t Valid Epoch: {} \t Lr:{:.4f},valid loss: {:.4f},valid_r2: {:.4f}".
                    format(xuhao, epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], val_loss, valid_r2))
            epoch_train_loss.append(loss - kl_loss)

        prediction = pd.DataFrame(
            {"id": val_target_id.detach().cpu().numpy(), "pred": val_target_y.detach().cpu().numpy(), "val_pred": val_pred_y.detach().cpu().numpy(),
             "cha": val_target_y.detach().cpu().numpy() - val_pred_y.detach().cpu().numpy(), 'val_var_y': val_var_y.detach().cpu().numpy(), })
        prediction.to_csv('./trained/prediction_{}.csv'.format(xuhao), index=False)

    end = time.perf_counter()
    print(str(end - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PE-GCN')
    # Data & model selection
    parser.add_argument('-d', '--dset', type=str, default='cali_housing',
                        choices=['cali_housing', 'election', 'air_temp', '3d_road'])
    parser.add_argument('-m', '--model_name', type=str, default='pegcn', choices=['gcn', 'pegcn'])
    # Utilities
    parser.add_argument('-s', '--random_state', type=int, default=42)
    parser.add_argument('-p', '--path', type=str, default='./')
    # Training setting
    parser.add_argument('-ts', '--train_size', type=float, default=0.2)
    parser.add_argument('-bt', '--batched_training', type=bool, default=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('-ne', '--n_epochs', type=int, default=1)
    parser.add_argument('-nep', '--n_epochs_res', type=int, default=1)
    parser.add_argument('-xuhao', '--n_xuhao', type=int, default=2)
    parser.add_argument('-loss', '--train_crit', type=str, default='mse', choices=['mse', 'l1'])
    parser.add_argument('-lrgnn', '--lrgnn', type=float, default=1e-3)
    parser.add_argument('-lrrnp', '--lrrnp', type=float, default=1e-3)
    # Model config
    parser.add_argument('-embd', '--emb_dim', type=float, default=128)
    parser.add_argument('-k', '--k', type=int, default=5)
    # Logging & evaluation
    parser.add_argument('-save', '--save_freq', type=int, default=5)
    parser.add_argument('-print', '--print_progress', type=bool, default=True)
    parser.add_argument('-f')  # Dummy to get parser to run in Colab

    args = parser.parse_args()

    out = train(args)
