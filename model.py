
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import warnings

warnings.filterwarnings("ignore")
from attention import *
import torch as torch
import torch.nn.parallel
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph
import random
import os
import time
import numpy as np

import math
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
start = time.perf_counter()
time.sleep(2)
def knn_to_adj(knn, n):
    adj_matrix = torch.zeros(n, n, dtype=float)
    for i in range(len(knn[0])):
        tow = knn[0][i]
        fro = knn[1][i]
        adj_matrix[tow, fro] = 1
    return adj_matrix.T


def Distance(a, b):
    if a.shape[0] == 2:
        x1, y1 = a[0], a[1]
        x2, y2 = b[0], b[1]
        d = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    return d

# Helper function for edge weights
def makeEdgeWeight(x, edge_index):
    to = edge_index[0]
    fro = edge_index[1]
    edge_weight = []
    for i in range(len(to)):
        edge_weight.append(Distance(x[to[i]], x[fro[i]]))
    max_val = max(edge_weight)
    rng = max_val - min(edge_weight)
    edge_weight = [(max_val - elem) / rng for elem in edge_weight]
    return torch.Tensor(edge_weight)

class GCN(nn.Module):
    """
        GCN
    """
    def __init__(self, num_features_in=3, num_features_out=1, k=20):
        super(GCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        # self.MAT = MAT
        self.conv1 = GCNConv(num_features_in, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, num_features_out)

    def forward(self, x, c, ei, ew):
        x = x.float()
        c = c.float()
        if torch.is_tensor(ei) & torch.is_tensor(ew):
            edge_index = ei
            edge_weight = ew
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            edge_weight = makeEdgeWeight(c, edge_index).to(self.device)
        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, training=self.training)
        output = self.fc(h2)
        if self.MAT:
            morans_output = self.fc_morans(h2)
            return output, morans_output
        else:
            return output

class PEGCN(nn.Module):
    """
        GCN with positional encoder and auxiliary tasks
    """

    def __init__(self, num_features_in=3, num_features_out=1, emb_hidden_dim=128, emb_dim=16, k=20):
        super(PEGCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb_hidden_dim = emb_hidden_dim
        self.emb_dim = emb_dim
        self.k = k
        self.spenc = locationencoder(input_dim=2, num_hidden=self.emb_hidden_dim)

        self.conv1 = GCNConv(num_features_in + emb_dim, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, num_features_out)


    def forward(self, x, c, ei, ew):
        x = x.float()
        c = c.float()
        if torch.is_tensor(ei) & torch.is_tensor(ew):
            edge_index = ei
            edge_weight = ew
        else:
            edge_index = knn_graph(c, k=self.k).to(self.device)
            edge_weight = makeEdgeWeight(c, edge_index).to(self.device)

        c = c.reshape(1, c.shape[0], c.shape[1])

        emb = self.spenc(c)  # attention
        emb = emb.reshape(emb.shape[1], emb.shape[2]).float()
        x = torch.cat((x, emb), dim=1)

        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, training=self.training)
        output = self.fc(h2)

        return output


class LossWrapper(nn.Module):
    def __init__(self, model,  loss='mse', k=5, batch_size=32):
        super(LossWrapper, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model

        self.k = k
        self.batch_size = batch_size

        if loss == "mse":
            self.criterion = nn.MSELoss()
        elif loss == "l1":
            self.criterion = nn.L1Loss()

    def forward(self, input, targets, coords, edge_index, edge_weight, morans_input):

        outputs = self.model(input, coords, edge_index, edge_weight)
        loss = self.criterion(targets.float().reshape(-1), outputs.float().reshape(-1))
        return loss

