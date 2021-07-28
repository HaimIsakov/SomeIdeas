import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class JustGraphStructure(nn.Module):
    def __init__(self, data_size, RECEIVED_PARAMS, device):
        super(JustGraphStructure, self).__init__()
        self.data_size = data_size
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        # משקול לפני שמכניסים לרשת
        self.pre_weighting = nn.Linear(self.data_size, self.RECEIVED_PARAMS["preweight"])
        self.fc1 = nn.Linear(self.RECEIVED_PARAMS["preweight"], self.RECEIVED_PARAMS["layer_1"])  # input layer
        self.fc2 = nn.Linear(self.RECEIVED_PARAMS["layer_1"], self.RECEIVED_PARAMS["layer_2"])
        self.fc3 = nn.Linear(self.RECEIVED_PARAMS["layer_2"], 1)
        self.device = device
        self.alpha = torch.rand(1, requires_grad=True).to(self.device)
        self.activation_func = self.RECEIVED_PARAMS['activation']

    def forward(self, x, adjacency_matrix):
        # x = x.view(-1, self.data_size)
        # alpha_A = adjacency_matrix * self.alpha.expand_as(adjacency_matrix)
        alpha_A = torch.mul(adjacency_matrix, self.alpha)
        # alpha_A = torch.matmul(adjacency_matrix, self.alpha.expand_as(adjacency_matrix))
        a, b, c = alpha_A.shape
        d, e = x.shape
        I = torch.eye(b).to(self.device)
        alpha_A_plus_I = alpha_A + I
        x = torch.reshape(x, (d, e, 1))
        x = torch.matmul(alpha_A_plus_I, x)
        x = torch.reshape(x, (d, e))
        x = torch.sign(x)

        if self.activation_func == 'relu':
            x = F.relu(self.pre_weighting(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        elif self.activation_func == 'elu':
            x = F.elu(self.pre_weighting(x))
            x = F.elu(self.fc1(x))
            x = F.elu(self.fc2(x))
        elif self.activation_func == 'tanh':
            x = torch.tanh(self.pre_weighting(x))
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
        # x = torch.sigmoid(x) # BCE loss automatically applies sigmoid
        x = self.fc3(x)
        return x

    def calculate_adjacency_matrix(self, adjacency_matrix):
        size = adjacency_matrix.shape[0]
        # D^(-0.5)
        def calc_d_minus_root_sqr(adjacency_matrix):
            return np.diag([1 / np.sqrt(np.sum(adjacency_matrix[i, :])) for i in range(size)])

        modified_adj = adjacency_matrix + np.identity(size)
        D__minus_sqrt = calc_d_minus_root_sqr(modified_adj)
        normalized_adjacency = D__minus_sqrt * np.matrix(modified_adj) * D__minus_sqrt
        return normalized_adjacency
