import torch
import torch.nn as nn
import torch.nn.functional as F


class ValuesAndGraphStructure(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device):
        super(ValuesAndGraphStructure, self).__init__()
        self.feature_size = feature_size
        self.nodes_number = nodes_number
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.pre_weighting = nn.Linear(self.feature_size, int(self.RECEIVED_PARAMS["preweight"]))
        self.fc1 = nn.Linear(int(self.RECEIVED_PARAMS["preweight"]) * self.nodes_number, int(self.RECEIVED_PARAMS["layer_1"]))  # input layer
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["layer_2"]), 1)
        self.device = device
        self.alpha = nn.Parameter(torch.rand(1, requires_grad=True)).to(self.device)
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])

    def forward(self, x, adjacency_matrix):
        # multiply the matrix adjacency_matrix by (learnt scalar) self.alpha
        alpha_A = torch.mul(adjacency_matrix, self.alpha)  # 𝛼A
        a, b, c = alpha_A.shape
        d, e, f = x.shape
        I = torch.eye(b).to(self.device)
        alpha_A_plus_I = alpha_A + I  # 𝛼A + I
        normalized_adjacency_matrix = self.calculate_adjacency_matrix(alpha_A_plus_I)
        x = torch.matmul(normalized_adjacency_matrix, x)  # (𝛼A + I)·x
        if self.activation_func == 'relu':
            x = F.relu(self.pre_weighting(x))  # (𝛼A + I)·x·W
            # a/d variables are the batch size
            # x = x.view(d, 1, -1)
            x = torch.flatten(x, start_dim=1)  # flatten the tensor
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
        elif self.activation_func == 'elu':
            x = F.elu(self.pre_weighting(x))   # (𝛼A + I)·x·W
            # x = x.view(d, 1, -1)
            x = torch.flatten(x, start_dim=1)  # flatten the tensor
            x = F.elu(self.fc1(x))
            x = self.dropout(x)
            x = F.elu(self.fc2(x))
        elif self.activation_func == 'tanh':
            x = torch.tanh(self.pre_weighting(x))   # (𝛼A + I)·x·W
            # x = x.view(d, 1, -1)
            x = torch.flatten(x, start_dim=1)  # flatten the tensor
            x = torch.tanh(self.fc1(x))
            x = self.dropout(x)
            x = torch.tanh(self.fc2(x))
        elif self.activation_func == 'srss':
            x = self.srss(self.pre_weighting(x))   # (𝛼A + I)·x·W
            # x = x.view(d, 1, -1)
            x = torch.flatten(x, start_dim=1)  # flatten the tensor
            x = self.srss(self.fc1(x))
            x = self.dropout(x)
            x = self.srss(self.fc2(x))
        x = self.fc3(x)
        return x

    def srss(self, x):
        return 1 - 2 / (x**2 + 1)

    def calculate_adjacency_matrix(self, batched_adjacency_matrix):
        # D^(-0.5)
        def calc_d_minus_root_sqr(batched_adjacency_matrix):
            return torch.stack([torch.diag(torch.pow(adjacency_matrix.sum(1), -0.5)) for adjacency_matrix in batched_adjacency_matrix])
        D__minus_sqrt = calc_d_minus_root_sqr(batched_adjacency_matrix)
        normalized_adjacency = torch.matmul(torch.matmul(D__minus_sqrt, batched_adjacency_matrix), D__minus_sqrt)
        return normalized_adjacency
