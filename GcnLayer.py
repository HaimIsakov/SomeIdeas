import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device):
        super(GCNLayer, self).__init__()
        self.feature_size = feature_size
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.pre_weighting = nn.Linear(self.feature_size, int(self.RECEIVED_PARAMS["preweight"]))
        self.alpha = nn.Parameter(torch.rand(1, requires_grad=True)).to(self.device)

    def forward(self, x, adjacency_matrix):
        alpha_A = torch.mul(adjacency_matrix, self.alpha)  # 𝛼A
        a, b, c = alpha_A.shape
        # d, e, f = x.shape
        I = torch.eye(b).to(self.device)
        alpha_A_plus_I = alpha_A + I  # 𝛼A + I
        normalized_adjacency_matrix = self.calculate_adjacency_matrix(alpha_A_plus_I)
        x = torch.matmul(normalized_adjacency_matrix, x)  # (𝛼A + I)·x

        if self.activation_func == 'relu':
            x = F.relu(self.pre_weighting(x))  # (𝛼A + I)·x·W
        elif self.activation_func == 'elu':
            x = F.elu(self.pre_weighting(x))   # (𝛼A + I)·x·W
        elif self.activation_func == 'tanh':
            x = torch.tanh(self.pre_weighting(x))   # (𝛼A + I)·x·W
        return x

    def calculate_adjacency_matrix(self, batched_adjacency_matrix):
        # D^(-0.5)
        def calc_d_minus_root_sqr(batched_adjacency_matrix):
            return torch.stack([torch.diag(torch.pow(adjacency_matrix.sum(1), -0.5)) for adjacency_matrix in batched_adjacency_matrix])
        D__minus_sqrt = calc_d_minus_root_sqr(batched_adjacency_matrix)
        normalized_adjacency = torch.matmul(torch.matmul(D__minus_sqrt, batched_adjacency_matrix), D__minus_sqrt)
        return normalized_adjacency
