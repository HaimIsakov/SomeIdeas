import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class AttentionGCN(nn.Module):
    '''
    input_dim: dimension of each vertex in the graph
    hl: dim of hidden layer
    p: dropout probability
    '''

    def __init__(self, nodes_number, data_size, RECEIVED_PARAMS, device):
        super(AttentionGCN, self).__init__()
        self.input_dim = data_size
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])
        self.device = device
        self.nodes_number = nodes_number
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.pre_weighting = nn.Linear(self.input_dim, int(self.RECEIVED_PARAMS["preweight"]))
        self.alpha = nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))
        self.fc1 = nn.Linear(int(self.RECEIVED_PARAMS["preweight"]), int(self.RECEIVED_PARAMS["layer_1"]))
        # self.embedding_state_dict = embedding_state_dict
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.keys = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), 1)
        self.output = nn.Linear(self.input_dim, 1)

        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}
        # GCN layer
        self.nodes_embedding_model = nn.Sequential(
            self.pre_weighting,
            self.activation_func_dict[self.activation_func]
        )

    '''
    padded input dim : N x input_dim
    '''
    def forward(self, x, adjacency_matrix):
        a, b, c = adjacency_matrix.shape
        d, e, f = x.shape
        I = torch.eye(b).to(self.device)
        alpha_I = I * self.alpha.expand_as(I)  # 𝛼I
        normalized_adjacency_matrix = self.calculate_adjacency_matrix(adjacency_matrix)  # Ã
        alpha_I_plus_A = alpha_I + normalized_adjacency_matrix  # 𝛼I + Ã
        # x = torch.squeeze(x, 0)
        x = torch.matmul(alpha_I_plus_A, x)  # (𝛼I + Ã)·x
        print(x.shape)
        # N x [feature size]
        x = self.nodes_embedding_model(x)
        print(x.shape)
        # x = torch.flatten(x, start_dim=1)  # flatten the tensor
        # values : N x input_dim
        # values = x.view(-1, self.input_dim)
        values = x
        # x : N x hl
        # x = F.relu(self.fc1(self.dropout(values)))

        # queries : N x hl
        queries = F.relu(self.fc1(self.dropout(x)))
        print(x.shape)
        # mat1 : N x 1
        mat1 = self.keys(queries)
        # mat2 : 1 x N
        mat2 = torch.transpose(mat1, 0, 1)
        mat2 = torch.div(mat2, math.sqrt(2))
        # scores : 1 x N
        scores = F.softmax(mat2, dim=1)
        # vect: 1 x input_dim
        vect = torch.matmul(scores, values)
        # vect = vect.view(1, self.input_dim)
        # y : 1 x 1
        y = self.output(vect)
        return y

    def calculate_adjacency_matrix(self, batched_adjacency_matrix):
        # D^(-0.5)
        def calc_d_minus_root_sqr(batched_adjacency_matrix):
            r = []
            for adjacency_matrix in batched_adjacency_matrix:
                sum_of_each_row = adjacency_matrix.sum(1)
                sum_of_each_row_plus_one = torch.where(sum_of_each_row != 0, sum_of_each_row, torch.tensor(1.0, device=self.device))
                r.append(torch.diag(torch.pow(sum_of_each_row_plus_one, -0.5)))
            s = torch.stack(r)
            if torch.isnan(s).any():
                print("Alpha when stuck", self.alpha.item())
                print("batched_adjacency_matrix", torch.isnan(batched_adjacency_matrix).any())
                print("The model is stuck", torch.isnan(s).any())
            return s
        D__minus_sqrt = calc_d_minus_root_sqr(batched_adjacency_matrix)
        normalized_adjacency = torch.matmul(torch.matmul(D__minus_sqrt, batched_adjacency_matrix), D__minus_sqrt)
        return normalized_adjacency
