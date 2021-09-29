import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Model(nn.Module):
    '''
    input_dim: dimension of each vertex in the graph
    hl: dim of hidden layer
    p: dropout probability
    '''

    def __init__(self, RECEIVED_PARAMS,input_dim, hl, p, embedding_state_dict):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])
        self.activation_func = self.RECEIVED_PARAMS['activation']

        self.fc1 = nn.Linear(self.input_dim, int(self.RECEIVED_PARAMS["layer_1"]))
        self.embedding_state_dict = embedding_state_dict
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.keys = nn.Linear(int(self.RECEIVED_PARAMS["layer_2"]), 1)
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
        x = torch.matmul(alpha_I_plus_A, x)  # (𝛼I + Ã)·x

        # values : N x input_dim
        values = padded_input.view(-1, self.input_dim)
        # x : N x hl
        x = F.relu(self.fc1(self.dropout(values)))
        # queries : N x hl
        queries = F.relu(self.fc2(self.dropout(x)))
        # mat1 : N x 1
        mat1 = self.keys(queries)
        # mat2 : 1 x N
        mat2 = torch.transpose(mat1, 0, 1)
        mat2 = torch.div(mat2, math.sqrt(self.hl))
        # scores : 1 x N
        scores = F.softmax(mat2, dim=1)
        # vect: 1 x input_dim
        vect = torch.matmul(scores, values)
        vect = vect.view(1, self.input_dim)
        # y : 1 x 1
        y = self.output(vect)
        return y
