import numpy as np
import torch
import torch.nn as nn


class ConcatValuesAndGraphStructure(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device, num_classes=1):
        super(ConcatValuesAndGraphStructure, self).__init__()
        self.feature_size = feature_size
        self.nodes_number = nodes_number
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.min_value = torch.tensor([1e-10], device=device).float()
        self.pre_weighting = nn.Linear(1, int(self.RECEIVED_PARAMS["preweight"]))
        self.fc1 = nn.Linear(1, int(self.RECEIVED_PARAMS["layer_1"]))  # input layer
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["preweight"]) * self.nodes_number +
                             int(self.RECEIVED_PARAMS["layer_2"]) * self.nodes_number, num_classes)
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])

        # self.alpha = nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))
        noise = np.random.normal(0, 0.1)
        self.alpha = nn.Parameter(torch.tensor([1+noise], requires_grad=True, device=self.device).float())

        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}

        self.gcn_layer = nn.Sequential(
            self.pre_weighting,
            self.activation_func_dict[self.activation_func]
        )
        self.classifier = nn.Sequential(
            self.fc1,
            self.activation_func_dict[self.activation_func],
            self.dropout,
            self.fc2,
            self.activation_func_dict[self.activation_func],
        )

    def forward(self, x, adjacency_matrix):
        # multiply the matrix adjacency_matrix by (learnt scalar) self.alpha
        a, b, c = adjacency_matrix.shape
        d, e, f = x.shape
        I = torch.eye(b).to(self.device)
        if self.alpha.item() < self.min_value.item():
            print("In min_value")
            print("alpha value", self.alpha.item(), "min_value", self.min_value.item())
            alpha_I = I * self.min_value.expand_as(I)  # min_value * I
        else:
            alpha_I = I * self.alpha.expand_as(I)  # 𝛼I

        normalized_adjacency_matrix = self.calculate_adjacency_matrix(adjacency_matrix)  # Ã
        alpha_I_plus_normalized_A = alpha_I + normalized_adjacency_matrix  # 𝛼I + Ã

        ones_vector = torch.ones(x.shape).to(self.device)
        gcn_output = torch.matmul(alpha_I_plus_normalized_A, ones_vector)  # (𝛼I + Ã)·1

        gcn_output = self.gcn_layer(gcn_output)
        gcn_output = torch.flatten(gcn_output, start_dim=1)  # flatten the tensor

        fc_output = self.classifier(x)
        fc_output = torch.flatten(fc_output, start_dim=1)  # flatten the tensor
        concat_graph_and_values = torch.cat((gcn_output, fc_output), 1)
        final_output = self.fc3(concat_graph_and_values)
        return final_output

    def calculate_adjacency_matrix(self, batched_adjacency_matrix):
        # Here we normalize A

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
