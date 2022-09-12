import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoGVM(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device, normalize_adj=False):
        super(TwoGVM, self).__init__()
        self.feature_size = feature_size
        self.nodes_number = nodes_number
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.minimum = 1e-10
        self.normalize_adj = normalize_adj  # TODO: Change to True by default

        self.pre_weighting = nn.Linear(1, int(self.RECEIVED_PARAMS["preweight"]))
        self.pre_weighting2 = nn.Linear(int(self.RECEIVED_PARAMS["preweight"]), int(self.RECEIVED_PARAMS["preweight"]))

        self.fc1 = nn.Linear(1, int(self.RECEIVED_PARAMS["layer_1"]))  # input layer
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["preweight"]) * self.nodes_number +
                             int(self.RECEIVED_PARAMS["layer_2"]) * self.nodes_number, 1)
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])

        # self.alpha = nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))

        noise = np.random.normal(0, 0.1)
        self.alpha = nn.Parameter(torch.tensor([1+noise], requires_grad=True, device=self.device).float())
        # self.alpha = torch.tensor([1], device=self.device)

        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}
        # self.activation_func = "srss"
        if self.feature_size > 1:
            self.transform_mat_to_vec = nn.Linear(self.feature_size, 1)

        self.gcn_layer = nn.Sequential(
            self.pre_weighting,
            self.activation_func_dict[self.activation_func]
        )
        self.gcn_layer2 = nn.Sequential(
            self.pre_weighting2,
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
        if self.feature_size > 1:
            # For abide dataset where the feature matrix is matrix. We want to transform the matrix into a vector.
            x = self.transform_mat_to_vec(x)
        I = torch.eye(b).to(self.device)

        if self.alpha.item() < self.minimum:
            print("In min_value")
            # print("alpha value", self.alpha.item(), "min_value", self.min_value.item())
            # self.alpha = deepcopy(self.min_value)
            self.alpha.data = torch.clamp(self.alpha, min=self.minimum)
            print("new alpha", self.alpha.item())
            # alpha_I = I * self.min_value.expand_as(I)  # min_value * I
        alpha_I = I * self.alpha.expand_as(I)  # 𝛼I
        if self.normalize_adj:
            normalized_adjacency_matrix = self.calculate_adjacency_matrix(adjacency_matrix)  # Ã
        else:
            normalized_adjacency_matrix = adjacency_matrix

        # normalized_adjacency_matrix = self.calculate_adjacency_matrix(adjacency_matrix)  # Ã
        alpha_I_plus_normalized_A = alpha_I + normalized_adjacency_matrix  # 𝛼I + Ã

        ones_vector = torch.ones(x.shape).to(self.device)
        gcn_output = torch.matmul(alpha_I_plus_normalized_A, ones_vector)  # (𝛼I + Ã)·1

        gcn_output = self.gcn_layer(gcn_output)
        # gcn_output = self.gcn_layer2(gcn_output)

        gcn_output = torch.flatten(gcn_output, start_dim=1)  # flatten the tensor

        fc_output = self.classifier(x)
        fc_output = torch.flatten(fc_output, start_dim=1)  # flatten the tensor
        concat_graph_and_values = torch.cat((gcn_output, fc_output), 1)
        final_output = self.fc3(concat_graph_and_values)
        return final_output

    # def forward(self, x, adjacency_matrix):
    #     # multiply the matrix adjacency_matrix by (learnt scalar) self.alpha
    #     # alpha_A = torch.mul(adjacency_matrix, self.alpha)  # 𝛼A  - this function does not forward gradients
    #     alpha_A = adjacency_matrix * self.alpha.expand_as(adjacency_matrix)  # 𝛼A
    #     a, b, c = alpha_A.shape
    #     d, e, f = x.shape
    #     I = torch.eye(b).to(self.device)
    #     alpha_A_plus_I = alpha_A + I  # 𝛼A + I
    #     normalized_adjacency_matrix = self.calculate_adjacency_matrix(alpha_A_plus_I)
    #     x = torch.sign(x)
    #     x = torch.matmul(normalized_adjacency_matrix, x)  # (𝛼A + I)·x
    #     if self.activation_func == 'relu':
    #         x = F.relu(self.pre_weighting(x))  # (𝛼A + I)·x·W
    #         # a/d variables are the batch size
    #         # x = x.view(d, 1, -1)
    #         x = torch.flatten(x, start_dim=1)  # flatten the tensor
    #         x = F.relu(self.fc1(x))
    #         x = self.dropout(x)
    #         x = F.relu(self.fc2(x))
    #     elif self.activation_func == 'elu':
    #         x = F.elu(self.pre_weighting(x))   # (𝛼A + I)·x·W
    #         # x = x.view(d, 1, -1)
    #         x = torch.flatten(x, start_dim=1)  # flatten the tensor
    #         x = F.elu(self.fc1(x))
    #         x = self.dropout(x)
    #         x = F.elu(self.fc2(x))
    #     elif self.activation_func == 'tanh':
    #         x = torch.tanh(self.pre_weighting(x))   # (𝛼A + I)·x·W
    #         # x = x.view(d, 1, -1)
    #         x = torch.flatten(x, start_dim=1)  # flatten the tensor
    #         x = torch.tanh(self.fc1(x))
    #         x = self.dropout(x)
    #         x = torch.tanh(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

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
