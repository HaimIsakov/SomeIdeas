import torch
import torch.nn as nn
import torch.nn.functional as F

from GcnModule import AlphaGcn


class GmicVUsingAlphaGcn(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device, num_classes=1):
        super(GmicVUsingAlphaGcn, self).__init__()
        self.feature_size = feature_size
        self.alpha_gcn = AlphaGcn(feature_size, RECEIVED_PARAMS, device)

        self.nodes_number = nodes_number
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.min_value = torch.tensor([1e-10], device=device).float()
        self.pre_weighting = nn.Linear(1, int(self.RECEIVED_PARAMS["preweight"]))
        self.fc1 = nn.Linear(int(self.RECEIVED_PARAMS["preweight"]) * self.nodes_number, int(self.RECEIVED_PARAMS["layer_1"]))  # input layer
        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["layer_2"]), num_classes)
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])

        self.alpha = self.alpha_gcn.alpha
        self.gcn_weights = self.alpha_gcn.pre_weighting

        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}
        if self.feature_size > 1:
            self.transform_mat_to_vec = nn.Linear(self.feature_size, 1)

        # self.gcn_layer = nn.Sequential(
        #     self.pre_weighting,
        #     self.activation_func_dict[self.activation_func]
        # )
        self.classifier = nn.Sequential(
            self.fc1,
            self.activation_func_dict[self.activation_func],
            self.dropout,
            self.fc2,
            self.activation_func_dict[self.activation_func],
        )

    def forward(self, x, adjacency_matrix):
        if self.feature_size > 1:
            # For abide dataset where the feature matrix is matrix. We want to transform the matrix into a vector.
            x = self.transform_mat_to_vec(x)

        x = self.alpha_gcn(x, adjacency_matrix)
        x = self.activation_func_dict[self.activation_func](x)
        x = torch.flatten(x, start_dim=1)  # flatten the tensor
        x = self.classifier(x)
        x = self.fc3(x)
        return x
