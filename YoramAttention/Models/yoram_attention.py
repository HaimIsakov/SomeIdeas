import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YoramAttention(nn.Module):
    def __init__(self, nodes_number, feature_size, RECEIVED_PARAMS, device):
        super(YoramAttention, self).__init__()
        self.feature_size = feature_size
        self.nodes_number = nodes_number
        self.device = device
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.embedding_size = self.RECEIVED_PARAMS["embedding_size"]
        if self.feature_size > 1:
            self.transform_mat_to_vec = nn.Linear(self.feature_size, 1)

        self.keys = nn.Linear(self.embedding_size, int(self.RECEIVED_PARAMS["layer_1"]))
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])
        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}

        self.fc2 = nn.Linear(int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"]))
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["layer_2"]), 1)
        self.classifier = nn.Sequential(
            self.fc2,
            self.activation_func_dict[self.activation_func],
            self.dropout,
        )
        self.attention = None

    def forward(self, frequencies, node_embedding_vector):
        # print(self.keys)
        # print(node_embedding_vector.shape)
        attention_vector = F.softmax(self.keys(node_embedding_vector), dim=1)
        self.attention = attention_vector
        # self.get_attention_hist(attention_vector)
        if self.feature_size > 1:
            # For abide dataset where the feature matrix is matrix. We want to transform the matrix into a vector.
            frequencies = self.transform_mat_to_vec(frequencies)
        output = torch.transpose(frequencies, 1, 2) @ attention_vector
        x = self.classifier(output)
        x = self.fc3(x)
        x = x.squeeze(1)  # TODO: Check this squeeze
        return x

    def get_attention_hist(self, tensor, name, calc):
        if calc:
            for i, mat in enumerate(tensor):
                for dim in range(0, int(self.RECEIVED_PARAMS["layer_1"])):
                    indices = torch.tensor([dim])
                    vector_in_dim = torch.index_select(mat, 1, indices).cpu().detach().numpy()
                    plt.hist(vector_in_dim, bins=50)
                    plt.savefig(os.path.join("HIST", name + f"mat_{i}" + f"dim_{dim}" + ".png"))
                    plt.clf()
                    # plt.show()
