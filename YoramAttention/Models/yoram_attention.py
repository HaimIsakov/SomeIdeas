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

        self.keys = nn.Linear(self.feature_size, int(self.RECEIVED_PARAMS["layer1"]))
        self.activation_func = self.RECEIVED_PARAMS['activation']
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"])
        self.activation_func_dict = {'relu': nn.ReLU(), 'elu': nn.ELU(), 'tanh': nn.Tanh()}

    def forward(self, frequencies, node_embedding_vector):
        attention_vector = F.softmax(self.keys(node_embedding_vector), dim=1)
        output = frequencies @ attention_vector
        return output
