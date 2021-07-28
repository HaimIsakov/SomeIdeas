import torch
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.nn import GCNConv

# Had never been tried
class JustGraphStructureGeometric(torch.nn.Module):
    def __init__(self, num_node_features, RECEIVED_PARAMS, device):
        super(JustGraphStructureGeometric, self).__init__()
        self.num_node_features = num_node_features
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        layer1_size = int(self.RECEIVED_PARAMS["layer_1"])
        layer2_size = int(self.RECEIVED_PARAMS["layer_2"])
        self.conv1 = GCNConv(num_node_features, layer1_size)
        self.conv2 = GCNConv(layer1_size, layer2_size)
        self.layer3 = nn.Linear(layer2_size, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.activation_func == 'relu':
            x = F.relu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.relu(self.conv2(x, edge_index))
        elif self.activation_func == 'elu':
            x = F.elu(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = F.elu(self.conv2(x, edge_index))
        elif self.activation_func == 'tanh':
            x = torch.tanh(self.conv1(x, edge_index))
            x = self.dropout(x)
            x = torch.tanh(self.conv2(x, edge_index))

        x = self.layer3(x)
        return x
