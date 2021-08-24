import os

import pandas as pd
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import DataLoader
# from torch_geometric.nn import GCNConv, global_max_pool, GATConv
# from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from taxonomy_tree_for_pytorch_geometric import create_tax_tree


class GCN(torch.nn.Module):
    def __init__(self, data_size, RECEIVED_PARAMS, device):
        super(GCN, self).__init__()
        size = 100
        # self.att = GATConv(data_size, size)
        self.conv1 = GCNConv(data_size, size, improved=False)
        self.conv2 = GCNConv(size, size)
        self.conv3 = GCNConv(32, 32)
        num_classes = 1
        self.lin = Linear(size, num_classes)

    def forward(self, x, edge_index, batch):
        # print("Hi")
        # 1. Obtain node embeddings
        # x = self.conv1(x, edge_index)
        x = self.conv1(x, edge_index)
        x = torch.tanh(x)
        sh = x.shape
        # x = x.relu()
        x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5)
        x = self.lin(x)
        return x
