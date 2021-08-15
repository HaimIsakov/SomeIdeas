import os

import pandas as pd
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from taxonomy_tree_for_pytorch_geometric import create_tax_tree


class GCN(torch.nn.Module):
    def __init__(self, data_size, RECEIVED_PARAMS, device):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        num_classes = 2
        self.lin = Linear(32, num_classes)

    def forward(self, x, edge_index):
        print("Hi")
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5)
        x = self.lin(x)
        return x

# data_file_path = os.path.join('IBD_split_dataset', 'OTU_IBD_after_mipmlp_Genus.csv')
# microbiome_df = pd.read_csv(data_file_path, index_col='ID')
# nodes_number = []
# graphs = []
# for i, mom in tqdm(enumerate(microbiome_df.iterrows()), desc='Create graphs', total=len(microbiome_df)):
#     # cur_graph = create_tax_tree(microbiome_df.iloc[i], ignore_values=0, ignore_flag=True)
#     cur_graph = create_tax_tree(microbiome_df.iloc[i])
#     graphs.append(cur_graph)
#     nodes_number.append(cur_graph.number_of_nodes())
#     # print("Nodes Number", cur_graph.number_of_nodes())
#
# data_list = []
# for g in graphs:
#     data = from_networkx(g, group_node_attrs=['val'])  # Notice: convert file was changed explicitly
#     data_list.append(data)
# loader = DataLoader(data_list, batch_size=32, exclude_keys=['val'], shuffle=True)
#
# for step, data in enumerate(loader):
#     print(f'Step {step + 1}:')
# print()
#
# model = GCN(64, 318)
# print(model)