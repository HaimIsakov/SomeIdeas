import networkx as nx
import torch
from torch.utils.data import Dataset
from torch import Tensor, FloatTensor
#from torch_geometric.utils import from_networkx

from MicrobiomeDataset import MicrobiomeDataset
from create_microbiome_graphs import CreateMicrobiomeGraphs


class GraphDataset(Dataset):
    def __init__(self, data_file_path, tag_file_path, mission, add_attributes, geometric_or_not=False):
        super(GraphDataset, self).__init__()
        # for dataset handling
        self.microbiome_dataset = MicrobiomeDataset(data_file_path, tag_file_path)
        microbiome_df = self.microbiome_dataset.microbiome_df
        # for graphs' creation
        self.create_microbiome_graphs = CreateMicrobiomeGraphs(microbiome_df, add_attributes)
        self.samples_len = microbiome_df.shape[0]
        self.dataset_dict = {}
        self.mission = mission
        self.geometric_or_not = geometric_or_not
        self.add_attributes = add_attributes

    def set_dataset_dict(self):
        dataset_dict = {}
        for i in range(self.samples_len):
            graph = self.create_microbiome_graphs.get_graph(i)
            if not self.geometric_or_not:
                dataset_dict[i] = {'graph': graph,
                                   'label': self.microbiome_dataset.get_label(i),
                                   'values_on_leaves': self.microbiome_dataset.get_leaves_values(i),
                                   'values_on_nodes': self.create_microbiome_graphs.get_values_on_nodes_ordered_by_nodes(graph),
                                   'adjacency_matrix': nx.adjacency_matrix(graph).todense()
                                   }
            else:
                data = from_networkx(graph, group_node_attrs=['val'])  # Notice: convert file has been changed explicitly
                data.y = torch.tensor(self.microbiome_dataset.get_label(i))
                dataset_dict[i] = {'data': data}
        return dataset_dict

    def get_joint_nodes(self):
        return self.create_microbiome_graphs.find_common_nodes().keys()

    def update_graphs(self):
        # self.create_microbiome_graphs.create_graphs_with_common_nodes(union_train_and_test)
        self.dataset_dict = self.set_dataset_dict()  # create dataset dict only after we added the missing nodes to the graph

    def get_all_groups(self):
        return self.microbiome_dataset.groups

    def get_leaves_number(self):
        return self.microbiome_dataset.get_leaves_number()

    def get_vector_size(self):
        return self.create_microbiome_graphs.get_vector_size()

    def nodes_number(self):
        return self.create_microbiome_graphs.nodes_number()

    def __len__(self):
        return self.samples_len

    def __getitem__(self, index):
        index_value = self.dataset_dict[index]
        if not self.geometric_or_not:
            label = index_value['label']
            adjacency_matrix = index_value['adjacency_matrix']
            # sparse_adjacency_matrix = nx.adjacency_matrix(gnx).tocoo()
            # values = sparse_adjacency_matrix.data
            # indices = np.vstack((sparse_adjacency_matrix.row, sparse_adjacency_matrix.col))
            # i = torch.LongTensor(indices)
            # v = torch.FloatTensor(values)
            # shape = sparse_adjacency_matrix.shape
            # x = torch.sparse.FloatTensor(i, v, torch.Size(shape))
            if self.mission == "just_values":
                values = index_value['values_on_leaves']
            else:
                values = index_value['values_on_nodes']
            return Tensor(values), Tensor(adjacency_matrix), label
        else:
            return index_value['data']
