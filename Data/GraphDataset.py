import sys
from copy import copy, deepcopy
import networkx as nx
import torch
import tqdm
from torch.utils.data import Dataset
from torch import Tensor, FloatTensor
#from torch_geometric.utils import from_networkx
# sys.path.insert(2, 'Missions')

from FiedlerVector.calc_fiedler_vector import return_k_first_eigen_vectors
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
        self.train_graphs_list = []

    def set_dataset_dict(self, **kwargs):
        dataset_dict = {}
        for i in range(self.samples_len):
            graph = self.create_microbiome_graphs.get_graph(i)
            if not self.geometric_or_not:
                dataset_dict[i] = {'graph': graph,
                                   'label': self.microbiome_dataset.get_label(i),
                                   'values_on_leaves': self.microbiome_dataset.get_leaves_values(i),
                                   'values_on_nodes': self.create_microbiome_graphs.get_values_on_nodes_ordered_by_nodes(
                                       graph),
                                   'adjacency_matrix': nx.adjacency_matrix(graph).todense()}
                if 'X' in kwargs:
                    X = kwargs['X']
                    dataset_dict[i]['graph_embed'] = X
            else:
                data = from_networkx(graph, group_node_attrs=['val'])  # Notice: convert file has been changed explicitly
                data.y = torch.tensor(self.microbiome_dataset.get_label(i))
                dataset_dict[i] = {'data': data}
        return dataset_dict

    # def get_joint_nodes(self):
    #     return self.create_microbiome_graphs.find_common_nodes().keys()

    def set_train_graphs_list(self, train_graphs_list):
        self.train_graphs_list = train_graphs_list

    def set_graph_embed_in_dataset_dict(self, embed_mat):
        for i in range(self.samples_len):
            if not self.geometric_or_not:
                self.dataset_dict[i]['graph_embed'] = embed_mat

    def set_fiedler_vector(self):
        for i in tqdm.tqdm(range(self.samples_len), total=self.samples_len, desc='Calculate fiedler vector'):
            graph = self.dataset_dict[i]['graph']
            fiedler_vectors = return_k_first_eigen_vectors(graph, k=1)
            self.dataset_dict[i]['fiedler_vector'] = fiedler_vectors

    def update_graphs(self, **kwargs):
        # self.create_microbiome_graphs.create_graphs_with_common_nodes(union_train_and_test)
        self.dataset_dict = self.set_dataset_dict(**kwargs)  # create dataset dict only after we added the missing nodes to the graph

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

            if self.mission == "just_values":
                values = deepcopy(index_value['values_on_leaves'])
                adjacency_matrix = deepcopy(index_value['adjacency_matrix'])
            elif self.mission == "just_graph" or self.mission == "graph_and_values" \
                    or self.mission == "double_gcn_layer" or self.mission == "concat_graph_and_values":
                values = deepcopy(index_value['values_on_nodes'])
                # print(np.array(index_value['values_on_nodes']) + 1e-300)
                # values = np.log(np.array(index_value['values_on_nodes']) + 1e-300)
                adjacency_matrix = deepcopy(index_value['adjacency_matrix'])
            elif self.mission == "yoram_attention":
                values = deepcopy(index_value['values_on_nodes'])
                adjacency_matrix = deepcopy(index_value['graph_embed'])  # TODO: it is not the actual adj mat - so Fix it
            elif self.mission == "fiedler_vector":
                values = deepcopy(index_value['values_on_nodes'])
                adjacency_matrix = deepcopy(index_value['fiedler_vector'])
            else:
                print("The task was not added here")
            return Tensor(values), Tensor(adjacency_matrix), label
        else:
            return index_value['data']
