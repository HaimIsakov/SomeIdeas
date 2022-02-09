import os
from copy import deepcopy

from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm
from shahar_gdm_utils_func import *


class ShaharGdmDataset(Dataset):
    def __init__(self, data_path, label_path, subject_list, mission):
        self.data_path = data_path
        self.label_path = label_path
        self.subject_list = subject_list
        self.networks_dict = self.load_or_create_network()
        self.label_dict = self.load_or_create_label_dict()
        self.dataset_dict = {}
        self.mission = mission
        self.train_graphs_list = []
        # self.set_dataset_dict()
        x=1

    def __getitem__(self, index):
        index_value = self.dataset_dict[index]
        if self.mission == "just_values" or self.mission == "just_graph" or self.mission == "graph_and_values"\
                or self.mission == "double_gcn_layer" or self.mission == "concat_graph_and_values":
            values = deepcopy(index_value['values'])
            adjacency_matrix = deepcopy(index_value['adjacency_matrix'])
        if self.mission == "yoram_attention":
            values = deepcopy(index_value['values'])
            adjacency_matrix = deepcopy(index_value['graph_embed'])  # TODO: it is not the actual adj mat - so Fix it

        label = self.dataset_dict[index]['label']
        return Tensor(values), Tensor(adjacency_matrix), label

    def set_train_graphs_list(self, train_graphs_list):
        self.train_graphs_list = train_graphs_list

    def set_graph_embed_in_dataset_dict(self, embed_mat):
        for i, subject in enumerate(self.subject_list):
            self.dataset_dict[i]['graph_embed'] = embed_mat

    def __len__(self):
        return len(self.dataset_dict)

    def __repr__(self):
        return "Shahar Gdm Dataset " + "len" + str(len(self))

    def get_vector_size(self):
        return 1
        # return self.dataset_dict[0]['values'].shape[1]

    def nodes_number(self):
        return self.dataset_dict[0]['adjacency_matrix'].shape[0]

    def get_leaves_number(self):
        return self.nodes_number()

    def get_graph_list(self):
        graphs_list = [0]*len(self.subject_list)
        for i, subject in enumerate(self.subject_list):
            graph_from_adj_matrix = nx.from_numpy_matrix(self.networks_dict[subject])
            graphs_list[i] = graph_from_adj_matrix
        return graphs_list

    def set_dataset_dict(self, **kwargs):
        for i, subject in enumerate(self.subject_list):
            graph = self.networks_dict[subject]
            self.dataset_dict[i] = {'subject': subject,
                                    'label': self.label_dict[subject],
                                    'adjacency_matrix': nx.adjacency_matrix(graph).todense(),
                                    'graph': graph,
                                    'values': self.get_values_on_nodes(graph),
                                    }
            if 'X' in kwargs:
                X = kwargs['X']
                self.dataset_dict[i]['graph_embed'] = X

    def update_graphs(self, **kwargs):
        self.set_dataset_dict(**kwargs)

    def load_or_create_network(self):
        samples_path = os.path.join(self.data_path)
        samples_df = load_dataset(samples_path)
        vals_df = samples_df.copy(deep=True).fillna(0)
        existence_df = samples_df.copy(deep=True).isnull().astype(int).apply(lambda x: 1 - x)
        corr, corr_combined_df = find_corr(vals_df, existence_df)
        networks_dict = df_to_graphs(samples_df, corr)
        return networks_dict

    def load_or_create_label_dict(self):
        labels_path = os.path.join(self.label_path)
        label_dict = pd.read_csv(labels_path, index_col=0)
        label_dict = label_dict.to_dict(orient='index')
        for k, v in label_dict.items():
            label_dict[k] = label_dict[k]["GDMA"]
        return label_dict

    def get_values_on_nodes(self, graph):
        nodes_and_values = graph.nodes(data=True)
        values_matrix = [[feature_value for feature_name, feature_value in value_dict.items()] for node, value_dict
                         in nodes_and_values]
        return values_matrix


# if __name__ == "__main__":
#    data_path = "week_14_new.csv"
#    label_path = "gdm.csv"
#    phenotype_dataset = pd.read_csv(label_path)
#    subject_list = list(phenotype_dataset.index)
#    mission = "graph_and_values"
#    shahar_gdm_dataset = ShaharGdmDataset(data_path, label_path, subject_list, mission)
