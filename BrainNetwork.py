import os
import csv
import networkx as nx
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm
from AbideDatasetUtils import *


class AbideDataset(Dataset):
    def __init__(self, data_path, label_path, subject_list, mission):
        self.data_path = data_path
        self.label_path = label_path
        self.subject_list = subject_list
        self.networks_dict = self.load_or_create_brain_network()
        # calc_avg_degree(self.networks_dict)
        self.label_dict = self.load_or_create_label_dict()
        # self.graphs_list = self.get_graph_list()
        self.dataset_dict = {}
        self.mission = mission
        self.train_graphs_list = []
        # self.update_graphs(data_path, label_path)

    def __getitem__(self, index):
        index_value = self.dataset_dict[index]
        if self.mission == "just_values" or self.mission == "just_graph" or self.mission == "graph_and_values"\
                or self.mission == "double_gcn_layer" or self.mission == "concat_graph_and_values":
            # values = self.dataset_dict[index]['adjacency_matrix']
            values = index_value['values']
            # In GAT-Li paper they applied absolute value on adj matrix, and put 0 on the diagonal
            adjacency_matrix = transform_adj_mat(index_value['adjacency_matrix'])
        if self.mission == "yoram_attention":
            values = index_value['values']
            adjacency_matrix = index_value['graph_embed']  # TODO: it is not the actual adj mat - so Fix it

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
        return "Abide Dataset" + "len" + str(len(self))

    def get_vector_size(self):
        # return 1
        return self.dataset_dict[0]['adjacency_matrix'].shape[1]

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
            self.dataset_dict[i] = {'subject': subject,
                                    'label': self.label_dict[subject],
                                    'adjacency_matrix': self.networks_dict[subject],
                                    'values': self.networks_dict[subject].copy(),
                                    'graph': nx.from_numpy_matrix(self.networks_dict[subject])}
            # 'values': calc_sum_abs_corr(self.networks_dict[subject])
            # 'values': self.networks_dict[subject].copy()
            # 'values': calc_first_eigen_vector(self.networks_dict[subject])
            if 'X' in kwargs:
                X = kwargs['X']
                self.dataset_dict[i]['graph_embed'] = X

    def update_graphs(self, **kwargs):
        self.set_dataset_dict(**kwargs)

    def load_or_create_brain_network(self):
        networks_dict = {}
        for i, subject in tqdm(enumerate(self.subject_list), desc='Create Abide Networks', total=len(self.subject_list)):
            file_path = os.path.join(self.data_path, f"{subject}_rois_ho.1D")
            input_matrix = np.loadtxt(file_path)
            # load_connectivity_binary - Take only the correlations above some threshold
            # network = load_connectivity_binary(input_matrix)
            network = load_connectivity_origin(input_matrix)
            networks_dict[subject] = network
        return networks_dict

    def load_or_create_label_dict(self):
        label_dict = {}
        with open(self.label_path) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if row['FILE_ID'] in self.subject_list:
                    label = int(row['DX_GROUP'])
                    if label == 2:
                        label_dict[row['FILE_ID']] = 0
                    else:
                        label_dict[row['FILE_ID']] = 1
        return label_dict

    def get_all_groups(self):
        return [self.dataset_dict[i]['subject'] for i in range(len(self.subject_list))]

#if __name__ == "__main__":
#    data_path = "rois_ho"
#    label_path = "Phenotypic_V1_0b_preprocessed1.csv"
#    phenotype_dataset = pd.read_csv("Phenotypic_V1_0b_preprocessed1.csv")
#    subject_list = [value for value in phenotype_dataset["FILE_ID"].tolist() if value != "no_filename"]
#    abide_dataset = AbideDataset(data_path, label_path, subject_list)
