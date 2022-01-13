import os
import csv
import networkx as nx
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm


class CancerDataset(Dataset):
    def __init__(self, adj_mat_path, data_path, label_path, subject_list, mission):
        self.adj_mat_path = adj_mat_path
        self.data_path = data_path
        self.label_path = label_path
        self.subject_list = subject_list
        self.network = self.load_or_create_cancer_network()
        self.label_mat = self.load_or_create_label_dict()
        self.values_mat = self.load_or_create_values()
        # self.graphs_list = self.get_graph_list()
        self.dataset_dict = {}
        self.mission = mission
        self.train_graphs_list = []

    def __getitem__(self, index):
        index_value = self.dataset_dict[index]
        if self.mission == "just_values" or self.mission == "just_graph" or self.mission == "graph_and_values":
            values = index_value['values']
            adjacency_matrix = self.network
        if self.mission == "yoram_attention":
            values = index_value['values']
            adjacency_matrix = index_value['graph_embed']  # TODO: it is not the actual adj mat - so Fix it

        label = self.dataset_dict[index]['label'][0][0]
        # adjacency_matrix = [5]
        return Tensor(values), Tensor(adjacency_matrix), Tensor(label)

    def set_train_graphs_list(self, train_graphs_list):
        self.train_graphs_list = train_graphs_list

    def set_graph_embed_in_dataset_dict(self, embed_mat):
        for i, subject in enumerate(self.subject_list):
            self.dataset_dict[i]['graph_embed'] = embed_mat

    def __len__(self):
        return len(self.dataset_dict)

    def __repr__(self):
        return "cancer"

    def get_vector_size(self):
        # return 1
        return self.values_mat[1]

    def nodes_number(self):
        return self.dataset_dict[0]['adjacency_matrix'].shape[0]

    def get_leaves_number(self):
        return self.nodes_number()

    # def get_graph_list(self):
    #     graphs_list = [0]*len(self.subject_list)
    #     for i, subject in enumerate(self.subject_list):
    #         graph_from_adj_matrix = nx.from_numpy_matrix(self.networks_dict[subject])
    #         graphs_list[i] = graph_from_adj_matrix
    #     return graphs_list

    def set_dataset_dict(self, **kwargs):
        common_graph = nx.from_numpy_matrix(self.network)
        for i, subject in enumerate(self.subject_list):
            self.dataset_dict[i] = {'label': self.label_mat[i][0],
                                    'values': self.values_mat[i],
                                    'adjacency_matrix': self.network,
                                    'graph': common_graph}
            if 'X' in kwargs:
                X = kwargs['X']
                self.dataset_dict[i]['graph_embed'] = X

    def update_graphs(self, **kwargs):
        self.set_dataset_dict(**kwargs)

    def load_or_create_cancer_network(self):
        adj_mat = pd.read_csv(self.adj_mat_path, header=None)
        return np.matrix(adj_mat.values)

    def load_or_create_label_dict(self):
        label_df = pd.read_csv(self.label_path)
        return np.matrix(label_df.values)

    def load_or_create_values(self):
        values_df = pd.read_csv(self.data_path)
        return np.matrix(values_df.values)

    # def get_all_groups(self):
    #     return [self.dataset_dict[i]['subject'] for i in range(len(self.subject_list))]


# if __name__ == "__main__":
    # train_data = pd.read_csv("new_cancer_train_data.csv", header=None)
    # test_data = pd.read_csv("new_cancer_test_data.csv", header=None)
    # train_label = pd.read_csv("new_cancer_train_label.csv", header=None)
    # test_label = pd.read_csv("new_cancer_test_label.csv", header=None)
    # Train_test_Data = pd.concat((train_data, test_data), ignore_index=True, axis=0)
    # Train_test_Label = pd.concat((train_label, test_label), ignore_index=True, axis=0)
    # Train_test_Data.to_csv("new_cancer_data.csv", index=False)
    # Train_test_Label.to_csv("new_cancer_label.csv", index=False)
    #
    # mission = "graph_and_values"
    # data_path = "new_cancer_data.csv"  # It contains both train and test set
    # label_path = "new_cancer_label.csv"  # It contains both train and test set
    # adj_mat_path = "new_cancer_adj_matrix.csv"
    # subject_list = range(0, 10)
    #
    # cancer_dataset = CancerDataset(adj_mat_path, data_path, label_path, subject_list, mission)
    # cancer_dataset.set_dataset_dict()
    # a = cancer_dataset[5]
    # x=5
