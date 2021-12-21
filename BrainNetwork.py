import os
import csv
import networkx as nx
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm
from AbideDatasetUtils import load_connectivity


class AbideDataset(Dataset):
    def __init__(self, subject_list, mission):
        self.dataset_dict = {}
        self.subject_list = subject_list
        self.mission = mission
        self.graphs_list = []*len(self.subject_list)
        #self.update_graphs(data_path, label_path)

    def __getitem__(self, index):
        label = self.dataset_dict[index]['label']
        # In GAT-Li paper they applied absolute value on adj matrix
        adjacency_matrix = np.absolute(self.dataset_dict[index]['adjacency_matrix'])
        values = self.dataset_dict[index]['adjacency_matrix']
        return Tensor(values), Tensor(adjacency_matrix), label

    def __len__(self):
        return len(self.dataset_dict)

    def __repr__(self):
        return "Abide Dataset" + "len" + str(len(self))

    def get_vector_size(self):
        return self.dataset_dict[0]['adjacency_matrix'].shape[1]

    def nodes_number(self):
        return self.dataset_dict[0]['adjacency_matrix'].shape[0]

    def get_leaves_number(self):
        return self.nodes_number()

    def update_graph_list(self, data_path):
        networks_dict = self.load_or_create_brain_network(data_path)
        for i, subject in enumerate(self.subject_list):
            graph_from_adj_matrix = nx.from_numpy_matrix(self.dataset_dict[i]['adjacency_matrix'])
            self.graphs_list[i] = graph_from_adj_matrix

    def set_dataset_dict(self,data_path, label_path, **kwargs):
        networks_dict = self.load_or_create_brain_network(data_path)
        label_dict = self.load_or_create_label_dict(label_path)
        for i, subject in enumerate(self.subject_list):
            self.dataset_dict[i] = {'subject': subject,
                                    'label': label_dict[subject],
                                    'adjacency_matrix': networks_dict[subject]}
            graph_from_adj_matrix = nx.from_numpy_matrix(self.dataset_dict[i]['adjacency_matrix'])
            self.graphs_list[i] = graph_from_adj_matrix

    def update_graphs(self, data_path, label_path, **kwargs):
        self.set_dataset_dict(data_path, label_path, **kwargs)

    def load_or_create_brain_network(self, data_path):
        networks_dict = {}
        for i, subject in tqdm(enumerate(self.subject_list), desc='Create Abide Networks', total=len(self.subject_list)):
            file_path = os.path.join(data_path, f"{subject}_{data_path}.1D")
            input_matrix = np.loadtxt(file_path)
            network = load_connectivity(input_matrix)
            networks_dict[subject] = network
        return networks_dict

    def load_or_create_label_dict(self, label_path):
        label_dict = {}
        with open(label_path) as csv_file:
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
