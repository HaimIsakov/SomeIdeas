import os
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
from colorama import Fore
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm


class HLA_TCR(Dataset):
    def __init__(self, train_data_path, test_data_path, label_path, subject_list, mission, graph_model,
                 allele, dall):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.label_path = label_path
        self.subject_list = subject_list
        self.dataset_name = "TCR"
        self.adj_mat = None
        self.values_dict = None
        self.networks_dict = None
        self.graph_model = graph_model
        self.allele = allele
        self.dall = dall

        self.label_dict = self.load_or_create_label_dict()
        self.dataset_dict = {}
        self.mission = mission
        self.train_graphs_list = []
        self.run_number = 0

    def calc_avg_degree(self):
        avg_degree_graphs = []
        for k, v in self.dataset_dict.items():
            cur_graph = v['graph']
            degree_list = [val for (node, val) in list(cur_graph.degree())]
            avg_degree = np.mean(degree_list)
            avg_degree_graphs.append(avg_degree)
        x = np.mean(avg_degree_graphs)
        print("Average degree of all graphs", x)
        return x

    def __getitem__(self, index):
        index_value = self.dataset_dict[index]
        values = np.expand_dims(deepcopy(index_value['values']), axis=1)
        adjacency_matrix = deepcopy(index_value['adjacency_matrix'])
        label = self.dataset_dict[index]['label']
        return Tensor(values), Tensor(adjacency_matrix), label

    def set_train_graphs_list(self, train_graphs_list):
        self.train_graphs_list = train_graphs_list

    def __len__(self):
        return len(self.subject_list)

    def __repr__(self):
        return f"{self.dataset_name} Dataset" + "_len" + str(len(self))

    def get_vector_size(self):
        return 1

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

    def from_distance_mat_to_adj_matrix(self, adj_mat_path):
        adj_mat_df = pd.read_csv(adj_mat_path, index_col=0)
        # network_values = distance_mat_df.values
        # if self.graph_model == "projection":
        #     np.fill_diagonal(network_values, 1)
        #     TODO: Make the comment to real code when running the first version of tcrs' graphs creation
        # network_values = 1 / network_values
        # np.fill_diagonal(network_values, 0)
        # else:
        #     np.fill_diagonal(network_values, 0)
        # adj_mat_df = pd.DataFrame(network_values, index=distance_mat_df.index, columns=distance_mat_df.index)
        return adj_mat_df

    def set_dataset_dict(self, **kwargs):
        for i, subject in enumerate(self.subject_list):
            self.dataset_dict[i] = {'subject': subject,
                                    'label': self.label_dict[subject],
                                    'adjacency_matrix': self.networks_dict[subject].values,
                                    'values': self.values_dict[subject],
                                    'graph': nx.from_numpy_matrix(self.networks_dict[subject].values)}

    def update_graphs(self, **kwargs):
        if self.adj_mat is not None:
            self.set_dataset_dict(**kwargs)
            self.calc_avg_degree()

    def load_or_create_tcr_network(self):
        networks_dict = {}
        values_dict = {}
        golden_tcrs = set(list(self.adj_mat.index))

        for i, subject in tqdm(enumerate(self.subject_list), desc='Create TCR Networks', total=len(self.subject_list),
                               bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTGREEN_EX, Fore.RESET)):
            # file_path = os.path.join(self.data_path, f"final_{subject}.csv")
            # tcr_sample_df = pd.read_csv(file_path, index_col=0)
            # file_path = os.path.join(self.data_path, subject + ".csv")
            if subject in self.dall:
                file_path = os.path.join(self.dall[subject])
            else:
                continue

            # file_path = self.dall[subject]
            samples_df = pd.read_csv(file_path, usecols=["combined", "frequency"])
            no_rep_sample_df = samples_df.groupby("combined").sum()  # sum the repetitions
            cur_sample_tcrs = set(list(no_rep_sample_df.index))
            intersec_golden_and_sample_tcrs = list(golden_tcrs & cur_sample_tcrs)
            network = self.adj_mat.copy(deep=True)
            for tcr in list(golden_tcrs):
                if tcr not in intersec_golden_and_sample_tcrs:
                    network[tcr] = np.zeros(network.shape[1])
            for tcr in intersec_golden_and_sample_tcrs:
                network.loc[tcr] = network[tcr]
            # network.to_csv(f"tcr_new_graph_{subject}.csv")
            networks_dict[subject] = network
            no_rep_sample_df['frequency'] = np.log(no_rep_sample_df['frequency'] + 1e-300)
            tcr_sample_dict = {}
            for tcr in golden_tcrs:
                if tcr in intersec_golden_and_sample_tcrs:
                    tcr_sample_dict[tcr] = no_rep_sample_df.loc[tcr]['frequency']
                else:
                    tcr_sample_dict[tcr] = 0
            values_list = []
            # To make all values among all samples the same order
            for tcr in golden_tcrs:
                values_list.append(tcr_sample_dict[tcr])
            values_dict[subject] = values_list
        tqdm._instances.clear()

        return networks_dict, values_dict

    def load_or_create_label_dict(self):
        label_df = pd.read_csv(self.label_path, index_col=0)
        label_dict = label_df.to_dict()[self.allele]
        return label_dict

    def calc_golden_tcrs(self, adj_mat_path=None):
        if adj_mat_path is None:
            adj_mat_path = f"dist_mat_{self.run_number}.csv"
        self.adj_mat = self.from_distance_mat_to_adj_matrix(adj_mat_path + ".csv")
        # self.values_df = self.load_or_create_values_dict()
        self.networks_dict, self.values_dict = self.load_or_create_tcr_network()
