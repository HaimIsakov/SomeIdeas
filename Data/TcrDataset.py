import os
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd
from colorama import Fore
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm

from ofek_files_utils_functions import HistoMaker


class TCRDataset(Dataset):
    def __init__(self, dataset_name, data_path, label_path, subject_list, mission):
        self.data_path = data_path
        self.label_path = label_path
        self.subject_list = subject_list
        self.dataset_name = dataset_name if dataset_name != "tcr" else "TCR"
        self.adj_mat = None
        self.values_dict = None
        self.networks_dict = None
        # self.adj_mat = self.from_distance_mat_to_adj_matrix(adj_mat_path)
        self.label_dict = self.load_or_create_label_dict()
        # self.values_df = self.load_or_create_values_dict()
        # self.networks_dict, self.values_dict = self.load_or_create_tcr_network()
        # calc_avg_degree(self.networks_dict)
        # self.graphs_list = self.get_graph_list()
        self.dataset_dict = {}
        self.mission = mission
        self.train_graphs_list = []
        self.run_number = 0
        # self.update_graphs(data_path, label_path)

    def __getitem__(self, index):
        index_value = self.dataset_dict[index]
        if self.mission == "just_values" or self.mission == "just_graph" or self.mission == "graph_and_values"\
                or self.mission == "double_gcn_layer" or self.mission == "concat_graph_and_values":
            values = np.expand_dims(deepcopy(index_value['values']), axis=1)
            adjacency_matrix = deepcopy(index_value['adjacency_matrix'])
        if self.mission == "yoram_attention":
            # values = index_value['values']
            values = np.expand_dims(deepcopy(index_value['values']), axis=1)
            adjacency_matrix = deepcopy(index_value['graph_embed'])  # TODO: it is not the actual adj mat - so Fix it

        label = self.dataset_dict[index]['label']
        return Tensor(values), Tensor(adjacency_matrix), label

    def set_train_graphs_list(self, train_graphs_list):
        self.train_graphs_list = train_graphs_list

    def set_graph_embed_in_dataset_dict(self, embed_mat):
        for i, subject in enumerate(self.subject_list):
            self.dataset_dict[i]['graph_embed'] = embed_mat

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
        distance_mat_df = pd.read_csv(adj_mat_path, index_col=0)
        network_values = distance_mat_df.values
        np.fill_diagonal(network_values, 1)
        # TODO: Make the comment to real code when running the first version of tcrs' graphs creation
        # network_values = 1 / network_values
        np.fill_diagonal(network_values, 0)
        adj_mat_df = pd.DataFrame(network_values, index=distance_mat_df.index, columns=distance_mat_df.index)
        return adj_mat_df

    def set_dataset_dict(self, **kwargs):
        for i, subject in enumerate(self.subject_list):
            self.dataset_dict[i] = {'subject': subject,
                                    'label': self.label_dict[subject],
                                    'adjacency_matrix': self.networks_dict[subject].values,
                                    'values': self.values_dict[subject],
                                    'graph': nx.from_numpy_matrix(self.networks_dict[subject].values)}
            if 'X' in kwargs:
                X = kwargs['X']
                self.dataset_dict[i]['graph_embed'] = X

    def update_graphs(self, **kwargs):
        if self.adj_mat is not None:
            self.set_dataset_dict(**kwargs)
    #
    # def load_or_create_values_dict(self):
    #     values_dict = {}
    #     for i, subject in enumerate(self.subject_list):
    #         values_list = []
    #         path = os.path.join(self.data_path, subject + ".csv")
    #         sample_df = pd.read_csv(path, usecols=["combined", "frequency"])
    #         all_tcrs = list(self.adj_mat.index)
    #         intersec = set(all_tcrs) & set(sample_df["combined"])
    #         new_sample_df = sample_df[["combined", "frequency"]]
    #         new_sample_df.set_index("combined", inplace=True)
    #         new_sample_df_only_tcr_list = new_sample_df.loc[intersec]
    #         tcr_sample_df = new_sample_df_only_tcr_list
    #         # file_path = os.path.join(self.data_path, f"final_{subject}.csv")
    #         # tcr_sample_df = pd.read_csv(file_path, index_col=0)
    #         tcr_sample_df = np.log(tcr_sample_df.groupby("combined").sum() + 1e-300)
    #         golden_tcr = list(tcr_sample_df.index)
    #         # print(tcr_sample_df.to_dict())
    #         tcr_sample_dict = tcr_sample_df.to_dict()['frequency']
    #         all_tcrs_minus_golden_tcr = list(set(all_tcrs) - set(golden_tcr))
    #         for tcr in all_tcrs_minus_golden_tcr:
    #             tcr_sample_dict[tcr] = 0
    #         # To make all values among all samples the same order
    #         for tcr in all_tcrs:
    #             values_list.append(tcr_sample_dict[tcr])
    #         values_dict[subject] = values_list
    #     return values_dict

    def load_or_create_tcr_network(self):
        networks_dict = {}
        values_dict = {}
        golden_tcrs = set(list(self.adj_mat.index))

        for i, subject in tqdm(enumerate(self.subject_list), desc='Create TCR Networks', total=len(self.subject_list),
                               bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTGREEN_EX, Fore.RESET)):
            # file_path = os.path.join(self.data_path, f"final_{subject}.csv")
            # tcr_sample_df = pd.read_csv(file_path, index_col=0)
            file_path = os.path.join(self.data_path, subject + ".csv")
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
        return networks_dict, values_dict

    def load_or_create_label_dict(self):
        label_df = pd.read_csv(self.label_path, usecols=["sample", 'status'])
        label_df["sample"] = label_df["sample"] + "_" + label_df['status']
        label_df.set_index("sample", inplace=True)
        label_df = label_df.loc[self.subject_list]
        label_df["status"] = label_df["status"].map({"negative": 0, "positive": 1})
        label_dict = label_df.to_dict()['status']
        return label_dict

    def calc_golden_tcrs(self, adj_mat_path=None):
        if adj_mat_path is None:
            adj_mat_path = f"dist_mat_{self.run_number}.csv"
        self.adj_mat = self.from_distance_mat_to_adj_matrix(adj_mat_path + ".csv")
        # self.values_df = self.load_or_create_values_dict()
        self.networks_dict, self.values_dict = self.load_or_create_tcr_network()

    # def get_all_groups(self):
    #     return [self.dataset_dict[i]['subject'] for i in range(len(self.subject_list))]


# if __name__ == "__main__":
#     mission = 1
#     data_path = os.path.join("TCR_dataset", "final_sample_files", "Final_Test")
#     label_path = os.path.join("TCR_dataset", "samples.csv")
#     phenotype_dataset = pd.read_csv(label_path)
#     phenotype_dataset = phenotype_dataset[phenotype_dataset["test/train"] == "test"]
#     subject_list = [sample + "_" + sign for sample, sign in zip(phenotype_dataset["sample"].tolist(),
#                                                                phenotype_dataset["status"].tolist())]
#     adj_mat_path = os.path.join("TCR_dataset", "distance_matrix.csv")
#     tcr_dataset = TCRDataset(adj_mat_path, data_path, label_path, subject_list, mission)
