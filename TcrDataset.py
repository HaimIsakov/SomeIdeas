import os
import networkx as nx
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm


class TCRDataset(Dataset):
    def __init__(self, data_path, label_path, subject_list, mission):
        self.data_path = data_path
        self.label_path = label_path
        self.subject_list = subject_list
        self.adj_mat = self.from_distance_mat_to_adj_matrix()
        self.label_dict = self.load_or_create_label_dict()
        self.values_df = self.load_or_create_values_dict()
        self.networks_dict = self.load_or_create_tcr_network()
        # calc_avg_degree(self.networks_dict)

        # self.graphs_list = self.get_graph_list()
        self.dataset_dict = {}
        self.mission = mission
        self.train_graphs_list = []
        # self.update_graphs(data_path, label_path)

    def __getitem__(self, index):
        index_value = self.dataset_dict[index]
        if self.mission == "just_values" or self.mission == "just_graph" or self.mission == "graph_and_values"\
                or self.mission == "double_gcn_layer" or self.mission == "concat_graph_and_values":
            values = index_value['values']
            adjacency_matrix = index_value['adjacency_matrix']
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
        return "TCR Dataset" + "_len" + str(len(self))

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

    def from_distance_mat_to_adj_matrix(self):
        distance_mat_df = pd.read_csv(os.path.join("TCR_dataset", "distance_matrix.csv"), index_col=0)
        network_values = distance_mat_df.values
        np.fill_diagonal(network_values, 1)
        network_values = 1 / network_values
        np.fill_diagonal(network_values, 0)
        adj_mat_df = pd.DataFrame(network_values, index=distance_mat_df.index, columns=distance_mat_df.index)
        return adj_mat_df

    def set_dataset_dict(self, **kwargs):
        for i, subject in enumerate(self.subject_list):
            self.dataset_dict[i] = {'subject': subject,
                                    'label': self.label_dict[subject],
                                    'adjacency_matrix': self.networks_dict[subject],
                                    'values': self.values_df[subject],
                                    'graph': nx.from_numpy_matrix(self.networks_dict[subject])}
            if 'X' in kwargs:
                X = kwargs['X']
                self.dataset_dict[i]['graph_embed'] = X

    def update_graphs(self, **kwargs):
        self.set_dataset_dict(**kwargs)

    def load_or_create_values_dict(self):
        values_dict = {}
        for i, subject in enumerate(self.subject_list):
            values_list = []
            file_path = os.path.join(self.data_path, f"final_{subject}.csv")
            tcr_sample_df = pd.read_csv(file_path, index_col=0)
            golden_tcr = list(tcr_sample_df.index)
            all_tcrs = list(self.adj_mat.index)
            tcr_sample_dict = tcr_sample_df.to_dict()['frequency']
            all_tcrs_minus_golden_tcr = list(set(all_tcrs) - set(golden_tcr))
            for tcr in all_tcrs_minus_golden_tcr:
                tcr_sample_dict[tcr] = 0
            # To make all values among all samples the same order
            for tcr in all_tcrs:
                values_list.append(tcr_sample_dict[tcr])
            values_dict[subject] = values_list
        return values_dict

    def load_or_create_tcr_network(self):
        networks_dict = {}
        for i, subject in tqdm(enumerate(self.subject_list), desc='Create TCR Networks', total=len(self.subject_list)):
            file_path = os.path.join(self.data_path, f"final_{subject}.csv")
            tcr_sample_df = pd.read_csv(file_path, index_col=0)
            tcr_sample_df = tcr_sample_df.groupby("combined").sum()  # sum the repetitions
            golden_tcr = list(tcr_sample_df.index)
            all_tcrs = list(self.adj_mat.index)
            all_tcrs_minus_golden_tcr = list(set(all_tcrs) - set(golden_tcr))
            network = self.adj_mat.copy()
            for tcr in all_tcrs_minus_golden_tcr:
                network[tcr] = np.zeros(self.adj_mat.shape[1])
            for tcr in golden_tcr:
                network.loc[tcr] = network[tcr]
            networks_dict[subject] = network
        return networks_dict

    def load_or_create_label_dict(self):
        label_df = pd.read_csv(self.label_path, usecols=["sample", 'status'])
        label_df["sample"] = label_df["sample"] + "_" + label_df['status']
        label_df.set_index("sample", inplace=True)
        label_df["status"] = label_df["status"].map({"negative": 0, "positive": 1})
        label_dict = label_df.to_dict()['status']
        return label_dict

    # def get_all_groups(self):
    #     return [self.dataset_dict[i]['subject'] for i in range(len(self.subject_list))]


if __name__ == "__main__":
    mission = 1
    data_path = os.path.join("TCR_dataset", "final_sample_files", "Final_Test")
    label_path = os.path.join("TCR_dataset", "samples.csv")
    phenotype_dataset = pd.read_csv(label_path)
    phenotype_dataset = phenotype_dataset[phenotype_dataset["test/train"] == "test"]
    subject_list = [sample + "_" + sign for sample, sign in zip(phenotype_dataset["sample"].tolist(),
                                                               phenotype_dataset["status"].tolist())]
    tcr_dataset = TCRDataset(data_path, label_path, subject_list, mission)
