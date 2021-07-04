from torch.utils.data import Dataset
import pandas as pd
from taxonomy_tree import *
import numpy as np
from tqdm import tqdm


class ArrangeGDMDataset(Dataset):
    def __init__(self, data_file_path, tag_file_path, mission, adjacency_normalization):
        self.mission = mission
        self._microbiome_df = pd.read_csv(data_file_path, index_col='ID')
        self._tags = pd.read_csv(tag_file_path, index_col='ID')
        self.graphs_list = []
        self.groups, self.labels = [], []  # for "sklearn.model_selection.GroupShuffleSplit, Stratify"
        # lambda_func_trimester = lambda x: True
        lambda_func_trimester = lambda x: x == 1
        # lambda_func_repetition = lambda x: True
        lambda_func_repetition = lambda x: x == 1
        self.arrange_dataframes(lambda_func_repetition=lambda_func_repetition, lambda_func_trimester=lambda_func_trimester)
        self.create_graphs_with_common_nodes()
        self.node_order = self.set_node_order()
        self.adjacency_normalization = adjacency_normalization

    def arrange_dataframes(self, lambda_func_repetition=lambda x: True, lambda_func_trimester=lambda x: True):
        self.remove_na()
        self._tags['Tag'] = self._tags['Tag'].astype(int)
        self._tags['trimester'] = self._tags['trimester'].astype(int)
        self.split_id_col()
        self._microbiome_df = self._microbiome_df[self._tags['trimester'].apply(lambda_func_trimester)]
        self._tags = self._tags[self._tags['trimester'].apply(lambda_func_trimester)]
        self._tags = self._tags[self._microbiome_df['Repetition'].apply(lambda_func_repetition)]
        self._microbiome_df = self._microbiome_df[self._microbiome_df['Repetition'].apply(lambda_func_repetition)]
        self._microbiome_df.sort_index(inplace=True)
        self._tags.sort_index(inplace=True)
        self.add_groups()  # It is important to verify that the order of instances is correct
        self.add_labels()
        del self._tags['trimester']
        del self._microbiome_df['Repetition']
        del self._microbiome_df['Code']

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def split_id_col(self):
        self._microbiome_df['Code'] = [cur_id.split('-')[0] for cur_id in self._microbiome_df.index]
        self._microbiome_df['Repetition'] = [int(cur_id.split('-')[-1]) for cur_id in self._microbiome_df.index]

    def add_groups(self):
        self.groups = list(self._microbiome_df['Code'])

    def get_groups(self, indexes):
        return [self.groups[i] for i in indexes]

    def add_labels(self):
        self.labels = list(self._tags['Tag'])

    def get_labels(self, indexes):
        return [self.labels[i] for i in indexes]

    def remove_na(self):
        index = self._tags['Tag'].index[self._tags['Tag'].apply(np.isnan)]
        self._tags.drop(index, inplace=True)
        self._microbiome_df.drop(index, inplace=True)

    def count_each_class(self):
        counter_dict = self._tags['Tag'].value_counts()
        return counter_dict[0], counter_dict[1]

    def create_tax_trees(self):
        for i, mom in tqdm(enumerate(self._microbiome_df.iterrows()), desc='Create graphs'):
            cur_graph = create_tax_tree(self._microbiome_df.iloc[i], ignore_values=0, ignore_flag=True)
            self.graphs_list.append(cur_graph)

    def find_common_nodes(self):
        nodes_dict = {}
        j = 0
        for graph in self.graphs_list:
            nodes = graph.nodes(data=False)
            for name, value in nodes:
                if name not in nodes_dict:
                    nodes_dict[name] = j
                    j = j + 1
        return nodes_dict

    def create_graphs_with_common_nodes(self):
        self.create_tax_trees()
        nodes_dict = self.find_common_nodes()
        for graph in tqdm(self.graphs_list, desc='Add to graphs the common nodes set'):
            nodes_and_values = graph.nodes()
            nodes = [node_name for node_name, value in nodes_and_values]
            for node_name in nodes_dict:
                # if there is a node that exists in other graph but not in the current graph we want to add it
                if node_name not in nodes:
                    graph.add_node((node_name, 0.0))

        # sort the node, so that every graph has the same order of graph.nodes()
        self.sort_all_graphs()

    def sort_all_graphs(self):
        temp_graph_list = []
        for graph in self.graphs_list:
            temp_graph = nx.Graph()
            temp_graph.add_nodes_from(sorted(graph.nodes(data=True)))
            temp_graph.add_edges_from(graph.edges(data=True))
            # temp_graph.add_edges_from(sorted(graph.edges(data=True)))
            temp_graph_list.append(temp_graph)
        self.graphs_list = temp_graph_list

    def set_node_order(self):
        nodes_and_values = sorted(self.graphs_list[0].nodes())
        return [node_name for node_name, value in nodes_and_values]
