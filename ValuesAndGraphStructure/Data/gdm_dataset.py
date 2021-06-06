from torch.utils.data import Dataset
import pandas as pd
from torch import Tensor
from JustValues.Data.taxonomy_tree import *
import numpy as np
from tqdm import tqdm

class GDMDataset(Dataset):
    def __init__(self, data_file_path, tag_file_path, mission='values_and_graph_structure'):
        self.mission = mission
        self._df = pd.read_csv(data_file_path, index_col='ID')
        self._tags = pd.read_csv(tag_file_path, index_col='ID')
        self.graphs_list = []
        self.arrange_dataframes()
        self.create_graphs_with_common_nodes()

    def arrange_dataframes(self):
        self.remove_na()
        self._tags['Tag'] = self._tags['Tag'].astype(int)
        self._tags['trimester'] = self._tags['trimester'].astype(int)
        self.split_id_col()
        self._df = self._df[self._tags['trimester'] > 2]
        self._tags = self._tags[self._tags['trimester'] > 2]
        self._tags = self._tags[self._df['Repetition'] == 1]
        self._df = self._df[self._df['Repetition'] == 1]
        self._df.sort_index(inplace=True)
        self._tags.sort_index(inplace=True)
        del self._tags['trimester']
        del self._df['Repetition']
        del self._df['Code']

    def __getitem__(self, index):
        # need to return A - adjacency matrix as well as values on each node and label
        values = self.get_values_on_nodes_ordered_by_nodes(index)
        # values = list(self._df.iloc[index])
        adjacency_matrix = nx.adjacency_matrix(self.graphs_list[index]).todense()
        # adjacency_matrix = [5]
        label = int(self._tags.iloc[index]['Tag'])
        return Tensor(adjacency_matrix), Tensor(values), label

    def __len__(self):
        a, b = self._df.shape
        return a

    def split_id_col(self):
        self._df['Code'] = [cur_id.split('-')[0] for cur_id in self._df.index]
        self._df['Repetition'] = [int(cur_id.split('-')[-1]) for cur_id in self._df.index]

    def remove_na(self):
        index = self._tags['Tag'].index[self._tags['Tag'].apply(np.isnan)]
        self._tags.drop(index, inplace=True)
        self._df.drop(index, inplace=True)

    def get_vector_size(self):
        nodes_dict = self.find_common_nodes()
        vector_size = len(nodes_dict)
        return vector_size
        # a, b = self._df.shape
        # return b

    def count_each_class(self):
        counter_dict = self._tags['Tag'].value_counts()
        return counter_dict[0], counter_dict[1]

    def create_tax_trees(self):
        for i, mom in tqdm(enumerate(self._df.iterrows()), desc='Create graphs'):
            cur_graph = create_tax_tree(self._df.iloc[i], ignore_values=0, ignore_flag=True)
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

    def get_values_on_nodes_ordered_by_nodes(self, index):
        cur_graph = self.graphs_list[index]
        nodes_and_values = cur_graph.nodes()
        values = [value for node_name, value in nodes_and_values]
        return values



    # def check_graphs(self):
    ## check adjacency_matrix
    #     [print(i) for i in nx.adjacency_matrix(self.graphs_list[0]).todense()]
    #     print("-------------------------------------------------------------------------------")
    #     [print(i) for i in nx.adjacency_matrix(self.graphs_list[-1]).todense()]

    ## check that all graphs contain all nodes in the same order
        # f = open('nodes_check.txt', 'wt')
        # for g in self.graphs_list:
        #     f.write(';'.join([str(a) for a, b in g.nodes()]))
        #     f.write('\n')
        # f.close()
