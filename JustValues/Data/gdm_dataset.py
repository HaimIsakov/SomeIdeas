from torch import Tensor
from arrange_gdm_dataset import *


class GDMDataset(ArrangeGDMDataset):
    def __init__(self, data_file_path, tag_file_path, mission='just_values'):
        super().__init__(data_file_path, tag_file_path, mission)

    def __getitem__(self, index):
        # need to return A - adjacency matrix as well as values on each node and label
        # values = self.get_values_on_nodes_ordered_by_nodes(index)
        values = list(self._microbiome_df.iloc[index])
        adjacency_matrix = nx.adjacency_matrix(self.graphs_list[index]).todense()
        # adjacency_matrix = [5]  # random value only for speed
        label = int(self._tags.iloc[index]['Tag'])
        return Tensor(adjacency_matrix), Tensor(values), label

    def __len__(self):
        a, b = self._microbiome_df.shape
        return a

    def get_vector_size(self):
        a, b = self._microbiome_df.shape
        return b

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
