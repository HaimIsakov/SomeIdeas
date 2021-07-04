from torch import Tensor
from arrange_gdm_dataset import *


class GDMDataset(ArrangeGDMDataset):
    def __init__(self, data_file_path, tag_file_path, mission, adjacency_normalization):
        super().__init__(data_file_path, tag_file_path, mission, adjacency_normalization)

    def __getitem__(self, index):
        # need to return A - adjacency matrix as well as values on each node and label
        # values = self.get_values_on_nodes_ordered_by_nodes(index)
        if self.mission == 'JustValues':
            values = list(self._microbiome_df.iloc[index])
            A = [5]  # random value only for speed
        else:
            values = self.get_values_on_nodes_ordered_by_nodes(index)
            # adjacency_matrix = nx.adjacency_matrix(self.graphs_list[index]).todense()
            gnx = self.graphs_list[index]
            A = self.normalize_adjacency(gnx, self.node_order)
        label = int(self._tags.iloc[index]['Tag'])
        return Tensor(A), Tensor(values), label

    def __len__(self):
        a, b = self._microbiome_df.shape
        return a

    def get_leaves_number(self):
        a, b = self._microbiome_df.shape
        return b

    def get_vector_size(self):
        nodes_dict = self.find_common_nodes()
        vector_size = len(nodes_dict)
        return vector_size

    def get_values_on_nodes_ordered_by_nodes(self, index):
        cur_graph = self.graphs_list[index]
        nodes_and_values = cur_graph.nodes()
        values = [value for node_name, value in nodes_and_values]
        return values

    def normalize_adjacency(self, gnx, node_order):
        D = self.degree_matrix(gnx, nodelist=node_order)
        A = nx.adjacency_matrix(gnx).todense()
        D__minus_sqrt = np.matrix(np.reciprocal(np.sqrt(D)))
        if self.adjacency_normalization == "symmetric_adjacency":
            # D^(-0.5) * (A + A.T + I) * D^(-0.5)
            adjacency = D__minus_sqrt * np.matrix(A + A.T + np.identity(A.shape[0])) * D__minus_sqrt
        elif self.adjacency_normalization == "just_A":
            return A
        return adjacency

    # calculate degree matrix
    def degree_matrix(self, gnx, nodelist):
        degrees = gnx.degree(gnx.nodes)
        return np.diag([degrees[d] for d in nodelist])

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
