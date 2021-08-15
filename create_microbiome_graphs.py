from tqdm import tqdm
# from taxonomy_tree import *
from taxonomy_tree_for_pytorch_geometric import *


class CreateMicrobiomeGraphs:
    def __init__(self, df):
        self.microbiome_df = df
        self.graphs_list = []
        # self.create_graphs_with_common_nodes()
        self.create_tax_trees()
        self.sort_all_graphs()
        self.union_nodes = []

    def create_tax_trees(self):
        for i, mom in tqdm(enumerate(self.microbiome_df.iterrows()), desc='Create graphs', total=len(self.microbiome_df)):
            # cur_graph = create_tax_tree(self.microbiome_df.iloc[i], flag=0, keepFlagged=False)
            cur_graph = create_tax_tree(self.microbiome_df.iloc[i], zeroflag=True)
            self.graphs_list.append(cur_graph)
            # print("Number of Nodes in graph", cur_graph.number_of_nodes())

    def find_common_nodes(self):
        nodes_dict = {}
        j = 0
        for graph in self.graphs_list:
            nodes = graph.nodes(data=False)
            for node_name in nodes:
                if node_name not in nodes_dict:
                    nodes_dict[node_name] = j
                    j = j + 1
        return nodes_dict

    # def create_graphs_with_common_nodes(self, union_nodes):
    #     self.union_nodes = union_nodes
    #     # self.create_tax_trees()
    #     # nodes_dict = self.find_common_nodes()
    #     for graph in tqdm(self.graphs_list, desc='Add to graphs the common nodes set'):
    #         nodes = graph.nodes()
    #         # nodes = [node_name for node_name, value in nodes_and_values]
    #         for node_name in union_nodes:
    #             # if there is a node that exists in other graph but not in the current graph we want to add it
    #             if node_name not in nodes:
    #                 graph.add_node(node_name, val=0)
    #
    #     # sort the node, so that every graph has the same order of graph.nodes()
    #     self.sort_all_graphs()

    def create_graphs_with_common_nodes(self):
        # old version of taxonomy tree format
        self.create_tax_trees()
        nodes_dict = self.find_common_nodes()
        for graph in tqdm(self.graphs_list, desc='Add to graphs the common nodes set', total=len(self.graphs_list)):
            nodes_and_values = graph.nodes()
            nodes = [node_name for node_name, value in nodes_and_values]
            for node_name in nodes_dict:
                # if there is a node that exists in other graph but not in the current graph we want to add it
                if node_name not in nodes:
                    graph.add_node((node_name, 0.0))

        # sort the node, so that every graph has the same order of graph.nodes()
        self.sort_all_graphs()

    # def sort_all_graphs(self):
    #     temp_graph_list = []
    #     for graph in self.graphs_list:
    #         temp_graph = nx.Graph()
    #         temp_graph.add_nodes_from(sorted(graph.nodes(data=True), key=lambda tup: tup[0]))
    #         temp_graph.add_edges_from(graph.edges(data=True))
    #         # temp_graph.add_edges_from(sorted(graph.edges(data=True)))
    #         temp_graph_list.append(temp_graph)
    #     self.graphs_list = temp_graph_list

    def sort_all_graphs(self):
        temp_graph_list = []
        for graph in self.graphs_list:
            temp_graph = nx.Graph()
            temp_graph.add_nodes_from(sorted(graph.nodes(data=True)))
            temp_graph.add_edges_from(graph.edges(data=False))
            # temp_graph.add_edges_from(sorted(graph.edges(data=True)))
            temp_graph_list.append(temp_graph)
        self.graphs_list = temp_graph_list

    def get_graphs_list(self, index):
        return self.graphs_list[index]

    def get_vector_size(self):
        nodes_number = []
        return_number = 0
        for g in self.graphs_list:
            nodes_number.append(g.number_of_nodes())
        if all(x == nodes_number[0] for x in nodes_number):
            return_number = nodes_number[0]
        else:
            return_number = len(self.union_nodes)
        return return_number

    def get_values_on_nodes_ordered_by_nodes(self, gnx):
        nodes_and_values = gnx.nodes(data=True)
        values = [value_dict['val'] for node, value_dict in nodes_and_values]
        return values

    # def get_values_on_nodes_ordered_by_nodes(self, gnx):
    #     nodes_and_values = gnx.nodes()
    #     values = [value for node_name, value in nodes_and_values]
    #     return values

