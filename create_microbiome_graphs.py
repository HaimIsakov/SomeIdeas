import networkx as nx
from tqdm import tqdm
from taxonomy_tree import *


class CreateMicrobiomeGraphs:
    def __init__(self, df):
        self.microbiome_df = df
        self.graphs_list = []
        self.create_graphs_with_common_nodes()

    def __getitem__(self, index):
        return self.graphs_list[index]

    def create_tax_trees(self):
        for i, mom in tqdm(enumerate(self.microbiome_df.iterrows()), desc='Create graphs'):
            cur_graph = create_tax_tree(self.microbiome_df.iloc[i], ignore_values=0, ignore_flag=True)
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
        print(f"Common nodes: {len(self.find_common_nodes())}")
        for graph in self.graphs_list:
            temp_graph = nx.Graph()
            sorted_nodes = sorted(graph.nodes(data=True))
            print(f"Number of Nodes in unsorted graph {len(sorted_nodes)}")
            # temp_graph.add_nodes_from(sorted_nodes)
            for node_value, _ in sorted_nodes:
                node_name = node_value[0]
                value = node_value[1]
                temp_graph.add_node(node_name, frequency=value)
            print(f"Number of Nodes in after sorting graph {temp_graph.number_of_nodes()}")
            temp_graph.add_edges_from(graph.edges(data=True))

            # temp_graph.add_edges_from(sorted(graph.edges(data=True)))
            temp_graph_list.append(temp_graph)
        self.graphs_list = temp_graph_list

    def get_graphs_list(self):
        return self.graphs_list

    def get_vector_size(self):
        nodes_dict = self.find_common_nodes()
        vector_size = len(nodes_dict)
        return vector_size

    def get_values_on_nodes_ordered_by_nodes(self, gnx):
        nodes_and_values = gnx.nodes()
        values = [value for node_name, value in nodes_and_values]
        return values
