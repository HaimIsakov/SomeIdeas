from collections import OrderedDict
from lol_graph import *


# Directed Lol Graph Wrapper
class DLGW:
    def __init__(self, weighted=True):
        self.lol_directed = LolGraph(directed=True, weighted=weighted)
        self.reversed_lol = LolGraph(directed=True, weighted=weighted)

    def convert(self, graph):
        self.lol_directed.convert(graph)
        self.reversed_lol.convert(self.reverse_edges(graph))

    def reverse_edges(self, graph):
        if self.reversed_lol.is_weighted():
            graph = [[edge[1], edge[0], edge[2]] for edge in graph]
        else:
            graph = [[edge[1], edge[0]] for edge in graph]
        return graph

    def is_directed(self):
        return self.lol_directed.is_directed()

    def is_weighted(self):
        return self.lol_directed.is_weighted()

    def number_of_edges(self):
        return self.lol_directed.number_of_edges()

    def number_of_nodes(self):
        return self.lol_directed.number_of_nodes()

    def copy(self):
        new_lol = DLGW()
        new_lol.lol_directed = self.lol_directed.copy()
        new_lol.reversed_lol = self.reversed_lol.copy()
        return new_lol

    def out_degree(self, node):
        return self.lol_directed.out_degree(node)

    def binary_search(self, arr, x):
        return self.lol_directed.binary_search(arr, x)

    def nodes_binary_search(self, arr, x):
        return self.lol_directed.nodes_binary_search(arr, x)

    def in_degree(self, node):
        return self.reversed_lol.out_degree(node)

    def predecessors(self, node):
        if self.reversed_lol.is_weighted():
            neighbors_list, weights_list = self.reversed_lol.neighbors(node)
        else:
            neighbors_list = self.reversed_lol.neighbors(node)
        return neighbors_list

    def nodes(self):
        return self.lol_directed.nodes()

    def edges(self):
        return self.lol_directed.edges()

    def is_edge_between_nodes(self, node1, node2):
        return self.lol_directed.is_edge_between_nodes(node1, node2)

    def size(self):
        return self.lol_directed.size()

    def get_edge_data(self, node1, node2, default=None):
        return self.lol_directed.get_edge_data(node1, node2, default)

    # convert back to [[5,1,0.1],[2,3,3],[5,3,0.2],[4,5,9]] format using self dicts
    def convert_back(self):
        return self.lol_directed.convert_back()

    # get neighbors of specific node n
    def neighbors(self, node):
        return self.lol_directed.neighbors(node)

    # get neighbors and weights for every node
    def graph_adjacency(self):
        return self.lol_directed.graph_adjacency()

    def add_edges(self, edges):
        return self.lol_directed.add_edges(edges)

    def swap_edge(self, edge_to_delete, edge_to_add):
        return self.lol_directed.swap_edge(edge_to_delete, edge_to_add)
