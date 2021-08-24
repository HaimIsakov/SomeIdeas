import os
import numpy as np
from collections import OrderedDict
import csv
import sys


class LolGraph:

    def __init__(self, directed=True, weighted=True):
        self._index_list = [0]
        self._neighbors_list = []
        self._weights_list = []
        self._map_node_to_number = OrderedDict()
        self._map_number_to_node = OrderedDict()
        self.directed = directed
        self.weighted = weighted

    def is_directed(self):
        return self.directed

    def is_weighted(self):
        return self.weighted

    def number_of_edges(self):
        if self.is_directed():
            return len(self._neighbors_list)
        return len(self._neighbors_list) / 2

    def number_of_nodes(self):
        return len(self._index_list) - 1

    def copy(self):
        new_lol_graph = LolGraph()
        new_lol_graph._index_list = self._index_list.copy()
        new_lol_graph._neighbors_list = self._neighbors_list.copy()
        new_lol_graph._weights_list = self._weights_list.copy()
        new_lol_graph._map_node_to_number = self._map_node_to_number.copy()
        new_lol_graph._map_number_to_node = self._map_number_to_node.copy()
        new_lol_graph.directed = self.directed
        new_lol_graph.weighted = self.weighted
        return new_lol_graph

    # outdegree
    def out_degree(self, node):
        number = self._map_node_to_number[node]
        idx = self._index_list[number]
        idx_end = self._index_list[number + 1]
        if self.is_weighted():
            weights_list = self._weights_list[idx: idx_end]
            return sum(weights_list)
        else:
            return idx_end - idx

    # Iterative Binary Search Function
    # It returns index of x in given array arr if present,
    # else returns -1
    def binary_search(self, arr, x):
        low = 0
        high = len(arr) - 1
        mid = 0

        while low <= high:
            mid = (high + low) // 2

            # Check if x is present at mid
            if arr[mid] < x:
                low = mid + 1

            # If x is greater, ignore left half
            elif arr[mid] > x:
                high = mid - 1

            # If x is smaller, ignore right half
            else:
                return mid

        # If we reach here, then the element was not present
        return -1

    def nodes_binary_search(self, arr, x):
        # num = self._map_node_to_number[x]
        # arr_num = [self._map_node_to_number[node] for node in arr]
        # num_found_index = self.binary_search(arr_num, num)
        # return num_found_index
        if x in arr:
            return arr.index(x)
        return -1

    # # indegree
    # def in_degree(self, node):
    #     sum = 0
    #     for node_from in self.nodes():
    #         if self.is_weighted():
    #             neighbors_list, weights_list = self.neighbors(node_from)
    #         else:
    #             neighbors_list = self.neighbors(node_from)
    #         x = self.nodes_binary_search(neighbors_list, node)
    #         if x != -1:
    #             if self.is_weighted():
    #                 sum += weights_list[x]
    #             else:
    #                 sum += 1
    #     return sum
    #
    # def predecessors(self, node):
    #     nodes_list = []
    #     for node_from in self.nodes():
    #         if self.is_edge_between_nodes(node_from, node):
    #             nodes_list.append(node_from)
    #     return nodes_list

    def nodes(self):
        return list(self._map_node_to_number.keys())

    def edges(self):
        return self.convert_back()

    def is_edge_between_nodes(self, node1, node2):
        number = self._map_node_to_number[node1]
        idx = self._index_list[number]
        idx_end = self._index_list[number + 1]
        return self._map_node_to_number[node2] in self._neighbors_list[idx: idx_end]
        # for neighbor in self._neighbors_list[idx: idx_end]:
        #     if node2 == self._map_number_to_node[neighbor]:
        #         return True
        # return False

    def size(self):
        if self.is_weighted() and not self.is_directed():
            edges = self.edges()
            size = 0
            for edge in edges:
                if edge[0] == edge[1]:
                    size += edge[2]
                else:
                    size += edge[2] / 2
            return size
        if self.is_weighted():
            return sum(self._weights_list)
        if not self.is_directed():
            return len(self._neighbors_list) / 2
        return len(self._neighbors_list)

    def get_edge_data(self, node1, node2, default=None):
        number = self._map_node_to_number[node1]
        idx = self._index_list[number]
        idx_end = self._index_list[number + 1]
        node1_neighbors = self._neighbors_list[idx: idx_end]
        node2_index = self.binary_search(node1_neighbors, self._map_node_to_number[node2])
        if node2_index == -1:
            return default
        if self.is_weighted():
            return {"weight": self._weights_list[idx + node2_index]}
        return {}

    # input: csv file containing edges list, in the form of [[5,1],[2,3],[5,3],[4,5]]
    def convert_with_csv(self, files_name, header=True):
        self._map_node_to_number = OrderedDict()
        graph = []
        for i in range(len(files_name)):
            file = files_name[i]
            with open(file, "r") as csvfile:
                datareader = csv.reader(csvfile)
                if header:
                    next(datareader, None)  # skip the headers
                for edge in datareader:
                    edge[2] = float(edge[2])
                    graph.append(edge)
                csvfile.close()
        self.convert(graph)

    # input: np array of edges, in the form of np array [[5,1,0.1],[2,3,3],[5,3,0.2],[4,5,9]]
    def convert(self, graph):
        free = 0
        '''create dictionary to self._map_node_to_number from edges to our new numbering'''

        for edge in graph:
            if self._map_node_to_number.get(edge[0], None) is None:
                self._map_node_to_number[edge[0]] = free
                free += 1
            if self._map_node_to_number.get(edge[1], None) is None:
                self._map_node_to_number[edge[1]] = free
                free += 1

        '''create the opposite dictionary'''
        self._map_number_to_node = OrderedDict((y, x) for x, y in self._map_node_to_number.items())

        d = OrderedDict()
        '''starting to create the index list. Unordered is important'''
        for idx, edge in enumerate(graph):
            d[self._map_node_to_number[edge[0]]] = d.get(self._map_node_to_number[edge[0]], 0) + 1
            if not self.is_directed():
                if edge[0] != edge[1]:
                    d[self._map_node_to_number[edge[1]]] = d.get(self._map_node_to_number[edge[1]], 0) + 1
            elif self._map_node_to_number[edge[1]] not in d.keys():
                d[self._map_node_to_number[edge[1]]] = 0

        '''transfer the dictionary to list'''
        for j in range(1, len(d.keys()) + 1):
            self._index_list.append(self._index_list[j - 1] + d.get(j - 1, 0))

        '''create the second list'''
        if self.is_directed():
            self._neighbors_list = [-1] * len(graph)
            if self.is_weighted():
                self._weights_list = [0] * len(graph)
        else:
            self._neighbors_list = [-1] * len(graph) * 2
            if self.weighted:
                self._weights_list = [0] * len(graph) * 2

        space = OrderedDict((x, -1) for x in self._map_number_to_node.keys())
        for idx, edge in enumerate(graph):
            left = self._map_node_to_number[edge[0]]
            right = self._map_node_to_number[edge[1]]
            if self.is_weighted():
                weight = float(edge[2])

            if space[left] != -1:
                space[left] += 1
                i = space[left]
            else:
                i = self._index_list[left]
                space[left] = i
            self._neighbors_list[i] = right
            if self.weighted:
                self._weights_list[i] = weight

            if not self.is_directed() and left != right:
                if space[right] != -1:
                    space[right] += 1
                    i = space[right]
                else:
                    i = self._index_list[right]
                    space[right] = i
                self._neighbors_list[i] = left
                if self.is_weighted():
                    self._weights_list[i] = weight
        self._neighbors_list, self._weights_list = self.sort_all(index_list=self._index_list,
                                                                 neighbors_list=self._neighbors_list,
                                                                 weights_list=self._weights_list)

    # convert back to [[5,1,0.1],[2,3,3],[5,3,0.2],[4,5,9]] format using self dicts
    def convert_back(self):
        graph = []
        for number in range(len(self._index_list) - 1):
            node = self._map_number_to_node[number]
            index = self._index_list[number]
            while index < self._index_list[number + 1]:
                to_node = self._map_number_to_node[self._neighbors_list[index]]
                if not self.is_directed() and to_node < node:
                    index += 1
                    continue
                if self.is_weighted():
                    weight = self._weights_list[index]
                    edge = [node, to_node, weight]
                else:
                    edge = [node, to_node]
                graph.append(edge)
                index += 1
        return graph

    # sort the neighbors for each node
    def sort_all(self, index_list=None, neighbors_list=None, weights_list=None):
        for number in range(len(index_list) - 1):
            start = index_list[number]
            end = index_list[number + 1]
            neighbors_list[start: end], weights_list[start: end] = self.sort_neighbors(neighbors_list[start: end], weights_list[start: end])
        return neighbors_list, weights_list

    def sort_neighbors(self, neighbors_list=None, weights_list=None):
        if self.weighted:
            neighbors_weights = {neighbors_list[i]: weights_list[i] for i in range(len(neighbors_list))}
            neighbors_weights = OrderedDict(sorted(neighbors_weights.items()))
            return neighbors_weights.keys(), neighbors_weights.values()
        else:
            return sorted(neighbors_list), weights_list

    # get neighbors of specific node n
    def neighbors(self, node):
        number = self._map_node_to_number[node]
        idx = self._index_list[number]
        idx_end = self._index_list[number+1]
        neighbors_list = [0] * (idx_end-idx)
        if self.is_weighted():
            weights_list = self._weights_list[idx: idx_end]
        for i, neighbor in enumerate(self._neighbors_list[idx: idx_end]):
            neighbors_list[i] = self._map_number_to_node[neighbor]
        if self.is_weighted():
            return neighbors_list, weights_list
        else:
            return neighbors_list

    # get neighbors and weights for every node
    def graph_adjacency(self):
        graph_adjacency_dict = dict()
        for number in range(len(self._index_list) - 1):
            node = self._map_number_to_node[number]
            if node not in graph_adjacency_dict.keys():
                graph_adjacency_dict[node] = dict()

            if self.is_weighted():
                neighbors_list, weights_list = self.neighbors(node)
                for neighbor, weight in zip(neighbors_list, weights_list):
                    graph_adjacency_dict[node][neighbor] = {'weight': weight}
            else:
                neighbors_list = self.neighbors(node)
                for neighbor in neighbors_list:
                    graph_adjacency_dict[node][neighbor] = {'weight': 1}
        return graph_adjacency_dict

    # Add new edges to the graph, but with limitations:
    # For example, if the edge is [w,v] and the graph is diracted, w can't be an existing node.
    # if the graph is not diracted, w and v can't be existing nodes.
    def add_edges(self, edges):
        last_index = self._index_list[-1]
        index_list = [last_index]
        neighbors_list = []
        weights_list = []

        nodes_amount = len(self._map_node_to_number.keys())
        free = nodes_amount
        map_node_to_number = OrderedDict()

        '''create dictionary to self._map_node_to_number from edges to our new numbering'''
        for edge in edges:
            if (self._map_node_to_number.get(edge[0], None) is not None or
                (not self.directed and self._map_node_to_number.get(edge[1], None) is not None)):
                print("Error: add_edges can't add edges from an existing node")
                return
            if map_node_to_number.get(edge[0], None) is None:
                map_node_to_number[edge[0]] = free
                free += 1
            if map_node_to_number.get(edge[1], None) is None and self._map_node_to_number.get(edge[1], None) is None:
                map_node_to_number[edge[1]] = free
                free += 1
        '''create the opposite dictionary'''
        map_number_to_node = OrderedDict((y, x) for x, y in map_node_to_number.items())

        """update the original dicts"""
        self._map_node_to_number.update(map_node_to_number)
        self._map_number_to_node.update(map_number_to_node)

        d = OrderedDict()
        '''starting to create the index list. Unordered is important'''
        for idx, edge in enumerate(edges):
            d[self._map_node_to_number[edge[0]]] = d.get(self._map_node_to_number[edge[0]], 0) + 1
            if not self.directed:
                d[self._map_node_to_number[edge[1]]] = d.get(self._map_node_to_number[edge[1]], 0) + 1
            elif self._map_node_to_number[edge[1]] not in d.keys() and edge[1] in map_node_to_number.keys():
                d[self._map_node_to_number[edge[1]]] = 0

        '''transfer the dictionary to list'''
        for j in range(1, len(d.keys()) + 1):
            index_list.append(index_list[j - 1] + d.get(nodes_amount + j - 1, 0))

        '''create the second list'''
        if self.is_directed():
            neighbors_list = [-1] * len(edges)
        else:
            neighbors_list = [-1] * len(edges) * 2
        if self.is_weighted():
            weights_list = [0] * len(edges)

        space = OrderedDict((x, -1) for x in self._map_number_to_node.keys())
        for idx, edge in enumerate(edges):
            left = self._map_node_to_number[edge[0]]
            right = self._map_node_to_number[edge[1]]
            if self.is_weighted():
                weight = float(edge[2])

            if space[left] != -1:
                space[left] += 1
                i = space[left]
            else:
                i = index_list[left - nodes_amount] - last_index
                space[left] = i
            neighbors_list[i] = right
            if self.is_weighted():
                weights_list[i] = weight

            if not self.is_directed():
                if space[right] != -1:
                    space[right] += 1
                    i = space[right]
                else:
                    i = index_list[right - len(self._index_list) - 1]
                    space[right] = i
                neighbors_list[i] = left
                if self.is_weighted():
                    weights_list[i] = weight

        """sort the neighbors"""
        neighbors_list, weights_list = self.sort_all(index_list=index_list,
                                                     neighbors_list=neighbors_list,
                                                     weights_list=weights_list)

        """update the original dicts"""
        self._index_list += index_list[1:]
        self._neighbors_list += neighbors_list
        self._weights_list += weights_list


    # swap between two edges, but with limitations.
    # For example, the graph can only be directed.
    # The swap can only be in the form of: from edge [a,b] to edge [a, c].
    def swap_edge(self, edge_to_delete, edge_to_add):
        if not self.directed or (self.directed and edge_to_delete[0] != edge_to_add[0]):
            print("Error: swap_edge can only be only on directed graph and from the same node")
            return
        if not self.is_edge_between_nodes(edge_to_delete[0], edge_to_delete[1]):
            print("Error: edge_to_delete was not found")
            return
        number = self._map_node_to_number[edge_to_add[0]]
        to_number = self._map_node_to_number[edge_to_add[1]]
        from_number = self._map_node_to_number[edge_to_delete[1]]
        start_index_of_source = self._index_list[number]
        end_index_of_source = self._index_list[number+1]

        neighbors_list = self._neighbors_list[start_index_of_source: end_index_of_source]
        neighbor_index = self.binary_search(neighbors_list, from_number)
        neighbors_list[neighbor_index] = to_number

        if self.is_weighted():
            weights_list = self._weights_list[start_index_of_source: end_index_of_source]
            weights_list[neighbor_index] = edge_to_add[2]
            neighbor_index, weights_list = self.sort_neighbors(neighbors_list, weights_list)
            self._weights_list[start_index_of_source: end_index_of_source] = weights_list
            # index_of_replacement = start_index_of_source + neighbor_index
            # neighbors_list[neighbor_index] = to_number
            # weights_list[neighbor_index] = edge_to_add[2]
        else:
            neighbor_index, weights_list = self.sort_neighbors(neighbors_list)
        self._neighbors_list[start_index_of_source: end_index_of_source] = neighbor_index

    # get memory usage of the lol object
    def get_memory(self):
        return sum([sys.getsizeof(var) for var in [self._index_list, self._neighbors_list, self._weights_list,
                                                   self._map_node_to_number, self._map_number_to_node,
                                                   self.directed, self.weighted]])





