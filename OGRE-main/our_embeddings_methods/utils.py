"""
Helper Functions for Static methods implementations.
"""


import networkx as nx
import time


def user_print(item, user_wish):
    """
    a function to show the user the state of the code. If you want a live update of the current state of the code and
    some details: set user wish to True else False
    """
    if user_wish is True:
        print(item, sep=' ', end='', flush=True)
        time.sleep(3)
        print(" ", end='\r')


def get_initial_proj_nodes_by_degrees(G, number):
    """
    Function to decide which nodes would be in the initial embedding by highest degree.
    :param G: Our graph
    :param number: Controls number of nodes in the initial projection
    :return: A list of the nodes that are in the initial projection
    """
    nodes = list(G.nodes())
    # a dictionary of the nodes and their degrees
    dict_degrees = dict(G.degree(G.nodes()))
    # a dictionary of the nodes and the average degrees
    dict_avg_neighbor_deg = nx.average_neighbor_degree(G)
    # sort the dictionary
    sort_degrees = sorted(dict_degrees.items(), key=lambda pw: (pw[1], pw[0]))  # list
    sort_degrees.reverse()
    new_dict_degrees = {}
    for i in range(len(sort_degrees)):
        new_dict_degrees.update({sort_degrees[i][0]: i})
    sort_avg_n_d = sorted(dict_avg_neighbor_deg.items(), key=lambda pw: (pw[1], pw[0]))  # list
    sort_avg_n_d.reverse()
    new_dict_avg_degrees = {}
    for i in range(len(sort_avg_n_d)):
        new_dict_avg_degrees.update({sort_avg_n_d[i][0]: i})
    new_dict = {}
    for node in nodes:
        new_dict.update({node: new_dict_degrees[node] + new_dict_avg_degrees[node]})
    x = {k: v for k, v in sorted(new_dict.items(), key=lambda item: item[1])}
    initial_nodes = []
    keys = list(x.keys())
    for i in range(number):
        initial_nodes.append(keys[i])
    return initial_nodes


def get_initial_proj_nodes_by_k_core(G, number):
    """
    Function to decide which nodes would be in the initial embedding by k-core score.
    :param G: Our graph
    :param number: Controls number of nodes in the initial projection
    :return: A list of the nodes that are in the initial projection
    """
    G.remove_edges_from(G.selfloop_edges())
    core_dict = nx.core_number(G)
    sorted_core_dict = {k: v for k, v in sorted(core_dict.items(), key=lambda item: item[1], reverse=True)}
    keys = list(sorted_core_dict.keys())
    chosen_nodes = keys[:number]
    return chosen_nodes


def create_sub_G(proj_nodes, G):
    """
    Creating a new graph from the final_nodes- so we can do the node2vec projection on it
    :param: proj_nodes: List of nodes in the initial embedding
    :param: G: Our networkx graph
    """
    sub_G = G.subgraph(list(proj_nodes))
    return sub_G


def create_dict_neighbors(G):
    """
    Create a dictionary where value==node and key==set_of_neighbors.
    :param: G: Our graph
    :return: Dictionary of neighbours as described above.
    """
    G_nodes = list(G.nodes())
    neighbors_dict = {}
    for i in range(len(G_nodes)):
        node = G_nodes[i]
        neighbors_dict.update({node: set(G[node])})
    return neighbors_dict


def create_dicts_of_connections(set_proj_nodes, set_no_proj_nodes, neighbors_dict):
    """
    a function that creates 3 dictionaries:
    1. dict_node_node (explained below)
    2. dict_node_enode (explained below)
    2. dict_enode_enode (explained below)
    :param: set_proj_nodes: Set of nodes that are in the embedding
    :param: set_no_proj_nodes: Set of nodes that are not in the embedding
    :param: Neighbors_dict: Dictionary where value==node and key==set_of_neighbors
    :return: 3 nodes that were mentioned before
    """
    # value == (node that isn't in the embedding), key == (set of its neighbours that are also not in the embedding)
    dict_node_node = {}
    # value == (node that isn't in the embedding), key == (set of neighbours thar are in the embedding)
    dict_node_enode = {}
    # key==(node that is in the projection and has neighbors in it), value==(set of neighbors that are in projection)
    dict_enode_enode = {}
    # nodes that are not in the projection
    list_no_proj = list(set_no_proj_nodes)
    list_proj = list(set_proj_nodes)
    for i in range(len(list_no_proj)):
        node = list_no_proj[i]
        # neighbors of the node that aren't in the projection
        set1 = neighbors_dict[node].intersection(set_no_proj_nodes)
        dict_node_node.update({node: set1})
        # neighbors of the node that are in the projection
        set2 = neighbors_dict[node].intersection(set_proj_nodes)
        if len(set2) > 0:
            dict_node_enode.update({node: set2})
    for i in range(len(list_proj)):
        node = list_proj[i]
        # neighbors of the node that are in the projection
        set1 = neighbors_dict[node].intersection(set_proj_nodes)
        if len(set1) > 0:
            dict_enode_enode.update({node: set1})
    return dict_node_node, dict_node_enode, dict_enode_enode


