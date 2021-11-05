"""
Implementation of DOGRE / WOGRE approach.
Notice projection==embedding.
"""

import numpy as np
from sklearn.linear_model import Ridge
import time
from state_of_the_art.state_of_the_art_embedding import final
from math import pow
from utils import get_initial_proj_nodes_by_k_core, get_initial_proj_nodes_by_degrees
import heapq
import networkx as nx


def user_print(item, user_wish):
    """
    A function to show the user the state of the our_embeddings_methods. If you want a live update of the current state
    of code and some details: set user wish to True else False
    """
    if user_wish is True:
        print(item, sep=' ', end='', flush=True)
        time.sleep(3)
        print(" ", end='\r')


def create_sub_G(proj_nodes, G):
    """
    Creating a new graph from the nodes in the initial embedding so we can do the initial embedding on it
    :param proj_nodes: The nodes in the initial embedding
    :param G: Our graph
    :return: A sub graph of G that its nodes are the nodes in the initial embedding.
    """
    sub_G = G.subgraph(list(proj_nodes))
    return sub_G


def create_dict_neighbors(G):
    """
    Create a dictionary of neighbors.
    :param G: Our graph
    :return: neighbors_dict where key==node and value==set_of_neighbors (both incoming and outgoing)
    """
    G_nodes = list(G.nodes())
    neighbors_dict = {}
    for i in range(len(G_nodes)):
        node = G_nodes[i]
        neighbors_dict.update({node: set(G[node])})
    return neighbors_dict


def create_dicts_same_nodes(my_set, neighbors_dict, node, dict_out, dict_in):
    """
    A function to create useful dictionaries to represent connections between nodes that have the same type, i.e between
    nodes that are in the embedding and between nodes that aren't in the embedding. It depends on the input.
    :param my_set: Set of the nodes that aren't currently in the embedding OR Set of the nodes that are currently in
            the embedding
    :param neighbors_dict: Dictionary of all nodes and neighbors (both incoming and outgoing)
    :param node: Current node
    :param dict_out: explained below
    :param dict_in: explained below
    :return: There are 4 possibilities (2 versions, 2 to every version):
            A) 1. dict_node_node_out: key == nodes not in embedding, value == set of outgoing nodes not in embedding
                 (i.e there is a directed edge (i,j) when i is the key node and j isn't in the embedding)
               2. dict_node_node_in: key == nodes not in embedding , value == set of incoming nodes not in embedding
                 (i.e there is a directed edge (j,i) when i is the key node and j isn't in the embedding)
            B) 1. dict_enode_enode_out: key == nodes in embedding , value == set of outgoing nodes in embedding
                 (i.e there is a directed edge (i,j) when i is the key node and j is in the embedding)
               2. dict_enode_enode_in: key == nodes in embedding , value == set of incoming nodes in embedding
                 (i.e there is a directed edge (j,i) when i is the key node and j is in the embedding)
    """
    set1 = neighbors_dict[node].intersection(my_set)
    count_1 = 0
    count_2 = 0
    if (len(set1)) > 0:
        count_1 += 1
        dict_out.update({node: set1})
        neigh = list(set1)
        for j in range(len(neigh)):
            if dict_in.get(neigh[j]) is None:
                dict_in.update({neigh[j]: set([node])})
            else:
                dict_in[neigh[j]].update(set([node]))
    else:
        count_2 += 1
    return dict_out, dict_in


def create_dict_node_enode(set_proj_nodes, neighbors_dict, H, node, dict_node_enode, dict_enode_node):
    """
    A function to create useful dictionaries to represent connections between nodes that are in the embedding and
    nodes that are not in the embedding.
    :param set_proj_nodes: Set of the nodes that are in the embedding
    :param neighbors_dict: Dictionary of all nodes and neighbors (both incoming and outgoing)
    :param H:  H is the undirected version of our graph
    :param node: Current node
    :param dict_node_enode: explained below
    :param dict_enode_node: explained below
    :return: 1. dict_node_enode: key == nodes not in embedding, value == set of outdoing nodes in embedding (i.e
                    there is a directed edge (i,j) when i is the key node and j is in the embedding)
             2. dict_enode_node: key == nodes not in embedding, value == set of incoming nodes in embedding (i.e
                    there is a directed edge (j,i) when i is the key node and j is in the embedding)
    """
    set2 = neighbors_dict[node].intersection(set_proj_nodes)
    set_all = set(H[node]).intersection(set_proj_nodes)
    set_in = set_all - set2
    if len(set2) > 0:
        dict_node_enode.update({node: set2})
    if len(set_in) > 0:
        dict_enode_node.update({node: set_in})
    return dict_node_enode, dict_enode_node


def create_dicts_of_connections(set_proj_nodes, set_no_proj_nodes, neighbors_dict, G):
    """
     A function that creates 6 dictionaries of connections between different types of nodes.
    :param set_proj_nodes: Set of the nodes that are in the projection
    :param set_no_proj_nodes: Set of the nodes that aren't in the projection
    :param neighbors_dict: Dictionary of neighbours
    :return: 6 dictionaries, explained above (in the two former functions)
    """
    dict_node_node_out = {}
    dict_node_node_in = {}
    dict_node_enode = {}
    dict_enode_node = {}
    dict_enode_enode_out = {}
    dict_enode_enode_in = {}
    list_no_proj = list(set_no_proj_nodes)
    list_proj = list(set_proj_nodes)
    H = G.to_undirected()
    for i in range(len(list_no_proj)):
        node = list_no_proj[i]
        dict_node_node_out, dict_node_node_in = create_dicts_same_nodes(set_no_proj_nodes, neighbors_dict, node,
                                                                        dict_node_node_out, dict_node_node_in)
        dict_node_enode, dict_enode_node = create_dict_node_enode(set_proj_nodes, neighbors_dict, H, node,
                                                                  dict_node_enode, dict_enode_node)
    for i in range(len(list_proj)):
        node = list_proj[i]
        dict_enode_enode_out, dict_enode_enode_in = create_dicts_same_nodes(set_proj_nodes, neighbors_dict, node,
                                                                            dict_enode_enode_out, dict_enode_enode_in)
    return dict_node_node_out, dict_node_node_in, dict_node_enode, dict_enode_node, dict_enode_enode_out, dict_enode_enode_in


def calculate_average_projection_second_order(dict_proj, node, dict_enode_enode, average_two_order_proj, dim, G):
    """
    A function to calculate the average embeddings of the second order neighbors, both outgoing and incoming,
    depends on the input.
    :param dict_proj: Dict of embeddings (key==node, value==its embedding)
    :param node: Current node we're dealing with
    :param dict_enode_enode: key==node in embedding , value==set of neighbors that are in the embedding. Direction
            (i.e outgoing or incoming depends on the input)
    :param average_two_order_proj: explained below
    :return: Average embedding of second order neighbours, outgoing or incoming.
    """
    two_order_neighs = dict_enode_enode.get(node)
    k2 = 0
    # if the neighbors in the projection also have neighbors in the projection calculate the average projection
    if two_order_neighs is not None:
        two_order_neighs_in = list(two_order_neighs)
        k2 += len(two_order_neighs_in)
        two_order_projs = []
        for i in range(len(two_order_neighs_in)):
            if G[node].get(two_order_neighs_in[i]) is not None:
                two_order_proj = G[node][two_order_neighs_in[i]]["weight"]*dict_proj[two_order_neighs_in[i]]

            else:
                two_order_proj = G[two_order_neighs_in[i]][node]["weight"]*dict_proj[two_order_neighs_in[i]]
            two_order_projs.append(two_order_proj)
        two_order_projs = np.array(two_order_projs)
        two_order_projs = np.mean(two_order_projs, axis=0)
    # else, the average projection is 0
    else:
        two_order_projs = np.zeros(dim)
    average_two_order_proj.append(two_order_projs)
    return average_two_order_proj, k2


def calculate_projection_of_neighbors(current_node, proj_nodes, dict_proj, dict_enode_enode_in, dict_enode_enode_out, dim, G):
    """
    A function to calculate average degree of first order neighbors and second order neighbors, direction
    (outgoing or incoming) depends on the input.
    :param proj_nodes: Neighbors that are in the embedding, direction depends on the input.
    :param dict_proj: Dict of embeddings (key==node, value==embedding)
    :return: Average degree of first order neighbors and second order neighbors
    """
    proj = []
    # average projections of the two order neighbors, both incoming and outgoing
    average_two_order_proj_in = []
    average_two_order_proj_out = []
    list_proj_nodes = list(proj_nodes)
    # the number of first order neighbors
    k1 = len(proj_nodes)
    # to calculate the number of the second order neighbors
    k2_in = 0
    k2_out = 0
    # to calculate the average projection of the second order neighbors
    for k in range(len(list_proj_nodes)):
        node = list_proj_nodes[k]
        average_two_order_proj_in, a = calculate_average_projection_second_order(dict_proj, node, dict_enode_enode_in,
                                                                              average_two_order_proj_in, dim, G)
        k2_in += a
        average_two_order_proj_out, b = calculate_average_projection_second_order(dict_proj, node, dict_enode_enode_out,
                                                                               average_two_order_proj_out, dim, G)
        k2_out += b
        proj.append(dict_proj[node])
        if G[current_node].get(node) is not None:
            proj.append(G[current_node][node]["weight"] * dict_proj[node])
        else:
            proj.append(G[node][current_node]["weight"] * dict_proj[node])
    # for every neighbor we have the average projection of its neighbors, so now do average on all of them
    average_two_order_proj_in = np.array(average_two_order_proj_in)
    average_two_order_proj_in = np.mean(average_two_order_proj_in, axis=0)
    average_two_order_proj_out = np.array(average_two_order_proj_out)
    average_two_order_proj_out = np.mean(average_two_order_proj_out, axis=0)
    proj = np.array(proj)
    # find the average proj
    proj = np.mean(proj, axis=0)
    return proj, average_two_order_proj_in, average_two_order_proj_out, k1, k2_in, k2_out


def calculate_projection(current_node, G, proj_nodes_in, proj_nodes_out, dict_proj, dict_enode_enode_in, dict_enode_enode_out, dim,
                         alpha1, alpha2, beta_11, beta_12, beta_21, beta_22):
    """
    A function to calculate the final embedding of the node by D-VERSE method as explained in the pdf file in github.
    :param proj_nodes_in: embeddings of first order incoming neighbors.
    :param proj_nodes_out: embeddings of first order outgoing neighbors.
    :param dict_proj: Dict of embeddings (key==node, value==projection)
    :param dim: Dimension of the embedding
    :param alpha1, alpha2, beta_11, beta_12, beta_21, beta_22: Parameters to calculate the final projection.
    :return: The final projection of our node.
    """
    if len(proj_nodes_in) > 0:
        x_1, z_11, z_12, k1, k2_in_in, k2_in_out = calculate_projection_of_neighbors(current_node,
            proj_nodes_in, dict_proj, dict_enode_enode_in, dict_enode_enode_out, dim, G)
    else:
        x_1, z_11, z_12 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    if len(proj_nodes_out) > 0:
        x_2, z_21, z_22, k1, k2_in_out, k2_out_out = calculate_projection_of_neighbors(current_node,
            proj_nodes_out, dict_proj, dict_enode_enode_in, dict_enode_enode_out, dim, G)
    else:
        x_2, z_21, z_22 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    # the final projection of the node
    final_proj = alpha1*x_1+alpha2*x_2 - beta_11*z_11 - beta_12*z_12 - \
                 beta_21*z_21 - beta_22*z_22
    return final_proj


def calculate_projection_weighted(current_node, G, proj_nodes_in, proj_nodes_out, dict_proj, dict_enode_enode_in, dict_enode_enode_out,
                                  dim, params):
    """
    A function to calculate the final embedding of the node by We-VERSE method as explained in the pdf file in github.
    :param proj_nodes_in: embeddings of first order incoming neighbors.
    :param proj_nodes_out: embeddings of first order outgoing neighbors.
    :param dict_proj: Dict of embeddings (key==node, value==projection)
    :param dict_enode_enode_in: key == nodes in embedding , value == set of incoming nodes in embedding
                 (i.e there is a directed edge (j,i) when i is the key node and j is in the embedding)
    :param dict_enode_enode_out: key == nodes in embedding , value == set of outgoing nodes in embedding
                 (i.e there is a directed edge (i,j) when i is the key node and j is in the embedding)
    :param dim: Dimension of the embedding space
    :param params: Optimal parameters to calculate the embedding
    :return: The final projection of our node.
    """
    if len(proj_nodes_in) > 0:
        x_1, z_11, z_12, k1_in, k2_in_in, k2_in_out = calculate_projection_of_neighbors(current_node,
            proj_nodes_in, dict_proj, dict_enode_enode_in, dict_enode_enode_out, dim, G)
    else:
        x_1, z_11, z_12 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
        k1_in = 0
        k2_in_in = 0
        k2_in_out = 0
    if len(proj_nodes_out) > 0:
        x_2, z_21, z_22, k1_out, k2_out_in, k2_out_out = calculate_projection_of_neighbors(current_node,
            proj_nodes_out, dict_proj, dict_enode_enode_in, dict_enode_enode_out, dim, G)
    else:
        x_2, z_21, z_22 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
        k1_out = 0
        k2_out_in = 0
        k2_out_out = 0
    # the final projection of the node
    proj_1 = params[0]*x_1 + params[1]*x_1*k1_in + params[2]*x_1*pow(k1_in, 2)
    proj_2 = params[3] * x_2 + params[4] * x_2 * k1_out + params[5] * x_2 * pow(k1_out, 2)
    proj_3 = params[6]*z_11 + params[7]*z_11*k2_in_in + params[8]*z_11*pow(k2_in_in, 2)
    proj_4 = params[9]*z_12 + params[10]*z_12*k2_in_out + params[11]*z_12*pow(k2_in_out, 2)
    proj_5 = params[12]*z_21 + params[13]*z_21*k2_out_in + params[14]*z_21*pow(k2_out_in, 2)
    proj_6 = params[15]*z_22 + params[16]*z_22*k2_out_out + params[17]*z_22*pow(k2_out_out, 2)
    final_proj = proj_1 + proj_2 + proj_3 + proj_4 + proj_5 + proj_6
    return final_proj


def first_changes(dict_1, dict_2, dict_3, node):
    """
    Technical changes need to be done after adding the node to the projection
    :param dict_1: dict_node_enode OR dict_enode_node
    :param dict_2: dict_enode_enode_out OR dict_enode_enode_in
    :param dict_3: dict_enode_enode_in OR dict_enode_enode_out
    :param node: Current node
    :return: Dicts after changes
    """
    if dict_1.get(node) is not None:
        enode = dict_1[node]
        dict_1.pop(node)
        dict_2.update({node: enode})
        enode = list(enode)
        for i in range(len(enode)):
            out_i = enode[i]
            if dict_3.get(out_i) is not None:
                dict_3[out_i].update(set([node]))
            else:
                dict_3.update({out_i: set([node])})
    return dict_1, dict_2, dict_3


def second_changes(dict_1, dict_2, dict_3, node):
    """
    Technical changes need to be done after adding the node to the projection
    :param dict_1: dict_node_node_out OR dict_node_node_in
    :param dict_2: dict_node_node_in OR dict_node_node_out
    :param dict_3: dict_enode_node OR dict_node_enode
    :param node: Current node
    :return: Dicts after changes
    """
    if dict_1.get(node) is not None:
        relevant_n_e = dict_1[node]
        dict_1.pop(node)
        if len(relevant_n_e) > 0:
            # loop of non embd neighbors
            relevant_n_e1 = list(relevant_n_e)
            for j in range(len(relevant_n_e)):
                tmp_append_n_n = dict_2.get(relevant_n_e1[j])
                if tmp_append_n_n is not None:
                    # if relevant_n_e1[j] in dict_node_node:
                    tmp_append_n_n = tmp_append_n_n-set([node])
                    dict_2[relevant_n_e1[j]] = tmp_append_n_n
                tmp_append = dict_3.get(relevant_n_e1[j])
                if tmp_append is not None:
                    # add our node to the set cause now our node is in embd
                    tmp_append.update(set([node]))
                    dict_3[relevant_n_e1[j]] = tmp_append
                else:
                    dict_3.update({relevant_n_e1[j]: set([node])})
    return dict_1, dict_2, dict_3


def one_iteration(params, dict_enode_proj, dict_node_enode, dict_enode_node, dict_node_node_in, dict_node_node_out,
                  dict_enode_enode_in, dict_enode_enode_out, set_n_e, current_node, dim, G):
    """
    One iteration of the final function. We calculate the projection and do necessary changes.
    Notice: All paranms dicts are explained above
    :param params: Best parameters to calculate a node's projection, explained in the git.
    :param set_n_e: Set of nodes that aren't in the projection
    :param current_node: The node we're dealing with at the moment.
    :param dim: The dimension of the projection
    :return: The dicts and the set of nodes not in projection because they are changed. Also return condition to tell
    if we still need to do iterations.
    """
    condition = 1
    if dict_node_enode.get(current_node) is not None:
        embd_neigh_out = dict_node_enode[current_node]
    else:
        embd_neigh_out = set()
    if dict_enode_node.get(current_node) is not None:
        embd_neigh_in = dict_enode_node[current_node]
    else:
        embd_neigh_in = set()
    if params.size < 10:
        # the final projection of the node
        final_proj = calculate_projection(current_node, G, embd_neigh_in, embd_neigh_out, dict_enode_proj, dict_enode_enode_in, dict_enode_enode_out, dim,
                                          alpha1=params[0], alpha2=params[1], beta_11=params[2],
                                          beta_12=params[3], beta_21=params[4], beta_22=params[5])
    else:
        final_proj = calculate_projection_weighted(current_node, G, embd_neigh_in, embd_neigh_out, dict_enode_proj, dict_enode_enode_in,
                                          dict_enode_enode_out, dim, params)

    dict_enode_proj.update({current_node: final_proj})

    # do first changes
    dict_node_enode, dict_enode_enode_out, dict_enode_enode_in = first_changes(
        dict_node_enode, dict_enode_enode_out, dict_enode_enode_in, current_node)
    dict_enode_node, dict_enode_enode_in, dict_enode_enode_out = first_changes(
        dict_enode_node, dict_enode_enode_in, dict_enode_enode_out, current_node)

    # do second changes
    dict_node_node_out, dict_node_node_in, dict_enode_node = second_changes(
        dict_node_node_out, dict_node_node_in, dict_enode_node, current_node)
    dict_node_node_in, dict_node_node_out, dict_node_enode = second_changes(
        dict_node_node_in, dict_node_node_out, dict_node_enode, current_node)

    # remove the node from the set of nodes that aren't in the projection
    set_n_e.remove(current_node)

    return condition, dict_enode_proj, dict_node_enode, dict_enode_node, dict_node_node_out, dict_node_node_in,\
           dict_enode_enode_out, dict_enode_enode_in, set_n_e


def final_function(pre_params, dict_enode_proj, dict_node_enode, dict_enode_node, dict_node_node_out, dict_node_node_in,
                   dict_enode_enode_out, dict_enode_enode_in, set_n_e, batch_precent, dim, G):
    """
    The final function that iteratively divided the dictionary of nodes without embedding into number of batches
    determined by batch_precent. It does by building a heap every iteration so that we enter the nodes to the
    projection from the nodes which have the most neighbors in the embedding to the least. This way the projection
    gets more accurate.
    """
    condition = 1
    k = 0
    set_n_e2 = set_n_e.copy()
    while condition > 0:
        condition = 0
        k += 1
        #print(k)
        batch_size = int(batch_precent * len(set_n_e2))
        if batch_size > len(set_n_e):
            num_times = len(set_n_e)
        else:
            num_times = batch_size
        list_n_e = list(set_n_e)
        heap = []
        dict_node_enode_all = dict_node_enode.copy()
        keys = list(dict_enode_node.keys())
        for key in keys:
            if dict_node_enode.get(key) is None:
                dict_node_enode_all.update({key: dict_enode_node[key]})
            else:
                dict_node_enode_all[key].update(dict_enode_node[key])
        for i in range(len(list_n_e)):
            my_node = list_n_e[i]
            a = dict_node_enode_all.get(my_node)
            if a is not None:
                num_neighbors = len(dict_node_enode_all[my_node])
            else:
                num_neighbors = 0
            heapq.heappush(heap, [-num_neighbors, my_node])
        for i in range(len(set_n_e))[:num_times]:
            # look on node number i in the loop
            current_node = heapq.heappop(heap)[1]
            if dict_node_enode_all.get(current_node) is not None:
                condition, dict_enode_proj, dict_node_enode, dict_enode_node, dict_node_node_out, dict_node_node_in, \
                dict_enode_enode_out, dict_enode_enode_in, set_n_e = \
                    one_iteration(pre_params, dict_enode_proj, dict_node_enode, dict_enode_node,
                                  dict_node_node_in, dict_node_node_out,
                  dict_enode_enode_in, dict_enode_enode_out, set_n_e, current_node, dim, G)
    return dict_enode_proj, set_n_e


def crate_data_for_linear_regression(initial_proj_nodes, dict_intial_projections, dict_enode_enode_in,
                                     dict_enode_enode_out, dim, G):
    """
    In order to find the best parameters to calculate a node's vector, we perform linear regression. In this function
    we prepare the data for this. Here we create it by D-VERSE method.
    :param initial_proj_nodes: The nodes that are in the initial projection
    :param dict_intial_projections: Dictionary of nodes and their projection
    :param dict_enode_enode_in: key == nodes in projection , value == set of incoming nodes in projection
                 (i.e there is a directed edge (j,i) when i is the key node and j is in the projection)
    :param dict_enode_enode_out: key == nodes in projection , value == set of outgoing nodes in projection
                (i.e there is a directed edge (i,j) when i is the key node and j is in the projection)
    :param dim: The dimension of the projection
    :return: Data for linear regression
    """
    nodes = []
    dict_node_params = {}
    for i in range(len(initial_proj_nodes)):
        node = initial_proj_nodes[i]
        if dict_enode_enode_in.get(node) is not None:
            x_1, z_11, z_12, k1, k2, k2_in_in = calculate_projection_of_neighbors(node,
                dict_enode_enode_in[node], dict_intial_projections, dict_enode_enode_in, dict_enode_enode_out, dim, G)
        else:
            x_1, z_11, z_12 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
        if dict_enode_enode_out.get(node) is not None:
            x_2, z_21, z_22, k1, k2, k2_in_in = calculate_projection_of_neighbors(node,
                dict_enode_enode_out[node], dict_intial_projections, dict_enode_enode_in, dict_enode_enode_out, dim, G)
        else:
            x_2, z_21, z_22 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
        a = np.zeros(dim * 6)
        b = np.concatenate((x_1, x_2, z_11, z_12, z_21, z_22))
        if np.array_equal(a, b) is False:
            nodes.append(node)
            X = np.column_stack((x_1, x_2, z_11, z_12, z_21, z_22, np.ones(dim)))
            dict_node_params.update({node: X})
        else:
            a = 1
    return dict_node_params, nodes


def crate_data_for_linear_regression_weighted(initial_proj_nodes, dict_intial_projections, dict_enode_enode_in,
                                              dict_enode_enode_out, dim, G):
    """
    In order to find the best parameters to calculate a node's vector, we perform linear regression. In this function
    we prepare the data for this. Here we create it by We-VERSE method.
    :param initial_proj_nodes: The nodes that are in the initial projection
    :param dict_intial_projections: Dictionary of nodes and their projection
    :param dict_enode_enode_in: key == nodes in projection , value == set of incoming nodes in projection
                 (i.e there is a directed edge (j,i) when i is the key node and j is in the projection)
    :param dict_enode_enode_out: key == nodes in projection , value == set of outgoing nodes in projection
                (i.e there is a directed edge (i,j) when i is the key node and j is in the projection)
    :param dim: The dimension of the projection
    :return: Data for linear regression
    """
    nodes = []
    dict_node_params = {}
    for i in range(len(initial_proj_nodes)):
        node = initial_proj_nodes[i]
        if dict_enode_enode_in.get(node) is not None:
            x_1, z_11, z_12, k1_in, k2_in_in, k2_in_out = calculate_projection_of_neighbors(node,
                dict_enode_enode_in[node], dict_intial_projections, dict_enode_enode_in, dict_enode_enode_out, dim, G)
        else:
            x_1, z_11, z_12 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
            k1_in = 0
            k2_in_in = 0
            k2_in_out = 0
        if dict_enode_enode_out.get(node) is not None:
            x_2, z_21, z_22, k1_out, k2_out_in, k2_out_out = calculate_projection_of_neighbors(node,
                dict_enode_enode_out[node], dict_intial_projections, dict_enode_enode_in, dict_enode_enode_out, dim, G)
        else:
            x_2, z_21, z_22 = np.zeros(dim), np.zeros(dim), np.zeros(dim)
            k1_out = 0
            k2_out_in = 0
            k2_out_out = 0
        a = np.zeros(dim * 18)
        x1_all = np.concatenate((x_1, x_1 * k1_in, x_1 * pow(k1_in, 2)))
        x2_all = np.concatenate((x_2, x_2 * k1_out, x_2 * pow(k1_out, 2)))
        z11_all = np.concatenate((z_11, z_11 * k2_in_in, z_11 * pow(k2_in_in, 2)))
        z12_all = np.concatenate((z_12, z_12 * k2_in_out, z_12 * pow(k2_in_out, 2)))
        z21_all = np.concatenate((z_21, z_21 * k2_out_in, z_21 * pow(k2_out_in, 2)))
        z22_all = np.concatenate((z_22, z_22 * k2_out_out, z_22 * pow(k2_out_out, 2)))
        b = np.concatenate((x1_all, x2_all, z11_all, z12_all, z21_all, z22_all))
        if np.array_equal(a, b) is False:
            nodes.append(node)
            X = np.column_stack((x_1, x_1 * k1_in, x_1 * pow(k1_in, 2), x_2, x_2 * k1_out, x_2 * pow(k1_out, 2),
                                 z_11, z_11 * k2_in_in, z_11 * pow(k2_in_in, 2),
                                 z_12, z_12 * k2_in_out, z_12 * pow(k2_in_out, 2),
                                 z_21, z_21 * k2_out_in, z_21 * pow(k2_out_in, 2),
                                 z_22, z_22 * k2_out_out, z_22 * pow(k2_out_out, 2), np.ones(dim)))
            dict_node_params.update({node: X})
    return dict_node_params, nodes


def calculate_weights_degrees_of_initial(sub_G, nodes, dim):
    """
    For weighted regression this function calculates each node degree and return a list of degrees.
    :param sub_G:
    :param nodes:
    :param dim:
    :return:
    """
    d = sub_G.degree(nodes)
    degrees = np.array(list(d.values()))
    all_degrees = np.tile(degrees, dim)
    return all_degrees


def linear_regression(dict_params, nodes, dict_proj, regu_val=0, degrees=None):
    """
    In order to find the best parameters to calculate a node's vector, we perform linear regression.
    :param dict_params: key==nodes in initial projection, value==A matrix of size 10*6
    :param nodes: The nodes that are in the initial projection
    :param dict_proj: Dict of projections (key==node, value==projection)
    :param degrees: If weighted regression is wanted, then degrees is a list degrees of the nodes that are in the
            initial embedding. Else, None.
    :param regu_val: Regularization value for regression. 0 if not inserted.
    :return: Best parameters to calculate the final projection.
    """
    all_x_list = []
    all_y_list = []
    for node in nodes:
        all_x_list.append(dict_params[node])
        all_y_list.append(dict_proj[node].T)
    X = np.concatenate(all_x_list)
    Y = np.concatenate(all_y_list)
    reg = Ridge(alpha=regu_val)
    if degrees is None:
        reg.fit(X, Y)
    else:
        reg.fit(X, Y, sample_weight=degrees)
    params = reg.coef_
    reg_score = reg.score(X, Y)
    return params, reg_score


def main_D_W_OGRE(name, G, initial_method, method, initial, dim, params, choose="degrees", regu_val=0, weighted_reg=False,
                  file_tags=None, F=None):
    """
    Main Function
    :param name: Name of graph
    :param G: Graph to project
    :param initial_method: State-of-the-art method to project the initial nodes
    :param method: One of our embedding methods - DOGRE / WOGRE
    :param initial: How many initial nodes to project. It is a float representing percentages.
    :param dim: Embedding dimension
    :param params: Dictionary of parameters. Full explanation in state_of_the_art_embedding.py.
    :param choose: How to choose nodes in initial embedding - by highest degree or by highest k-core scores.
            options are: degrees / k-core. Valid is degrees.
    :param regu_val: Regularization value for regression. Default is 0.
    :param weighted_reg: True if regression is weighted by degree, else False. Default is False.
    :return: Dictionary of projections where keys==nodes and values==embeddings
    """
    user_wish = True

    list_dicts = []
    list_initial_proj_nodes = []

    times = []

    for l in initial:
        t = time.time()
        # get the initial projection, number of nodes can be changed, see documentation of the function above
        if F is not None:
            initial_proj_nodes = list(F.nodes())
        else:
            if choose == "degrees":
                initial_proj_nodes = get_initial_proj_nodes_by_degrees(G, l)
            elif choose == "k-core":
                initial_proj_nodes = get_initial_proj_nodes_by_k_core(G, l)
            # if non valid choice is inserted, choose by highest degree
            else:
                initial_proj_nodes = get_initial_proj_nodes_by_degrees(G, l)
        list_initial_proj_nodes.append(initial_proj_nodes)
        print(len(initial_proj_nodes))
        user_print("number of nodes in initial projection is: " + str(len(initial_proj_nodes)), user_wish)
        n = G.number_of_nodes()
        e = G.number_of_edges()
        user_print("number of nodes in graph is: " + str(n), user_wish)
        user_print("number of edges in graph is: " + str(e), user_wish)
        # the nodes of our graph
        G_nodes = list(G.nodes())
        set_G_nodes = set(G_nodes)
        set_proj_nodes = set(initial_proj_nodes)
        G_edges = [list(i) for i in G.edges()]
        user_print("make a sub graph of the embedding nodes, it will take a while...", user_wish)
        # creating sub_G to do node2vec on it later
        if F is None:
            sub_G = create_sub_G(initial_proj_nodes, G)
        else:
            sub_G = F.copy()
        user_print("calculate the projection of the sub graph with {}...".format(initial_method), user_wish)
        # create dictionary of nodes and their projections after running node2vec on the sub graph
        if initial_method == "GF":
            my_iter = params["max_iter"]
            params["max_iter"] = 1500
            _, dict_projections, _ = final(name, sub_G, initial_method, params)
            params["max_iter"] = my_iter
        elif initial_method == "GCN":
            _, dict_projections, _ = final(name, sub_G, initial_method, params, file_tags)
        else:
            _, dict_projections, _ = final(name, sub_G, initial_method, params)
        neighbors_dict = create_dict_neighbors(G)
        set_nodes_no_proj = set_G_nodes - set_proj_nodes
        # create dicts of connections
        dict_node_node_in, dict_node_node_out, dict_node_enode, dict_enode_node, dict_enode_enode_in, dict_enode_enode_out\
            = create_dicts_of_connections(set_proj_nodes, set_nodes_no_proj, neighbors_dict, G)
        # create data for linear regression
        if method == "DOGRE":
            params_estimate_dict, labeled_nodes = crate_data_for_linear_regression(initial_proj_nodes, dict_projections, dict_enode_enode_in,
                                             dict_enode_enode_out, dim, G)
        else:
            params_estimate_dict, labeled_nodes = crate_data_for_linear_regression_weighted(initial_proj_nodes, dict_projections,
                                                                                   dict_enode_enode_in,
                                                                                   dict_enode_enode_out, dim, G)
        # calculate best parameters for the final projection calculation
        if weighted_reg is True:
            degrees = calculate_weights_degrees_of_initial(sub_G, labeled_nodes, dim)
            pre_params, score = linear_regression(params_estimate_dict, labeled_nodes, dict_projections,
                                                  regu_val, degrees)
        else:
            pre_params, score = linear_regression(params_estimate_dict, labeled_nodes, dict_projections, regu_val)

        print(pre_params, score)
        # create the final dictionary of nodes and their dictionaries
        final_dict_projections, set_no_proj = final_function(pre_params, dict_projections,
                                                        dict_node_enode, dict_enode_node, dict_node_node_out,
                                                        dict_node_node_in, dict_enode_enode_out,
                                                        dict_enode_enode_in, set_nodes_no_proj, 0.01, dim, G)

        print("The number of nodes that aren't in the final projection:", len(set_no_proj))
        if len(set_no_proj) != 0:
            set_n_e_sub_g = nx.subgraph(G, list(set_no_proj))
            if initial_method == "GCN":
                _, projections, _ = final(name, set_n_e_sub_g, initial_method, params, file_tags=file_tags)
            elif initial_method == "HOPE":
                if len(set_no_proj) < int(dim/2):
                    params = {"dimension": dim, "walk_length": 80, "num_walks": 16, "workers": 2}
                    _, projections, _ = final(name, set_n_e_sub_g, "node2vec", params)
                else:
                    _, projections, _ = final(name, set_n_e_sub_g, initial_method, params)
            else:
                _, projections, _ = final(name, set_n_e_sub_g, initial_method, params)
        else:
            projections = {}
        z = {**final_dict_projections, **projections}
        print("The number of nodes that are in the final projection:", len(z))
        # to calculate running time
        elapsed_time = time.time() - t
        times.append(elapsed_time)
        print("running time: ", elapsed_time)
        list_dicts.append(z)

    return list_dicts, times, list_initial_proj_nodes
