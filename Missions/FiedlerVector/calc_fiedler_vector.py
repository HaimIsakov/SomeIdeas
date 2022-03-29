from random import random

import networkx as nx
import numpy as np
import pandas as pd
from networkx import fast_gnp_random_graph
from numpy import linalg as LA
from copy import deepcopy

def calc_lap_mat(graph):
    lap = nx.laplacian_matrix(graph).todense()
    return lap


def calc_eigen_vectors(mat):
    w1, v1 = LA.eig(mat)
    return v1


def exclude_real_part(vector):
    real_part = np.real(vector)
    img_part = np.imag(vector)
    # print("All number is vector are real numbers:", not np.any(img_part))
    return real_part


def return_k_first_eigen_vectors(graph, k=1):
    lap = calc_lap_mat(graph)
    v1 = calc_eigen_vectors(lap)
    k_first_eigen_vectors = v1.T[:k]  # Take the k-first columns
    k_first_eigen_vectors = exclude_real_part(k_first_eigen_vectors).T
    # print(k_first_eigen_vectors)
    return k_first_eigen_vectors


def check_fiedler_vector():
    # In this section, we are goint to check Fiedler vector projection. We suspect that the fiedler vector can be mirror
    # image, and thus randomly return +eigen_vector or -eigen_vector. Hence, we will generate a graph, and each time
    # the number of edges will be reduced in 10%, so that the graphs will be quite the same.
    n = 200
    p = 0.5
    random_graph = fast_gnp_random_graph(n, p)
    fiedler_vectors_matrix = []
    fiedler_vector = return_k_first_eigen_vectors(random_graph, k=1)
    fiedler_vectors_matrix.append(fiedler_vector)
    num_experiments = 50
    p_remove = 0.4
    for i in range(num_experiments):
        print(i)
        cur_random_graph = deepcopy(random_graph)
        # remove an edge with a probability of p_remove
        for edge in cur_random_graph.edges():
            if random() < p_remove:
                # print("Number of edges before remove", cur_random_graph.number_of_edges())
                cur_random_graph.remove_edge(*edge)
                # print("Number of edges after", cur_random_graph.number_of_edges())

        print("Number of edges", cur_random_graph.number_of_edges())
        cur_fiedler_vector = return_k_first_eigen_vectors(cur_random_graph, k=1)
        fiedler_vectors_matrix.append(cur_fiedler_vector)
    fiedler_vectors_df = pd.DataFrame(data=np.squeeze(fiedler_vectors_matrix))
    fiedler_vectors_corr_mat = fiedler_vectors_df.corr()
    print(fiedler_vectors_corr_mat)
    return fiedler_vectors_corr_mat


if __name__ == "__main__":
    fiedler_vectors_corr_mat = check_fiedler_vector()
