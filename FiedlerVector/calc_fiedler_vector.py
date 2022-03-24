import networkx as nx
import numpy as np
from numpy import linalg as LA


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
