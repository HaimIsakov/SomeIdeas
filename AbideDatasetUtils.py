import networkx as nx
import numpy as np
from nilearn import connectome

Threshold = 0.2
def load_connectivity_origin(input_matrix):
    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    conn_array = conn_measure.fit_transform([input_matrix])[0]
    conn_array = np.delete(conn_array, 82, axis=0)
    conn_array = np.delete(conn_array, 82, axis=1)
    # np.fill_diagonal(conn_array, 0)  # Since the diagonal has 1 correlation value
    network = conn_array
    return network


def load_connectivity_binary(input_matrix):
    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    conn_array = conn_measure.fit_transform([input_matrix])[0]
    conn_array = np.delete(conn_array, 82, axis=0)
    conn_array = np.delete(conn_array, 82, axis=1)
    network = conn_array

    new_network = (np.abs(network) > Threshold).astype(int)
    # np.fill_diagonal(new_network, 0)  # Since the diagonal has 1 correlation value
    return new_network

def transform_adj_mat(adj_matrix):
    new_network = (np.abs(adj_matrix) > Threshold).astype(int)
    np.fill_diagonal(new_network, 0)
    return new_network

def calc_first_eigen_vector(corr_mat):
    corr_mat_copy = corr_mat.copy()
    eig_vals, eig_vecs = np.linalg.eig(corr_mat_copy)
    eig_vals, eig_vecs = eig_vals.real, eig_vecs.real
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    first_eigen_vector = eig_pairs[0][1]
    first_eigen_vector = np.expand_dims(first_eigen_vector, axis=1)
    return first_eigen_vector

def calc_sum_abs_corr(corr_mat):
    corr_mat_copy = corr_mat.copy()
    corr_mat_copy = np.absolute(corr_mat_copy)
    sum_of_correlations = corr_mat_copy.sum(axis=1)
    sum_of_correlations = np.expand_dims(sum_of_correlations, axis=1)
    return sum_of_correlations

def calc_avg_degree(networks_dict):
    thresholds = [.1, .2, .3, .4, .5, .6]
    avg_degree_thresh_dict = {}
    for thresh in thresholds:
        avg_degree_thresh_dict[thresh] = []
        for subject, network in networks_dict.items():
            new_network = (np.abs(network) > thresh).astype(int)
            a, b = new_network.shape
            nodes_degree_vec = new_network.sum(axis=1)
            avg_degree = nodes_degree_vec.sum()/a
            avg_degree_thresh_dict[thresh].append(avg_degree)

    for thresh in thresholds:
        avg_degree_all_graphs = sum(avg_degree_thresh_dict[thresh]) / len(networks_dict)
        print(f"Threshold is {thresh} and the average degree in all graphs is {avg_degree_all_graphs}")
