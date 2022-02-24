import pickle
import random

import matplotlib.pyplot as plt

# from modules import *
# from data import *
import numpy as np
import pandas as pd
import scipy
from scipy.stats import ortho_group
import stellargraph
from sklearn import model_selection, preprocessing
from sklearn.decomposition import PCA
from stellargraph import StellarGraph

MU_FACTOR = 0.1
EPSILON_FACTOR = 0.3
z_features_dim = 5
x_features_dim = 4
alpha = 0.8
p0 = 0.001
PERCENT_EDGES_LOSS = 0.3
def create_hyper_dict():
    global z_features_dim
    global x_features_dim
    global alpha
    global p0
    global data_size
    global MU_FACTOR
    global EPSILON_FACTOR


    data_size = 5000
    MU_FACTOR = 0.1
    EPSILON_FACTOR = 0.3

def create_real_features(feature_dim, diff_Mu_factor=MU_FACTOR):
    m = ortho_group.rvs(dim=feature_dim)
    orthogonal_mat = np.matmul(m.T, m)
    sigma_vec = np.abs(np.random.randn(feature_dim))
    covariance_matrix = np.matmul(np.transpose(orthogonal_mat), np.matmul(np.diag(sigma_vec), orthogonal_mat))

    Mu1 = np.array([random.randint(-10, 10) for _ in range(feature_dim)])
    Mu2 = np.array([np.random.normal(Mu1[i] + diff_Mu_factor*sigma_vec[i], 1/20*sigma_vec[i], 1)[0] for i in range(feature_dim)])

    class1 = np.random.multivariate_normal(Mu1, covariance_matrix, size=int(data_size))
    class2 = np.random.multivariate_normal(Mu2, covariance_matrix, size=int(data_size))

    return class1, class2, sigma_vec, Mu1, Mu2

def add_noise(features_class0, features_class1, new_dim, sigma_vec, epsilon_factor=EPSILON_FACTOR):
    """
    The function get Zi and return Xi = A*Zi + epsilon
    :param features: original features
    :param new_dim: the dimension of the new features
    :return: Xi
    """
    n_examples = features_class0.shape[0]
    old_dim = features_class0.shape[1]
    A = np.random.rand(new_dim, old_dim)
    epsilon = [np.random.normal(0, sigma*epsilon_factor, n_examples) for sigma in sigma_vec]
    epsilon = np.array(epsilon)
    new_features0 = np.matmul(A, features_class0.T + epsilon)
    new_features1 = np.matmul(A, features_class1.T + epsilon)
    return new_features0.T, new_features1.T


def remove_dimentions(G, node_subjects, model):
    # get the features for the nodes
    graph_nodes = node_subjects.index
    graph_nodes_features = G.node_features(graph_nodes)

    # create new indexing
    node_ind = {graph_nodes[i]: i for i in range(len(graph_nodes))}

    min_max_scaler = preprocessing.MinMaxScaler()
    graph_nodes_features = min_max_scaler.fit_transform(graph_nodes_features)

    # remove dimentions:
    nodes_features = model.fit_transform(graph_nodes_features)

    nodes_features = scipy.stats.zscore(nodes_features, axis=1)


    edges = G.edges()
    relevnat_edges = [edge for edge in edges if edge[0] in graph_nodes and edge[1] in graph_nodes]

    first_node = list(map(lambda x: x[0], relevnat_edges))
    # node_not_in_graph = [x for x in first_node if x not in fulfilled_graph_nodes])]
    second_node = list(map(lambda x: x[1], relevnat_edges))

    first_node = [node_ind[first_node[i]] for i in range(len(first_node))]
    second_node = [node_ind[second_node[i]] for i in range(len(second_node))]
    square_node_data = pd.DataFrame(nodes_features, columns=list(range(nodes_features.shape[1])))
    square_edges = pd.DataFrame({"source": first_node, "target": second_node})

    new_ind = [node_ind[node_subjects.index[i]] for i in range(len(node_subjects))]
    node_subjects = node_subjects.set_axis(new_ind)

    return StellarGraph(square_node_data, square_edges), node_subjects, node_ind

def get_graphs(G, node_subjects, dataset_size, n_dimensions):
    fulfilled_graph_subjects, deficiency_graph_subjects = model_selection.train_test_split(
        node_subjects, train_size=round(dataset_size * (1 - PERCENT_EDGES_LOSS)), test_size=None, stratify=node_subjects
    )

    # create 2 graphs and remove n_dimensions:
    pca = PCA(n_components=n_dimensions)
    # dict from node to node index:
    graph1, fulfilled_graph_subjects, node_dict1 = remove_dimentions(G, fulfilled_graph_subjects, pca)
    graph2, deficiency_graph_subjects, node_dict2 = remove_dimentions(G, deficiency_graph_subjects, pca)
    return graph1, fulfilled_graph_subjects

def create_graph(features, class_1_ind, class_2_ind, alpha_prob, prob_to_edge):
    """

    :param features:
    :param ind: [class_1_ind, class_2_ind]
    :param class2_ind:
    :return:
    """
    edge_possibility_same_class = alpha_prob * prob_to_edge
    edge_possibility_diff_class = (1-alpha_prob) * prob_to_edge

    square_node_data = pd.DataFrame(features, columns=list(range(features.shape[1])))

    first_node = []
    second_node = []

    for node in class_1_ind:
        class_1 = np.random.uniform(0, 1, len(class_1_ind))
        has_edge = np.where(class_1 < edge_possibility_same_class)[0]
        first_node = first_node + [node]*len(has_edge)
        second_node = second_node + list(has_edge)

        class_2 = np.random.uniform(0, 1, len(class_2_ind))
        has_edge = np.where(class_2 < edge_possibility_diff_class)[0] + len(class_1)
        first_node = first_node + [node]*len(has_edge)
        second_node = second_node + list(has_edge)

    for node in class_2_ind:
        class_1 = np.random.uniform(0, 1, len(class_1_ind))
        has_edge = np.where(class_1 < edge_possibility_diff_class)[0]
        first_node = first_node + [node]*len(has_edge)
        second_node = second_node + list(has_edge)

        class_2 = np.random.uniform(0, 1, len(class_2_ind))
        has_edge = np.where(class_2 < edge_possibility_same_class)[0] + len(class_1)
        first_node = first_node + [node]*len(has_edge)
        second_node = second_node + list(has_edge)

    square_edges = pd.DataFrame({"source": first_node, "target": second_node})
    return stellargraph.StellarGraph(square_node_data, square_edges)


def baseline_wrapper(fulfilled_graph_subjects, train, graph1, validation, deficiency_graph_subjects, graph2):
    n_classes = len(np.unique(fulfilled_graph_subjects.array))
    baseline.create_hyper_dict(n_classes)

    train_nodes = train.index
    train_features = graph1.node_features(train_nodes)

    validation_nodes = validation.index
    validation_features = graph1.node_features(validation_nodes)

    partial_graph_nodes = deficiency_graph_subjects.index
    partial_graph_nodes_features = graph2.node_features(partial_graph_nodes)

    second_model, f1_macro_model = train_and_test_FC(train_features,
                                     train.array, validation_features, validation.array)
    return f1_macro_model

def plot_f1_macro(grid, f1_score):
    noise_vec = [x[0] for x in grid if x[0] < 2]
    Mu_diff_vec = [x[1] for x in grid if x[1] < 2]

    X, Y = np.meshgrid(noise_vec, Mu_diff_vec)
    Z = np.ndarray([len(noise_vec), len(Mu_diff_vec)])
    for i, item in enumerate(zip(noise_vec, Mu_diff_vec)):
        noise, Mu_diff = item
        noise_ind = noise_vec.index(noise)
        Mu_ind = Mu_diff_vec.index(Mu_diff)
        Z[noise_ind, Mu_ind] = f1_score[i]
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    label_size = 8
    matplotlib.rcParams['xtick.labelsize'] = label_size
    ax.view_init(12, 50)
    ax.set_title('F1 macro VS Noise and Mu distance')
    ax.set_xlabel('Noise Factor', fontsize=8)
    ax.set_ylabel('Mu Distance Facror', fontsize=8)
    ax.set_zlabel(r'F1 Macro After 10 Epochs', fontsize=8)
    plt.savefig("plots/toy_model_factors_2")
    plt.show()

def grid_baseline():
    noise_factor = np.arange(0.1, 4, 0.2)
    Mu_factor = np.arange(0.1, 4, 0.2)
    mesh = np.array(np.meshgrid(noise_factor, Mu_factor))
    grid = mesh.T.reshape(-1, 2)
    grid = tuple(map(tuple, grid))
    f1_score = []
    for noise, Mu_diff in grid:
        real_features, class1ind, class2ind, sigma_vector, Mu1, Mu2 = create_real_features(data_size, z_features_dim, Mu_diff)
        noisy_features = add_noise(real_features, x_features_dim, sigma_vector, noise)
        labels = [0 if ind in class1ind else 1 for ind in range(data_size)]
        my_graph = create_graph(noisy_features, class1ind, class2ind)
        graph_subsets = pd.Series(data=labels, index=class1ind+class2ind)
        graph1, fulfilled_graph_subjects, graph2, deficiency_graph_subjects, train, validation = get_graphs(my_graph, graph_subsets, data_size, noisy_features.shape[1])
        # find baseline
        f1_score.append(baseline_wrapper(fulfilled_graph_subjects, train, graph1, validation, deficiency_graph_subjects, graph2))
    file = open("toy_model_f1macro", "wb")
    pickle.dump([grid, f1_score], file)

def build_all_model(mu_facor, epsilon_factor, z_dim, x_dim, alpha, p0):
    real_features_class0, real_features_class1, sigma_vector, Mu1, Mu2 = create_real_features(z_dim, mu_facor)
    noisy_features0, noisy_features1 = add_noise(real_features_class0, real_features_class1, x_dim, sigma_vector, epsilon_factor)
    noisy_features = np.concatenate((noisy_features0, noisy_features1))
    labels = [0] * data_size + [1] * data_size
    class1index = list(range(data_size))
    class2index = list(range(data_size, 2*data_size))
    my_graph = create_graph(noisy_features, class1index, class2index, alpha, p0)
    graph_subsets = pd.Series(data=labels, index=class1index+class2index)
    params = {'sigma_vector': sigma_vector, 'Mu1': Mu1, 'Mu2': Mu2}
    return get_graphs(my_graph, graph_subsets, data_size, noisy_features.shape[1]), params


if __name__ == '__main__':
    create_hyper_dict()
    graphs, params = build_all_model(mu_facor=MU_FACTOR, epsilon_factor=EPSILON_FACTOR,
                                                     z_dim=z_features_dim, x_dim=x_features_dim, alpha=alpha, p0=p0)

    graph1, fulfilled_graph_subjects = graphs
    networkx_graphs = graph1.to_networkx()
    x = 5
    # file = open("datasets/toy_model_ver2", "wb")
    # data = {'graph1': graph1, 'fulfilled_graph_subjects': fulfilled_graph_subjects, 'graph2': graph2, 'deficiency_graph_subjects':deficiency_graph_subjects, 'train':train, 'validation':validation}
    #
    # pickle.dump({'data': data, 'params': params}, file)
    #
