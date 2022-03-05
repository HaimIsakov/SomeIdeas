import os
from functools import reduce

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from GraphDataset import GraphDataset
from graph_measures.features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
from graph_measures.features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator
from graph_measures.features_algorithms.vertices.closeness_centrality import ClosenessCentralityCalculator
from graph_measures.features_algorithms.vertices.clustering_coefficient import ClusteringCoefficientCalculator
from graph_measures.features_algorithms.vertices.general import GeneralCalculator
from graph_measures.features_algorithms.vertices.load_centrality import LoadCentralityCalculator
from graph_measures.features_algorithms.vertices.louvain import LouvainCalculator
from graph_measures.features_algorithms.vertices.motifs import MotifsNodeCalculator
from graph_measures.features_infra.feature_calculators import FeatureMeta
from graph_measures.features_infra.graph_features import GraphFeatures
from graph_measures.loggers import PrintLogger
import seaborn as sns
from networkx import closeness_centrality, average_neighbor_degree, core_number, betweenness_centrality
plt.rcParams["font.family"] = "Times New Roman"

import networkx as nx

# def add_nodes_attributes(graphs_list):
#     logger = PrintLogger("MyLogger")
#     for graph in graphs_list:
#         features_meta = {
#             "general": FeatureMeta(GeneralCalculator, {"general"}),
#             "average_neighbor_degree": FeatureMeta(AverageNeighborDegreeCalculator, {"nd_avg"}),
#             "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
#             "closeness_centrality": FeatureMeta(ClosenessCentralityCalculator, {"closeness"}),
#             "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"})
#             # ,"motifs": FeatureMeta(MotifsNodeCalculator, {"motifs"}),
#             # "clustering_coefficient": FeatureMeta(ClusteringCoefficientCalculator, {"clustering"})
#         }
#
#         features = GraphFeatures(graph, features_meta, dir_path="stamdir", logger=logger)
#         # _feature_to_dict
#         features.build()
#         mx_dict = features.to_dict()
#         for node, graph_feature_matrix in mx_dict.items():
#             feature_matrix_0 = graph_feature_matrix.tolist()[0]  # the first row in graph_feature_matrix
#             for ind, feature in enumerate(feature_matrix_0):
#                 cur_feature_name = f"feature{ind}"
#                 graph.nodes[node][cur_feature_name] = feature  # add node attributes

def calc_attributes_by_myself(graphs_list):
    graphs_attributes_mean = {}
    for i, graph in enumerate(graphs_list):
        # edge number
        edges_number = graph.number_of_edges()
        # closeness_centrality
        closeness_centrality_dict = closeness_centrality(graph)
        closeness_centrality_df = pd.DataFrame.from_dict(closeness_centrality_dict, orient='index',
                                                         columns=["closeness_centrality"])
        # degree
        degree_dict = {node: val for (node, val) in graph.degree()}
        degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=["degree"])
        # average_neighbor_degree
        average_neighbor_degree_dict = average_neighbor_degree(graph)
        average_neighbor_degree_df = pd.DataFrame.from_dict(average_neighbor_degree_dict, orient='index',
                                                     columns=["average_neighbor_degree"])
        # core_number
        core_number_dict = core_number(graph)
        core_number_df = pd.DataFrame.from_dict(core_number_dict, orient='index',
                                                        columns=["core_number"])
        # betweenness_centrality
        betweenness_centrality_dict = betweenness_centrality(graph)
        betweenness_centrality_df = pd.DataFrame.from_dict(betweenness_centrality_dict, orient='index',
                                            columns=["betweenness_centrality"])
        graphs_attributes_mean[i] = {"closeness_centrality": closeness_centrality_df.mean(axis=0).values[0],
                                     "degree": degree_df.mean(axis=0).values[0],
                                     "average_neighbor_degree": average_neighbor_degree_df.mean(axis=0).values[0],
                                     "core_number": core_number_df.mean(axis=0).values[0],
                                     "betweenness_centrality": betweenness_centrality_df.mean(axis=0).values[0],
                                     "edges_number": edges_number}

    graphs_attributes_mean_df = pd.DataFrame.from_dict(graphs_attributes_mean, orient='index')
    return graphs_attributes_mean_df

# def merge_df_attributes_labels(graphs_attributes_mean_df_0, graphs_attributes_mean_df_1):
#     merged = pd.merge(graphs_attributes_mean_df_0, graphs_attributes_mean_df_1, left_index=True, right_index=True,
#                       how='inner', suffixes=("_0", "_1"))
#     merged = merged.reindex(sorted(merged.columns), axis=1)
#     return merged

def get_graphs_list(dataset_dict):
    graphs_list = []
    for k, v in dataset_dict.items():
        graphs_list.append(v['graph'])
    return graphs_list

def seperate_graphs_list_according_to_its_label(dataset_dict):
    graphs_list_0, graphs_list_1 = [], []
    for k, v in dataset_dict.items():
        graph = v['graph']
        label = v['label']
        if label == 0:
            graphs_list_0.append(graph)
        else:
            graphs_list_1.append(graph)
    return graphs_list_0, graphs_list_1

def plot_comparison_between_graph_and_labels(graphs_attributes_mean_df_0, graphs_attributes_mean_df_1, save_fig):
    a, b = graphs_attributes_mean_df_0.shape
    fig, axs = plt.subplots(b, 2, figsize=(25,25))
    ind = [(i, j) for i in range(b) for j in range(2)]
    p = 0
    for col in list(graphs_attributes_mean_df_0.columns):
        # for ax1
        ax1 = axs[ind[p]]
        x = graphs_attributes_mean_df_0[col]
        mu = x.mean()
        median = np.median(x)
        sigma = x.std()
        textstr = '\n'.join((
            r'$\mu=%.4f$' % (mu, ),
            r'$\mathrm{median}=%.4f$' % (median, ),
            r'$\sigma=%.4f$' % (sigma, )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.70, 0.95, textstr, transform=ax1.transAxes, fontsize=15, verticalalignment='top', bbox=props)
        sns.histplot(graphs_attributes_mean_df_0[col], stat='probability', ax=ax1)
        ax1.set_title(col+"_0", size=18)
        ax1.set(xlabel=None)
        # for ax2
        ax2 = axs[ind[p + 1]]
        x = graphs_attributes_mean_df_1[col]
        mu = x.mean()
        median = np.median(x)
        sigma = x.std()

        textstr = '\n'.join((
            r'$\mu=%.4f$' % (mu, ),
            r'$\mathrm{median}=%.4f$' % (median, ),
            r'$\sigma=%.4f$' % (sigma, )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.70, 0.95, textstr, transform=ax2.transAxes, fontsize=15, verticalalignment='top', bbox=props)
        sns.histplot(graphs_attributes_mean_df_1[col], stat='probability', ax=ax2)
        ax2.set_title(col+"_1", size=18)
        ax2.set(xlabel=None)

        p += 2
    plt.tight_layout()
    plt.savefig(save_fig + ".pdf")
    plt.show()


if __name__ == "__main__":
    dataset_name = "IBD"
    origin_dir = "split_datasets_new"
    train_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                        f'train_val_set_{dataset_name}_microbiome.csv')
    train_tag_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                       f'train_val_set_{dataset_name}_tags.csv')

    test_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                       f'test_set_{dataset_name}_microbiome.csv')
    test_tag_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                      f'test_set_{dataset_name}_tags.csv')

    add_attributes, geometric_mode = False, False
    data_path = train_data_file_path
    label_path = train_tag_file_path
    mission = "just_values"
    cur_dataset = GraphDataset(data_path, label_path, mission, add_attributes, geometric_mode)
    cur_dataset.update_graphs()
    graphs_list_0, graphs_list_1 = seperate_graphs_list_according_to_its_label(cur_dataset.dataset_dict)
    graphs_attributes_mean_df_0 = calc_attributes_by_myself(graphs_list_0)
    # graphs_attributes_mean_df_0.to_csv("graphs_attributes_mean_df_0.csv")
    graphs_attributes_mean_df_1 = calc_attributes_by_myself(graphs_list_1)
    # graphs_attributes_mean_df_1.to_csv("graphs_attributes_mean_df_1.csv")

    # graphs_attributes_mean_df_0 = pd.read_csv("graphs_attributes_mean_df_0.csv", index_col=0)
    # graphs_attributes_mean_df_1 = pd.read_csv("graphs_attributes_mean_df_1.csv", index_col=0)
    save_fig = "merged"
    plot_comparison_between_graph_and_labels(graphs_attributes_mean_df_0, graphs_attributes_mean_df_1, save_fig)
    # add_nodes_attributes(graphs_list)
    x = 1
