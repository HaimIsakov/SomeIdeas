import os
import sys
import numpy as np
import pandas as pd
from colorama import Fore
from matplotlib import pyplot as plt
from tqdm import tqdm

for path_name in [os.path.join(os.path.dirname(__file__)),
                  os.path.join(os.path.dirname(__file__), 'Data'),
                  os.path.join(os.path.dirname(__file__), 'Missions')]:
    sys.path.append(path_name)


from GraphDataset import GraphDataset
# from graph_measures.features_algorithms.vertices.average_neighbor_degree import AverageNeighborDegreeCalculator
# from graph_measures.features_algorithms.vertices.betweenness_centrality import BetweennessCentralityCalculator
# from graph_measures.features_algorithms.vertices.closeness_centrality import ClosenessCentralityCalculator
# from graph_measures.features_algorithms.vertices.clustering_coefficient import ClusteringCoefficientCalculator
# from graph_measures.features_algorithms.vertices.general import GeneralCalculator
# from graph_measures.features_algorithms.vertices.load_centrality import LoadCentralityCalculator
# from graph_measures.features_algorithms.vertices.louvain import LouvainCalculator
# from graph_measures.features_algorithms.vertices.motifs import MotifsNodeCalculator
# from graph_measures.features_infra.feature_calculators import FeatureMeta
# from graph_measures.features_infra.graph_features import GraphFeatures
# from graph_measures.loggers import PrintLogger
import seaborn as sns
from networkx import closeness_centrality, average_neighbor_degree, core_number, betweenness_centrality

from TcrDataset import TCRDataset

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

def calc_attributes_by_myself_degree_only(graphs_list):
    graphs_attributes_mean = {}
    for i, graph in enumerate(graphs_list):
        # degree
        degree_dict = {node: val for (node, val) in graph.degree()}
        degree_df = pd.DataFrame.from_dict(degree_dict, orient='index', columns=["degree"])
        graphs_attributes_mean[i] = {"degree": degree_df.mean(axis=0).values[0]}
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



def arrange_new_tcr_dataset():
    def load_or_create_tcr_network(subject_list, adj_mat):
        networks_dict = {}
        values_dict = {}
        for i, subject in tqdm(enumerate(subject_list), desc='Create TCR Networks', total=len(subject_list),
                               bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.LIGHTGREEN_EX, Fore.RESET)):
            # file_path = os.path.join(self.data_path, f"final_{subject}.csv")
            # tcr_sample_df = pd.read_csv(file_path, index_col=0)
            file_path = os.path.join("TCR_Dataset2", "Train", subject + ".csv")
            samples_df = pd.read_csv(file_path, usecols=["combined", "frequency"])
            no_rep_sample_df = samples_df.groupby("combined").sum()  # sum the repetitions
            golden_tcrs = set(list(adj_mat.index))
            cur_sample_tcrs = set(list(no_rep_sample_df.index))
            intersec_golden_and_sample_tcrs = list(golden_tcrs & cur_sample_tcrs)
            network = adj_mat.copy(deep=True)
            for tcr in list(golden_tcrs):
                if tcr not in intersec_golden_and_sample_tcrs:
                    network[tcr] = np.zeros(network.shape[1])
            for tcr in intersec_golden_and_sample_tcrs:
                network.loc[tcr] = network[tcr]
            networks_dict[subject] = network
            no_rep_sample_df['frequency'] = np.log(no_rep_sample_df['frequency'] + 1e-300)
            tcr_sample_dict = {}
            for tcr in golden_tcrs:
                if tcr in intersec_golden_and_sample_tcrs:
                    tcr_sample_dict[tcr] = no_rep_sample_df.loc[tcr]['frequency']
                else:
                    tcr_sample_dict[tcr] = 0
            values_list = []
            # To make all values among all samples the same order
            for tcr in golden_tcrs:
                values_list.append(tcr_sample_dict[tcr])
            values_dict[subject] = values_list
        return networks_dict, values_dict

    train_data_file_path = os.path.join("TCR_Dataset2", "Train")
    train_tag_file_path = os.path.join("TCR_dataset", "samples.csv")

    label_df = pd.read_csv(train_tag_file_path, usecols=["sample", 'status'])
    label_df["sample"] = label_df["sample"] + "_" + label_df['status']
    label_df.set_index("sample", inplace=True)
    label_df["status"] = label_df["status"].map({"negative": 0, "positive": 1})
    label_dict = label_df.to_dict()['status']




def plot_comparison_between_graph_and_labels(graphs_attributes_mean_df_0, graphs_attributes_mean_df_1, save_fig):
    a, b = graphs_attributes_mean_df_0.shape
    fig, axs = plt.subplots(b, 1, figsize=(25, 25))
    ind = [(i, j) for i in range(b) for j in [0]]
    p = 0
    for col in list(graphs_attributes_mean_df_0.columns):
        # for ax1
        ax1 = axs[p]
        x_0 = graphs_attributes_mean_df_0[col]
        mu_0 = x_0.mean()
        median_0 = np.median(x_0)
        sigma_0 = x_0.std()

        x_1 = graphs_attributes_mean_df_1[col]
        mu_1 = x_1.mean()
        median_1 = np.median(x_1)
        sigma_1 = x_1.std()
        textstr = '\n'.join((
            r'$\mu_0=%.4f$' % (mu_0, ),
            r'$\mu_1=%.4f$' % (mu_1,),
            r'$\sigma_0=%.4f$' % (sigma_0,),
            r'$\sigma_1=%.4f$' % (sigma_1,),
            r'$\mathrm{median}_0=%.4f$' % (median_0, ),
            r'$\mathrm{median}_1=%.4f$' % (median_1,)))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.88, 0.95, textstr, transform=ax1.transAxes, fontsize=15, verticalalignment='top', bbox=props)
        sns.histplot(x_0, stat='probability', ax=ax1, color="k")
        sns.histplot(x_1, stat='probability', ax=ax1, color="y")

        ax1.set_title(col, size=18)
        ax1.set(xlabel=None)
        # for ax2
        # ax2 = axs[ind[p + 1]]
        # x = graphs_attributes_mean_df_1[col]
        # mu = x.mean()
        # median = np.median(x)
        # sigma = x.std()
        #
        # textstr = '\n'.join((
        #     r'$\mu=%.4f$' % (mu, ),
        #     r'$\mathrm{median}=%.4f$' % (median, ),
        #     r'$\sigma=%.4f$' % (sigma, )))
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # ax2.text(0.70, 0.95, textstr, transform=ax2.transAxes, fontsize=15, verticalalignment='top', bbox=props)
        # sns.histplot(graphs_attributes_mean_df_1[col], stat='probability', ax=ax2)
        # ax2.set_title(col+"_1", size=18)
        # ax2.set(xlabel=None)
        p += 1
    plt.tight_layout()
    plt.savefig(save_fig + ".jpeg")
    plt.show()

def plot_comparison_between_graph_and_labels_all_datasets(datasets, save_fig):
    fig, axs = plt.subplots(len(datasets), 1, figsize=(25, 25))
    p = 0

    for dataset_name in datasets.keys():
        graphs_attributes_mean_df_0, graphs_attributes_mean_df_1 = load_microbiome_dataset(dataset_name)
        # ind = [(i, j) for i in range(len_datasets) for j in [0]]
        for col in list(graphs_attributes_mean_df_0.columns):
            # for ax1
            print(p)
            ax1 = axs[p]
            x_0 = graphs_attributes_mean_df_0[col]
            mu_0 = x_0.mean()
            median_0 = np.median(x_0)
            sigma_0 = x_0.sem()

            x_1 = graphs_attributes_mean_df_1[col]
            mu_1 = x_1.mean()
            median_1 = np.median(x_1)
            sigma_1 = x_1.sem()
            # textstr = '\n'.join((
            #     r'$\mu_0=%.4f$' % (mu_0, ),
            #     r'$\mu_1=%.4f$' % (mu_1,),
            #     r'$\sigma_0=%.4f$' % (sigma_0,),
            #     r'$\sigma_1=%.4f$' % (sigma_1,)))
            #     # r'$\mathrm{median}_0=%.4f$' % (median_0, ),
            #     # r'$\mathrm{median}_1=%.4f$' % (median_1,)))
            textstr = '\n'.join((f"Class 0: {mu_0:.4f} ± {sigma_0:.4f}", f"Class 1: {mu_1:.4f} ± {sigma_1:.4f}"))

            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax1.text(0.82, 0.95, textstr, transform=ax1.transAxes, fontsize=25, verticalalignment='top', bbox=props)
            sns.histplot(x_0, stat='probability', ax=ax1, color="k", binrange=(0,1), bins=8)
            sns.histplot(x_1, stat='probability', ax=ax1, color="y", binrange=(0,1), bins=8)
            ax1.tick_params(axis='both', which='major', labelsize=25)

            ax1.set_title(datasets[dataset_name], size=30)
            ax1.set(xlabel=None)
            ax1.set(ylabel=None)

            p += 1
    plt.tight_layout()
    plt.savefig(save_fig + ".jpeg")
    plt.show()


def get_tcr_files():
    train_data_file_path = os.path.join("TCR_Dataset2", "Train")
    train_tag_file_path = os.path.join("TCR_dataset", "samples.csv")
    test_data_file_path = os.path.join("TCR_Dataset2", "Test")
    test_tag_file_path = os.path.join("TCR_dataset", "samples.csv")
    label_df = pd.read_csv(train_tag_file_path)
    label_df["sample"] = label_df["sample"] + "_" + label_df['status']
    label_df.set_index("sample", inplace=True)
    train_subject_list = list(label_df[label_df["test/train"] == "train"].index)
    test_subject_list = list(label_df[label_df["test/train"] == "test"].index)
    return train_data_file_path, train_tag_file_path, train_subject_list

def get_ISB_files():
    train_val_test_data_file_path = os.path.join("covid", "new_ISB")
    train_val_test_label_file_path = os.path.join("covid", "ISB_samples.csv")
    label_df = pd.read_csv(train_val_test_label_file_path)
    label_df["sample"] = label_df["sample"] + "_" + label_df['status']
    label_df.set_index("sample", inplace=True)
    subject_list = list(label_df.index)
    return train_val_test_data_file_path, train_val_test_label_file_path, subject_list

def get_NIH_files():
    train_val_test_data_file_path = os.path.join("covid", "new_NIH")
    train_val_test_label_file_path = os.path.join("covid", "NIH_samples.csv")
    label_df = pd.read_csv(train_val_test_label_file_path)
    label_df["sample"] = label_df["sample"] + "_" + label_df['status']
    label_df.set_index("sample", inplace=True)
    subject_list = list(label_df.index)
    return train_val_test_data_file_path, train_val_test_label_file_path, subject_list

def load_microbiome_dataset(dataset_name):
    origin_dir = os.path.join("Data", "split_datasets_new")
    train_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                        f'train_val_set_{dataset_name}_microbiome.csv')
    train_tag_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                       f'train_val_set_{dataset_name}_tags.csv')

    test_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                       f'test_set_{dataset_name}_microbiome.csv')
    test_tag_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
                                      f'test_set_{dataset_name}_tags.csv')
    data_path = train_data_file_path
    label_path = train_tag_file_path

    add_attributes, geometric_mode = False, False
    mission = "just_values"
    cur_dataset = GraphDataset(data_path, label_path, mission, add_attributes, geometric_mode)

    cur_dataset.update_graphs()
    graphs_list_0, graphs_list_1 = seperate_graphs_list_according_to_its_label(cur_dataset.dataset_dict)
    graphs_attributes_mean_df_0 = calc_attributes_by_myself_degree_only(graphs_list_0)
    graphs_attributes_mean_df_1 = calc_attributes_by_myself_degree_only(graphs_list_1)
    return graphs_attributes_mean_df_0, graphs_attributes_mean_df_1


if __name__ == "__main__":
    # dataset_name = "bw"
    # origin_dir = os.path.join("Data", "split_datasets_new")
    # train_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
    #                                     f'train_val_set_{dataset_name}_microbiome.csv')
    # train_tag_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
    #                                    f'train_val_set_{dataset_name}_tags.csv')
    #
    # test_data_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
    #                                    f'test_set_{dataset_name}_microbiome.csv')
    # test_tag_file_path = os.path.join(origin_dir, f'{dataset_name}_split_dataset',
    #                                   f'test_set_{dataset_name}_tags.csv')
    #
    # # train_data_file_path = os.path.join("covid", f"new_{dataset_name}")
    # # train_tag_file_path = os.path.join("covid", f"{dataset_name}_samples.csv")
    #
    # data_path = train_data_file_path
    # label_path = train_tag_file_path
    # # data_path, label_path, subject_list = get_tcr_files()
    # # data_path, label_path, subject_list = get_ISB_files()
    # # data_path, label_path, subject_list = get_NIH_files()
    #
    # add_attributes, geometric_mode = False, False
    # mission = "just_values"
    # cur_dataset = GraphDataset(data_path, label_path, mission, add_attributes, geometric_mode)
    # # label_df = pd.read_csv(train_tag_file_path)
    # # label_df["sample"] = label_df["sample"] + "_" + label_df['status']
    # # label_df.set_index("sample", inplace=True)
    # # subject_list = list(label_df.index)
    #
    # # cur_dataset = TCRDataset(dataset_name, data_path, label_path, subject_list, mission)
    # # adj_mat_path = "dist_mat_with_sample_size_135_run_number_0"
    # # cur_dataset.calc_golden_tcrs(adj_mat_path=adj_mat_path)
    # cur_dataset.update_graphs()
    # graphs_list_0, graphs_list_1 = seperate_graphs_list_according_to_its_label(cur_dataset.dataset_dict)
    # graphs_attributes_mean_df_0 = calc_attributes_by_myself(graphs_list_0)
    # # graphs_attributes_mean_df_0.to_csv("graphs_attributes_mean_df_0.csv")
    # graphs_attributes_mean_df_1 = calc_attributes_by_myself(graphs_list_1)
    # # graphs_attributes_mean_df_1.to_csv("graphs_attributes_mean_df_1.csv")
    #
    # # graphs_attributes_mean_df_0 = pd.read_csv("graphs_attributes_mean_df_0.csv", index_col=0)
    # # graphs_attributes_mean_df_1 = pd.read_csv("graphs_attributes_mean_df_1.csv", index_col=0)
    # save_fig = f"{dataset_name}_comparison_graphs_attributes"
    # plot_comparison_between_graph_and_labels(graphs_attributes_mean_df_0, graphs_attributes_mean_df_1, save_fig)
    # # add_nodes_attributes(graphs_list)

    datasets = ["Cirrhosis", "IBD", "bw", "IBD_Chrone", "male_female", "nugent", "milk", "nut", "peanut"]
    datasets_dict = {"Cirrhosis":"Cirrhosis", "IBD":"IBD", "bw":"CA", "IBD_Chrone":"IBD_Chrone",
                     "male_female":"Male Female", "nugent":"Nugent", "milk":"Milk", "nut":"Nut", "peanut":"Peanut"}

    # datasets = ["Cirrhosis", "IBD"]
    # datasets_dict = {"Cirrhosis":"Cirrhosis", "IBD":"IBD"}

    save_fig = "degree_distribution_compare"
    plot_comparison_between_graph_and_labels_all_datasets(datasets_dict, save_fig)