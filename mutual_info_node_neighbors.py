import os
import random
import sys

# from Data.GraphDataset import GraphDataset
import numpy as np


for path_name in [os.path.join(os.path.dirname(__file__)),
                  os.path.join(os.path.dirname(__file__), 'Data'),
                  os.path.join(os.path.dirname(__file__), 'Missions')]:
    sys.path.append(path_name)
from GraphDataset import *
from BrainNetwork import *
import pandas as pd
from tqdm import tqdm
from taxonomy_tree_average_sons import *
# from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.metrics import mutual_info_score as MIC
from networkx import all_neighbors
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def calc_mutual_info_for_feature_and_tag(dataset, graphs, labels, feature="value"):
    d = {}
    mutual_information_list, shuffled_mutual_information_list = [],[]
    for graph, label in zip(graphs, labels):
        if feature == "value":
            feature_dict = dict(graph.nodes.data())
        elif feature == "degree":
            feature_dict = dict(graph.degree())
        for ind, node in enumerate(feature_dict.keys()):
            if feature == "value":
                value = feature_dict[node]['val']
            elif feature == "degree":
                value = feature_dict[node]

            if ind not in d:
                d[ind] = ([value], [label])
            else:
                values_list, labels_list = d[ind]
                values_list.append(value)
                labels_list.append(label)

    for node_number, (values_list, labels_list) in d.items():
        X = values_list
        y = labels_list
        mutual_information = MIC(X, y)
        mutual_information_list.append(mutual_information)
        random.shuffle(y)
        shuffled_mutual_information = MIC(X, y)
        shuffled_mutual_information_list.append(shuffled_mutual_information)
        # print(mutual_information)
    # print(dataset, "Mean", np.mean(mutual_information_list))
    # print(dataset, " shuffled Mean", np.mean(shuffled_mutual_information_list))

    return mutual_information_list, shuffled_mutual_information_list


def calc_mutual_info_for_values_and_neighbors_values(dataset, graphs):
    mutual_information_list, shuffled_mutual_information_list = [], []
    for graph in graphs:
        all_edges = graph.edges
        node_neighbor_tuples = []
        all_nodes_and_values = dict(graph.nodes.data())
        for edge in all_edges:
            node1, node2 = edge
            value1, value2 = all_nodes_and_values[node1]['val'], all_nodes_and_values[node2]['val']
            cur_tuple = (value1, value2)
            node_neighbor_tuples.append(cur_tuple)
        X = [i2 for i1, i2 in node_neighbor_tuples]
        y = [i1 for i1, i2 in node_neighbor_tuples]
        mutual_information = MIC(X, y)
        mutual_information_list.append(mutual_information)

        random.shuffle(y)
        shuffled_mutual_information = MIC(X, y)
        shuffled_mutual_information_list.append(shuffled_mutual_information)
        # print(mutual_information)
    return mutual_information_list, shuffled_mutual_information_list


def abide_dataset():
    train_data_file_path = os.path.join("Data", "rois_ho", "final_sample_files", "Final_Train")
    train_tag_file_path = os.path.join("Data", "Phenotypic_V1_0b_preprocessed1.csv")
    train_subject_list = []
    for subdir, dirs, files in os.walk(train_data_file_path):
        for file in files:
            file_id = file.split("_rois_ho")[0]
            train_subject_list.append(file_id)

    cur_graph_dataset = AbideDataset(train_data_file_path, train_tag_file_path, train_subject_list, mission)
    cur_graph_dataset.set_dataset_dict()
    return cur_graph_dataset


if __name__ == "__main__":
    # datasets = ["Cirrhosis", "IBD", "bw", "IBD_Chrone", "male_female", "nugent", "milk"]
    datasets = ["abide"]
    for dataset in datasets:
        mission = "just_values"
        print(dataset)
        # Microbiome dataset
        data_file_path = os.path.join("Data", 'split_datasets_new', f"{dataset}_split_dataset",
                                      f'train_val_set_{dataset}_microbiome.csv')
        tag_file_path = os.path.join("Data", 'split_datasets_new', f"{dataset}_split_dataset",
                                    f'train_val_set_{dataset}_tags.csv')
        cur_graph_dataset = GraphDataset(data_file_path, tag_file_path, mission)
        cur_graph_dataset.dataset_dict = cur_graph_dataset.set_dataset_dict()


        graphs, labels = [], []
        for i in range(len(cur_graph_dataset.dataset_dict)):
            cur_sample = cur_graph_dataset.dataset_dict[i]
            cur_graph = cur_sample['graph']
            cur_label = cur_sample['label']
            graphs.append(cur_graph)
            labels.append(cur_label)

        # mutual_information_list, shuffled_mutual_information_list = \
        #     calc_mutual_info_for_feature_and_tag(cur_graph_dataset, graphs, labels, feature="degree")
        # mean_real_mi, std_real_mi = np.mean(mutual_information_list), np.std(mutual_information_list)
        # mean_random_mi, std_random_mi = np.mean(shuffled_mutual_information_list), np.std(shuffled_mutual_information_list)
        # print(f"Mean of real MI score {mean_real_mi:.4f}")
        # print(f"Std of real MI score {std_real_mi:.4f}")
        #
        # print(f"Mean of random MI score {mean_random_mi:.4f}")
        # print(f"Std of random MI score {std_random_mi:.4f}")
        mutual_information_list, shuffled_mutual_information_list = calc_mutual_info_for_values_and_neighbors_values(cur_graph_dataset, graphs)
        mean_real_mi, std_real_mi = np.mean(mutual_information_list), np.std(mutual_information_list)
        mean_random_mi, std_random_mi = np.mean(shuffled_mutual_information_list), np.std(shuffled_mutual_information_list)

        print(f"Mean of real MI score {mean_real_mi:.4f}")
        print(f"Std of real MI score {std_real_mi:.4f}")

        print(f"Mean of random MI score {mean_random_mi:.4f}")
        print(f"Std of random MI score {std_random_mi:.4f}")
