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
import pandas as pd
from tqdm import tqdm
from taxonomy_tree_average_sons import *
# from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.metrics import mutual_info_score as MIC
from networkx import all_neighbors
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

if __name__ == "__main__":
    datasets = ["Cirrhosis", "IBD", "bw", "IBD_Chrone", "male_female", "nugent", "milk"]
    for dataset in datasets:
        print(dataset)
        mutual_information_list, shuffled_mutual_information_list = [], []
        data_file_path = os.path.join("Data", 'split_datasets_new', f"{dataset}_split_dataset",
                                      f'train_val_set_{dataset}_microbiome.csv')
        tag_file_path = os.path.join("Data", 'split_datasets_new', f"{dataset}_split_dataset",
                                    f'train_val_set_{dataset}_tags.csv')
        mission = 1
        cur_graph_dataset = GraphDataset(data_file_path, tag_file_path, mission)
        cur_graph_dataset.dataset_dict = cur_graph_dataset.set_dataset_dict()
        # microbiome_df = pd.read_csv(data_file_path, index_col='ID')
        graphs, labels = [], []
        # for i, mom in tqdm(enumerate(microbiome_df.iterrows()), desc='Create graphs', total=len(microbiome_df)):
        #     cur_graph = create_tax_tree(microbiome_df.iloc[i])
        #     graphs.append(cur_graph)
        #     # break

        for i in range(len(cur_graph_dataset.dataset_dict)):
            cur_sample = cur_graph_dataset.dataset_dict[i]
            cur_graph = cur_sample['graph']
            cur_label = cur_sample['label']
            graphs.append(cur_graph)
            labels.append(cur_label)

        d1 = {}
        for graph, label in zip(graphs, labels):
            all_nodes_and_values = dict(graph.nodes.data())
            # degree_dict = dict(graph.degree())
            for ind, node in enumerate(all_nodes_and_values.keys()):
                value = all_nodes_and_values[node]['val']
                # value = degree_dict[node]
                if ind not in d1:
                    d1[ind] = ([value], [label])
                else:
                    values_list, labels_list = d1[ind]
                    values_list.append(value)
                    labels_list.append(label)

        for node_number, (values_list, labels_list) in d1.items():
            X = values_list
            y = labels_list
            mutual_information = MIC(X, y)
            mutual_information_list.append(mutual_information)
            random.shuffle(y)
            shuffled_mutual_information = MIC(X, y)
            shuffled_mutual_information_list.append(shuffled_mutual_information)
            # print(mutual_information)
        print(dataset, "Mean", np.mean(mutual_information_list))
        print(dataset, " shuffled Mean", np.mean(shuffled_mutual_information_list))


# all_edges = graph.edges
    # graph = graphs[0]
    # node_neighbor_tuples = []
    # for edge in all_edges:
    #     node1, node2 = edge
    #     value1, value2 = all_nodes_and_values[node1]['val'], all_nodes_and_values[node2]['val']
    #     cur_tuple = (value1, value2)
    #     node_neighbor_tuples.append(cur_tuple)
    # X = [i2 for i1, i2 in node_neighbor_tuples]
    # y = [i1 for i1, i2 in node_neighbor_tuples]
    # mutual_information = MIC(X, y)
    # print(mutual_information)
