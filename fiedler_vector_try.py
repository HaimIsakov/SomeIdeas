import os

import numpy as np
import pandas as pd
import networkx as nx
from numpy import linalg as LA

from GraphDataset import GraphDataset


def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)


dataset_name = "bw"
origin_dir = "split_datasets_new"
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
first_graph = cur_dataset.dataset_dict[0]['graph']
second_graph = cur_dataset.dataset_dict[1]['graph']

# connected_components = nx.connected_components(first_graph)
# S = [first_graph.subgraph(c).copy() for c in connected_components]
adj1 = nx.adjacency_matrix(first_graph).todense()
adj2 = nx.adjacency_matrix(second_graph).todense()

print(check_symmetric(adj2, tol=1e-8))
lap1 = nx.laplacian_matrix(first_graph).todense()
w1, v1 = LA.eig(lap1)
first_first_vector = v1.T[0]
print(first_first_vector)
lap2 = nx.laplacian_matrix(second_graph).todense()
w2, v2 = LA.eig(lap2)
second_first_vector = v2.T[0]
print(second_first_vector)

x = 1
