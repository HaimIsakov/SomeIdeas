from datetime import datetime
import torch
from torch_geometric.data import DataLoader, Dataset
from torch_geometric.utils.convert import from_networkx
import os
from arrange_gdm_dataset import ArrangeGDMDataset


def create_gdm_dataset(data_file_path, tag_file_path, mission, category):
    gdm_dataset = ArrangeGDMDataset(data_file_path, tag_file_path, mission, category)
    return gdm_dataset

class GraphDataset(Dataset):
    def __init__(self, graphs_list):
        self.graphs_list = graphs_list

    def __len__(self):
        return len(self.graphs_list)

    def __getitem__(self, index):
        return self.graphs_list[index]


directory_name = "ValuesAndGraphStructure"
mission = 'GraphStructure&Values'
params_file_path = os.path.join(directory_name, 'Models', "values_and_graph_structure_on_nodes_params_file.json")

train_data_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_microbiome.csv')
train_tag_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_tags.csv')

test_data_file_path = os.path.join('GDM_split_dataset', 'test_set_gdm_microbiome.csv')
test_tag_file_path = os.path.join('GDM_split_dataset', 'test_set_gdm_tags.csv')

result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

device = "cuda:2" if torch.cuda.is_available() else "cpu"
print("device", device)
number_of_runs = 1

train_val_dataset = create_gdm_dataset(train_data_file_path, train_tag_file_path, mission, "just_A")
test_dataset = create_gdm_dataset(test_data_file_path, test_tag_file_path, mission, "just_A")

graphs_list = train_val_dataset.graphs_list
data_list = []
for graph in graphs_list:
    data = from_networkx(graph)
    data.x = data.frequency
    del data.frequency
    # print(data)
    data_list.append(data)

graph_dataset = GraphDataset(data_list)
loader = DataLoader(graph_dataset, batch_size=32)

# for graph in loader:
#     # print(graph)
# print()
