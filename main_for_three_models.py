import json
from datetime import datetime
import os
import numpy as np
import torch
from GraphDataset import GraphDataset
from train_test_val_ktimes import TrainTestValKTimes
# import warnings
# warnings.simplefilter(action='ignore', category=UserWarning)


def load_params_file(file_path):
    RECEIVED_PARAMS = json.load(open(file_path, 'r'))
    return RECEIVED_PARAMS


def create_gdm_dataset(data_file_path, tag_file_path, mission):
    # gdm_dataset = ArrangeGDMDataset(data_file_path, tag_file_path, mission)
    gdm_dataset = GraphDataset(data_file_path, tag_file_path, mission)
    return gdm_dataset


if __name__ == '__main__':
    # Just Values
    directory_name = "JustValues"
    mission = 'JustValues'
    params_file_path = os.path.join(directory_name, 'Models', "just_values_on_nodes_params_file.json")
    # Just Graph Structure
    # directory_name = "JustGraphStructure"
    # mission = 'JustGraphStructure'
    # params_file_path = os.path.join(directory_name, 'Models', "graph_structure_params_file.json")
    # Values And Graph Structure
    # directory_name = "ValuesAndGraphStructure"
    # mission = 'GraphStructure&Values'
    # params_file_path = os.path.join(directory_name, 'Models', "values_and_graph_structure_on_nodes_params_file.json")

    # data_file_path = os.path.join(directory_name, 'Data', 'OTU_merged_Mucositis_Genus_after_mipmlp_eps_1.csv')
    # tag_file_path = os.path.join(directory_name, 'Data', "tag_gdm_file.csv")
    train_data_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_microbiome.csv')
    train_tag_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_tags.csv')

    test_data_file_path = os.path.join('GDM_split_dataset', 'test_set_gdm_microbiome.csv')
    test_tag_file_path = os.path.join('GDM_split_dataset', 'test_set_gdm_tags.csv')

    result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print("device", device)
    number_of_runs = 1

    train_val_dataset = create_gdm_dataset(train_data_file_path, train_tag_file_path, mission)
    test_dataset = create_gdm_dataset(test_data_file_path, test_tag_file_path, mission)

    union_nodes_set_trainval = train_val_dataset.get_joint_nodes()
    union_nodes_set_test = test_dataset.get_joint_nodes()

    union_train_and_test = set(union_nodes_set_trainval) | set(union_nodes_set_test)
    train_val_dataset.update_graphs(union_train_and_test)
    test_dataset.update_graphs(union_train_and_test)

    RECEIVED_PARAMS = load_params_file(params_file_path)
    trainer_and_tester = TrainTestValKTimes(RECEIVED_PARAMS, number_of_runs, device, train_val_dataset,
                                            test_dataset, result_directory_name)
    val_metric = trainer_and_tester.train_group_k_cross_validation(k=10)

    mean_val_metric = np.average(val_metric)
    print("\n \n \n Mean_val_metric: ", mean_val_metric)
