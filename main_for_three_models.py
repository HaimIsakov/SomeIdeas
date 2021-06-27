import json
from datetime import datetime
import os
import torch
from gdm_dataset import GDMDataset
from train_test_val_ktimes import TrainTestValKTimes

# import warnings
# warnings.simplefilter(action='ignore', category=UserWarning)

def load_params_file(file_path):
    RECEIVED_PARAMS = json.load(open(file_path, 'r'))
    return RECEIVED_PARAMS


def create_dataset(data_file_path, tag_file_path, mission):
    gdm_dataset = GDMDataset(data_file_path, tag_file_path, mission)
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

    data_file_path = os.path.join(directory_name, 'Data', 'OTU_merged_Mucositis_Genus_after_mipmlp_eps_1.csv')
    tag_file_path = os.path.join(directory_name, 'Data', "tag_gdm_file.csv")
    result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print("device", device)
    number_of_runs = 1
    gdm_dataset = create_dataset(data_file_path, tag_file_path, mission)
    RECEIVED_PARAMS = load_params_file(params_file_path)
    trainer_and_tester = TrainTestValKTimes(mission, RECEIVED_PARAMS, number_of_runs, device, gdm_dataset, result_directory_name)
    # trainer_and_tester.train_k_splits_of_dataset()
    trainer_and_tester.train_k_cross_validation_of_dataset(k=5)
