import json
from datetime import datetime
import os

import torch

from gdm_dataset import GDMDataset
from train_test_val_ktimes import TrainTestValKTimes


def load_params_file(file_path):
    RECEIVED_PARAMS = json.load(open(file_path, 'r'))
    return RECEIVED_PARAMS


def create_dataset(data_file_path, tag_file_path, mission):
    gdm_dataset = GDMDataset(data_file_path, tag_file_path, mission)
    return gdm_dataset


if __name__ == '__main__':
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    data_file_path = os.path.join("JustValues",'Data', 'OTU_merged_Mucositis_Genus_after_mipmlp_eps_1.csv')
    tag_file_path = os.path.join("JustValues",'Data', "tag_gdm_file.csv")
    params_file_path = os.path.join("JustValues", 'Models', "just_values_on_nodes_params_file.json")
    mission = 'JustValues'
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    result_directory_name = os.path.join("JustValues","Result_After_Proposal")
    number_of_runs = 10
    gdm_dataset = create_dataset(data_file_path, tag_file_path, mission)
    RECEIVED_PARAMS = load_params_file(params_file_path)
    trainer_and_tester = TrainTestValKTimes(mission, RECEIVED_PARAMS, number_of_runs, device, gdm_dataset, result_directory_name)
    trainer_and_tester.train_k_splits_of_dataset()
    print()
