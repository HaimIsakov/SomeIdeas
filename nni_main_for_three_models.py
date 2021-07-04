import json
from datetime import datetime
import os
import torch
from gdm_dataset import GDMDataset
from train_test_val_ktimes import TrainTestValKTimes
import logging
import numpy as np
# import warnings
# warnings.simplefilter(action='ignore', category=UserWarning)
import nni

LOG = logging.getLogger('nni_logger')

def load_params_file(file_path):
    RECEIVED_PARAMS = json.load(open(file_path, 'r'))
    return RECEIVED_PARAMS


def create_dataset(data_file_path, tag_file_path, mission, category):
    gdm_dataset = GDMDataset(data_file_path, tag_file_path, mission, category)
    return gdm_dataset


if __name__ == '__main__':

    # get parameters from tuner
    RECEIVED_PARAMS = nni.get_next_parameter()

    # print('loading data')

    # with open('parameters.json') as f:
    #     params_file_path = json.load(f)

    try:
        # Just Values
        directory_name = "JustValues"
        mission = 'JustValues'
        # params_file_path = os.path.join(directory_name, 'Models', "just_values_on_nodes_params_file.json")
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
        print("Device", device)
        number_of_runs = 1
        gdm_dataset = create_dataset(data_file_path, tag_file_path, mission, "symmetric_adjacency")
        # RECEIVED_PARAMS = load_params_file(params_file_path)
        trainer_and_tester = TrainTestValKTimes(mission, RECEIVED_PARAMS, number_of_runs, device, gdm_dataset, result_directory_name)
        # test_metric = trainer_and_tester.train_k_cross_validation_of_dataset(k=5)
        test_metric = trainer_and_tester.stratify_train_val_test_ksplits(n_splits=2, n_repeats=1)
        mean_test_metric = np.average(test_metric)
        # report final result
        LOG.debug("\n \n \n Mean_test_metric: ", mean_test_metric)
        nni.report_final_result(mean_test_metric)

    except Exception as e:
        LOG.exception(e)
        raise
