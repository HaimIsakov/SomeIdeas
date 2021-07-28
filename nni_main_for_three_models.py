import json
from datetime import datetime
import os
import torch
from arrange_gdm_dataset import ArrangeGDMDataset
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


def create_gdm_dataset(data_file_path, tag_file_path, mission, category):
    gdm_dataset = ArrangeGDMDataset(data_file_path, tag_file_path, mission, category)
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

        train_data_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_microbiome.csv')
        train_tag_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_tags.csv')

        test_data_file_path = os.path.join('GDM_split_dataset', 'test_set_gdm_microbiome.csv')
        test_tag_file_path = os.path.join('GDM_split_dataset', 'test_set_gdm_tags.csv')

        result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
        date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("device", device)
        number_of_runs = 1

        train_val_dataset = create_gdm_dataset(train_data_file_path, train_tag_file_path, mission, "just_A")
        test_dataset = create_gdm_dataset(test_data_file_path, test_tag_file_path, mission, "just_A")

        trainer_and_tester = TrainTestValKTimes(mission, RECEIVED_PARAMS, number_of_runs, device, train_val_dataset,
                                                test_dataset, result_directory_name)
        test_metric = trainer_and_tester.train_group_k_cross_validation(k=5)

        mean_test_metric = np.average(test_metric)
        # print("\n \n \n Mean_test_metric: ", mean_test_metric)
        # report final result
        LOG.debug("\n \n \n Mean_test_metric: ", mean_test_metric)
        nni.report_final_result(mean_test_metric)

    except Exception as e:
        LOG.exception(e)
        raise
