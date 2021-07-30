import json
import logging
from datetime import datetime
import os

import nni
import numpy as np
import torch
from GraphDataset import GraphDataset
from train_test_val_ktimes import TrainTestValKTimes
# import warnings
# warnings.simplefilter(action='ignore', category=UserWarning)

LOG = logging.getLogger('nni_logger')

def load_params_file(file_path):
    RECEIVED_PARAMS = json.load(open(file_path, 'r'))
    return RECEIVED_PARAMS


def create_gdm_dataset(data_file_path, tag_file_path, mission):
    gdm_dataset = GraphDataset(data_file_path, tag_file_path, mission)
    return gdm_dataset

def just_values():
    directory_name = "JustValues"
    mission = 'JustValues'
    params_file_path = os.path.join(directory_name, 'Models', "just_values_on_nodes_params_file.json")
    return directory_name, mission, params_file_path

def just_graph_structure():
    directory_name = "JustGraphStructure"
    mission = 'JustGraphStructure'
    params_file_path = os.path.join(directory_name, 'Models', "graph_structure_params_file.json")
    return directory_name, mission, params_file_path

def values_and_graph_structure():
    directory_name = "ValuesAndGraphStructure"
    mission = 'GraphStructure&Values'
    params_file_path = os.path.join(directory_name, 'Models', "values_and_graph_structure_on_nodes_params_file.json")
    return directory_name, mission, params_file_path

def gdm_files():
    train_data_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_microbiome.csv')
    train_tag_file_path = os.path.join('GDM_split_dataset', 'train_val_set_gdm_tags.csv')

    test_data_file_path = os.path.join('GDM_split_dataset', 'test_set_gdm_microbiome.csv')
    test_tag_file_path = os.path.join('GDM_split_dataset', 'test_set_gdm_tags.csv')
    return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path


if __name__ == '__main__':
    nni_flag = True
    try:
        # Just Values
        directory_name, mission, params_file_path = just_values()
        # # Just Graph Structure
        # directory_name, mission, params_file_path = just_graph_structure()
        # # Values And Graph Structure
        # directory_name, mission, params_file_path = values_and_graph_structure()
        print("Mission:", mission)

        train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = gdm_files()
        result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
        date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("device", device)
        number_of_runs = 1

        train_val_dataset = create_gdm_dataset(train_data_file_path, train_tag_file_path, mission)
        test_dataset = create_gdm_dataset(test_data_file_path, test_tag_file_path, mission)

        union_nodes_set_trainval = train_val_dataset.get_joint_nodes()
        union_nodes_set_test = test_dataset.get_joint_nodes()

        union_train_and_test = set(union_nodes_set_trainval) | set(union_nodes_set_test)
        train_val_dataset.update_graphs(union_train_and_test)
        test_dataset.update_graphs(union_train_and_test)

        if nni_flag:
            RECEIVED_PARAMS = nni.get_next_parameter()
        else:
            RECEIVED_PARAMS = load_params_file(params_file_path)
        trainer_and_tester = TrainTestValKTimes(RECEIVED_PARAMS, number_of_runs, device, train_val_dataset,
                                                test_dataset, result_directory_name)
        val_metric, test_metric = trainer_and_tester.train_group_k_cross_validation(k=5)

        mean_val_metric = np.average(val_metric)
        std_val_metric = np.std(val_metric)
        mean_test_metric = np.average(test_metric)
        std_test_metric = np.std(test_metric)

        if nni_flag:
            LOG.debug("\n \nMean Validation Set AUC: ", mean_val_metric)
            LOG.debug("\nMean Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)
            nni.report_intermediate_result(mean_test_metric)
            nni.report_final_result(mean_val_metric)

        else:
            print("\n \nMean Validation Set AUC: ", mean_val_metric, " +- ", std_val_metric)
            print("Mean Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)

    except Exception as e:
        LOG.exception(e)
        raise
