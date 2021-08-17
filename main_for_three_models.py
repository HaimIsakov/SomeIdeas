import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import json
import logging
import time
import sys
from datetime import datetime

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


def create_dataset(data_file_path, tag_file_path, mission, geometric_or_not):
    full_dataset = GraphDataset(data_file_path, tag_file_path, mission, geometric_or_not=geometric_or_not)
    return full_dataset


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


def corrhosis_files():
    train_data_file_path = os.path.join('Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_microbiome.csv')
    train_tag_file_path = os.path.join('Cirrhosis_split_dataset', 'train_val_set_Cirrhosis_tags.csv')

    test_data_file_path = os.path.join('Cirrhosis_split_dataset', 'test_set_Cirrhosis_microbiome.csv')
    test_tag_file_path = os.path.join('Cirrhosis_split_dataset', 'test_set_Cirrhosis_tags.csv')
    return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path


def ibd_files():
    train_data_file_path = os.path.join('IBD_split_dataset', 'train_val_set_IBD_microbiome.csv')
    train_tag_file_path = os.path.join('IBD_split_dataset', 'train_val_set_IBD_tags.csv')

    test_data_file_path = os.path.join('IBD_split_dataset', 'test_set_IBD_microbiome.csv')
    test_tag_file_path = os.path.join('IBD_split_dataset', 'test_set_IBD_tags.csv')
    return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path


def bw_files():
    train_data_file_path = os.path.join('Black_vs_White_split_dataset', 'train_val_set_Black_vs_White_microbiome.csv')
    train_tag_file_path = os.path.join('Black_vs_White_split_dataset', 'train_val_set_Black_vs_White_tags.csv')

    test_data_file_path = os.path.join('Black_vs_White_split_dataset', 'test_set_Black_vs_White_microbiome.csv')
    test_tag_file_path = os.path.join('Black_vs_White_split_dataset', 'test_set_Black_vs_White_tags.csv')
    return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main script of all models')
    parser.add_argument("--dataset", help="Dataset name", default="gdm", type=str)
    parser.add_argument("--task_number", help="Task number", default=1, type=int)
    parser.add_argument("--device_num", help="Cuda Device Number", default=0, type=int)
    parser.add_argument("--nni", help="is nni mode", default=0, type=int)

    # Set some default values for the hyper-parameters
    args = parser.parse_args()
    dataset_name = args.dataset
    mission_number = args.task_number
    cuda_number = args.device_num
    nni_flag = False if args.nni == 0 else True
    pytorch_geometric_mode = False
    try:
        # mission_number = int(sys.argv[1])
        # if mission_number == 1:
            # Just Values
        directory_name, mission, params_file_path = just_values()
        if mission_number == 2:
            # Just Graph Structure
            directory_name, mission, params_file_path = just_graph_structure()
        elif mission_number == 3:
            # Values And Graph Structure
            directory_name, mission, params_file_path = values_and_graph_structure()
        elif mission_number == 4:
            # pytorch geometric gcn
            directory_name, mission, params_file_path = values_and_graph_structure()
            pytorch_geometric_mode = True
        print("Mission:", mission)

        print("Dataset:", dataset_name)
        # if dataset_name == "gdm":
        train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = gdm_files()
        if dataset_name == "cirrhosis":
            train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = corrhosis_files()
        if dataset_name == "IBD":
            train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = ibd_files()
        if dataset_name == "bw":
            train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = bw_files()

        result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
        date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

        device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
        print("device", device)
        number_of_runs = 1

        print("Train Graphs")
        train_val_dataset = create_dataset(train_data_file_path, train_tag_file_path, mission, geometric_or_not=pytorch_geometric_mode)
        print("Test Graphs")
        test_dataset = create_dataset(test_data_file_path, test_tag_file_path, mission, geometric_or_not=pytorch_geometric_mode)

        # union_nodes_set_trainval = train_val_dataset.get_joint_nodes()
        # union_nodes_set_test = test_dataset.get_joint_nodes()
        #
        # union_train_and_test = set(union_nodes_set_trainval) | set(union_nodes_set_test)
        train_val_dataset.update_graphs()
        test_dataset.update_graphs()

        if nni_flag:
            RECEIVED_PARAMS = nni.get_next_parameter()
        else:
            RECEIVED_PARAMS = load_params_file(params_file_path)
        # start = time.time()
        trainer_and_tester = TrainTestValKTimes(RECEIVED_PARAMS, number_of_runs, device, train_val_dataset,
                                                test_dataset, result_directory_name,
                                                nni_flag=nni_flag,
                                                geometric_or_not=pytorch_geometric_mode)
        val_metric, test_metric = trainer_and_tester.train_group_k_cross_validation(k=5)
        # end = time.time()
        # print(end-start)
        mean_val_metric = np.average(val_metric)
        std_val_metric = np.std(val_metric)
        mean_test_metric = np.average(test_metric)
        std_test_metric = np.std(test_metric)

        if nni_flag:
            LOG.debug("\n \nMean Validation Set AUC: ", mean_val_metric)
            LOG.debug("\nMean Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)
            nni.report_intermediate_result(mean_test_metric)
            nni.report_final_result(mean_val_metric)

        print("\n \nMean Validation Set AUC: ", mean_val_metric, " +- ", std_val_metric)
        print("Mean Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)

    except Exception as e:
        LOG.exception(e)
        raise
