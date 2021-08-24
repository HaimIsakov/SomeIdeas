import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import pandas as pd
import numpy
import argparse
import json
import logging
import csv

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


def create_dataset(data_file_path, tag_file_path, mission, geometric_or_not, add_attributes):
    full_dataset = GraphDataset(data_file_path, tag_file_path, mission, add_attributes, geometric_or_not=geometric_or_not)
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


def pytorch_geometric():
    directory_name = "PytorchGeometric"
    mission = 'GraphAttentionModel'
    params_file_path = os.path.join(directory_name, 'Models', "pytorch_geometric_params_file.json")
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

def ibd_chrone_files():
    train_data_file_path = os.path.join('IBD_Chrone_split_dataset', 'train_val_set_IBD_Chrone_microbiome.csv')
    train_tag_file_path = os.path.join('IBD_Chrone_split_dataset', 'train_val_set_IBD_Chrone_tags.csv')

    test_data_file_path = os.path.join('IBD_Chrone_split_dataset', 'test_set_IBD_Chrone_microbiome.csv')
    test_tag_file_path = os.path.join('IBD_Chrone_split_dataset', 'test_set_IBD_Chrone_tags.csv')
    return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path

def tasks_menu(task_number):

    tasks_dict = {1: just_values, 2: just_graph_structure, 3: values_and_graph_structure,
                  4: pytorch_geometric}
    try:
        directory_name, mission, params_file_path = tasks_dict[task_number]()
        print("Mission:", mission)
    except Exception as e:
        print("Task number missing or do not exist")
        raise
    return directory_name, mission, params_file_path

datasets_dict = {"gdm": gdm_files, "cirrhosis": corrhosis_files, "IBD": ibd_files,
                 "bw": bw_files, "IBD_Chrone": ibd_chrone_files}

def datasets_menu(dataset_name):
    print("Dataset:", dataset_name)
    global datasets_dict
    try:
        train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = datasets_dict[dataset_name]()
    except Exception as e:
        print("Dataset missing or do not exist")
        raise
    return train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path


def set_arguments():
    parser = argparse.ArgumentParser(description='Main script of all models')
    parser.add_argument("--dataset", help="Dataset name", default="gdm", type=str)
    parser.add_argument("--task_number", help="Task number", default=1, type=int)
    parser.add_argument("--device_num", help="Cuda Device Number", default=0, type=int)
    parser.add_argument("--nni", help="is nni mode", default=0, type=int)
    return parser

def train_from_main(RECEIVED_PARAMS, device, train_val_dataset, test_dataset, result_directory_name, nni_flag, geometric_or_not):
    trainer_and_tester = TrainTestValKTimes(RECEIVED_PARAMS, device, train_val_dataset,
                                            test_dataset, result_directory_name,
                                            nni_flag=nni_flag,
                                            geometric_or_not=geometric_or_not)
    val_metric, test_metric = trainer_and_tester.train_group_k_cross_validation(k=10)
    return val_metric, test_metric

def prepare_datasets_from_maim(train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path,
                               mission, pytorch_geometric_mode, add_attributes):
    print("Train Graphs")
    train_val_dataset = create_dataset(train_data_file_path, train_tag_file_path, mission, pytorch_geometric_mode, add_attributes)
    print("Test Graphs")
    test_dataset = create_dataset(test_data_file_path, test_tag_file_path, mission, pytorch_geometric_mode, add_attributes)

    train_val_dataset.update_graphs()
    test_dataset.update_graphs()
    return train_val_dataset, test_dataset

def main(dataset_name, mission_number, nni_flag, pytorch_geometric_mode, cuda_number, add_attributes):
    train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = datasets_menu(dataset_name)
    directory_name, mission, params_file_path = tasks_menu(mission_number)
    result_directory_name = os.path.join(directory_name, "Result_After_Proposal")

    if nni_flag:
        RECEIVED_PARAMS = nni.get_next_parameter()
    else:
        RECEIVED_PARAMS = load_params_file(params_file_path)

    device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
    print("device", device)

    train_val_dataset, test_dataset = prepare_datasets_from_maim(train_data_file_path, train_tag_file_path,
                                                                 test_data_file_path, test_tag_file_path,
                                                                 mission, pytorch_geometric_mode, add_attributes)

    val_metric, test_metric = train_from_main(RECEIVED_PARAMS, device, train_val_dataset, test_dataset,
                                              result_directory_name, nni_flag, pytorch_geometric_mode)
    return val_metric, test_metric

def results_dealing(val_metric, test_metric, nni_flag, RECEIVED_PARAMS, result_file_name):
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
        result_file_name = f"{result_file_name}_val_mean_{mean_val_metric:.3f}_test_mean_{mean_test_metric:.3f}.csv"
        f = open(result_file_name, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow([","] + [f"Run{i}" for i in range(len(val_metric))] + ["", "Mean+-std"])
        writer.writerow(['val_auc'] + val_metric + ["", str(mean_val_metric) + "+-" + str(std_val_metric)])
        writer.writerow(['test_auc'] + test_metric + ["", str(mean_test_metric) + "+-" + str(std_test_metric)])
        writer.writerow([])
        writer.writerow([])
        for key, value in RECEIVED_PARAMS.items():
            writer.writerow([key, value])
        f.close()


def run_again_from_nni_results_csv(file, n_rows=10):
    result_df = pd.read_csv(file, header=0)
    result_df.sort_values(by=['reward'], inplace=True, ascending=False)
    del result_df["trialJobId"]
    del result_df["intermediate"]
    del result_df["reward"]
    first_n_rows = result_df[0:n_rows]
    params_list = [{} for i in range(n_rows)]
    for i in range(n_rows):
        for j in first_n_rows.columns:
            params_list[i][j] = int(first_n_rows.iloc[i][j]) if type(first_n_rows.iloc[i][j]) is np.int64 else first_n_rows.iloc[i][j]
    return params_list


if __name__ == '__main__':
    parser = set_arguments()
    args = parser.parse_args()

    dataset_name = args.dataset
    mission_number = args.task_number
    cuda_number = args.device_num
    nni_flag = False if args.nni == 0 else True
    pytorch_geometric_mode = False
    add_attributes = False
    try:

        # nni_result_file = os.path.join("nni_results", "bw", "bw_nni_just_values.csv")
        # params_list = run_again_from_nni_results_csv(nni_result_file, n_rows=10)
        # dataset_name = "bw"
        # mission_number = 3
        # nni_flag = False
        # pytorch_geometric_mode = False
        # result_file_name = "bw_nni_graph_and_values"
        # for RECEIVED_PARAMS in params_list:
        #     val_metric, test_metric = \
        #         main(dataset_name, mission_number, nni_flag, pytorch_geometric_mode, cuda_number, add_attributes,
        #              RECEIVED_PARAMS)
        #     results_dealing(val_metric, test_metric, nni_flag, RECEIVED_PARAMS, result_file_name)
        if mission_number == 4:
            pytorch_geometric_mode = True

        if dataset_name == 'all':
            for dataset in datasets_dict:
                val_metric, test_metric = \
                    main(dataset, mission_number, nni_flag, pytorch_geometric_mode, cuda_number, add_attributes)
                # results_dealing(val_metric, test_metric, nni_flag, RECEIVED_PARAMS, result_file_name)
        else:
            val_metric, test_metric = main(dataset_name, mission_number, nni_flag, pytorch_geometric_mode, cuda_number, add_attributes)
            # results_dealing(val_metric, test_metric, nni_flag, RECEIVED_PARAMS, result_file_name)

        mean_val_metric = np.average(val_metric)
        std_val_metric = np.std(val_metric)
        mean_test_metric = np.average(test_metric)
        std_test_metric = np.std(test_metric)

        print("\n \nMean Validation Set AUC: ", mean_val_metric, " +- ", std_val_metric)
        print("Mean Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)
    except Exception as e:
        LOG.exception(e)
        raise
