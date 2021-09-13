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
from MyTasks import *
from MyDatasets import *
# import warnings
# warnings.simplefilter(action='ignore', category=UserWarning)

LOG = logging.getLogger('nni_logger')
K = 10  # For k-cross-validation
datasets_dict = {"gdm": MyDatasets.gdm_files, "cirrhosis": MyDatasets.cirrhosis_files, "IBD": MyDatasets.ibd_files,
                 "bw": MyDatasets.bw_files, "IBD_Chrone": MyDatasets.ibd_chrone_files,
                 "allergy_or_not": MyDatasets.allergy_or_not_files,
                 "allergy_milk_or_not": MyDatasets.allergy_milk_or_not_files,
                 "male_vs_female": MyDatasets.male_vs_female,
                 "male_vs_female_species": MyDatasets.male_vs_female_species}

tasks_dict = {1: MyTasks.just_values, 2: MyTasks.just_graph_structure, 3: MyTasks.values_and_graph_structure,
              4: MyTasks.pytorch_geometric}


class Main:
    def __init__(self, dataset_name, task_number, RECEIVED_PARAMS, device, nni_mode=False, geometric_mode=False,
                 add_attributes=False, plot_figures=False):
        self.dataset_name = dataset_name
        self.task_number = task_number
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.device = device
        self.nni_mode = nni_mode
        self.geometric_mode = geometric_mode
        self.add_attributes = add_attributes
        self.plot_figures = plot_figures

    def create_dataset(self, data_file_path, tag_file_path, mission):
        cur_dataset = GraphDataset(data_file_path, tag_file_path, mission, self.add_attributes, self.geometric_mode)
        return cur_dataset

    def turn_on_train(self):
        my_tasks = MyTasks(tasks_dict)
        my_datasets = MyDatasets(datasets_dict)

        directory_name, mission, params_file_path = my_tasks.get_task_files(self.task_number)
        result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
        train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = \
            my_datasets.get_dataset_files(self.dataset_name)

        print("Training-Validation Sets Graphs")
        train_val_dataset = self.create_dataset(train_data_file_path, train_tag_file_path, mission)
        print("Test set Graphs")
        test_dataset = self.create_dataset(test_data_file_path, test_tag_file_path, mission)
        train_val_dataset.update_graphs()
        test_dataset.update_graphs()

        trainer_and_tester = TrainTestValKTimes(self.RECEIVED_PARAMS, self.device, train_val_dataset, test_dataset,
                                                result_directory_name, nni_flag=self.nni_mode,
                                                geometric_or_not=self.geometric_mode)
        train_metric, val_metric, test_metric, min_train_val_metric = trainer_and_tester.train_group_k_cross_validation(k=K)
        return train_metric, val_metric, test_metric, min_train_val_metric


def set_arguments():
    parser = argparse.ArgumentParser(description='Main script of all models')
    parser.add_argument("--dataset", help="Dataset name", default="gdm", type=str)
    parser.add_argument("--task_number", help="Task number", default=1, type=int)
    parser.add_argument("--device_num", help="Cuda Device Number", default=0, type=int)
    parser.add_argument("--nni", help="is nni mode", default=0, type=int)
    return parser


def results_dealing(train_metric, val_metric, test_metric, min_train_val_metric, nni_flag, RECEIVED_PARAMS, result_file_name):
    mean_train_metric = np.average(train_metric)
    std_train_metric = np.std(train_metric)
    mean_min_train_val_metric = np.average(min_train_val_metric)
    std_min_train_val_metric = np.std(min_train_val_metric)
    mean_val_metric = np.average(val_metric)
    std_val_metric = np.std(val_metric)
    mean_test_metric = np.average(test_metric)
    std_test_metric = np.std(test_metric)

    if nni_flag:
        LOG.debug("\n \nMean Validation Set AUC: ", mean_min_train_val_metric, " +- ", std_min_train_val_metric)
        LOG.debug("\nMean Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)
        nni.report_intermediate_result(mean_test_metric)
        nni.report_final_result(mean_min_train_val_metric)
    else:
        result_file_name = f"{result_file_name}_val_mean_{mean_val_metric:.3f}_test_mean_{mean_test_metric:.3f}.csv"
        f = open(result_file_name, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow([","] + [f"Run{i}" for i in range(len(val_metric))] + ["", "Mean+-std"])
        writer.writerow(['val_auc'] + val_metric + ["", str(mean_val_metric) + "+-" + str(std_val_metric)])
        writer.writerow(['test_auc'] + test_metric + ["", str(mean_test_metric) + "+-" + str(std_test_metric)])
        writer.writerow(['train_auc'] + train_metric + ["", str(mean_train_metric) + "+-" + str(std_train_metric)])
        writer.writerow([])
        writer.writerow([])
        for key, value in RECEIVED_PARAMS.items():
            writer.writerow([key, value])
        f.close()

    print("\n \nMean minimum Validation and Train Sets AUC: ", mean_min_train_val_metric, " +- ", std_min_train_val_metric)
    print("Mean Validation Set AUC: ", mean_val_metric, " +- ", std_val_metric)
    print("Mean Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)
    print("Mean Train Set AUC: ", mean_train_metric, " +- ", std_train_metric)


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


def run_again_from_nni_results_csv_format2(file, n_rows=10):
    result_df = pd.read_csv(file, header=0, index_col=0)
    result_df.sort_values(by=['mean_val_nni'], inplace=True, ascending=False)
    del result_df["std_val_nni"]
    del result_df["mean_test_nni"]
    del result_df["std_test_nni"]
    del result_df["mean_val_nni"]
    first_n_rows = result_df[0:n_rows]
    params_list = [{} for i in range(n_rows)]
    for i in range(n_rows):
        for j in first_n_rows.columns:
            params_list[i][j] = int(first_n_rows.iloc[i][j]) if type(first_n_rows.iloc[i][j]) is np.int64 else first_n_rows.iloc[i][j]
    return params_list


def reproduce_from_nni(nni_result_file, dataset_name, mission_number):
    mission_dict = {1: "just_values", 2: "just_graph", 3: "graph_and_values"}
    params_list = run_again_from_nni_results_csv(nni_result_file, n_rows=5)
    # params_list = run_again_from_nni_results_csv_format2(nni_result_file, n_rows=5)
    nni_flag = False
    pytorch_geometric_mode = False
    add_attributes = False
    cuda_number = 0
    device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
    print("Device", device)
    for RECEIVED_PARAMS in params_list:
        main_runner = Main(dataset_name, mission_number, RECEIVED_PARAMS, device, nni_mode=nni_flag,
                           geometric_mode=pytorch_geometric_mode, add_attributes=add_attributes, plot_figures=False)
        train_metric, val_metric, test_metric, min_train_val_metric = main_runner.turn_on_train()
        result_file_name = f"{dataset_name}_{mission_dict[mission_number]}"
        results_dealing(train_metric, val_metric, test_metric, min_train_val_metric, nni_flag, RECEIVED_PARAMS, result_file_name)
        for k, v in RECEIVED_PARAMS.items():
            print(type(v))


def run_regular():
    parser = set_arguments()
    args = parser.parse_args()
    dataset_name = args.dataset
    mission_number = args.task_number
    cuda_number = args.device_num
    nni_flag = False if args.nni == 0 else True
    pytorch_geometric_mode = False
    add_attributes = False

    my_tasks = MyTasks(tasks_dict)
    directory_name, mission, params_file_path = my_tasks.get_task_files(mission_number)
    if nni_flag:
        RECEIVED_PARAMS = nni.get_next_parameter()
    else:
        RECEIVED_PARAMS = json.load(open(params_file_path, 'r'))
    RECEIVED_PARAMS["learning_rate"] = np.float64(RECEIVED_PARAMS["learning_rate"])
    RECEIVED_PARAMS["dropout"] = np.float64(RECEIVED_PARAMS["dropout"])
    RECEIVED_PARAMS["regularization"] = np.float64(RECEIVED_PARAMS["regularization"])
    RECEIVED_PARAMS["train_frac"] = np.float64(RECEIVED_PARAMS["train_frac"])
    RECEIVED_PARAMS["test_frac"] = np.float64(RECEIVED_PARAMS["test_frac"])

    device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
    main_runner = Main(dataset_name, mission_number, RECEIVED_PARAMS, device, nni_mode=nni_flag,
                       geometric_mode=pytorch_geometric_mode, add_attributes=add_attributes, plot_figures=False)
    train_metric, val_metric, test_metric, min_train_val_metric = main_runner.turn_on_train()
    result_file_name = f"{dataset_name}_{mission}"
    results_dealing(train_metric, val_metric, test_metric, min_train_val_metric, nni_flag, RECEIVED_PARAMS, result_file_name)


if __name__ == '__main__':
    try:
        run_regular()
        # try:
        #     print("cirrhosis_nni_graph_and_values")
        #     reproduce_from_nni(os.path.join("nni_results_fixed", "cirrhosis_nni_fixed_graph_and_values.csv"), "cirrhosis", 3)
        # except Exception as e:
        #     print(e)
        #     pass
        # try:
        #     print("bw_nni_graph_and_values")
        #     reproduce_from_nni(os.path.join("nni_results_fixed", "bw_nni_fixed_graph_and_values.csv"), "bw", 3)
        # except Exception as e:
        #     print(e)
        #     pass
        # try:
        #     print("gdm_nni_graph_and_values")
        #     reproduce_from_nni(os.path.join("nni_results_fixed", "gdm_nni_fixed_graph_and_values.csv"), "gdm", 3)
        # except Exception as e:
        #     print(e)
        #     pass
        # try:
        #     print("ibd_nni_graph_and_values")
        #     reproduce_from_nni(os.path.join("nni_results_fixed", "ibd_nni_fixed_graph_and_values.csv"), "IBD", 3)
        # except Exception as e:
        #     print(e)
        #     pass
        # try:
        #     print("ibd_chrone_graph_and_values")
        #     reproduce_from_nni(os.path.join("nni_results_fixed", "ibd_chrone_nni_fixed_graph_and_values.csv"), "IBD_Chrone", 3)
        # except Exception as e:
        #     print(e)
        #     pass
    except Exception as e:
        LOG.exception(e)
        raise
