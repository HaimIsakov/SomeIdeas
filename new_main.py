import os
import random
import sys

from tqdm import tqdm

from exclude_hyper_parameters import get_hyper_parameters_as_dict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
for path_name in [os.path.join(os.path.dirname(__file__)),
                  os.path.join(os.path.dirname(__file__), 'Data'),
                  os.path.join(os.path.dirname(__file__), 'Missions')]:
    sys.path.append(path_name)

# sys.path.insert(1, 'Data')

from ShaharGdmDataset import ShaharGdmDataset
from TcrDataset import TCRDataset
from GraphDataset import GraphDataset
from BrainNetwork import AbideDataset
# from cancer_data.CancerDataset import CancerDataset
from nni_functions_utils import run_again_from_nni_results_csv
from train_test_val_ktimes_no_external_test import TrainTestValKTimesNoExternalTest
from train_test_no_val_ktimes import TrainTestValKTimesNoExternalTestNoVal
from train_test_val_ktimes import TrainTestValKTimes

import argparse
import json
import logging
import csv
from datetime import date
import nni
import numpy as np
import torch
import itertools
from MyTasks import *
from MyDatasets import *
# import warnings
# warnings.simplefilter(action='ignore', category=UserWarning)

LOG = logging.getLogger('nni_logger')
K = 10  # For k-cross-validation
# # "gdm": MyDatasets.gdm_files,"male_vs_female_species": MyDatasets.male_vs_female_species,"allergy_or_not": MyDatasets.allergy_or_not_files,"allergy_milk_or_not": MyDatasets.allergy_milk_or_not_files

datasets_dict = {"Cirrhosis": MyDatasets.cirrhosis_files, "IBD": MyDatasets.ibd_files,
                 "bw": MyDatasets.bw_files, "IBD_Chrone": MyDatasets.ibd_chrone_files,
                 "Male_vs_Female": MyDatasets.male_vs_female, "male_female": MyDatasets.male_vs_female,
                 "nugent": MyDatasets.nugent}
datasets = ["Cirrhosis", "IBD", "bw", "IBD_Chrone", "male_female", "nugent", "milk", "nut", "peanut"]
tasks_dict = {1: MyTasks.just_values, 2: MyTasks.just_graph_structure, 3: MyTasks.values_and_graph_structure,
              4: MyTasks.double_gcn_layers, 5: MyTasks.one_head_attention, 6: MyTasks.yoram_attention,
              7: MyTasks.concat_graph_and_values, 8: MyTasks.fiedler_vector}

mission_dict = {1: "just_values", 2: "just_graph", 3: "graph_and_values", 4: "double_gcn_layer",
                6: "yoram_attention", 7: "concat_graph_and_values", 8: "fiedler_vector"}

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

    def create_dataset(self, data_path, label_path, subject_list, mission):
        if self.dataset_name == "abide":
            cur_dataset = AbideDataset(data_path, label_path, subject_list, mission)
        elif self.dataset_name == "cancer":
            adj_mat_path = os.path.join("cancer_data", "new_cancer_adj_matrix.csv")
            cur_dataset = CancerDataset(adj_mat_path, data_path, label_path, subject_list, mission)
        elif self.dataset_name == "tcr" or dataset_name == "ISB" or dataset_name == "NIH":
            # adj_mat_path = os.path.join("TCR_dataset", "distance_matrix.csv")
            # adj_mat_path = os.path.join("TCR_dataset", "distance_matrix2.csv")
            cur_dataset = TCRDataset(self.dataset_name, data_path, label_path, subject_list, mission)
        elif self.dataset_name == "gdm":
            cur_dataset = ShaharGdmDataset(data_path, label_path, subject_list, mission)
        else:
            cur_dataset = GraphDataset(data_path, label_path, mission, self.add_attributes, self.geometric_mode)
        cur_dataset.update_graphs()
        return cur_dataset

    def play(self, external_test=True):
        # kwargs = {}
        my_tasks = MyTasks(tasks_dict, self.dataset_name)
        my_datasets = MyDatasets(datasets_dict)

        directory_name, mission, params_file_path = my_tasks.get_task_files(self.task_number)
        result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
        if self.dataset_name == "ISB" or self.dataset_name == "NIH":
            train_test_data_file_path, train_test_label_file_path, subject_list\
                = my_datasets.get_dataset_files_no_external_test(self.dataset_name)
            train_test_dataset = self.create_dataset(train_test_data_file_path,
                                                         train_test_label_file_path,
                                                         subject_list, mission)
            # for ISB or NIH we do not want validation set
            trainer_and_tester = TrainTestValKTimesNoExternalTestNoVal(self.RECEIVED_PARAMS, self.device, train_test_dataset,
                                                    result_directory_name, nni_flag=self.nni_mode,
                                                    geometric_or_not=self.geometric_mode, plot_figures=self.plot_figures,
                                                                  **kwargs)
        else:
            if not external_test:
                print("No external test set")
                train_val_test_data_file_path, train_val_test_label_file_path, subject_list\
                    = my_datasets.get_dataset_files_no_external_test(self.dataset_name)
                train_val_test_dataset = self.create_dataset(train_val_test_data_file_path,
                                                             train_val_test_label_file_path,
                                                             subject_list, mission)

                trainer_and_tester = TrainTestValKTimesNoExternalTest(self.RECEIVED_PARAMS, self.device, train_val_test_dataset,
                                                        result_directory_name, nni_flag=self.nni_mode,
                                                        geometric_or_not=self.geometric_mode, plot_figures=self.plot_figures,
                                                                      **kwargs)
            else:
                print("With external test set")
                train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path, train_subject_list, \
                test_subject_list = my_datasets.get_dataset_files_yes_external_test(self.dataset_name)

                test_dataset = self.create_dataset(test_data_file_path, test_tag_file_path, test_subject_list, mission)
                train_val_dataset = self.create_dataset(train_data_file_path, train_tag_file_path, train_subject_list, mission)
                # kwargs = {"samples": 500}
                trainer_and_tester = TrainTestValKTimes(self.RECEIVED_PARAMS, self.device, train_val_dataset, test_dataset,
                                                        result_directory_name, nni_flag=self.nni_mode,
                                                        geometric_or_not=self.geometric_mode, plot_figures=self.plot_figures,
                                                        **kwargs)
        # In GDM dataset we don't need 10-cv
        if self.dataset_name == "gdm":
            K = 1
        elif self.dataset_name == "tcr" or self.dataset_name == "ISB" or self.dataset_name == "NIH":
            K = 5
        else:
            K = 10
        return_lists = trainer_and_tester.train_group_k_cross_validation(k=K)
        return return_lists

def set_arguments():
    parser = argparse.ArgumentParser(description='Main script of all models')
    parser.add_argument("--dataset", help="Dataset name", default="IBD", type=str)
    parser.add_argument("--task_number", help="Task number", default=1, type=int)
    parser.add_argument("--device_num", help="Cuda Device Number", default=3, type=int)
    parser.add_argument("--nni", help="is nni mode", default=0, type=int)
    parser.add_argument("--samples", help="sample in tcr dataset", default=-1, type=int)
    parser.add_argument("--thresh", help="tcr correlation matrix threshold", default=0.05,  type=float)
    return parser


def calc_mean_and_std(dataset_metric_list):
    mean_dataset_metric = np.average(dataset_metric_list)
    std_dataset_metric = np.std(dataset_metric_list)
    return mean_dataset_metric, std_dataset_metric


def results_dealing(return_lists, nni_flag, RECEIVED_PARAMS, result_file_name):
    train_auc, val_auc, test_auc, min_train_val_metric, alpha_list, \
    train_f1_micro, val_f1_micro, test_f1_micro, \
    train_f1_macro, val_f1_macro, test_f1_macro = return_lists
    # train_metric, val_metric, test_metric, min_train_val_metric, alpha_list = return_lists
    mean_train_metric, std_train_metric = calc_mean_and_std(train_auc)
    mean_min_train_val_metric, std_min_train_val_metric = calc_mean_and_std(min_train_val_metric)
    mean_val_metric, std_val_metric = calc_mean_and_std(val_auc)
    mean_test_metric, std_test_metric = calc_mean_and_std(test_auc)

    mean_train_f1_micro, std_train_f1_micro = calc_mean_and_std(train_f1_micro)
    mean_val_f1_micro, std_val_f1_micro = calc_mean_and_std(val_f1_micro)
    mean_test_f1_micro, std_test_f1_micro = calc_mean_and_std(test_f1_micro)

    mean_train_f1_macro, std_train_f1_macro = calc_mean_and_std(train_f1_macro)
    mean_val_f1_macro, std_val_f1_macro = calc_mean_and_std(val_f1_macro)
    mean_test_f1_macro, std_test_f1_macro = calc_mean_and_std(test_f1_macro)

    mean_alpha, std_alpha = calc_mean_and_std(alpha_list)
    if nni_flag:
        LOG.debug("\n \nMean Validation Set metric: ", mean_min_train_val_metric, " +- ", std_min_train_val_metric)
        LOG.debug("\nMean Final_Test Set metric: ", mean_test_metric, " +- ", std_test_metric)
        nni.report_intermediate_result(mean_test_metric)
        nni.report_final_result(mean_min_train_val_metric)
    else:
        result_file_name = f"{result_file_name}_val_mean_{mean_val_metric:.3f}_test_mean_{mean_test_metric:.3f}.csv"
        f = open(result_file_name, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow([","] + [f"Run{i}" for i in range(len(train_auc))] + ["", "Auc_Mean", "Auc_std"])
        writer.writerow(['Train_auc'] + train_auc + ["", str(mean_train_metric), str(std_train_metric)])
        writer.writerow(['Val_auc'] + val_auc + ["", str(mean_val_metric), str(std_val_metric)])
        writer.writerow(['Test_auc'] + test_auc + ["", str(mean_test_metric), str(std_test_metric)])
        writer.writerow([])

        writer.writerow([","] + [f"Run{i}" for i in range(len(train_auc))] + ["", "F1_micro_Mean", "F1_micro_std"])
        writer.writerow(['Train_F1_micro'] + train_f1_micro + ["", str(mean_train_f1_micro), str(std_train_f1_micro)])
        writer.writerow(['Val_F1_micro'] + val_f1_micro + ["", str(mean_val_f1_micro), str(std_val_f1_micro)])
        writer.writerow(['Test_F1_micro'] + test_f1_micro + ["", str(mean_test_f1_micro), str(std_test_f1_micro)])
        writer.writerow([])

        writer.writerow([","] + [f"Run{i}" for i in range(len(train_auc))] + ["", "F1_macro_Mean", "F1_macro_std"])
        writer.writerow(['Train_F1_macro'] + train_f1_macro + ["", str(mean_train_f1_macro), str(std_train_f1_macro)])
        writer.writerow(['Val_F1_macro'] + val_f1_macro + ["", str(mean_val_f1_macro), str(std_val_f1_macro)])
        writer.writerow(['Test_F1_macro'] + test_f1_macro + ["", str(mean_test_f1_macro), str(std_test_f1_macro)])

        writer.writerow([])
        writer.writerow([])
        for key, value in RECEIVED_PARAMS.items():
            writer.writerow([key, value])
        writer.writerow([])
        writer.writerow(['Alpha_Value'] + alpha_list + ["", str(mean_alpha), str(std_alpha)])

        f.close()

    print("\n \nMean minimum Validation and Train Sets AUC: ", mean_min_train_val_metric, " +- ", std_min_train_val_metric)
    print("Mean Train Set AUC: ", mean_train_metric, " +- ", std_train_metric)
    print("Mean Validation Set AUC: ", mean_val_metric, " +- ", std_val_metric)
    print("Mean Final_Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)
    return return_lists


# def reproduce_from_nni(nni_result_file, dataset_name, mission_number):
#     # mission_dict = {1: "just_values", 2: "just_graph", 3: "graph_and_values"}
#     params_list = run_again_from_nni_results_csv(nni_result_file, n_rows=5)
#     # params_list = run_again_from_nni_results_csv_format2(nni_result_file, n_rows=5)
#     nni_flag = False
#     pytorch_geometric_mode = False
#     add_attributes = False
#     cuda_number = 3
#     device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
#     print("Device", device)
#     for RECEIVED_PARAMS in params_list:
#         main_runner = Main(dataset_name, mission_number, RECEIVED_PARAMS, device, nni_mode=nni_flag,
#                            geometric_mode=pytorch_geometric_mode, add_attributes=add_attributes, plot_figures=False)
#         train_metric, val_metric, test_metric, min_train_val_metric = main_runner.play()
#         result_file_name = f"{dataset_name}_{mission_dict[mission_number]}"
#         results_dealing(train_metric, val_metric, test_metric, min_train_val_metric, nni_flag, RECEIVED_PARAMS, result_file_name)
#         for k, v in RECEIVED_PARAMS.items():
#             print(type(v))


def rerun_from_grid_search(directory, cuda_number, dataset_name, mission_number, nni_flag=False,
                           pytorch_geometric_mode=False, add_attributes=False, **kwargs):
    device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
    max_mean_val_metric = 0
    max_results_params = {}
    save_params_file_name = f"params_{dataset_name}_{mission_dict[mission_number]}.json"
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            path = os.path.join(directory, file)
            print(path)
            RECEIVED_PARAMS = get_hyper_parameters_as_dict(path)
            return_lists = runner_backbone(RECEIVED_PARAMS, device, dataset_name, mission_number, nni_flag,
                                           pytorch_geometric_mode, add_attributes, **kwargs)
            train_auc, val_auc, test_auc, min_train_val_metric, alpha_list, \
            train_f1_micro, val_f1_micro, test_f1_micro, \
            train_f1_macro, val_f1_macro, test_f1_macro = return_lists
            mean_val_metric, std_val_metric = calc_mean_and_std(val_auc)
            if mean_val_metric > max_mean_val_metric:
                print("New max_mean_val_metric", mean_val_metric, "+-", std_val_metric)
                max_mean_val_metric = mean_val_metric
                max_results_params = RECEIVED_PARAMS
                print("New Hyper parameters", max_results_params)
    with open(save_params_file_name, 'w') as fp:
        json.dump(dict(sorted(max_results_params.items())), fp)
    return max_results_params


def run_all_dataset(mission_number, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes, datasets, **kwargs):
    datasets_results_dict = {}
    for dataset_name in datasets:
        print(dataset_name)
        try:
            return_lists = runner(dataset_name, mission_number, cuda_number, nni_flag,
                                  pytorch_geometric_mode, add_attributes, **kwargs)
            train_auc, val_auc, test_auc, min_train_val_metric, alpha_list, \
            train_f1_micro, val_f1_micro, test_f1_micro, \
            train_f1_macro, val_f1_macro, test_f1_macro = return_lists
            # train_metric, val_metric, test_metric, min_train_val_metric, alpha_list = return_lists
            mean_train_metric, std_train_metric = calc_mean_and_std(train_auc)
            mean_min_train_val_metric, std_min_train_val_metric = calc_mean_and_std(min_train_val_metric)
            mean_val_metric, std_val_metric = calc_mean_and_std(val_auc)
            mean_test_metric, std_test_metric = calc_mean_and_std(test_auc)

            mean_train_f1_micro, std_train_f1_micro = calc_mean_and_std(train_f1_micro)
            mean_val_f1_micro, std_val_f1_micro = calc_mean_and_std(val_f1_micro)
            mean_test_f1_micro, std_test_f1_micro = calc_mean_and_std(test_f1_micro)

            mean_train_f1_macro, std_train_f1_macro = calc_mean_and_std(train_f1_macro)
            mean_val_f1_macro, std_val_f1_macro = calc_mean_and_std(val_f1_macro)
            mean_test_f1_macro, std_test_f1_macro = calc_mean_and_std(test_f1_macro)

            datasets_results_dict[dataset_name] = {"train_auc_mean": mean_train_metric,
                                                   "val_metric_mean": mean_val_metric,
                                                   "test_metric_mean": mean_test_metric,

                                                   "train_auc_std": std_train_metric,
                                                   "val_auc_std": std_val_metric,
                                                   "test_auc_std": std_test_metric,

                                                   "mean_train_f1_micro": mean_train_f1_micro,
                                                   "mean_val_f1_micro": mean_val_f1_micro,
                                                   "mean_test_f1_micro": mean_test_f1_micro,

                                                   "std_train_f1_micro": std_train_f1_micro,
                                                   "std_val_f1_micro": std_val_f1_micro,
                                                   "std_test_f1_micro": std_test_f1_micro,

                                                   "mean_train_f1_macro": mean_train_f1_macro,
                                                   "mean_val_f1_macro": mean_val_f1_macro,
                                                   "mean_test_f1_macro": mean_test_f1_macro,

                                                   "std_train_f1_macro": std_train_f1_macro,
                                                   "std_val_f1_macro": std_val_f1_macro,
                                                   "std_test_f1_macro": std_test_f1_macro,
                                                   }
            print("Test list", test_auc)
            if len(alpha_list) > 0 and mission_number != 1:
                print("Alpha_list", alpha_list)
                mean_alpha_value, std_alpha_value = calc_mean_and_std(alpha_list)
                datasets_results_dict[dataset_name]["alpha_value_mean"] = mean_alpha_value
                datasets_results_dict[dataset_name]["alpha_value_std"] = std_alpha_value

        except Exception as e:
            # raise
            print(e)
    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    all_missions_results_df = pd.DataFrame.from_dict(datasets_results_dict, orient='index')
    all_missions_results_df.to_csv(f"{mission_dict[mission_number]}_all_datasets_results_train_val_test_{d1}.csv")


def run_all_missions(dataset_name, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes, missions, **kwargs):
    missions = [1, 2, 3, 4, 6, 7]
    missions_results_dict = {}
    for mission in missions:
        print("Mission", mission_dict[mission])
        try:
            return_lists = runner(dataset_name, mission, cuda_number, nni_flag,
                                  pytorch_geometric_mode, add_attributes, **kwargs)
            train_metric, val_metric, test_metric, _, alpha_list = return_lists
            mean_train_metric, std_train_metric = calc_mean_and_std(train_metric)
            mean_val_metric, std_val_metric = calc_mean_and_std(val_metric)
            mean_test_metric, std_test_metric = calc_mean_and_std(test_metric)

            missions_results_dict[mission] = {"train_metric_mean": mean_train_metric,
                                                   "val_metric_mean": mean_val_metric,
                                                   "test_metric_mean": mean_test_metric,
                                                   "train_metric_std": std_train_metric,
                                                   "val_metric_std": std_val_metric,
                                                   "test_metric_std": std_test_metric}
            print("Test list", test_metric)
            if len(alpha_list) > 0:
                print("Alpha_list", alpha_list)
                mean_alpha_value, std_alpha_value = calc_mean_and_std(alpha_list)
                missions_results_dict[mission]["alpha_value_mean"] = mean_alpha_value
                missions_results_dict[mission]["alpha_value_std"] = std_alpha_value
        except Exception as e:
            print(e)

    today = date.today()
    d1 = today.strftime("%d_%m_%Y")
    all_missions_results_df = pd.DataFrame.from_dict(missions_results_dict, orient='index')
    all_missions_results_df.to_csv(f"{dataset_name}_all_missions_results_train_val_test_{d1}.csv")


def run_all_datasets_missions(cuda_number, nni_flag, pytorch_geometric_mode, add_attributes):
    # mission_dict = {1: "just_values", 2: "just_graph", 3: "graph_and_values", 6: "yoram_attention"}
    for mission_number in mission_dict.keys():
        run_all_dataset(mission_number, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes)


def get_model_hyper_parameters(dataset_name, mission_number):
    if dataset_name == "abide" or dataset_name == "abide1":
        params_file_path = os.path.join("Data", "abide_dataset_params.json")  # TODO: add best parameters from nni
        if nni_flag:
            RECEIVED_PARAMS = nni.get_next_parameter()
        else:
            RECEIVED_PARAMS = json.load(open(params_file_path, 'r'))

        print("Dataset", dataset_name)
        print("Mission", mission_number)
    elif dataset_name == "cancer":
        my_tasks = MyTasks(tasks_dict, dataset_name)
        directory_name, mission, params_file_path = my_tasks.get_task_files(mission_number)
        params_file_path = os.path.join(directory_name, 'Models', f"{mission}_params_file.json")
        print(params_file_path)
        if nni_flag:
            RECEIVED_PARAMS = nni.get_next_parameter()
        else:
            RECEIVED_PARAMS = json.load(open(params_file_path, 'r'))
    elif dataset_name == "tcr" or dataset_name == "ISB" or dataset_name == "NIH":
        my_tasks = MyTasks(tasks_dict, dataset_name)
        directory_name, mission, params_file_path = my_tasks.get_task_files(mission_number)
        # params_file_path = os.path.join(directory_name, 'Models', f"{mission}_params_file.json")
        print(params_file_path)
        if nni_flag:
            RECEIVED_PARAMS = nni.get_next_parameter()
        else:
            RECEIVED_PARAMS = json.load(open(params_file_path, 'r'))
    else:
        my_tasks = MyTasks(tasks_dict, dataset_name)
        directory_name, mission, params_file_path = my_tasks.get_task_files(mission_number)
        if nni_flag:
            RECEIVED_PARAMS = nni.get_next_parameter()
        else:
            RECEIVED_PARAMS = json.load(open(params_file_path, 'r'))
    return RECEIVED_PARAMS


def runner(dataset_name, mission_number, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes, **kwargs):
    device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
    RECEIVED_PARAMS = get_model_hyper_parameters(dataset_name, mission_number)
    return_lists = runner_backbone(RECEIVED_PARAMS, device, dataset_name, mission_number, nni_flag,
                                   pytorch_geometric_mode, add_attributes, **kwargs)
    return return_lists


def runner_backbone(RECEIVED_PARAMS, device, dataset_name, mission_number, nni_flag, pytorch_geometric_mode,
                    add_attributes, **kwargs):
    print("Device", device)
    print("Hyper-parameters", RECEIVED_PARAMS)
    print("Mission", mission_dict[mission_number])
    print("Dataset Name", dataset_name)
    main_runner = Main(dataset_name, mission_number, RECEIVED_PARAMS, device, nni_mode=nni_flag,
                       geometric_mode=pytorch_geometric_mode, add_attributes=add_attributes, plot_figures=False)
    return_lists = main_runner.play(kwargs)
    result_file_name = f"{dataset_name}_{mission_dict[mission_number]}"
    results_dealing(return_lists, nni_flag, RECEIVED_PARAMS, result_file_name)
    return return_lists


# def tcr_runner_hyper_parameters(dataset_name, mission_number, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes,
#                             **kwargs):
#     device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
#     print("Device", device)
#     RECEIVED_PARAMS = get_model_hyper_parameters(dataset_name, mission_number)
#     print("Mission", mission_dict[mission_number])
#     print("Hyper-parameters", RECEIVED_PARAMS)
#
#     results_dict = {}
#     for thresh in np.linspace(0, 0.1, num=10):
#         try:
#             print("Thresold", thresh)
#             RECEIVED_PARAMS["thresh"] = thresh
#             main_runner = Main(dataset_name, mission_number, RECEIVED_PARAMS, device, nni_mode=nni_flag,
#                                geometric_mode=pytorch_geometric_mode, add_attributes=add_attributes, plot_figures=False)
#             return_lists = main_runner.play(kwargs)
#             result_file_name = f"{dataset_name}_{mission_dict[mission_number]}"
#             results_dealing(return_lists, nni_flag, RECEIVED_PARAMS, result_file_name)
#             train_metric, val_metric, test_metric, _, alpha_list = return_lists
#             mean_train_metric, std_train_metric = calc_mean_and_std(train_metric)
#             mean_val_metric, std_val_metric = calc_mean_and_std(val_metric)
#             mean_test_metric, std_test_metric = calc_mean_and_std(test_metric)
#
#             results_dict[thresh] = {"train_metric_mean": mean_train_metric,
#                                                    "val_metric_mean": mean_val_metric,
#                                                    "test_metric_mean": mean_test_metric,
#                                                    "train_metric_std": std_train_metric,
#                                                    "val_metric_std": std_val_metric,
#                                                    "test_metric_std": std_test_metric}
#             print("Test list", test_metric)
#             if len(alpha_list) > 0:
#                 print("Alpha_list", alpha_list)
#                 mean_alpha_value, std_alpha_value = calc_mean_and_std(alpha_list)
#                 results_dict[thresh]["alpha_value_mean"] = mean_alpha_value
#                 results_dict[thresh]["alpha_value_std"] = std_alpha_value
#         except Exception as e:
#             # raise
#             print(e)
#     today = date.today()
#     d1 = today.strftime("%d_%m_%Y")
#     all_missions_results_df = pd.DataFrame.from_dict(results_dict, orient='index')
#     all_missions_results_df.to_csv(f"{mission_dict[mission_number]}_hyper_parameters_results_train_val_test_{d1}.csv")


# def run_grid_search(dataset_name, mission_number, cuda_number, hyper_parameters_dict):
#     device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
#     print("Device", device)
#     print("Mission", mission_dict[mission_number])
#     print("Dataset", dataset_name)
#     # RECEIVED_PARAMS = get_model_hyper_parameters(dataset_name, mission_number)
#     # print("Original Hyper-parameters", RECEIVED_PARAMS)
#     d1 = date.today().strftime("%d_%m_%Y")
#     results_file = open(f"{dataset_name}_{mission_dict[mission_number]}_hyper_parameters_results_train_val_test_{d1}.csv", "w")
#     headers = list(hyper_parameters_dict.keys()) + ["train_metric_mean", "val_metric_mean", "test_metric_mean",
#                                               "train_metric_std", "val_metric_std", "test_metric_std"]
#     headers_str = ",".join(headers)
#     results_file.write(f"{headers_str}\n")
#     hyper_parameters_names = list(hyper_parameters_dict.keys())
#     a = list(hyper_parameters_dict.values())
#     all_combination_hyper_parameters = list(itertools.product(*a))
#     random.shuffle(all_combination_hyper_parameters)
#     for hyper_parameters_set in tqdm(all_combination_hyper_parameters):
#         RECEIVED_PARAMS = {}
#         # create RECEIVED_PARAMS dict
#         for i, hyper_parameter in enumerate(hyper_parameters_set):
#             RECEIVED_PARAMS[hyper_parameters_names[i]] = hyper_parameter
#         print(RECEIVED_PARAMS)
#         try:
#             # run the model for mission and dataset and a set of hyper-parameters
#             main_runner = Main(dataset_name, mission_number, RECEIVED_PARAMS, device, nni_mode=False,
#                                geometric_mode=False, add_attributes=False, plot_figures=False)
#             return_lists = main_runner.play(kwargs)
#             result_file_name = f"{dataset_name}_{mission_dict[mission_number]}"
#             results_dealing(return_lists, nni_flag, RECEIVED_PARAMS, result_file_name)
#             # get models results
#             train_metric, val_metric, test_metric, _, alpha_list = return_lists
#             # calculate some statistics
#             mean_train_metric, std_train_metric = calc_mean_and_std(train_metric)
#             mean_val_metric, std_val_metric = calc_mean_and_std(val_metric)
#             mean_test_metric, std_test_metric = calc_mean_and_std(test_metric)
#             # result_str represents the line will be written to result file
#             result_str = ""
#             for i, (k, v) in enumerate(RECEIVED_PARAMS.items()):
#                 result_str += str(v) + ","
#             result_str += str(mean_train_metric) + ","
#             result_str += str(mean_val_metric) + ","
#             result_str += str(mean_test_metric) + ","
#             result_str += str(std_train_metric) + ","
#             result_str += str(std_val_metric) + ","
#             result_str += str(std_test_metric)
#             result_str += "\n"
#             print("Test list", test_metric)
#             results_file.write(result_str)
#             # if len(alpha_list) > 0:
#             #     print("Alpha_list", alpha_list)
#             #     mean_alpha_value, std_alpha_value = calc_mean_and_std(alpha_list)
#             #     results_dict[thresh]["alpha_value_mean"] = mean_alpha_value
#             #     results_dict[thresh]["alpha_value_std"] = std_alpha_value
#         except Exception as e:
#             # raise
#             print(e)
#     results_file.close()


def get_hyper_parameters_for_grid_search(mission):
    hyper_parameters_dict = {}
    if mission == 1:
        hyper_parameters_dict = {"learning_rate": [1e-6, 16-5, 1e-4, 1e-3],
                                 "batch_size": [8, 16, 32],
                                 "dropout": [0.1, 0.2, 0.4, 0.5],
                                 "activation": ["relu", "tanh"],
                                 "regularization": [1e-6, 1e-4, 1e-3],
                                 "layer_1": [16, 32, 128],
                                 "layer_2": [16, 32, 128]}
    elif mission in [2, 3, 7]:
        hyper_parameters_dict = {"learning_rate": [1e-6, 16-5, 1e-4, 1e-3],
                                 "batch_size": [8, 16, 32],
                                 "dropout": [0.1, 0.2, 0.4, 0.5],
                                 "activation": ["relu", "tanh"],
                                 "regularization": [1e-6, 1e-4, 1e-3],
                                 "layer_1": [16, 32, 128],
                                 "layer_2": [16, 32, 128],
                                 "preweight": [5, 8, 10, 12]}
    elif mission == 4:
        hyper_parameters_dict = {"learning_rate": [1e-6, 16-5, 1e-4, 1e-3],
                                 "batch_size": [8, 16, 32],
                                 "dropout": [0.1, 0.2, 0.4, 0.5],
                                 "activation": ["relu", "tanh"],
                                 "regularization": [1e-6, 1e-4, 1e-3],
                                 "layer_1": [16, 32, 128],
                                 "layer_2": [16, 32, 128],
                                 "preweight": [5, 8, 10, 12],
                                 "preweight2": [5, 8, 10, 12]}
    return hyper_parameters_dict


if __name__ == '__main__':
    try:
        parser = set_arguments()
        args = parser.parse_args()
        dataset_name = args.dataset

        mission_number = args.task_number
        cuda_number = args.device_num
        samples = args.samples
        nni_flag = False if args.nni == 0 else True
        pytorch_geometric_mode = False
        add_attributes = False
        kwargs = {"samples": samples}
        # for dataset_name in ["nugent", "IBD_Chrone", "male_female"]:
        #     for option in ["_4sparse_97", "_4sparse_98", "_4sparse_99"]:
        #         runner(dataset_name + option, 3, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes, **kwargs)

        # runner(dataset_name, mission_number, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes, **kwargs)
        # datasets = ["nut", "peanut", "male_female", "milk", "Cirrhosis"]
        run_all_dataset(2, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes, datasets, **kwargs)
        # run_all_dataset(3, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes, datasets, **kwargs)
        # datasets = ["Cirrhosis", "IBD", "bw", "IBD_Chrone", "male_female", "nugent"]
        # datasets = ["nut", "peanut", "milk"]


        # # For grid search
        # hyper_parameters_dict = get_hyper_parameters_for_grid_search(mission_number)
        # run_grid_search(dataset_name, mission_number, cuda_number, hyper_parameters_dict)

        # For rerun from grid search
        # datasets = ["Cirrhosis", "IBD", "bw", "nugent", "milk"]
        # for dataset_name in datasets:
        #     directory = os.path.join("Missions", "DoubleGcnLayers", "params", "3best_from_grid_search", dataset_name)
        #     rerun_from_grid_search(directory, cuda_number, dataset_name, mission_number, nni_flag=False,
        #                            pytorch_geometric_mode=False, add_attributes=False, **kwargs)
    except Exception as e:
        # LOG.exception(e)
        raise
