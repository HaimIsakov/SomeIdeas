import os

from ShaharGdmDataset import ShaharGdmDataset
from TcrDataset import TCRDataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

from sklearn.model_selection import train_test_split
from BrainNetwork import AbideDataset
# from cancer_data.CancerDataset import CancerDataset
from nni_functions_utils import run_again_from_nni_results_csv
from node2vec_embed import find_embed
from train_test_val_ktimes_no_external_test import TrainTestValKTimesNoExternalTest

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
# # "gdm": MyDatasets.gdm_files,"male_vs_female_species": MyDatasets.male_vs_female_species,"allergy_or_not": MyDatasets.allergy_or_not_files,"allergy_milk_or_not": MyDatasets.allergy_milk_or_not_files

# datasets_dict = {"Cirrhosis": MyDatasets.cirrhosis_files, "IBD": MyDatasets.ibd_files,
#                  "bw": MyDatasets.bw_files, "IBD_Chrone": MyDatasets.ibd_chrone_files,
#                  "Male_vs_Female": MyDatasets.male_vs_female, "male_female": MyDatasets.male_vs_female,
#                  "nut": MyDatasets.nut, "peanut": MyDatasets.peanut, "nugent": MyDatasets.nugent,
#                  "milk_no_controls": MyDatasets.allergy_milk_no_controls}
datasets_dict = {"Cirrhosis": MyDatasets.cirrhosis_files, "IBD": MyDatasets.ibd_files,
                 "bw": MyDatasets.bw_files, "IBD_Chrone": MyDatasets.ibd_chrone_files,
                 "Male_vs_Female": MyDatasets.male_vs_female, "male_female": MyDatasets.male_vs_female,
                 "nugent": MyDatasets.nugent}

tasks_dict = {1: MyTasks.just_values, 2: MyTasks.just_graph_structure, 3: MyTasks.values_and_graph_structure,
              4: MyTasks.double_gcn_layers, 5: MyTasks.one_head_attention, 6: MyTasks.yoram_attention,
              7: MyTasks.concat_graph_and_values}

mission_dict = {1: "just_values", 2: "just_graph", 3: "graph_and_values", 4: "double_gcn_layer",
                6: "yoram_attention", 7: "concat_graph_and_values"}

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
        elif self.dataset_name == "tcr":
            # adj_mat_path = os.path.join("TCR_dataset", "distance_matrix.csv")
            adj_mat_path = os.path.join("TCR_dataset", "distance_matrix2.csv")
            cur_dataset = TCRDataset(adj_mat_path, data_path, label_path, subject_list, mission)
        elif self.dataset_name == "gdm":
            cur_dataset = ShaharGdmDataset(data_path, label_path, subject_list, mission)
        else:
            cur_dataset = GraphDataset(data_path, label_path, mission, self.add_attributes, self.geometric_mode)

        cur_dataset.update_graphs()
        return cur_dataset

    def play(self, external_test=True):
        kwargs = {}
        my_tasks = MyTasks(tasks_dict, self.dataset_name)
        my_datasets = MyDatasets(datasets_dict)

        directory_name, mission, params_file_path = my_tasks.get_task_files(self.task_number)
        result_directory_name = os.path.join(directory_name, "Result_After_Proposal")

        if not external_test:
            print("No external test set")
            train_val_test_data_file_path, train_val_test_label_file_path, subject_list\
                = my_datasets.get_dataset_files_no_external_test(self.dataset_name)
            train_val_test_dataset = self.create_dataset(train_val_test_data_file_path,
                                                         train_val_test_label_file_path,
                                                         subject_list, mission)

            trainer_and_tester = TrainTestValKTimesNoExternalTest(self.RECEIVED_PARAMS, self.device, train_val_test_dataset,
                                                    result_directory_name, nni_flag=self.nni_mode,
                                                    geometric_or_not=self.geometric_mode, plot_figures=self.plot_figures)
        else:
            print("With external test set")
            train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path, train_subject_list, \
            test_subject_list = my_datasets.get_dataset_files_yes_external_test(self.dataset_name)

            test_dataset = self.create_dataset(test_data_file_path, test_tag_file_path, test_subject_list, mission)
            train_val_dataset = self.create_dataset(train_data_file_path, train_tag_file_path, train_subject_list, mission)

            trainer_and_tester = TrainTestValKTimes(self.RECEIVED_PARAMS, self.device, train_val_dataset, test_dataset,
                                                    result_directory_name, nni_flag=self.nni_mode,
                                                    geometric_or_not=self.geometric_mode, plot_figures=self.plot_figures)
        train_metric, val_metric, test_metric, min_train_val_metric = trainer_and_tester.train_group_k_cross_validation(k=K)
        return train_metric, val_metric, test_metric, min_train_val_metric

    # def turn_on_train(self):
    #     kwargs = {}
    #     my_tasks = MyTasks(tasks_dict, self.dataset_name)
    #     my_datasets = MyDatasets(datasets_dict)
    #
    #     directory_name, mission, params_file_path = my_tasks.get_task_files(self.task_number)
    #     result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
    #     # train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = \
    #     #     my_datasets.get_dataset_files(self.dataset_name)
    #     train_data_file_path, train_tag_file_path, test_data_file_path, test_tag_file_path = \
    #         my_datasets.microbiome_files(self.dataset_name)
    #
    #     print("Training-Validation Sets Graphs")
    #     train_val_dataset = self.create_dataset(train_data_file_path, train_tag_file_path, mission)
    #     print("Final_Test set Graphs")
    #     test_dataset = self.create_dataset(test_data_file_path, test_tag_file_path, mission)
    #
    #     if mission == "yoram_attention":
    #         algorithm = "node2vec"
    #         print("Calculate embedding")
    #         graphs_list = train_val_dataset.create_microbiome_graphs.graphs_list
    #         X = find_embed(graphs_list, algorithm=algorithm)
    #         kwargs = {'X': X}
    #
    #     train_val_dataset.update_graphs(**kwargs)
    #     test_dataset.update_graphs(**kwargs)
    #
    #     trainer_and_tester = TrainTestValKTimes(self.RECEIVED_PARAMS, self.device, train_val_dataset, test_dataset,
    #                                             result_directory_name, nni_flag=self.nni_mode,
    #                                             geometric_or_not=self.geometric_mode, plot_figures=self.plot_figures)
    #     train_metric, val_metric, test_metric, min_train_val_metric = trainer_and_tester.train_group_k_cross_validation(k=K)
    #     return train_metric, val_metric, test_metric, min_train_val_metric

    # def turn_on_train_abide_dataset(self, mission):
    #     kwargs = {}
    #     data_path = "rois_ho"
    #     label_path = "Phenotypic_V1_0b_preprocessed1.csv"
    #     phenotype_dataset = pd.read_csv("Phenotypic_V1_0b_preprocessed1.csv")
    #     subject_list = [value for value in phenotype_dataset["FILE_ID"].tolist() if value != "no_filename"]
    #     # The reason for random state is that the test dataset and train-validation dataset will always be the same
    #     subject_list_train_val_index, subject_list_test_index = train_test_split(subject_list, test_size=0.3, random_state=0)
    #
    #     train_val_abide_dataset = AbideDataset(data_path, label_path, subject_list_train_val_index, mission)
    #     test_abide_dataset = AbideDataset(data_path, label_path, subject_list_test_index, mission)
    #
    #     if mission == "yoram_attention":
    #         algorithm = "node2vec"
    #         print("Calculate embedding")
    #         graphs_list = train_val_abide_dataset.graphs_list
    #         X = find_embed(graphs_list, algorithm=algorithm)
    #         kwargs = {'X': X}
    #
    #     print("ABIDE Dataset Training-Validation Sets Graphs")
    #     train_val_abide_dataset.update_graphs(**kwargs)
    #     print("ABIDE Dataset Final_Test set Graphs")
    #     test_abide_dataset.update_graphs(**kwargs)
    #     directory_name = ""
    #     result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
    #     trainer_and_tester = TrainTestValKTimes(self.RECEIVED_PARAMS, self.device, train_val_abide_dataset, test_abide_dataset,
    #                                             result_directory_name, nni_flag=self.nni_mode,
    #                                             geometric_or_not=self.geometric_mode)
    #     train_metric, val_metric, test_metric, min_train_val_metric = trainer_and_tester.train_group_k_cross_validation(k=K)
    #     return train_metric, val_metric, test_metric, min_train_val_metric


def set_arguments():
    parser = argparse.ArgumentParser(description='Main script of all models')
    parser.add_argument("--dataset", help="Dataset name", default="IBD", type=str)
    parser.add_argument("--task_number", help="Task number", default=1, type=int)
    parser.add_argument("--device_num", help="Cuda Device Number", default=3, type=int)
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
        LOG.debug("\n \nMean Validation Set metric: ", mean_min_train_val_metric, " +- ", std_min_train_val_metric)
        LOG.debug("\nMean Final_Test Set metric: ", mean_test_metric, " +- ", std_test_metric)
        nni.report_intermediate_result(mean_test_metric)
        nni.report_final_result(mean_min_train_val_metric)
    else:
        result_file_name = f"{result_file_name}_val_mean_{mean_val_metric:.3f}_test_mean_{mean_test_metric:.3f}.csv"
        f = open(result_file_name, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow([","] + [f"Run{i}" for i in range(len(val_metric))] + ["", "Mean", "std"])
        writer.writerow(['Train_metric'] + train_metric + ["", str(mean_train_metric), str(std_train_metric)])
        writer.writerow(['Val_metric'] + val_metric + ["", str(mean_val_metric), str(std_val_metric)])
        writer.writerow(['Test_metric'] + test_metric + ["", str(mean_test_metric), str(std_test_metric)])

        writer.writerow([])
        writer.writerow([])
        for key, value in RECEIVED_PARAMS.items():
            writer.writerow([key, value])
        f.close()

    print("\n \nMean minimum Validation and Train Sets AUC: ", mean_min_train_val_metric, " +- ", std_min_train_val_metric)
    print("Mean Train Set AUC: ", mean_train_metric, " +- ", std_train_metric)
    print("Mean Validation Set AUC: ", mean_val_metric, " +- ", std_val_metric)
    print("Mean Final_Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)
    return train_metric, val_metric, test_metric


def reproduce_from_nni(nni_result_file, dataset_name, mission_number):
    # mission_dict = {1: "just_values", 2: "just_graph", 3: "graph_and_values"}
    params_list = run_again_from_nni_results_csv(nni_result_file, n_rows=5)
    # params_list = run_again_from_nni_results_csv_format2(nni_result_file, n_rows=5)
    nni_flag = False
    pytorch_geometric_mode = False
    add_attributes = False
    cuda_number = 3
    device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
    print("Device", device)
    for RECEIVED_PARAMS in params_list:
        main_runner = Main(dataset_name, mission_number, RECEIVED_PARAMS, device, nni_mode=nni_flag,
                           geometric_mode=pytorch_geometric_mode, add_attributes=add_attributes, plot_figures=False)
        train_metric, val_metric, test_metric, min_train_val_metric = main_runner.play()
        result_file_name = f"{dataset_name}_{mission_dict[mission_number]}"
        results_dealing(train_metric, val_metric, test_metric, min_train_val_metric, nni_flag, RECEIVED_PARAMS, result_file_name)
        for k, v in RECEIVED_PARAMS.items():
            print(type(v))


def run_all_dataset(mission_number, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes):
    datasets = ["milk", "nut", "peanut"]
    datasets_results_dict = {}
    for dataset_name in datasets:
        try:
            train_metric, val_metric, test_metric = \
                runner(dataset_name, mission_number, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes)
            datasets_results_dict[dataset_name] = {"train_metric": train_metric, "val_metric": val_metric,
                                                   "test_metric": test_metric}
        except Exception as e:
            print(e)


def run_all_missions(dataset_name, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes):
    for mission in [1, 2, 3, 4, 6, 7]:
        try:
            runner(dataset_name, mission, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes)
        except Exception as e:
            print(e)


def run_all_datasets_missions(cuda_number, nni_flag, pytorch_geometric_mode, add_attributes):
    # mission_dict = {1: "just_values", 2: "just_graph", 3: "graph_and_values", 6: "yoram_attention"}
    for mission_number in mission_dict.keys():
        run_all_dataset(mission_number, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes)


def runner(dataset_name, mission_number, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes):
    device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
    print("Device", device)
    # mission_dict = {1: "just_values", 2: "just_graph", 3: "graph_and_values", 4: "double_gcn_layer",
    #                 6: "yoram_attention", 7: "concat_graph_and_values"}
    if dataset_name == "abide" or dataset_name == "abide1":
        params_file_path = "abide_dataset_params.json"  # TODO: add best parameters from nni
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

    elif dataset_name == "tcr":
        my_tasks = MyTasks(tasks_dict, dataset_name)
        directory_name, mission, params_file_path = my_tasks.get_task_files(mission_number)
        params_file_path = os.path.join(directory_name, 'Models', f"{mission}_params_file.json")
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
        print("Dataset", dataset_name)
        print("Task", mission)

    main_runner = Main(dataset_name, mission_number, RECEIVED_PARAMS, device, nni_mode=nni_flag,
                       geometric_mode=pytorch_geometric_mode, add_attributes=add_attributes, plot_figures=False)
    train_metric, val_metric, test_metric, min_train_val_metric = main_runner.play()
    result_file_name = f"{dataset_name}_{mission_dict[mission_number]}"
    results_dealing(train_metric, val_metric, test_metric, min_train_val_metric, nni_flag, RECEIVED_PARAMS, result_file_name)
    return train_metric, val_metric, test_metric


if __name__ == '__main__':
    try:
        parser = set_arguments()
        args = parser.parse_args()
        dataset_name = args.dataset

        mission_number = args.task_number
        cuda_number = args.device_num
        nni_flag = False if args.nni == 0 else True
        pytorch_geometric_mode = False
        add_attributes = False

        # run_all_datasets_missions(cuda_number, nni_flag, pytorch_geometric_mode, add_attributes)
        # run_all_missions(dataset_name, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes)

        runner(dataset_name, mission_number, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes)
        # run_all_dataset(7, cuda_number, nni_flag, pytorch_geometric_mode, add_attributes)

        # try:
        #     print("tcr_just_graph_nni.csv")
        #     reproduce_from_nni(os.path.join("tcr_just_graph_nni.csv"), "tcr", 2)
        # except Exception as e:
        #     print(e)
        #     raise
        #     # pass
        # try:
        #     print("tcr_graph_and_values_nni")
        #     reproduce_from_nni(os.path.join("tcr_graph_and_values_nni.csv"), "tcr", 3)
        # except Exception as e:
        #     print(e)
        #     pass
        # try:
        #     print("tcr_just_values_nni")
        #     reproduce_from_nni(os.path.join("tcr_just_values_nni.csv"), "tcr", 1)
        # except Exception as e:
        #     print(e)
        #     pass
    except Exception as e:
        LOG.exception(e)
        raise
