import argparse
import os
import sys
from os import listdir
from os.path import isfile, join

from train_test_val_ktimes_no_external_test import TrainTestValKTimesNoExternalTest

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
from train_test_val_ktimes import TrainTestValKTimes
import json
import csv
import torch
from TcrDataset import *
from HLA_TCR import *


class Main:
    def __init__(self, RECEIVED_PARAMS, device):
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.device = device

    def create_dataset(self, data_path, label_path, subject_list, mission, graph_model):
        cur_dataset = TCRDataset(data_path, label_path, subject_list, mission, graph_model)
        cur_dataset.update_graphs()
        return cur_dataset

    def play(self, kwargs):
        mission = 'concat_graph_and_values'
        graph_model = self.RECEIVED_PARAMS["graph_model"]
        train_data_file_path = config_dict["train_data_file_path"]

        train_tag_file_path = config_dict["train_tag_file_path"]
        test_data_file_path = config_dict["test_data_file_path"]

        test_tag_file_path = config_dict["test_tag_file_path"]
        label_df = pd.read_csv(train_tag_file_path)
        label_df["sample"] = label_df["sample"] + "_" + label_df['status']
        label_df.set_index("sample", inplace=True)
        train_subject_list = list(label_df[label_df["test/train"] == "train"].index)
        test_subject_list = list(label_df[label_df["test/train"] == "test"].index)

        test_dataset = self.create_dataset(test_data_file_path, test_tag_file_path, test_subject_list, mission, graph_model)
        train_val_dataset = self.create_dataset(train_data_file_path, train_tag_file_path, train_subject_list, mission, graph_model)
        trainer_and_tester = TrainTestValKTimes(self.RECEIVED_PARAMS, self.device, train_val_dataset, test_dataset, **kwargs)
        K = 20
        return_lists = trainer_and_tester.train_group_k_cross_validation(k=K)
        return return_lists


def calc_mean_and_std(dataset_metric_list):
    mean_dataset_metric = np.average(dataset_metric_list)
    std_dataset_metric = np.std(dataset_metric_list)
    return mean_dataset_metric, std_dataset_metric


def results_dealing(return_lists, RECEIVED_PARAMS, result_file_name):
    train_metric, val_metric, test_metric, min_train_val_metric, alpha_list, all_auc = return_lists
    mean_train_metric, std_train_metric = calc_mean_and_std(train_metric)
    mean_min_train_val_metric, std_min_train_val_metric = calc_mean_and_std(min_train_val_metric)
    mean_val_metric, std_val_metric = calc_mean_and_std(val_metric)
    mean_test_metric, std_test_metric = calc_mean_and_std(test_metric)

    result_file_name = f"{result_file_name}_val_mean_{mean_val_metric:.3f}_test_mean_{mean_test_metric:.3f}.csv"
    f = open(result_file_name, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow([","] + [f"Run{i}" for i in range(len(val_metric))] + ["", "Mean", "std"])
    writer.writerow(['Training_auc'] + train_metric + ["", str(mean_train_metric), str(std_train_metric)])
    writer.writerow(['Val_auc'] + val_metric + ["", str(mean_val_metric), str(std_val_metric)])
    writer.writerow(['Test_auc'] + test_metric + ["", str(mean_test_metric), str(std_test_metric)])

    writer.writerow([])
    writer.writerow([])
    for key, value in RECEIVED_PARAMS.items():
        writer.writerow([key, value])
    writer.writerow(["All auc", all_auc])
    f.close()

    print("\n \nMean minimum Validation and Train Sets AUC: ", mean_min_train_val_metric, " +- ", std_min_train_val_metric)
    print("Mean Training Set AUC: ", mean_train_metric, " +- ", std_train_metric)
    print("Mean Validation Set AUC: ", mean_val_metric, " +- ", std_val_metric)
    print("Mean Test Set AUC: ", mean_test_metric, " +- ", std_test_metric)
    return train_metric, val_metric, test_metric


def runner(**kwargs):
    cuda_number = int(kwargs["cuda_number"])
    device = f"cuda:{cuda_number}" if torch.cuda.is_available() else "cpu"
    print("Device", device)
    params_file_path = kwargs["params_file_path"]
    RECEIVED_PARAMS = json.load(open(params_file_path, 'r'))
    print("Hyper-parameters", RECEIVED_PARAMS)
    main_runner = Main(RECEIVED_PARAMS, device)
    return_lists = main_runner.play(kwargs)
    result_file_name = f"tcr_concat_graph_and_values"
    results_dealing(return_lists, RECEIVED_PARAMS, result_file_name)
    return return_lists

def set_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Config file path", default="config_file.json", type=str)
    parser.add_argument("--params", help="Hyper parameters file path", default="tcr_concat_graph_and_values.json", type=str)
    return parser


if __name__ == '__main__':
    try:
        parser = set_arguments()
        args = parser.parse_args()
        config_file_path = args.config
        params_file_path = args.params
        # config_file_path = "config_file.json"
        config_dict = json.load(open(config_file_path, 'r'))
        config_dict["params_file_path"] = params_file_path
        runner(**config_dict)
        # run_alleles(**config_dict)
    except Exception as e:
        # print(e)
        raise
