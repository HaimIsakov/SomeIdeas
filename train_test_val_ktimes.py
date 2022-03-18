

# import torch_geometric.data
# from pytorch_geometric import GCN
import json
import os
import pickle
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

from ConcatGraphAndValues.concat_graph_and_values import ConcatValuesAndGraphStructure
from DoubleGcnLayers.Models.two_gcn_layers_graph_and_values import TwoLayersGCNValuesGraph
from OneHeadAttention.Models.ofek_model import AttentionGCN
from YoramAttention.Models.yoram_attention import YoramAttention
from distance_matrix import create_distance_matrix
from node2vec_embed import find_embed
from ofek_files_utils_functions import HistoMaker
from train_test_val_one_time import TrainTestValOneTime
from JustGraphStructure.Models.just_graph_structure import JustGraphStructure
from JustValues.Models.just_values_fc_binary_classification import JustValuesOnNodes
from ValuesAndGraphStructure.Models.values_and_graph_structure import ValuesAndGraphStructure
from train_test_val_ktimes_utils import *

LOSS_PLOT = 'loss'
ACCURACY_PLOT = 'acc'
AUC_PLOT = 'auc'
TRAIN_JOB = 'train'
TEST_JOB = 'test'


class TrainTestValKTimes:
    def __init__(self, RECEIVED_PARAMS, device, train_val_dataset, test_dataset,
                 result_directory_name, nni_flag=False, geometric_or_not=False, plot_figures=False, **kwargs):
        # self.mission = mission
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.device = device
        self.train_val_dataset = train_val_dataset
        self.test_dataset = test_dataset
        self.result_directory_name = result_directory_name
        self.nni_flag = nni_flag
        self.geometric_or_not = geometric_or_not
        self.plot_figures = plot_figures
        self.kwargs = kwargs
        # self.node_order = self.dataset.node_order

    def train_group_k_cross_validation(self, k=5):
        # train_frac = float(self.RECEIVED_PARAMS['train_frac'])
        # val_frac = float(self.RECEIVED_PARAMS['test_frac'])
        dataset_len = len(self.train_val_dataset)
        train_metric, val_metric, test_metric, min_train_val_metric = [], [], [], []
        # gss_train_val = GroupShuffleSplit(n_splits=k, train_size=0.75)
        # gss_train_val = GroupKFold(n_splits=k)
        run = 0
        for i in range(k):
            indexes_array = np.array(range(dataset_len))
            # Add seed for tcr dataset hyper-parameters tuning
            # if self.nni_flag and "TCR" in str(self.train_val_dataset):
            # if "TCR" in str(self.train_val_dataset):
            # # TODO : Remove random state
            #     print("Random state", i)
            #     train_idx, val_idx = train_test_split(indexes_array, test_size=0.2, shuffle=True, random_state=i)
            # else:
            train_idx, val_idx = train_test_split(indexes_array, test_size=0.2, shuffle=True)
            print(f"Run {run}")
            # print("len of train set:", len(train_idx))
            # print("len of val set:", len(val_idx))
            if run == 0 and not self.nni_flag and self.plot_figures:
                date, directory_root = self.create_directory_to_save_results()

            train_loader, val_loader, test_loader = self.create_data_loaders(i, train_idx, val_idx)
            print("Train labels", get_labels_distribution(train_loader))
            print("val labels", get_labels_distribution(val_loader))
            # print("test labels", get_labels_distribution(test_loader))
            model = self.get_model().to(self.device)
            trainer_and_tester = TrainTestValOneTime(model, self.RECEIVED_PARAMS, train_loader, val_loader, test_loader,
                                                     self.device)

            early_stopping_results = self.start_training_process(trainer_and_tester, train_loader, val_loader, test_loader)
            if len(trainer_and_tester.alpha_list) > 0:
                print(trainer_and_tester.alpha_list)
            min_val_train_auc = min(early_stopping_results['val_auc'], early_stopping_results['train_auc'])
            print("Minimum Validation and Train Auc", min_val_train_auc)
            min_train_val_metric.append(min_val_train_auc)  # the minimum between the aucs between train set and validation set
            train_metric.append(early_stopping_results['train_auc'])
            val_metric.append(early_stopping_results['val_auc'])
            test_metric.append(early_stopping_results['test_auc'])

            # if not self.nni_flag and self.plot_figures:
            #     os.mkdir(os.path.join(directory_root, f"Run{run}"))
            #     root = os.path.join(directory_root, f"Run{run}")
            #     f = open(os.path.join(directory_root, f"Validation_Auc_{trainer_and_tester.val_auc:.9f}.txt"), 'w')
            #     f.close()
            #     self.plot_acc_loss_auc(root, date, trainer_and_tester)
            run += 1
        return train_metric, val_metric, test_metric, min_train_val_metric

    def start_training_process(self, trainer_and_tester, train_loader, val_loader, test_loader):
        rerun_counter = 0
        if not self.geometric_or_not:
            early_stopping_results = trainer_and_tester.train()
        else:
            early_stopping_results = trainer_and_tester.train_geometric()

        # # If the train auc is too low (under 0.5 for example) try to rerun the training process again
        flag = rerun_if_bad_train_result(early_stopping_results)
        while flag and rerun_counter <= 3:
            print(f"Rerun this train-val split again because train auc is:{early_stopping_results['train_auc']:.4f}")
            print(f"Rerun number {rerun_counter}")
            model = self.get_model().to(self.device)
            trainer_and_tester = TrainTestValOneTime(model, self.RECEIVED_PARAMS, train_loader, val_loader,
                                                     test_loader,
                                                     self.device)
            if not self.geometric_or_not:
                early_stopping_results = trainer_and_tester.train()
            else:
                early_stopping_results = trainer_and_tester.train_geometric()

            flag = rerun_if_bad_train_result(early_stopping_results)
            rerun_counter += 1  # rerun_counter - the number of chances we give the model to converge again
        return early_stopping_results

    def tcr_dataset_dealing(self, train_idx, i):
        # adj_mat_path = f"dist_mat_{i}.csv"
        # if not self.nni_flag or not os.path.isfile(adj_mat_path):
        # if not os.path.isfile(adj_mat_path):
        print(self.kwargs)
        if "samples" not in self.kwargs:
            random_sample_from_train = len(train_idx)
        elif self.kwargs["samples"] == -1:
            random_sample_from_train = len(train_idx)
        else:
            random_sample_from_train = int(self.kwargs["samples"])
        print(f"\nTake only {random_sample_from_train} from the training set\n")

        # if not self.nni_flag:
        print("Here, we ----do---- calculate again the golden-tcrs")
        train = HistoMaker("train", random_sample_from_train)
        file_directory_path = os.path.join("TCR_Dataset2", "Train")  # TCR_Dataset2 exists only in server
        # sample only some sample according to input sample size, and calc the golden tcrs only from them
        train_idx = random.sample(list(train_idx), random_sample_from_train)
        train_files = [Path(os.path.join(file_directory_path, self.train_val_dataset.subject_list[id] + ".csv"))
                       for id in train_idx]
        print("Length of chosen files", len(train_files))
        numrec = int(self.RECEIVED_PARAMS["numrec"])  # cutoff is also a hyper-parameter
        print("Number of golden-tcrs", numrec)
        # cutoff = 7.0
        train.save_data(file_directory_path, files=train_files)
        # train.outlier_finder(i, numrec=numrec, cutoff=cutoff)
        # save files' names
        outliers_pickle_name = f"nni_tcr_outliers_with_sample_size_{len(train_files)}_run_number_{i}"
        adj_mat_path = f"nni__tcr_dist_mat_with_sample_size_{len(train_files)}_run_number_{i}"
        train.new_outlier_finder(numrec, pickle_name=outliers_pickle_name)  # find outliers and save to pickle
        # create distance matrix between the projection of the found golden tcrs
        create_distance_matrix(self.device, outliers_file=outliers_pickle_name, adj_mat=adj_mat_path)
        self.train_val_dataset.run_number = i
        self.test_dataset.run_number = i
        self.train_val_dataset.calc_golden_tcrs(adj_mat_path=adj_mat_path)
        self.train_val_dataset.update_graphs()
        self.test_dataset.calc_golden_tcrs(adj_mat_path=adj_mat_path)
        self.test_dataset.update_graphs()
        # else:
        #     print("Here, we ----do not---- calculate again the golden-tcrs")
        #     pkl_train_dataset = os.path.join("tcr_samples_pkl_files", f"tcr_dataset_dict_train_{i}_samples_{random_sample_from_train}.pkl")  # f"dataset_dict_train_{i}.pkl"
        #     pkl_test_dataset = os.path.join("tcr_samples_pkl_files", f"tcr_dataset_dict_test_{i}_samples_{random_sample_from_train}.pkl") # f"dataset_dict_test_{i}.pkl"
        #     pkl_train_idx = os.path.join("tcr_samples_pkl_files", f"train_idx_{i}_samples_{random_sample_from_train}.pkl")
        #     print(f"Load train dataset pickle from {pkl_train_dataset}")
        #     print(f"Load train dataset pickle from {pkl_test_dataset}")
        #     print(f"Load train idx pickle from {pkl_train_idx}")
        #     self.train_val_dataset.dataset_dict = pickle.load(open(f"{pkl_train_dataset}", 'rb'))
        #     self.test_dataset.dataset_dict = pickle.load(open(f"{pkl_test_dataset}", 'rb'))
        #     train_files = pickle.load(open(f"{pkl_train_idx}", 'rb'))
        #     train_idx = [self.train_val_dataset.subject_list.index(train_file.replace(".csv"))
        #                    for train_file in train_files]
        return train_idx

    def create_data_loaders(self, i, train_idx, val_idx):
        batch_size = int(self.RECEIVED_PARAMS['batch_size'])
        if not self.geometric_or_not:
            # For Tcr dataset
            if "TCR" in str(self.train_val_dataset):
                train_idx = self.tcr_dataset_dealing(train_idx, i)
            # Datasets
            train_data = torch.utils.data.Subset(self.train_val_dataset, train_idx)
            print("len of train data", len(train_data))
            val_data = torch.utils.data.Subset(self.train_val_dataset, val_idx)
            print("len of val data", len(val_data))
            # Get and set train_graphs_list
            train_graphs_list = get_train_graphs_list(train_data)
            self.train_val_dataset.set_train_graphs_list(train_graphs_list)
            self.test_dataset.set_train_graphs_list(train_graphs_list)
            self.find_embed_for_attention()

            # random_sample_from_train = int(self.kwargs["samples"])
            # with open(f"tcr_dataset_dict_train_{i}_samples_{random_sample_from_train}.pkl", "wb") as f:
            #     pickle.dump(self.train_val_dataset.dataset_dict, f)

            # with open(f"tcr_dataset_dict_test_{i}_samples_{random_sample_from_train}.pkl", "wb") as f:
            #     pickle.dump(self.test_dataset.dataset_dict, f)

            # Dataloader
            train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        else:
            # Datasets
            train_data = torch.utils.data.Subset(self.train_val_dataset, train_idx)
            val_data = torch.utils.data.Subset(self.train_val_dataset, val_idx)
            # Dataloader
            train_loader = torch_geometric.data.DataLoader(train_data, shuffle=True, batch_size=batch_size, exclude_keys=['val'])
            val_loader = torch_geometric.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, exclude_keys=['val'])
            # Notice that the test data is loaded as one unit.
            test_loader = torch_geometric.data.DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False,exclude_keys=['val'])
        return train_loader, val_loader, test_loader

    def find_embed_for_attention(self):
        if "embedding_algorithm" not in self.RECEIVED_PARAMS or self.train_val_dataset.mission != "yoram_attention":
            return
        algorithm = self.RECEIVED_PARAMS["embedding_algorithm"]
        print(f"Calculate {algorithm} embedding")
        # We want to embed only the train set
        graphs_list = self.train_val_dataset.train_graphs_list
        graph_embedding_matrix = find_embed(graphs_list, algorithm=algorithm)
        self.train_val_dataset.set_graph_embed_in_dataset_dict(graph_embedding_matrix)
        self.test_dataset.set_graph_embed_in_dataset_dict(graph_embedding_matrix)

    def get_model(self):
        if not self.geometric_or_not:
            mission = self.train_val_dataset.mission
            if mission == "just_values":
                # nodes_number - changed only for abide dataset
                data_size = self.train_val_dataset.get_vector_size()
                nodes_number = self.train_val_dataset.get_leaves_number()
                model = JustValuesOnNodes(nodes_number, data_size, self.RECEIVED_PARAMS)
            elif mission == "just_graph":
                data_size = self.train_val_dataset.get_vector_size()
                nodes_number = self.train_val_dataset.nodes_number()
                model = JustGraphStructure(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
            elif mission == "graph_and_values":
                data_size = self.train_val_dataset.get_vector_size()
                nodes_number = self.train_val_dataset.nodes_number()
                model = ValuesAndGraphStructure(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
            elif mission == "double_gcn_layer":
                data_size = self.train_val_dataset.get_vector_size()
                nodes_number = self.train_val_dataset.nodes_number()
                model = TwoLayersGCNValuesGraph(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
            elif mission == "concat_graph_and_values":
                data_size = self.train_val_dataset.get_vector_size()
                nodes_number = self.train_val_dataset.nodes_number()
                model = ConcatValuesAndGraphStructure(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
            elif mission == "one_head_attention":
                # data_size = self.train_val_dataset.get_vector_size()
                data_size = 128
                nodes_number = self.train_val_dataset.nodes_number()
                model = AttentionGCN(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
            elif mission == "yoram_attention":
                data_size = self.train_val_dataset.get_vector_size()
                # data_size = 128
                nodes_number = self.train_val_dataset.nodes_number()
                model = YoramAttention(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
        else:
            data_size = self.train_val_dataset.get_vector_size()
            model = GCN(1, self.RECEIVED_PARAMS, self.device)
        return model

    def create_directory_to_save_results(self):
        root = self.result_directory_name
        date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
        directory_root = os.path.join(root, f'{self.train_val_dataset.mission}_{date}')
        if not os.path.exists(directory_root):
            os.mkdir(directory_root)
        else:
            directory_root = directory_root+"_extra"
            os.mkdir(directory_root)
        return date, directory_root

    def plot_measurement(self, root, date, trainer_and_tester, measurement=LOSS_PLOT):
        if measurement == LOSS_PLOT:
            plt.plot(range(len(trainer_and_tester.train_loss_vec)), trainer_and_tester.train_loss_vec, label=TRAIN_JOB, color='b')
            plt.plot(range(len(trainer_and_tester.val_loss_vec)), trainer_and_tester.val_loss_vec, label=TEST_JOB, color='g')
            plt.legend(loc='best')
            plt.savefig(os.path.join(root, f'loss_plot_{date}.png'))
            plt.clf()

        if measurement == AUC_PLOT:
            # final_auc_test = trainer_and_tester.test_auc_vec[-1]
            plt.plot(range(len(trainer_and_tester.train_auc_vec)), trainer_and_tester.train_auc_vec, label=TRAIN_JOB, color='b')
            plt.plot(range(len(trainer_and_tester.val_auc_vec)), trainer_and_tester.val_auc_vec, label=TEST_JOB, color='g')
            plt.ylim((0, 1))
            plt.legend(loc='best')
            plt.savefig(os.path.join(root, f'auc_plot_{date}_last_{round(trainer_and_tester.val_auc, 2)}.png'))
            plt.clf()

    def plot_acc_loss_auc(self, root, date, trainer_and_tester):
        # root = os.path.join(root, f'Values_and_graph_structure_on_nodes_model_{date}')
        # os.mkdir(root)
        with open(os.path.join(root, "params_file_1_gcn_just_values.json"), 'w') as pf:
            json.dump(self.RECEIVED_PARAMS, pf)
        # copyfile(params_file_1_gcn_just_values, os.path.join(root, "params_file_1_gcn_just_values.json"))
        self.plot_measurement(root, date, trainer_and_tester, LOSS_PLOT)
        # self.plot_measurement(root, date, ACCURACY_PLOT)
        self.plot_measurement(root, date, trainer_and_tester, AUC_PLOT)