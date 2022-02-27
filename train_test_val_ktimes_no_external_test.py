import os
import random
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader
from pathlib import Path

from distance_matrix import create_distance_matrix
from ofek_files_utils_functions import HistoMaker
from train_test_val_ktimes_utils import *
from ConcatGraphAndValues.concat_graph_and_values import ConcatValuesAndGraphStructure
from DoubleGcnLayers.Models.two_gcn_layers_graph_and_values import TwoLayersGCNValuesGraph
from OneHeadAttention.Models.ofek_model import AttentionGCN
from YoramAttention.Models.yoram_attention import YoramAttention
from train_test_val_one_time import TrainTestValOneTime
from JustGraphStructure.Models.just_graph_structure import JustGraphStructure
from JustValues.Models.just_values_fc_binary_classification import JustValuesOnNodes
from ValuesAndGraphStructure.Models.values_and_graph_structure import ValuesAndGraphStructure
from node2vec_embed import find_embed


class TrainTestValKTimesNoExternalTest:
    def __init__(self, RECEIVED_PARAMS, device, train_val_test_dataset, result_directory_name, nni_flag=False,
                 geometric_or_not=False, plot_figures=False, **kwargs):
        # self.mission = mission
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.device = device
        self.train_val_test_dataset = train_val_test_dataset
        self.result_directory_name = result_directory_name
        self.nni_flag = nni_flag
        self.geometric_or_not = geometric_or_not
        self.plot_figures = plot_figures
        self.kwargs = kwargs

    def train_group_k_cross_validation(self, k=5):
        dataset_len = len(self.train_val_test_dataset)
        train_metric, val_metric, test_metric, min_train_val_metric = [], [], [], []
        run = 0
        for i in range(k):
            indexes_array = np.array(range(dataset_len))
            train_idx, test_idx = train_test_split(indexes_array, test_size=0.2, shuffle=True)
            train_idx, val_idx = train_test_split(train_idx, test_size=0.25, shuffle=True)  # 0.25 x 0.8 = 0.2
            print("Val indexes", val_idx)
            print(f"Run {run}")
            # print("len of train set:", len(train_idx))
            # print("len of val set:", len(val_idx))

            train_loader, val_loader, test_loader = self.create_data_loaders(i, train_idx, val_idx, test_idx)
            print("Train labels", get_labels_distribution(train_loader))
            print("val labels", get_labels_distribution(val_loader))
            print("test labels", get_labels_distribution(test_loader))
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
            run += 1
        return train_metric, val_metric, test_metric, min_train_val_metric

    def tcr_dataset_dealing(self, train_idx, i):
        print(self.kwargs)
        if "samples" not in self.kwargs:
            random_sample_from_train = len(train_idx)
        elif self.kwargs["samples"] == -1:
            random_sample_from_train = len(train_idx)
        else:
            random_sample_from_train = int(self.kwargs["samples"])
        print(f"\nTake only {random_sample_from_train} from the training set\n")

        print("Here, we ----do---- calculate again the golden-tcrs")
        train = HistoMaker("train", random_sample_from_train)
        if "ISB" in str(self.train_val_test_dataset):
            file_directory_path = os.path.join("covid", "new_ISB")
        elif "NIH" in str(self.train_val_test_dataset):
            file_directory_path = os.path.join("covid", "new_NIH")
        # sample only some sample according to input sample size, and calc the golden tcrs only from them
        train_idx = random.sample(list(train_idx), random_sample_from_train)
        train_files = [Path(os.path.join(file_directory_path, self.train_val_test_dataset.subject_list[id] + ".csv"))
                       for id in train_idx]
        print("train_files", train_files)
        print("Length of chosen files", len(train_files))
        numrec = int(self.RECEIVED_PARAMS["numrec"])  # cutoff is also a hyper-parameter
        print("Number of golden-tcrs", numrec)
        train.save_data(file_directory_path, files=train_files)
        # save files' names
        outliers_pickle_name = f"nni_outliers_with_sample_size_{len(train_files)}_run_number_{i}"
        adj_mat_path = f"nni_dist_mat_with_sample_size_{len(train_files)}_run_number_{i}"
        train.new_outlier_finder(numrec, pickle_name=outliers_pickle_name)  # find outliers and save to pickle
        # create distance matrix between the projection of the found golden tcrs
        create_distance_matrix(self.device, outliers_file=outliers_pickle_name, adj_mat=adj_mat_path)
        self.train_val_test_dataset.run_number = i
        self.train_val_test_dataset.run_number = i
        self.train_val_test_dataset.calc_golden_tcrs(adj_mat_path=adj_mat_path)
        self.train_val_test_dataset.update_graphs()
        return train_idx

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

    def create_data_loaders(self, i, train_idx, val_idx, test_idx):
        batch_size = int(self.RECEIVED_PARAMS['batch_size'])
        # For Tcr dataset
        if "ISB" in str(self.train_val_test_dataset) or "NIH" in str(self.train_val_test_dataset):
            train_idx = self.tcr_dataset_dealing(train_idx, i)
        # Datasets
        train_data = torch.utils.data.Subset(self.train_val_test_dataset, train_idx)
        val_data = torch.utils.data.Subset(self.train_val_test_dataset, val_idx)
        test_data = torch.utils.data.Subset(self.train_val_test_dataset, test_idx)
        # Get and set train_graphs_list
        # train_graphs_list = get_train_graphs_list(train_data)
        # self.train_val_test_dataset.set_train_graphs_list(train_graphs_list)
        # self.find_embed_for_attention()
        # Dataloader
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
        print("train loader size:", len(train_idx))
        print("val loader size:", len(val_idx))
        print("test loader size:", len(test_idx))
        return train_loader, val_loader, test_loader

    # def get_train_graphs_list(self, train_data):
    #     dataset_dict = train_data.dataset.dataset_dict
    #     indices = train_data.indices
    #     graphs_dict = {}
    #     for index in indices:
    #         graphs_dict[index] = dataset_dict[index]['graph']
    #     train_graphs_list = list(graphs_dict.values())
    #     return train_graphs_list

    def find_embed_for_attention(self):
        if "embedding_algorithm" not in self.RECEIVED_PARAMS or self.train_val_test_dataset.mission != "yoram_attention":
            return
        algorithm = self.RECEIVED_PARAMS["embedding_algorithm"]
        print(f"Calculate {algorithm} embedding")
        # We want to embed only the train set
        graphs_list = self.train_val_test_dataset.train_graphs_list
        graph_embedding_matrix = find_embed(graphs_list, algorithm=algorithm)
        self.train_val_test_dataset.set_graph_embed_in_dataset_dict(graph_embedding_matrix)

    def get_model(self):
        if not self.geometric_or_not:
            if self.train_val_test_dataset.mission == "just_values":
                # nodes_number - changed only for abide dataset
                data_size = self.train_val_test_dataset.get_vector_size()
                nodes_number = self.train_val_test_dataset.get_leaves_number()
                model = JustValuesOnNodes(nodes_number, data_size, self.RECEIVED_PARAMS)
            elif self.train_val_test_dataset.mission == "just_graph":
                data_size = self.train_val_test_dataset.get_vector_size()
                nodes_number = self.train_val_test_dataset.nodes_number()
                model = JustGraphStructure(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
            elif self.train_val_test_dataset.mission == "graph_and_values":
                data_size = self.train_val_test_dataset.get_vector_size()
                nodes_number = self.train_val_test_dataset.nodes_number()
                model = ValuesAndGraphStructure(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
            elif self.train_val_test_dataset.mission == "double_gcn_layer":
                data_size = self.train_val_test_dataset.get_vector_size()
                nodes_number = self.train_val_test_dataset.nodes_number()
                model = TwoLayersGCNValuesGraph(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
            elif self.train_val_test_dataset.mission == "concat_graph_and_values":
                data_size = self.train_val_test_dataset.get_vector_size()
                nodes_number = self.train_val_test_dataset.nodes_number()
                model = ConcatValuesAndGraphStructure(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
            elif self.train_val_test_dataset.mission == "one_head_attention":
                # data_size = self.train_val_dataset.get_vector_size()
                data_size = 128
                nodes_number = self.train_val_test_dataset.nodes_number()
                model = AttentionGCN(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
            elif self.train_val_test_dataset.mission == "yoram_attention":
                data_size = self.train_val_test_dataset.get_vector_size()
                # data_size = 128
                nodes_number = self.train_val_test_dataset.nodes_number()
                model = YoramAttention(nodes_number, data_size, self.RECEIVED_PARAMS, self.device)
        else:
            data_size = self.train_val_test_dataset.get_vector_size()
            model = GCN(1, self.RECEIVED_PARAMS, self.device)
        return model
