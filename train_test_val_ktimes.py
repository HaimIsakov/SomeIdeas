import json
import os
from datetime import datetime

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, RepeatedStratifiedKFold, train_test_split, StratifiedShuffleSplit
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch
from train_test_val_one_time import TrainTestValOneTime
from JustGraphStructure.Models.just_graph_structure import JustGraphStructure
from JustValues.Models.just_values_fc_binary_classification import JustValuesOnNodes
from ValuesAndGraphStructure.Models.values_and_graph_structure import ValuesAndGraphStructure

LOSS_PLOT = 'loss'
ACCURACY_PLOT = 'acc'
AUC_PLOT = 'auc'
TRAIN_JOB = 'train'
TEST_JOB = 'test'


class TrainTestValKTimes:
    def __init__(self, mission, RECEIVED_PARAMS, number_of_runs, device, dataset, result_directory_name):
        self.mission = mission
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.device = device
        self.dataset = dataset
        self.result_directory_name = result_directory_name
        self.number_of_runs = number_of_runs
        self.node_order = self.dataset.node_order

    def train_k_splits_of_dataset(self):
        for i in range(self.number_of_runs):
            print(f"Run {i}")
            if i == 0:
                root = self.result_directory_name
                date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
                directory_root = os.path.join(root, f'{self.mission}_{date}')
                os.mkdir(directory_root)
            gss = self.create_gss(k=1)
            train_idx, test_idx = next(gss.split(self.dataset, groups=self.dataset.groups))
            train_loader, val_loader, test_loader = self.create_data_loaders(train_idx, test_idx)
            model = self.get_model()
            model = model.to(self.device)
            trainer_and_tester = TrainTestValOneTime(model, self.RECEIVED_PARAMS, train_loader, val_loader, test_loader,
                                                     self.device, self.node_order)
            trainer_and_tester.train()
            os.mkdir(os.path.join(directory_root, f"Run{i}"))
            root = os.path.join(directory_root, f"Run{i}")
            f = open(os.path.join(root, f"Test Auc {trainer_and_tester.test_auc}"))
            f.close()
            self.plot_acc_loss_auc(root, date, trainer_and_tester)

    def train_k_cross_validation_of_dataset(self, k=5):
        gss = self.create_gss(k=k)
        run = 0
        test_metric = []
        for train_idx, test_idx in gss.split(self.dataset, groups=self.dataset.groups):
            print(f"Run {run}")
            if run == 0:
                root = self.result_directory_name
                date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
                directory_root = os.path.join(root, f'{self.mission}_{date}')
                os.mkdir(directory_root)

            samples_len = len(self.dataset)
            val_idx = np.array(list(set(range(samples_len)) - set(test_idx) - set(train_idx)))
            train_loader, val_loader, test_loader = self.create_data_loaders(train_idx, val_idx, test_idx)
            model = self.get_model()
            model = model.to(self.device)
            trainer_and_tester = TrainTestValOneTime(model, self.RECEIVED_PARAMS, train_loader, val_loader, test_loader,
                                                     self.device, self.node_order)
            trainer_and_tester.train()
            print("Test Auc", trainer_and_tester.test_auc)
            test_metric.append(trainer_and_tester.test_auc)
            os.mkdir(os.path.join(directory_root, f"Run{run}"))
            root = os.path.join(directory_root, f"Run{run}")
            f = open(os.path.join(directory_root, f"Test_Auc_{trainer_and_tester.test_auc:.9f}.txt"), 'w')
            f.close()
            self.plot_acc_loss_auc(root, date, trainer_and_tester)
            run += 1
        return test_metric

    def stratify_train_val_test_ksplits(self, n_splits=5, n_repeats=5):
        test_metric = []
        all_labels = self.dataset.labels
        train_val_idx, test_idx, train_val_y, test_y = train_test_split(np.arange(len(self.dataset)), all_labels,
                                                                        test_size=0.2, train_size=0.8,
                                                                        stratify=all_labels, shuffle=True)
        for run in range(n_repeats):
            print(f"Run {run}")
            if run == 0:
                root = self.result_directory_name
                date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
                directory_root = os.path.join(root, f'{self.mission}_{date}')
                os.mkdir(directory_root)
            # now we split again to get the validation
            train_idx, val_idx, y_train, y_val = train_test_split(train_val_idx, train_val_y, train_size=0.84375,
                                                                  stratify=train_val_y, shuffle=True)
            train_loader, val_loader, test_loader = self.create_data_loaders(train_idx, val_idx, test_idx)
            model = self.get_model()
            model = model.to(self.device)
            trainer_and_tester = TrainTestValOneTime(model, self.RECEIVED_PARAMS, train_loader, val_loader, test_loader,
                                                     self.device, self.node_order)
            trainer_and_tester.train()
            print("Test Auc", trainer_and_tester.test_auc)
            test_metric.append(trainer_and_tester.test_auc)
            os.mkdir(os.path.join(directory_root, f"Run{run}"))
            root = os.path.join(directory_root, f"Run{run}")
            f = open(os.path.join(directory_root, f"Test_Auc_{trainer_and_tester.test_auc:.9f}.txt"), 'w')
            f.close()
            self.plot_acc_loss_auc(root, date, trainer_and_tester)
            run += 1
        return test_metric

    def create_gss(self, k=1):
        train_frac = self.RECEIVED_PARAMS['train_frac']
        test_frac = self.RECEIVED_PARAMS['test_frac']
        gss = GroupShuffleSplit(n_splits=k, train_size=train_frac, test_size=test_frac)
        return gss

    def create_data_loaders(self, train_idx, val_idx, test_idx):
        # len_train = int(samples_len * self.RECEIVED_PARAMS['train_frac'])
        # len_test = samples_len - len_train
        # train, test = random_split(self.dataset, [len_train, len_test])
        # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(test, batch_size=batch_size)
        # samples_len = len(self.dataset)
        batch_size = self.RECEIVED_PARAMS['batch_size']
        # val_idx = np.array(list(set(range(samples_len)) - set(test_idx) - set(train_idx)))
        # Datasets
        train_data = torch.utils.data.Subset(self.dataset, train_idx)
        test_data = torch.utils.data.Subset(self.dataset, test_idx)
        val_data = torch.utils.data.Subset(self.dataset, val_idx)
        # Dataloader
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        # Notice that the test data is loaded as one unit.
        test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=len(test_data))
        return train_loader, val_loader, test_loader

    def get_model(self):
        if self.mission == "JustValues":
            data_size = self.dataset.get_leaves_number()
            model = JustValuesOnNodes(data_size, self.RECEIVED_PARAMS)
        elif self.mission == "JustGraphStructure":
            data_size = self.dataset.get_vector_size()
            model = JustGraphStructure(data_size, self.RECEIVED_PARAMS, self.device)
        elif self.mission == "GraphStructure&Values":
            data_size = self.dataset.get_vector_size()
            model = ValuesAndGraphStructure(data_size, self.RECEIVED_PARAMS, self.device)
        return model

    def stratified_group_train_test_split(self, samples: pd.DataFrame, group: str, stratify_by: str, test_size: float):
        groups = samples[group].drop_duplicates()
        stratify = samples.drop_duplicates(group)[stratify_by].to_numpy()
        groups_train, groups_test = train_test_split(groups, stratify=stratify, test_size=test_size)

        samples_train = samples.loc[lambda d: d[group].isin(groups_train)]
        samples_test = samples.loc[lambda d: d[group].isin(groups_test)]

        samples_train.sort_index(inplace=True)
        samples_test.sort_index(inplace=True)
        return samples_train, samples_test

    def plot_measurement(self, root, date, trainer_and_tester, measurement=LOSS_PLOT):
        if measurement == LOSS_PLOT:
            plt.plot(range(len(trainer_and_tester.train_loss_vec)), trainer_and_tester.train_loss_vec, label=TRAIN_JOB, color='b')
            plt.plot(range(len(trainer_and_tester.test_loss_vec)), trainer_and_tester.test_loss_vec, label=TEST_JOB, color='g')
            plt.legend(loc='best')
            plt.savefig(os.path.join(root, f'loss_plot_{date}.png'))
            plt.clf()

        # if measurement == ACCURACY_PLOT:
        #     max_acc_test = np.max(self._accuracy_vec_test)
        #     plt.plot(range(len(self._accuracy_vec_train)), self._accuracy_vec_train, label=TRAIN_JOB, color='b')
        #     plt.plot(range(len(self._accuracy_vec_test)), self._accuracy_vec_test, label=TEST_JOB, color='g')
        #     plt.legend(loc='best')
        #     plt.savefig(os.path.join(root, f'acc_plot_{date}_max_{round(max_acc_test, 2)}.png'))
        #     plt.clf()

        if measurement == AUC_PLOT:
            # final_auc_test = trainer_and_tester.test_auc_vec[-1]
            plt.plot(range(len(trainer_and_tester.train_auc_vec)), trainer_and_tester.train_auc_vec, label=TRAIN_JOB, color='b')
            plt.plot(range(len(trainer_and_tester.test_auc_vec)), trainer_and_tester.test_auc_vec, label=TEST_JOB, color='g')
            plt.ylim((0, 1))
            plt.legend(loc='best')
            plt.savefig(os.path.join(root, f'auc_plot_{date}_last_{round(trainer_and_tester.test_auc, 2)}.png'))
            plt.clf()

    def _binary_acc(self, y_pred, y_test):
        # TODO: Fix accuracy calculation
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)
        return acc

    def plot_acc_loss_auc(self, root, date, trainer_and_tester):
        # root = os.path.join(root, f'Values_and_graph_structure_on_nodes_model_{date}')
        # os.mkdir(root)
        with open(os.path.join(root, "params_file.json"), 'w') as pf:
            json.dump(self.RECEIVED_PARAMS, pf)
        # copyfile(params_file, os.path.join(root, "params_file.json"))
        self.plot_measurement(root, date, trainer_and_tester, LOSS_PLOT)
        # self.plot_measurement(root, date, ACCURACY_PLOT)
        self.plot_measurement(root, date, trainer_and_tester, AUC_PLOT)
