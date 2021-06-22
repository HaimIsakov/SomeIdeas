import json
import os
from datetime import datetime
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

    def train_k_splits_of_dataset(self):
        final_results_vec = []
        for i in range(self.number_of_runs):
            print(f"Run {i}")
            if i == 0:
                root = self.result_directory_name
                date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
                directory_root = os.path.join(root, f'{self.mission}_{date}')
                os.mkdir(directory_root)

            # device = "cuda:2" if torch.cuda.is_available() else "cpu"
            # print(self.device)
            train_loader, test_loader = self.create_data_loaders()
            model = self.get_model()
            model = model.to(self.device)
            trainer_and_tester = TrainTestValOneTime(model, self.RECEIVED_PARAMS, train_loader, test_loader, self.device)
            final_results_vec.append(trainer_and_tester.test_auc_vec[-1])
            os.mkdir(os.path.join(directory_root, f"Run{i}"))
            root = os.path.join(directory_root, f"Run{i}")
            self.plot_acc_loss_auc(root, date, trainer_and_tester)
        print(f"Auc on test set {np.mean(final_results_vec)}")

    def create_data_loaders(self):
        batch_size = self.RECEIVED_PARAMS['batch_size']
        samples_len = len(self.dataset)
        len_train = int(samples_len * self.RECEIVED_PARAMS['train_frac'])
        len_test = samples_len - len_train

        train, test = random_split(self.dataset, [len_train, len_test])
        # ToDo:create dataloader by myself to separate train and test for moms with different repetitions
        #  and also trimesters!
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=batch_size)
        return train_loader, test_loader

    def get_model(self):
        if self.mission == "JustValues":
            data_size = self.dataset.get_leaves_number()
            model = JustValuesOnNodes(data_size, self.RECEIVED_PARAMS)
        elif self.mission == "JustGraphStructure":
            data_size = self.dataset.get_vector_size()
            model = JustGraphStructure(data_size, self.RECEIVED_PARAMS)
        elif self.mission == "GraphStructure&Values":
            data_size = self.dataset.get_vector_size()
            model = ValuesAndGraphStructure(data_size, self.RECEIVED_PARAMS)
        return model

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
            final_auc_test = trainer_and_tester.test_auc_vec[-1]
            plt.plot(range(len(trainer_and_tester.train_auc_vec)), trainer_and_tester.train_auc_vec, label=TRAIN_JOB, color='b')
            plt.plot(range(len(trainer_and_tester.test_auc_vec)), trainer_and_tester.test_auc_vec, label=TEST_JOB, color='g')
            plt.ylim((0, 1))
            plt.legend(loc='best')
            plt.savefig(os.path.join(root, f'auc_plot_{date}_last_{round(final_auc_test, 2)}.png'))
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
