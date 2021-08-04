import json
import os
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
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
    def __init__(self, RECEIVED_PARAMS, number_of_runs, device, train_val_dataset, test_dataset,
                 result_directory_name, nni_flag=False):
        # self.mission = mission
        self.RECEIVED_PARAMS = RECEIVED_PARAMS
        self.device = device
        self.train_val_dataset = train_val_dataset
        self.test_dataset = test_dataset
        self.result_directory_name = result_directory_name
        self.number_of_runs = number_of_runs
        self.nni_flag = nni_flag
        # self.node_order = self.dataset.node_order

    def train_group_k_cross_validation(self, k=5):
        train_frac = self.RECEIVED_PARAMS['train_frac']
        val_frac = self.RECEIVED_PARAMS['test_frac']

        val_metric, test_metric = [], []
        gss_train_val = GroupShuffleSplit(n_splits=k, train_size=0.80)
        run = 0
        for train_idx, val_idx in gss_train_val.split(self.train_val_dataset, groups=self.train_val_dataset.get_all_groups()):
            print(f"Run {run}")
            if run == 0 and not self.nni_flag:
                date, directory_root = self.create_directory_to_save_results()

            train_loader, val_loader, test_loader = self.create_data_loaders(train_idx, val_idx)
            model = self.get_model().to(self.device)
            trainer_and_tester = TrainTestValOneTime(model, self.RECEIVED_PARAMS, train_loader, val_loader, test_loader,
                                                     self.device)
            early_stopping_results = trainer_and_tester.train()
            min_val_train_auc = max(min(early_stopping_results['val_auc'], early_stopping_results['train_auc']), 0.5)
            print("Minimum Validation and Train Auc", min_val_train_auc)
            val_metric.append(min_val_train_auc)  # the minimum between the aucs between train set and validation set
            test_metric.append(early_stopping_results['test_auc'])
            if not self.nni_flag:
                os.mkdir(os.path.join(directory_root, f"Run{run}"))
                root = os.path.join(directory_root, f"Run{run}")
                f = open(os.path.join(directory_root, f"Validation_Auc_{trainer_and_tester.val_auc:.9f}.txt"), 'w')
                f.close()
                self.plot_acc_loss_auc(root, date, trainer_and_tester)
            run += 1
        return val_metric, test_metric

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

    def create_data_loaders(self, train_idx, val_idx):
        batch_size = self.RECEIVED_PARAMS['batch_size']
        # Datasets
        train_data = torch.utils.data.Subset(self.train_val_dataset, train_idx)
        val_data = torch.utils.data.Subset(self.train_val_dataset, val_idx)
        # Dataloader
        train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        # Notice that the test data is loaded as one unit.
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=len(self.test_dataset), shuffle=False)
        return train_loader, val_loader, test_loader

    def get_model(self):
        if self.train_val_dataset.mission == "JustValues":
            data_size = self.train_val_dataset.get_leaves_number()
            model = JustValuesOnNodes(data_size, self.RECEIVED_PARAMS)
        elif self.train_val_dataset.mission == "JustGraphStructure":
            data_size = self.train_val_dataset.get_vector_size()
            model = JustGraphStructure(data_size, self.RECEIVED_PARAMS, self.device)
        elif self.train_val_dataset.mission == "GraphStructure&Values":
            data_size = self.train_val_dataset.get_vector_size()
            model = ValuesAndGraphStructure(data_size, self.RECEIVED_PARAMS, self.device)
        return model

    def plot_measurement(self, root, date, trainer_and_tester, measurement=LOSS_PLOT):
        if measurement == LOSS_PLOT:
            plt.plot(range(len(trainer_and_tester.train_loss_vec)), trainer_and_tester.train_loss_vec, label=TRAIN_JOB, color='b')
            plt.plot(range(len(trainer_and_tester.val_loss_vec)), trainer_and_tester.val_loss_vec, label=TEST_JOB, color='g')
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
            plt.plot(range(len(trainer_and_tester.val_auc_vec)), trainer_and_tester.val_auc_vec, label=TEST_JOB, color='g')
            plt.ylim((0, 1))
            plt.legend(loc='best')
            plt.savefig(os.path.join(root, f'auc_plot_{date}_last_{round(trainer_and_tester.val_auc, 2)}.png'))
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
