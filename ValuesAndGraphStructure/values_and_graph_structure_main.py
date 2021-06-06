import json
from datetime import datetime
from shutil import copyfile
import torch
from JustValues.Data.gdm_dataset import GDMDataset
from torch.utils.data import DataLoader, random_split
from ValuesAndGraphStructure.Models.values_and_graph_structure import ValuesAndGraphStructure, _train
import matplotlib.pyplot as plt
import os
import numpy as np

LOSS_PLOT = 'loss'
ACCURACY_PLOT = 'acc'
AUC_PLOT = 'auc'
TRAIN_JOB = 'train'
TEST_JOB = 'test'


def load_params_file(file_path):
    RECEIVED_PARAMS = json.load(open(file_path, 'r'))
    return RECEIVED_PARAMS


def create_dataset(data_file_path, tag_file_path):
    gdm_dataset = GDMDataset(data_file_path, tag_file_path)
    return gdm_dataset

def count_pos_neg_train_set(train_loader):
    count_ones = 0
    count_zeros = 0
    for batch_index, (A, data, target) in enumerate(train_loader):
        for i in target:
            if i.item() == 1:
                count_ones += 1
            if i.item() == 0:
                count_zeros += 1
    return count_zeros, count_ones


def train_model(gdm_dataset, RECEIVED_PARAMS):
    params_file_path = os.path.join('Models', "values_and_graph_structure_on_nodes_params_file.json")

    batch_size = RECEIVED_PARAMS['batch_size']
    number_of_runs = RECEIVED_PARAMS['runs_number']

    samples_len = len(gdm_dataset)
    len_train = int(samples_len * RECEIVED_PARAMS['train_frac'])
    len_test = samples_len - len_train

    final_results_vec = []
    for i in range(number_of_runs):
        if i == 0:
            root = os.path.join("Results_Gdm_Genus")
            date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
            root = os.path.join(root, f'Values_and_graph_structure_on_nodes_model_{date}')
            os.mkdir(root)
        train, test = random_split(gdm_dataset, [len_train, len_test])
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=batch_size)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        data_size = gdm_dataset.get_vector_size()
        count_zeros, count_ones = count_pos_neg_train_set(train_loader)
        loss_weights = [1 / count_zeros, 1 / count_ones]

        model = ValuesAndGraphStructure(data_size, RECEIVED_PARAMS)
        model = model.to(device)
        train_loss_vec, train_auc_vec, test_loss_vec, test_auc_vec = \
            _train(model, RECEIVED_PARAMS, train_loader, test_loader, loss_weights, device)
        final_results_vec.append(test_auc_vec[-1])
        os.mkdir(f"Run{i}")
        root = os.path.join(root, f"Run{i}")
        plot_acc_loss_auc(root, train_loss_vec, train_auc_vec, test_loss_vec, test_auc_vec, params_file_path)
        print(f"Average auc on test sets {np.mean(final_results_vec)}")


def plot_measurement(root, date, train_loss_vec, train_auc_vec, test_loss_vec, test_auc_vec, measurement=LOSS_PLOT):
    if measurement == LOSS_PLOT:
        plt.plot(range(len(train_loss_vec)), train_loss_vec, label=TRAIN_JOB, color='b')
        plt.plot(range(len(test_loss_vec)), test_loss_vec, label=TEST_JOB, color='g')
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
        final_auc_test = test_auc_vec[-1]
        plt.plot(range(len(train_auc_vec)), train_auc_vec, label=TRAIN_JOB, color='b')
        plt.plot(range(len(test_auc_vec)), test_auc_vec, label=TEST_JOB, color='g')
        plt.ylim((0, 1))
        plt.legend(loc='best')
        plt.savefig(os.path.join(root, f'auc_plot_{date}_max_{round(final_auc_test, 2)}.png'))
        plt.clf()


def plot_acc_loss_auc(root, train_loss_vec, train_auc_vec, test_loss_vec, test_auc_vec, params_file):
    # root = os.path.join(root, f'Values_and_graph_structure_on_nodes_model_{date}')
    # os.mkdir(root)
    copyfile(params_file, os.path.join(root, "params_file.json"))
    plot_measurement(root, date, train_loss_vec, train_auc_vec, test_loss_vec, test_auc_vec, LOSS_PLOT)
    # self.plot_measurement(root, date, ACCURACY_PLOT)
    plot_measurement(root, date, train_loss_vec, train_auc_vec, test_loss_vec, test_auc_vec, AUC_PLOT)


if __name__ == '__main__':
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    data_file_path = os.path.join('Data', 'OTU_merged_Mucositis_Genus_after_mipmlp_eps_1.csv')
    # data_file_path = os.path.join('Data', 'taxonomy_gdm_file.csv')
    tag_file_path = os.path.join('Data', "tag_gdm_file.csv")
    params_file_path = os.path.join('Models', "values_and_graph_structure_on_nodes_params_file.json")
    gdm_dataset = create_dataset(data_file_path, tag_file_path)
    RECEIVED_PARAMS = load_params_file(params_file_path)
    train_model(gdm_dataset, RECEIVED_PARAMS)
    # root = os.path.join("Results_Gdm_Genus")
    # plot_acc_loss_auc(root, date, train_loss_vec, train_auc_vec, test_loss_vec, test_auc_vec, params_file_path)

    print()
