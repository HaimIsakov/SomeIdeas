import json
import torch
from JustValues.Data.gdm_dataset import GDMDataset
from torch.utils.data import DataLoader, random_split
from JustValues.Models.fc_binary_classification import _train, JustValuesOnNodes
import matplotlib.pyplot as plt
import os


def load_params_file(file_path):
    RECEIVED_PARAMS = json.load(open(file_path, 'r'))
    return RECEIVED_PARAMS


def create_dataset(data_file_path, tag_file_path):
    gdm_dataset = GDMDataset(data_file_path, tag_file_path)
    return gdm_dataset


def train_model(gdm_dataset, RECEIVED_PARAMS):
    batch_size = RECEIVED_PARAMS['batch_size']
    samples_len = len(gdm_dataset)
    len_train = int(samples_len * RECEIVED_PARAMS['train_frac'])
    len_test = samples_len - len_train
    train, test = random_split(gdm_dataset, [len_train, len_test])
    # set train loader
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    # set test loader
    test_loader = DataLoader(test, batch_size=batch_size)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_size = gdm_dataset.get_vector_size()
    count_zeros, count_ones = gdm_dataset.count_each_class()
    loss_weights = [1 / count_zeros, 1 / count_ones]

    model = JustValuesOnNodes(data_size, RECEIVED_PARAMS)
    model = model.to(device)

    train_loss_vec, train_auc_vec, test_loss_vec, test_auc_vec = \
        _train(model, RECEIVED_PARAMS, train_loader, test_loader, device, loss_weights)


if __name__ == '__main__':
    data_file_path = os.path.join('Data', 'OTU_merged_Mucositis_Genus_after_mipmlp_new.csv')
    tag_file_path = os.path.join('Data', "tag_gdm_file.csv")
    gdm_dataset = create_dataset(data_file_path, tag_file_path)
    values, label = gdm_dataset[0]
    print()
