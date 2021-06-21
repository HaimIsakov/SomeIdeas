import json
from datetime import datetime
import os


def load_params_file(file_path):
    RECEIVED_PARAMS = json.load(open(file_path, 'r'))
    return RECEIVED_PARAMS


def create_dataset(data_file_path, tag_file_path):
    gdm_dataset = GDMDataset(data_file_path, tag_file_path)
    return gdm_dataset


if __name__ == '__main__':
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    data_file_path = os.path.join('Data', 'OTU_merged_Mucositis_Genus_after_mipmlp_eps_1.csv')
    # data_file_path = os.path.join('Data', 'taxonomy_gdm_file.csv')
    tag_file_path = os.path.join('Data', "tag_gdm_file.csv")
    params_file_path = os.path.join('Models', "graph_structure_params_file.json")
    gdm_dataset = create_dataset(data_file_path, tag_file_path)
    RECEIVED_PARAMS = load_params_file(params_file_path)
    train_model(gdm_dataset, RECEIVED_PARAMS)
    # root = os.path.join("Results_Gdm_Genus")
    # plot_acc_loss_auc(root, date, train_loss_vec, train_auc_vec, test_loss_vec, test_auc_vec, params_file_path)

    print()
