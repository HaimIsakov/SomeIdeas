import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gdm_dataset import GDMDataset


def create_dataset(data_file_path, tag_file_path, mission, category):
    gdm_dataset = GDMDataset(data_file_path, tag_file_path, mission, category)
    return gdm_dataset


def stratified_group_train_test_split(samples: pd.DataFrame, group: str, stratify_by: str, test_size: float):
    groups = samples[group].drop_duplicates()
    stratify = samples.drop_duplicates(group)[stratify_by].to_numpy()
    groups_train, groups_test = train_test_split(groups, stratify=stratify, test_size=test_size)

    samples_train = samples.loc[lambda d: d[group].isin(groups_train)]
    samples_test = samples.loc[lambda d: d[group].isin(groups_test)]

    samples_train.sort_index(inplace=True)
    samples_test.sort_index(inplace=True)
    return samples_train, samples_test


directory_name = "JustValues"
mission = 'JustValues'
params_file_path = os.path.join(directory_name, 'Models', "just_values_on_nodes_params_file.json")

data_file_path = os.path.join(directory_name, 'Data', 'OTU_merged_Mucositis_Genus_after_mipmlp_eps_1.csv')
tag_file_path = os.path.join(directory_name, 'Data', "tag_gdm_file.csv")
result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

gdm_dataset = create_dataset(data_file_path, tag_file_path, mission, "symmetric_adjacency")
df_tag = gdm_dataset._tags
for i in range(10):
    # df_train, df_val = StratifiedGroupShuffleSplit(df_tag)
    df_train, df_val = stratified_group_train_test_split(df_tag, "Code", "Tag", 0.2)
    print("train \n", df_train['Tag'].value_counts())
    print("test \n", df_val['Tag'].value_counts())
print()
