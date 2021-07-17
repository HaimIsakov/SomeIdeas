import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from arrange_gdm_dataset import ArrangeGDMDataset


def stratified_group_train_test_split(samples: pd.DataFrame, group: str, stratify_by: str, test_size: float):
    groups = samples[group].drop_duplicates()
    stratify = samples.drop_duplicates(group)[stratify_by].to_numpy()
    groups_train, groups_test = train_test_split(groups, stratify=stratify, test_size=test_size)

    samples_train = samples.loc[lambda d: d[group].isin(groups_train)]
    samples_test = samples.loc[lambda d: d[group].isin(groups_test)]

    samples_train.sort_index(inplace=True)
    samples_test.sort_index(inplace=True)
    return samples_train, samples_test

def create_gdm_dataset(data_file_path, tag_file_path, mission, category):
    gdm_dataset = ArrangeGDMDataset(data_file_path, tag_file_path, mission, category)
    return gdm_dataset

def split_gdm_dataset():
    directory_name = "JustValues"
    data_file_path = os.path.join(directory_name, 'Data', 'OTU_merged_Mucositis_Genus_after_mipmlp_eps_1.csv')
    tag_file_path = os.path.join(directory_name, 'Data', "tag_gdm_file.csv")
    result_directory_name = os.path.join(directory_name, "Result_After_Proposal")
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    gdm_dataset = create_gdm_dataset(data_file_path, tag_file_path, "", "just_A")

    np.random.seed(42)
    train_val_tags_df, test_tags_df = stratified_group_train_test_split(gdm_dataset._tags, "Code", "Tag", 0.2)

    train_val_microbiome_df = gdm_dataset._microbiome_df.merge(train_val_tags_df, left_index=True, right_index=True)
    del train_val_microbiome_df["Tag"]
    del train_val_microbiome_df["Code"]
    test_microbiome_df = gdm_dataset._microbiome_df.merge(test_tags_df, left_index=True, right_index=True)
    del test_microbiome_df["Tag"]
    del test_microbiome_df["Code"]
    train_val_microbiome_df.to_csv("train_val_set_gdm_microbiome.csv")
    train_val_tags_df.to_csv("train_val_set_gdm_tags.csv")
    test_microbiome_df.to_csv("test_set_gdm_microbiome.csv")
    test_tags_df.to_csv("test_set_gdm_tags.csv")

split_gdm_dataset()
