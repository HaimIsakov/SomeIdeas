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


def StratifiedGroupShuffleSplit(df_main):
    df_main = df_main.reindex(np.random.permutation(df_main.index))  # shuffle dataset

    # create empty train, val and test datasets
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()

    hparam_mse_wgt = 1 # must be between 0 and 1
    assert(0 <= hparam_mse_wgt <= 1)
    train_proportion = 0.8 # must be between 0 and 1
    assert(0 <= train_proportion <= 1)
    val_test_proportion = 0.2

    subject_grouped_df_main = df_main.groupby(['Code'], sort=False, as_index=False)
    category_grouped_df_main = df_main.groupby('Tag').count()[['Code']]/len(df_main)*100

    def calc_mse_loss(df):
        grouped_df = df.groupby('Tag').count()[['Code']]/len(df)*100
        df_temp = category_grouped_df_main.join(grouped_df, on='Tag', how='left', lsuffix='_main')
        df_temp.fillna(0, inplace=True)
        df_temp['diff'] = (df_temp['Code_main'] - df_temp['Code'])**2
        mse_loss = np.mean(df_temp['diff'])
        return mse_loss

    i = 0
    for _, group in subject_grouped_df_main:

        if (i < 2):
            if (i == 0):
                df_train = df_train.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
            elif (i == 1):
                df_val = df_val.append(pd.DataFrame(group), ignore_index=True)
                i += 1
                continue
            # else:
            #     df_test = df_test.append(pd.DataFrame(group), ignore_index=True)
            #     i += 1
            #     continue

        mse_loss_diff_train = calc_mse_loss(df_train) - calc_mse_loss(df_train.append(pd.DataFrame(group), ignore_index=True))
        mse_loss_diff_val = calc_mse_loss(df_val) - calc_mse_loss(df_val.append(pd.DataFrame(group), ignore_index=True))
        # mse_loss_diff_test = calc_mse_loss(df_test) - calc_mse_loss(df_test.append(pd.DataFrame(group), ignore_index=True))

        total_records = len(df_train) + len(df_val)

        len_diff_train = (train_proportion - (len(df_train)/total_records))
        len_diff_val = (val_test_proportion - (len(df_val)/total_records))
        # len_diff_test = (val_test_proportion - (len(df_test)/total_records))

        len_loss_diff_train = len_diff_train * abs(len_diff_train)
        len_loss_diff_val = len_diff_val * abs(len_diff_val)
        # len_loss_diff_test = len_diff_test * abs(len_diff_test)

        loss_train = (hparam_mse_wgt * mse_loss_diff_train) + ((1-hparam_mse_wgt) * len_loss_diff_train)
        loss_val = (hparam_mse_wgt * mse_loss_diff_val) + ((1-hparam_mse_wgt) * len_loss_diff_val)
        # loss_test = (hparam_mse_wgt * mse_loss_diff_test) + ((1-hparam_mse_wgt) * len_loss_diff_test)

        if (max(loss_train,loss_val) == loss_train):
            df_train = df_train.append(pd.DataFrame(group), ignore_index=True)
        elif (max(loss_train,loss_val) == loss_val):
            df_val = df_val.append(pd.DataFrame(group), ignore_index=True)
        # else:
        #     df_test = df_test.append(pd.DataFrame(group), ignore_index=True)

        # print("Group " + str(i) + ". loss_train: " + str(loss_train) + " | " + "loss_val: " + str(loss_val) + " | ")
        i += 1

    return df_train.sort_index(inplace=False), df_val.sort_index(inplace=False)



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
