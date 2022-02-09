import os
import pandas as pd
from sklearn.model_selection import train_test_split


def divide_to_train_test_dirs(train_inx, test_inx, sample_df, label_df):
    trainval_sample_df = sample_df.loc[train_inx]
    test_sample_df = sample_df.loc[test_inx]

    trainval_sample_df.to_csv("train_val_set_gdm.csv")
    test_sample_df.to_csv("test_set_gdm.csv")

    trainval_label_df = label_df.loc[train_inx]
    test_label_df = label_df.loc[test_inx]

    trainval_label_df.to_csv("train_val_set_gdm_tags.csv")
    test_label_df.to_csv("test_set_gdm_tags.csv")


if __name__ == "__main__":
    sample_df = pd.read_csv("week_14_new.csv", index_col=0)
    label_df = pd.read_csv("gdm.csv", index_col=0)
    subject_list = list(sample_df.index)
    trainval_inx, test_inx = train_test_split(subject_list, test_size=0.2, random_state=42)
    divide_to_train_test_dirs(trainval_inx, test_inx, sample_df, label_df)
