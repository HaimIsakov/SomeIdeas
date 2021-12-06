import pandas as pd
import os

f1 = "split_datasets/Male_vs_Female_split_dataset/train_val_set_Male_vs_Female_microbiome.csv"
f2 = "split_datasets/Male_vs_Female_split_dataset/test_set_Male_vs_Female_microbiome.csv"

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)

# print(df1.columns[(df1 == 0).all()])
# print(df2.columns[(df2 == 0).all()])

zero_cols1 = set(df1.columns[(df1 == 0).all()])
zero_cols2 = set(df2.columns[(df2 == 0).all()])

intersec = list(zero_cols1 & zero_cols2)

# print()
df1.drop(intersec, inplace=True, axis=1)
df2.drop(intersec, inplace=True, axis=1)

df1.to_csv("split_datasets/Male_vs_Female_split_dataset/train_val_set_Male_vs_Female_no_zero_cols_microbiome.csv")
df2.to_csv("split_datasets/Male_vs_Female_split_dataset/test_set_Male_vs_Female_no_zero_cols_microbiome.csv")
