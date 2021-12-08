import pandas as pd
import os


def remove_zero_cols_from_train_files(datasets_list):
    for dataset in datasets_list:
        print(dataset)
        train_file_path = os.path.join("split_datasets", f"{dataset}_split_dataset", f"train_val_set_{dataset}_microbiome.csv")
        test_file_path = os.path.join("split_datasets", f"{dataset}_split_dataset", f"test_set_{dataset}_microbiome.csv")

        train_microbiome_df = pd.read_csv(train_file_path, index_col='ID')
        test_microbiome_df = pd.read_csv(test_file_path, index_col='ID')

        zero_columns_train = train_microbiome_df.columns[(train_microbiome_df == 0).all()]
        print("Number of zero columns", len(zero_columns_train))
        train_microbiome_df.drop(zero_columns_train, axis=1, inplace=True)
        test_microbiome_df.drop(zero_columns_train, axis=1, inplace=True)

        # train_microbiome_df.to_csv(f"train_val_set_{dataset}_microbiome_no_zero_cols.csv")
        print("Shape of train_microbiome_df after remove of zero columns", train_microbiome_df.shape[1])
        # test_microbiome_df.to_csv(f"test_set_{dataset}_microbiome_no_zero_cols.csv")


if __name__ == "__main__":
    datasets_list = ["milk_no_controls", "nut", "peanut", "bw",
                     "Cirrhosis", "IBD_Chrone", "IBD", "Male_vs_Female", "nugent"]
    remove_zero_cols_from_train_files(datasets_list)
