import os
import pandas as pd


def save_microbiome_file(tag_df, microbiome_df):
    # Take only samples that have tags
    microbiome_df = microbiome_df.merge(tag_df, left_index=True, right_index=True)
    del microbiome_df["Tag"]
    del microbiome_df["Group"]
    # Delete zero columns
    microbiome_df = microbiome_df.loc[:, (microbiome_df != 0).any(axis=0)]
    return microbiome_df


# datasets = ["bw", "Cirrhosis", "IBD", "IBD_Chrone", "male_female", "nugent"]
datasets = ["nut", "peanut", "milk"]
for dataset_name in datasets:
    print(dataset_name)
    microbiome_df_path = os.path.join("..", f"{dataset_name}_split_dataset", f"OTU_merged_{dataset_name}_after_mipmlp_taxonomy_7_group_sub PCA_epsilon_1_normalizaion_log_After_mean_zeroing.csv")

    tags_df_path = os.path.join("..", f"{dataset_name}_split_dataset",  f"tag_{dataset_name}_file.csv")

    microbiome_df = pd.read_csv(microbiome_df_path, index_col='ID')
    tags_df = pd.read_csv(tags_df_path, index_col='ID')

    new_microbiome_df = save_microbiome_file(tags_df, microbiome_df)
    new_microbiome_df_name = os.path.join("..", f"{dataset_name}_split_dataset", f"OTU_merged_{dataset_name}_after_mipmlp_taxonomy_7_group_sub PCA_epsilon_1_" \
                             f"normalization_log_After_mean_zeroing_after_arrangement.csv")
    new_microbiome_df.to_csv(new_microbiome_df_name)
