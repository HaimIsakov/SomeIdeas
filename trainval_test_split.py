import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
class StratifiedGroupSplit:

    def __init__(self, microbiome_df, tags_df, group, stratify_by, test_size):
        self.microbiome_df = microbiome_df
        self.tags_df = tags_df
        self.group = group
        self.stratify_by = stratify_by
        self.test_size = test_size

    def stratified_group_train_test_split(self):
        # origin : https://stackoverflow.com/questions/56872664/complex-dataset-split-stratifiedgroupshufflesplit
        groups = self.tags_df[self.group].drop_duplicates()
        stratify = self.tags_df.drop_duplicates(self.group)[self.stratify_by].to_numpy()
        groups_train, groups_test = train_test_split(groups, stratify=stratify, test_size=self.test_size)

        samples_train = self.tags_df.loc[lambda d: d[self.group].isin(groups_train)]
        samples_test = self.tags_df.loc[lambda d: d[self.group].isin(groups_test)]

        samples_train.sort_index(inplace=True)
        samples_test.sort_index(inplace=True)
        return samples_train, samples_test

    def split_dataframes(self):
        np.random.seed(SEED)
        train_val_tags_df, test_tags_df = self.stratified_group_train_test_split()

        train_val_microbiome_df = self.microbiome_df.merge(train_val_tags_df, left_index=True, right_index=True)
        del train_val_microbiome_df[self.stratify_by]
        del train_val_microbiome_df[self.group]
        test_microbiome_df = self.microbiome_df.merge(test_tags_df, left_index=True, right_index=True)
        del test_microbiome_df[self.stratify_by]
        del test_microbiome_df[self.group]
        return train_val_microbiome_df, train_val_tags_df, test_microbiome_df, test_tags_df

    def remove_zero_columns_from_train_test(self, train_val_microbiome_df, test_microbiome_df):
        # Find zero columns on train set
        zero_cols_on_train = set(train_val_microbiome_df.columns[(train_val_microbiome_df == 0).all()])
        # Delete zero columns on both train and test set
        train_val_microbiome_df.drop(zero_cols_on_train, inplace=True, axis=1)
        test_microbiome_df.drop(zero_cols_on_train, inplace=True, axis=1)
        return train_val_microbiome_df, test_microbiome_df


if __name__ == "__main__":
    # dataset_name = "gdm"
    # microbiome_df_path = os.path.join("GDM_split_dataset", "final_OTU_merged_Mucositis_Genus_after_mipmlp_eps_1.csv")
    # tags_df_path = os.path.join("GDM_split_dataset", "tag_gdm_file_final.csv")

    # dataset_name = "Cirrhosis_split_dataset"
    # microbiome_df_path = os.path.join("Cirrhosis_split_dataset", "OTU_Cirrhosis_after_mipmlp_Genus_no_viruses.csv")
    # tags_df_path = os.path.join("Cirrhosis_split_dataset", "tag_Cirrhosis_file.csv")

    # dataset_name = "IBD_split_dataset"
    # microbiome_df_path = os.path.join('IBD_split_dataset', 'OTU_IBD_after_mipmlp_Genus.csv')
    # tags_df_path = os.path.join('IBD_split_dataset', 'tag_IBD_file.csv')

    # dataset_name = "bw_split_dataset"
    # microbiome_df_path = os.path.join('bw_split_dataset', 'OTU_Black_vs_White_after_mipmlp_Genus_same_ids.csv')
    # tags_df_path = os.path.join('bw_split_dataset', 'tag_bw_file.csv')

    # dataset_name = "IBD_Chrone_split_dataset"
    # microbiome_df_path = os.path.join("IBD_Chrone_split_dataset", "OTU_IBD_after_mipmlp_Genus.csv")
    # tags_df_path = os.path.join("IBD_Chrone_split_dataset", "tag_IBD_Chrone_file.csv")

    # dataset_name = "Allergy_or_not"
    # microbiome_df_path = os.path.join("Allergy_or_not_split_dataset", "OTU_Allergy_after_mipmlp_Genus_same_ids_new.csv")
    # tags_df_path = os.path.join("Allergy_or_not_split_dataset", "tag_allergy_or_not_file.csv")

    # dataset_name = "Allergy_milk_or_not"
    # microbiome_df_path = os.path.join("Allergy_milk_split_dataset", "OTU_Allergy_after_mipmlp_Genus_same_ids_new.csv")
    # tags_df_path = os.path.join("Allergy_milk_split_dataset", "tag_allergy_milk_or_not_file.csv")

    # dataset_name = "Male_vs_Female"
    # microbiome_df_path = os.path.join("Male_vs_Female_split_dataset", "OTU_Male_vs_Female_after_mipmlp_same_ids.csv")
    # tags_df_path = os.path.join("Male_vs_Female_split_dataset", "tag_male_female_file.csv")

    # dataset_name = "Male_vs_Female_Species"
    # microbiome_df_path = os.path.join("Male_vs_Female_Species_split_dataset", "OTU_Male_vs_Female_Species_after_mipmlp_same_ids.csv")
    # tags_df_path = os.path.join("Male_vs_Female_Species_split_dataset", "tag_male_female_file.csv")

    # dataset_name = "Black_vs_White_Species"
    # microbiome_df_path = os.path.join("Black_vs_White_Species_split_dataset", "OTU_Black_vs_White_after_mipmlp_Species_same_ids.csv")
    # tags_df_path = os.path.join("Black_vs_White_Species_split_dataset", "tag_bw_file.csv")

    # dataset_name = "nugent"
    # microbiome_df_path = os.path.join("split_datasets", "nugent_split_dataset", "OTU_Nugent_after_mipmlp_Genus_same_ids.csv")
    # tags_df_path = os.path.join("split_datasets", "nugent_split_dataset", "tag_nugent_file.csv")

    # dataset_name = "Cirrhosis"
    # microbiome_df_path = os.path.join("split_datasets_new", f"{dataset_name}_split_dataset", "OTU_merged_cirrhosis_after_mipmlp_taxonomy_5_group_sub PCA_epsilon_1_normalizaion_log_After_mean_zeroing.csv")
    # tags_df_path = os.path.join("split_datasets_new", f"{dataset_name}_split_dataset", "tag_Cirrhosis_file.csv")

    # dataset_name = "bw"
    # microbiome_df_path = os.path.join("split_datasets_new", f"{dataset_name}_split_dataset",
    #                                   f"OTU_merged_{dataset_name}_after_mipmlp_taxonomy_7_group_sub PCA_epsilon_1_normalizaion_log_After_mean_zeroing_same_ids.csv")
    # tags_df_path = os.path.join("split_datasets_new", f"{dataset_name}_split_dataset", "tag_bw_file.csv")


    # datasets = ["bw", "Cirrhosis", "IBD", "IBD_Chrone", "male_female", "nugent"]
    datasets = ["nut", "peanut", "milk"]
    for dataset in datasets:
        microbiome_df_path = os.path.join("split_datasets_new", f"{dataset}_split_dataset",
                                          f"OTU_merged_{dataset}_after_mipmlp_taxonomy_7_group_sub PCA_epsilon_1_"
                                          "normalization_log_After_mean_zeroing_after_arrangement.csv")
        tags_df_path = os.path.join("split_datasets_new", f"{dataset}_split_dataset",  f"tag_{dataset}_file.csv")
        microbiome_df = pd.read_csv(microbiome_df_path, index_col='ID')
        tags_df = pd.read_csv(tags_df_path, index_col='ID')
        group = "Group"
        stratify_by = "Tag"
        test_size = 0.2
        split_data = StratifiedGroupSplit(microbiome_df, tags_df, group, stratify_by, test_size)

        train_val_microbiome_df, train_val_tags_df, test_microbiome_df, test_tags_df = split_data.split_dataframes()
        train_val_microbiome_df, test_microbiome_df = split_data.remove_zero_columns_from_train_test(train_val_microbiome_df, test_microbiome_df)
        train_val_microbiome_df.to_csv(os.path.join("split_datasets_new", f"{dataset}_split_dataset", f"train_val_set_{dataset}_microbiome.csv"))
        train_val_tags_df.to_csv(os.path.join("split_datasets_new", f"{dataset}_split_dataset", f"train_val_set_{dataset}_tags.csv"))
        test_microbiome_df.to_csv(os.path.join("split_datasets_new", f"{dataset}_split_dataset", f"test_set_{dataset}_microbiome.csv"))
        test_tags_df.to_csv(os.path.join("split_datasets_new", f"{dataset}_split_dataset", f"test_set_{dataset}_tags.csv"))


    # microbiome_df = pd.read_csv(microbiome_df_path, index_col='ID')
    # tags_df = pd.read_csv(tags_df_path, index_col='ID')
    # group = "Group"
    # stratify_by = "Tag"
    # test_size = 0.2
    # split_data = StratifiedGroupSplit(microbiome_df, tags_df, group, stratify_by, test_size)
    #
    # train_val_microbiome_df, train_val_tags_df, test_microbiome_df, test_tags_df = split_data.split_dataframes()
    # train_val_microbiome_df, test_microbiome_df = split_data.remove_zero_columns_from_train_test(
    #     train_val_microbiome_df, test_microbiome_df)
    # train_val_microbiome_df.to_csv(f"train_val_set_{dataset_name}_microbiome.csv")
    # train_val_tags_df.to_csv(f"train_val_set_{dataset_name}_tags.csv")
    # test_microbiome_df.to_csv(f"test_set_{dataset_name}_microbiome.csv")
    # test_tags_df.to_csv(f"test_set_{dataset_name}_tags.csv")
